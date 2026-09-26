"""VL-JEPA (reimplementación abierta `open-vljepa`, checkpoint MSRVTT Stage B).

Arquitectura (Chen et al., 2025, arXiv:2512.10942; reimpl. de J. Baek):
    X-Encoder  V-JEPA 2 ViT-L (congelado)                     imagen -> tokens visuales
    Predictor  últimas 8 capas de Llama-3.2-1B, no causal     (tokens visuales, consulta) -> ŝ_Y ∈ R^1536
    Y-Encoder  EmbeddingGemma-300M + proyección               texto -> s_Y ∈ R^1536

El modelo NO genera tokens: predice un embedding continuo del texto objetivo. Para producir texto
legible se usa un "decodificador" por recuperación, que es la opción ligera que el propio paradigma
permite: se codifica con el Y-Encoder un banco de descripciones del split de *entrenamiento*
(nunca de test) y se devuelve la más cercana a ŝ_Y en coseno. Además, con el mismo banco se
construyen prototipos por valor de atributo (media normalizada de los embeddings de las
descripciones con ese valor), lo que da una predicción por atributo sin pasar por texto.

Nota de dominio: el checkpoint procesa *video* (tubelets temporales). Una imagen se trata como un
video estático repitiendo el fotograma `num_frames` veces (8, como en Stage B).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image

from ..data.m3di import select_discrete_attributes
from ..data.manifest import load_manifest
from ..paths import CACHE, resolve
from .base import Description, Describer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LLAMA32_PAD = "<|finetune_right_pad_id|>"


def _ensure_pad_token(tok):
    """Igual que open-vljepa: Llama-3.2 no trae pad; usar su token reservado y no el EOS."""
    if tok.pad_token is not None:
        return tok
    vocab = tok.get_vocab()
    if LLAMA32_PAD in vocab:
        tok.pad_token = LLAMA32_PAD
    else:
        tok.pad_token = tok.eos_token
    return tok


class VLJEPADescriber(Describer):
    family = "vljepa"
    generative = False

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bank_texts: list[str] = []
        self.bank_emb: torch.Tensor | None = None
        self.prototypes: dict[str, tuple[list, torch.Tensor]] = {}

    # ---------------------------------------------------------------- carga
    def _checkpoint_path(self) -> Path:
        c = self.cfg
        if c.get("ckpt_path"):
            return resolve(c.ckpt_path)
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=c.hf_id, filename=c.ckpt_file, revision=c.get("revision")))

    def load(self) -> None:
        from torchvision import transforms as T
        from transformers import AutoTokenizer

        c = self.cfg
        repo = resolve(c.repo_path)
        if not (repo / "openvljepa").is_dir():
            raise FileNotFoundError(
                f"No se encontró open-vljepa en {repo}. Ejecuta: git submodule update --init --recursive"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from openvljepa.models.vljepa import OpenVLJEPA  # noqa: E402  (import tras ajustar sys.path)

        self.ckpt_path = self._checkpoint_path()
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.arch = ckpt["config"]
        # El checkpoint apunta a meta-llama/Llama-3.2-1B, repo con acceso restringido por Meta.
        # `llama_name` permite usar una réplica sin restricción (mismos pesos y tokenizador): el
        # predictor solo toma de ahí la arquitectura y el tokenizador de consultas, porque sus pesos
        # se sobrescriben con los del checkpoint al hacer load_state_dict.
        if c.get("llama_name"):
            self.arch["predictor"]["llama_name"] = c.llama_name

        model = OpenVLJEPA(self.arch["encoder"], self.arch["y_encoder"], self.arch["predictor"],
                           torch_dtype=torch.bfloat16)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        bad = [k for k in missing if not k.startswith("x_encoder.model.")]
        if bad or unexpected:
            raise RuntimeError(f"Checkpoint incompatible. missing={bad[:5]} unexpected={unexpected[:5]}")
        del ckpt

        model = model.to(dtype=torch.bfloat16).eval()
        if c.quant == "nf4":
            from .quant import swap_linear_to_nf4

            n = swap_linear_to_nf4(model.predictor.layers, self.device)
            print(f"[vljepa] NF4 aplicado a {n} capas lineales del predictor")
        elif c.quant != "bf16":
            raise ValueError("VL-JEPA admite quant = nf4 | bf16")
        self.model = model.to(self.device)
        if c.get("adapter_path"):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, c.adapter_path).eval()

        data_cfg = self.arch.get("data", {})
        self.max_query_len = int(c.get("max_query_len") or data_cfg.get("max_query_len", 64))
        self.max_caption_len = int(c.get("max_caption_len") or data_cfg.get("max_caption_len", 64))
        self.num_frames = int(c.get("num_frames") or data_cfg.get("num_frames", 8))
        size = int(c.get("image_size") or data_cfg.get("image_size", 256))

        self.query_tok = _ensure_pad_token(AutoTokenizer.from_pretrained(self.arch["predictor"]["llama_name"]))
        self.target_tok = _ensure_pad_token(AutoTokenizer.from_pretrained(self.arch["y_encoder"]["model_name"]))
        self.transform = T.Compose([
            T.Resize(size, antialias=True), T.CenterCrop(size), T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._build_bank()

    # ------------------------------------------------------------- banco
    def _tok(self, tok, texts: list[str], max_len: int):
        enc = tok(texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        """Embeddings normalizados del Y-Encoder (espacio compartido de 1536 D)."""
        out = []
        for i in range(0, len(texts), batch_size):
            ids, mask = self._tok(self.target_tok, texts[i:i + batch_size], self.max_caption_len)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                e = self.model.y_encoder(ids, mask)
            out.append(F.normalize(e.float(), dim=-1).half())
        return torch.cat(out)

    def _bank_key(self) -> str:
        b = self.cfg.bank
        st = self.ckpt_path.stat()
        ident = (f"{self.ckpt_path.name}|{st.st_size}|{b.split}|{b.max_captions}|{b.seed}|"
                 f"{self.max_caption_len}|{self.cfg.get('adapter_path')}")
        return hashlib.sha1(ident.encode()).hexdigest()[:12]

    def _build_bank(self) -> None:
        b = self.cfg.bank
        rows = load_manifest(b.split, limit=b.max_captions, subset_seed=b.seed)
        cache = CACHE / f"vljepa_bank_{self._bank_key()}.pt"
        if cache.exists():
            blob = torch.load(cache, map_location="cpu", weights_only=False)
            texts, emb = blob["texts"], blob["emb"]
        else:
            texts = sorted({r["caption"] for r in rows})
            print(f"[vljepa] codificando banco: {len(texts)} descripciones únicas de '{b.split}'")
            emb = self.encode_texts(texts).cpu()
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"texts": texts, "emb": emb}, cache)
        self.bank_texts = texts
        self.bank_emb = emb.to(self.device)

        # prototipos por valor de atributo (factores discretos del texto)
        index = {t: i for i, t in enumerate(texts)}
        row_idx = torch.tensor([index[r["caption"]] for r in rows])
        for attr in self._bank_attributes(rows):
            vals = np.array([r["latents_text"][attr] for r in rows])
            uniq = sorted(set(vals.tolist()))
            protos = torch.stack([
                F.normalize(emb[row_idx[torch.from_numpy(vals == v)]].float().mean(0), dim=-1) for v in uniq
            ])
            self.prototypes[attr] = (uniq, protos.half().to(self.device))

    def _bank_attributes(self, rows: list[dict]) -> list[str]:
        """`bank.attributes: auto` -> latentes de texto discretos según el criterio compartido con la
        evaluación (vlmfid.data.select_discrete_attributes). Una lista explícita se respeta, pero los
        atributos ausentes del manifiesto se omiten con aviso en vez de fallar."""
        wanted = self.cfg.bank.attributes
        keys = list(rows[0]["latents_text"]) if rows else []
        if wanted in (None, "auto"):
            out = select_discrete_attributes({k: [r["latents_text"][k] for r in rows] for k in keys})
            print(f"[vljepa] atributos detectados para prototipos: {out}")
            return out
        missing = [a for a in wanted if a not in keys]
        if missing:
            print(f"[vljepa] aviso: atributos ausentes en el manifiesto, se omiten: {missing}")
        return [a for a in wanted if a in keys]

    # -------------------------------------------------------- inferencia
    @torch.inference_mode()
    def predict_embeddings(self, images: list[Image.Image], prompts: list[str]) -> torch.Tensor:
        frames = torch.stack([self.transform(im) for im in images])                    # (B,3,H,W)
        pv = frames.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)                  # (B,T,3,H,W)
        pv = pv.to(self.device, dtype=torch.bfloat16)
        q = [self.cfg.get("query_override") or p for p in prompts]
        ids, mask = self._tok(self.query_tok, q, self.max_query_len)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            pred = self.model(pv, ids, mask)
        return F.normalize(pred.float(), dim=-1)

    def describe(self, images: list[Image.Image], prompts: list[str]) -> list[Description]:
        pred = self.predict_embeddings(images, prompts)
        k = int(self.cfg.get("topk", 3))
        sims = pred.half() @ self.bank_emb.T
        top_s, top_i = sims.topk(k, dim=-1)

        attr_pred, attr_margin = [{} for _ in images], [{} for _ in images]
        for attr, (vals, protos) in self.prototypes.items():
            s = (pred.half() @ protos.T).float()
            best = s.topk(min(2, len(vals)), dim=-1)
            for j in range(len(images)):
                attr_pred[j][attr] = vals[best.indices[j, 0].item()]
                if best.values.shape[1] > 1:
                    attr_margin[j][attr] = round((best.values[j, 0] - best.values[j, 1]).item(), 4)

        outs = []
        for j in range(len(images)):
            topk = [[self.bank_texts[i], round(s, 4)] for i, s in zip(top_i[j].tolist(), top_s[j].float().tolist())]
            outs.append(Description(
                text=topk[0][0],
                extra={"topk": topk, "attr_pred_proto": attr_pred[j], "attr_margin_proto": attr_margin[j]},
                embedding=pred[j].cpu().numpy().astype(np.float16) if self.cfg.get("return_embeddings") else None,
            ))
        return outs

    def lora_target_modules(self) -> str:
        return r"predictor\.layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

    def info(self) -> dict:
        out = super().info()
        out.update({"ckpt": str(self.ckpt_path), "bank_size": len(self.bank_texts),
                    "num_frames": self.num_frames, "attributes": list(self.prototypes)})
        return out
