"""VL-JEPA (reimplementación abierta `open-vljepa`, checkpoint MSRVTT Stage B).

Arquitectura (Chen et al., 2025, arXiv:2512.10942, §3.1; reimpl. de J. Baek):
    X-Encoder  V-JEPA 2 ViT-L (congelado)                     imagen -> tokens visuales S_V
    Predictor  últimas 8 capas de Llama-3.2-1B, no causal     (S_V, consulta) -> Ŝ_Y ∈ R^1536
    Y-Encoder  EmbeddingGemma-300M + proyección               texto -> S_Y ∈ R^1536

El modelo NO genera tokens: predice el embedding del texto objetivo. Para obtener texto, Ŝ_Y pasa
por un Y-Decoder (docs/metodologia.md §2.4), que se entrena aparte y se adjunta con
`attach_decoder`. Sin decodificador, `describe` falla: no hay forma de producir texto. El paper no
especifica ese decodificador; es diseño propio del proyecto.

Cuantización (docs/metodologia.md §2.1, D5): NF4 en las capas lineales de los tres componentes
congelados (X-Encoder, capas del Predictor y transformer del Y-Encoder). Quedan en precisión completa
las proyecciones lineales que la etapa 2 entrena: `predictor.vis_proj`, `predictor.out_proj` e
`y_encoder.projector`.

Puntos de medición para la pérdida de información (docs/metodologia.md §2.5), con `predict_stages`:
    E1  vis_mean      media de los tokens del X-Encoder
    E2  pred_hidden   estado del Predictor promediado sobre tokens reales, antes de `out_proj`
    E3  pred          Ŝ_Y
    T1  encode_texts  S_Y del caption
Todos se devuelven SIN normalizar; la comparación entre Ŝ_Y y S_Y usa coseno.

Nota de dominio: el checkpoint procesa *video* (tubelets temporales). Una imagen se trata como un
video estático repitiendo el fotograma `num_frames` veces (8, como en Stage B); el paper también
duplica la imagen para igualar la forma de entrada (§3.1).

El Y-Encoder recibe el caption tal cual, sin el prefijo de tarea de EmbeddingGemma: así se entrenó el
checkpoint (`openvljepa.models.y_encoder` usa el transformer base, no el envoltorio
SentenceTransformer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from ..paths import resolve
from .base import Description, Describer
from .quant import quantization_report, swap_linear_to_nf4

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LLAMA32_PAD = "<|finetune_right_pad_id|>"
# Proyecciones que la etapa 2 entrena en precisión completa (nunca se cuantizan)
PROJECTIONS = ("predictor.vis_proj", "predictor.out_proj", "y_encoder.projector")


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
        self.decoder = None            # Y-Decoder; se adjunta con attach_decoder
        self.quant_report: dict = {}

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
        # el X-Encoder no viene en el checkpoint: está congelado y se carga de su repo original
        bad = [k for k in missing if not k.startswith("x_encoder.model.")]
        if bad or unexpected:
            raise RuntimeError(f"Checkpoint incompatible. missing={bad[:5]} unexpected={unexpected[:5]}")
        del ckpt

        model = model.to(dtype=torch.bfloat16).eval()
        if c.quant == "nf4":
            # todo componente congelado en NF4; las proyecciones quedan fuera (ver PROJECTIONS)
            n = {"x_encoder": swap_linear_to_nf4(model.x_encoder.model, self.device),
                 "predictor": swap_linear_to_nf4(model.predictor.layers, self.device),
                 "y_encoder": swap_linear_to_nf4(model.y_encoder.model, self.device)}
            print(f"[vljepa] NF4 aplicado a capas lineales: {n}", flush=True)
            if not all(n.values()):
                raise RuntimeError(f"Algún componente quedó sin capas cuantizadas: {n}")
        elif c.quant != "bf16":
            raise ValueError("VL-JEPA admite quant = nf4 | bf16")
        self.model = model.to(self.device)
        self.quant_report = quantization_report(
            self.model, {"x_encoder": ["x_encoder."], "predictor": ["predictor.layers."],
                         "y_encoder": ["y_encoder.model."], "projections": list(PROJECTIONS)})
        if self.quant_report.get("quantized_in_projections"):
            raise RuntimeError("Las proyecciones de VL-JEPA quedaron cuantizadas; deben ir en precisión completa")
        if c.quant == "bf16" and self.quant_report["quantized_linears"]:
            raise RuntimeError("quant=bf16 pero hay capas cuantizadas")

        if c.get("adapter_path"):
            from peft import PeftModel

            if c.quant == "nf4":
                # PEFT crea la capa LoRA de bnb.Linear4bit solo si el modelo declara esta bandera (este
                # modelo no es un PreTrainedModel; la cuantización se aplicó a mano). Igual que al entrenar.
                self.model.is_loaded_in_4bit = True
            self.model = PeftModel.from_pretrained(self.model, str(resolve(c.adapter_path))).eval()

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

    def _inner(self):
        """El OpenVLJEPA sin el envoltorio de PEFT (si lo hay)."""
        m = self.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    # ---------------------------------------------------------------- texto
    def _tok(self, tok, texts: list[str], max_len: int):
        enc = tok(texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def token_lengths(self, texts: list[str], which: str = "query") -> list[int]:
        """Tokens reales de cada texto (sin truncar ni rellenar)."""
        tok = self.query_tok if which == "query" else self.target_tok
        return [len(tok([t], add_special_tokens=True)["input_ids"][0]) for t in texts]

    def _check_fits(self, texts: list[str], which: str) -> None:
        """Falla si algún texto excede el límite: truncar en silencio cambiaría la consulta o el objetivo."""
        limit = self.max_query_len if which == "query" else self.max_caption_len
        for text in set(texts):
            n = self.token_lengths([text], which)[0]
            if n > limit:
                raise ValueError(
                    f"Texto de {n} tokens > límite {limit} ({which}): {text[:80]!r}. "
                    "Sube max_query_len / max_caption_len en la config del modelo."
                )

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        """S_Y: embeddings del Y-Encoder en el espacio compartido. (N, 1536) float32, sin normalizar."""
        self._check_fits(texts, "target")
        out = []
        for i in range(0, len(texts), batch_size):
            ids, mask = self._tok(self.target_tok, texts[i:i + batch_size], self.max_caption_len)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                e = self._inner().y_encoder(ids, mask)
            out.append(e.float())
        return torch.cat(out)

    # ----------------------------------------------------------- inferencia
    def _pixel_values(self, images: list[Image.Image]) -> torch.Tensor:
        frames = torch.stack([self.transform(im) for im in images])                    # (B,3,H,W)
        pv = frames.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)                  # (B,T,3,H,W)
        return pv.to(self.device, dtype=torch.bfloat16)

    @torch.inference_mode()
    def predict_stages(self, images: list[Image.Image], prompts: list[str]) -> dict[str, torch.Tensor]:
        """Representación de cada etapa de la cadena, una fila por imagen (float32, sin normalizar).

            vis_mean     (B, 1024)  E1: media de los tokens del X-Encoder
            pred_hidden  (B, 2048)  E2: estado del Predictor promediado, antes de la proyección de salida
            pred         (B, 1536)  E3: Ŝ_Y
        """
        q = [self.cfg.get("query_override") or p for p in prompts]
        self._check_fits(q, "query")
        ids, mask = self._tok(self.query_tok, q, self.max_query_len)
        inner = self._inner()
        captured: dict[str, torch.Tensor] = {}

        def grab(_module, args):
            # `Predictor.forward` solo devuelve out_proj(promedio): el promedio es la entrada de out_proj
            captured["pred_hidden"] = args[0].detach().float()

        hook = inner.predictor.out_proj.register_forward_pre_hook(grab)
        try:
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vis = inner.x_encoder(self._pixel_values(images))                     # (B, Nv, 1024)
                pred = inner.predictor(vis, ids, mask)                                # (B, 1536)
        finally:
            hook.remove()
        out = {"vis_mean": vis.float().mean(1), "pred_hidden": captured["pred_hidden"], "pred": pred.float()}
        for k, v in out.items():
            if v.ndim != 2 or v.shape[0] != len(images):
                raise RuntimeError(f"Forma inesperada en {k}: {tuple(v.shape)} para {len(images)} imágenes")
        out["n_visual_tokens"] = torch.tensor(vis.shape[1])
        return out

    def predict_embeddings(self, images: list[Image.Image], prompts: list[str]) -> torch.Tensor:
        """Ŝ_Y sin normalizar, (B, 1536) float32."""
        return self.predict_stages(images, prompts)["pred"]

    def attach_decoder(self, decoder) -> None:
        """Adjunta el Y-Decoder: un objeto con `decode(embeddings: Tensor[B, D]) -> list[str]`."""
        self.decoder = decoder

    def describe(self, images: list[Image.Image], prompts: list[str]) -> list[Description]:
        if self.decoder is None:
            raise RuntimeError(
                "VL-JEPA no genera texto por sí mismo: falta el Y-Decoder (docs/metodologia.md §2.4). "
                "Adjúntalo con `attach_decoder` antes de llamar a describe()."
            )
        pred = self.predict_embeddings(images, prompts)
        texts = self.decoder.decode(pred)
        if len(texts) != len(images):
            raise RuntimeError(f"El Y-Decoder devolvió {len(texts)} textos para {len(images)} imágenes")
        keep = bool(self.cfg.get("return_embeddings"))
        return [Description(text=t, extra={},
                            embedding=pred[j].cpu().numpy().astype(np.float16) if keep else None)
                for j, t in enumerate(texts)]

    def lora_target_modules(self) -> str:
        # Etapa 2 (fase 5): falta agregar las capas lineales del Y-Encoder (opción A del paper).
        return r"predictor\.layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

    def info(self) -> dict:
        out = super().info()
        out.update(self.quant_report)
        out.update({"ckpt": str(self.ckpt_path), "num_frames": self.num_frames,
                    "max_query_len": self.max_query_len, "max_caption_len": self.max_caption_len,
                    "has_decoder": self.decoder is not None})
        return out
