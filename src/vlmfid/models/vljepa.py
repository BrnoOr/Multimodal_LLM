"""Adaptador de VL-JEPA (open-vljepa) al Protocol Describer.

open-vljepa NO genera tokens: predice un embedding y se entrena con InfoNCE
bidireccional contra los embeddings del Y-Encoder (EmbeddingGemma). No hay
decodificador de texto.

`describe()` se implementa por RECUPERACION sobre un pool enumerado de captions:
  1. se codifica el pool una sola vez con el Y-Encoder            -> (P, 1536)
  2. cada imagen + query pasa por X-Encoder + Predictor           -> (B, 1536)
  3. argmax de similitud coseno devuelve el caption recuperado

Arquitectura (segun el README del repo):
    X-Encoder  facebook/vjepa2-vitl-fpc64-256   304M  congelado
    Predictor  ultimas 8 capas de Llama-3.2-1B  490M  entrenable
    Y-Encoder  google/embeddinggemma-300m       310M  entrenable
    espacio compartido 1536-D, InfoNCE tau=0.07

Notas:
  - Las imagenes de M3DI son estaticas y el encoder espera video: se replica la
    imagen en la dimension temporal. Es un modo de uso previsto — el Stage A del
    repo entreno con CC3M alimentando cada imagen como "video" de 1 frame.
  - ~1.1B parametros ≈ 2.2 GiB en bf16: no se cuantiza. LLaVA y Qwen (7B) si van
    en NF4. Declararlo en el informe.
  - Llama-3.2-1B y EmbeddingGemma-300m son gated: aceptar licencias y exportar
    HF_TOKEN antes de cargar.

VERIFICAR: la normalizacion de frames se replica de openvljepa/data/msrvtt.py.
Comparar con las lineas 80-140 de ese archivo y ajustar MEAN/STD si difieren.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from vlmfid.eval.caption_pool import PoolEntry, build_pool
from vlmfid.models.base import Describer, GenConfig, ModelSpec

# Normalizacion estandar de V-JEPA2 / ImageNet. VERIFICAR contra msrvtt.py.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEFAULT_QUERY = "Describe the video."


class VLJepaDescriber(Describer):
    """Describe por recuperacion sobre un pool finito de captions."""

    name = "vljepa"

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.name = spec.name

        repo = Path(spec.extra.get("repo", "external/open-vljepa")).resolve()
        ckpt_path = Path(spec.extra.get("ckpt", repo / "checkpoints_msrvtt/best.pt"))
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from openvljepa.data.msrvtt import _ensure_pad_token
        from openvljepa.models.vljepa import OpenVLJEPA
        from transformers import AutoTokenizer

        self.device = torch.device(spec.device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.cfg = cfg

        self.model = OpenVLJEPA(cfg["encoder"], cfg["y_encoder"], cfg["predictor"],
                                torch_dtype=torch.bfloat16)
        missing, unexpected = self.model.load_state_dict(ckpt["model_state_dict"],
                                                         strict=False)
        # las claves de x_encoder.model.* faltan a proposito: el encoder congelado
        # se recarga desde HF, no viaja en el checkpoint
        real_missing = [k for k in missing if not k.startswith("x_encoder.model.")]
        if real_missing:
            print(f"  AVISO claves faltantes no congeladas: {real_missing[:5]}")
        if unexpected:
            print(f"  AVISO claves inesperadas: {list(unexpected)[:5]}")
        self.model.eval().to(self.device)

        # dos tokenizadores distintos: query -> Llama (predictor), target -> Gemma
        self.q_tok = _ensure_pad_token(
            AutoTokenizer.from_pretrained(cfg["predictor"]["llama_name"]))
        self.t_tok = _ensure_pad_token(
            AutoTokenizer.from_pretrained(cfg["y_encoder"]["model_name"]))

        data_cfg = cfg.get("data", {})
        self.num_frames = spec.extra.get("num_frames", data_cfg.get("num_frames", 16))
        self.image_size = spec.extra.get("image_size", data_cfg.get("image_size", 256))
        self.max_query_len = data_cfg.get("max_query_len", 512)
        self.max_caption_len = data_cfg.get("max_caption_len", 512)

        self.pool: list[PoolEntry] = []
        self.pool_embeds: torch.Tensor | None = None
        self.last_entries: list[PoolEntry] = []

    # ------------------------------------------------------------------ pool

    @torch.no_grad()
    def build_caption_pool(self, pool: list[PoolEntry] | None = None,
                           batch_size: int = 64, quiet: bool = False) -> None:
        """Codifica el pool con el Y-Encoder. Se hace una vez y se reutiliza."""
        self.pool = pool if pool is not None else build_pool(
            quoted=self.spec.extra.get("quoted_pool", False))
        embeds = []
        for i in range(0, len(self.pool), batch_size):
            texts = [e.text for e in self.pool[i:i + batch_size]]
            enc = self.t_tok(texts, max_length=self.max_caption_len,
                             padding="max_length", truncation=True,
                             return_tensors="pt").to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t = self.model.y_encoder(enc["input_ids"], enc["attention_mask"])
                t = F.normalize(t.float(), dim=-1)
            embeds.append(t.cpu())
        self.pool_embeds = torch.cat(embeds).to(self.device)
        if not quiet:
            print(f"  pool codificado: {self.pool_embeds.shape}")

    # ---------------------------------------------------------------- imagen

    def _pixel_values(self, images: list[Image.Image]) -> torch.Tensor:
        """(B, T, C, H, W). La imagen estatica se replica en el eje temporal."""
        import torchvision.transforms as T

        tf = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])
        frames = torch.stack([tf(im.convert("RGB")) for im in images])  # (B,C,H,W)
        return frames.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)

    # -------------------------------------------------------------- describe

    @torch.inference_mode()
    def describe(self, images: list[Image.Image], prompt: str,
                 cfg: GenConfig | None = None) -> list[str]:
        """Recupera del pool el caption mas similar al embedding predicho.

        `prompt` se usa como query del predictor, no como instruccion de formato:
        VL-JEPA no puede cambiar el formato de su salida, que lo fija el pool.
        Esta asimetria respecto a LLaVA y Qwen debe declararse en el informe.
        """
        if self.pool_embeds is None:
            self.build_caption_pool()

        pv = self._pixel_values(images).to(self.device)
        q = self.q_tok([prompt or DEFAULT_QUERY], max_length=self.max_query_len,
                       padding="max_length", truncation=True,
                       return_tensors="pt").to(self.device)
        B = pv.shape[0]
        q_ids = q["input_ids"].expand(B, -1)
        q_mask = q["attention_mask"].expand(B, -1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(pv, q_ids, q_mask)          # (B, 1536)
            pred = F.normalize(pred.float(), dim=-1)

        sim = pred @ self.pool_embeds.T                    # (B, P)
        idx = sim.argmax(dim=1).tolist()
        self.last_entries = [self.pool[i] for i in idx]
        return [self.pool[i].text for i in idx]

    @torch.inference_mode()
    def describe_topk(self, images: list[Image.Image], prompt: str, k: int = 5
                      ) -> list[list[tuple[str, float]]]:
        """Top-k con similitud. Util para diagnostico: si el correcto esta en el
        top-5 pero no en el top-1, el modelo percibe pero no discrimina."""
        if self.pool_embeds is None:
            self.build_caption_pool()
        pv = self._pixel_values(images).to(self.device)
        q = self.q_tok([prompt or DEFAULT_QUERY], max_length=self.max_query_len,
                       padding="max_length", truncation=True,
                       return_tensors="pt").to(self.device)
        B = pv.shape[0]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(pv, q["input_ids"].expand(B, -1),
                              q["attention_mask"].expand(B, -1))
            pred = F.normalize(pred.float(), dim=-1)
        sim = pred @ self.pool_embeds.T
        vals, idx = sim.topk(k, dim=1)
        return [[(self.pool[j].text, float(v)) for j, v in zip(ii, vv)]
                for ii, vv in zip(idx.tolist(), vals.tolist())]

    def memory_footprint_gib(self) -> float:
        return sum(p.numel() * p.element_size()
                   for p in self.model.parameters()) / 2**30
