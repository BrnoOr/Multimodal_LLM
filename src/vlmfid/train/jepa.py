"""Objetivo de entrenamiento de VL-JEPA, opción A del paper (docs/metodologia.md §2.3, C7/D3).

    X-Encoder   congelado: sus salidas se calculan sin gradiente, en trozos de `train.x_encoder_chunk`
    Predictor   (S_V, consulta) -> Ŝ_Y, con LoRA y proyecciones entrenables
    Y-Encoder   caption -> S_Y, con LoRA (LR × 0.05) y proyección entrenable; se codifica EN CADA PASO
                porque cambia durante el ajuste
    pérdida     InfoNCE bidireccional del lote ([5] §2; `objectives.infonce_loss`)

La pérdida depende del lote completo (negativos del lote), así que no se acumula gradiente ni se parte
el lote ante un OOM: el lote físico es el efectivo (compose_train lo exige). Partir el X-Encoder en trozos
sí es exacto, porque no lleva gradiente y cada muestra es independiente.

Selección del paso (P7, decisión del usuario 16-11-2026): la InfoNCE en val. Las 500 imágenes de val se
recorren siempre en el mismo orden y en lotes del tamaño del lote efectivo, de modo que el valor es
comparable entre pasos. El Y-Decoder no interviene: el de la etapa 2 se entrena después, sobre el
Y-Encoder ajustado.
"""

from __future__ import annotations

import torch

from .collate import JEPAFeaturizer
from .objectives import duplicate_stats, infonce_loss


class JEPAObjective:
    def __init__(self, cfg, describer, device: torch.device, amp):
        self.cfg, self.describer, self.device, self.amp = cfg, describer, device, amp
        self.temperature = float(cfg.objective.temperature)
        self.mask_duplicates = bool(cfg.objective.mask_duplicate_captions)
        self.chunk = int(cfg.train.get("x_encoder_chunk") or 0)

    def featurizer(self, prompt_text: str) -> JEPAFeaturizer:
        return JEPAFeaturizer(self.describer, prompt_text)

    def _inner(self):
        m = self.describer.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def _visual(self, pv: torch.Tensor) -> torch.Tensor:
        """Tokens del X-Encoder congelado, sin gradiente, en trozos (exacto: cada muestra es independiente)."""
        enc = self._inner().x_encoder
        step = self.chunk or pv.shape[0]
        with torch.no_grad(), self.amp():
            return torch.cat([enc(pv[i:i + step]) for i in range(0, pv.shape[0], step)])

    def embeddings(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(Ŝ_Y, S_Y) del lote, (B, D) cada uno, con gradiente hacia Predictor e Y-Encoder."""
        inner = self._inner()
        pv = batch["pixel_values"].to(self.device, dtype=torch.bfloat16)
        vis = self._visual(pv)
        qi, qm = batch["query_ids"].to(self.device), batch["query_mask"].to(self.device)
        ci, cm = batch["caption_ids"].to(self.device), batch["caption_mask"].to(self.device)
        with self.amp():
            pred = inner.predictor(vis, qi, qm)
            target = inner.y_encoder(ci, cm)
        if pred.shape != target.shape or pred.shape[0] != len(batch["captions"]):
            raise RuntimeError(f"Formas inesperadas: Ŝ_Y {tuple(pred.shape)}, S_Y {tuple(target.shape)}")
        return pred, target

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        pred, target = self.embeddings(batch)
        loss, stats = infonce_loss(pred, target, self.temperature,
                                   batch["captions"] if self.mask_duplicates else None)
        return loss, stats | duplicate_stats(batch["captions"])

    def info(self) -> dict:
        return {"objective": "infonce_bidirectional_batch", "temperature": self.temperature,
                "mask_duplicate_captions": self.mask_duplicates, "x_encoder_chunk": self.chunk}
