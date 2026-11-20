"""Objetivo de entrenamiento de VL-JEPA (lo que en los VLM es `objectives.lm_loss`).

Reúne lo que antes estaba repartido en `loop.Trainer` (`_target_table`, `_jepa_pred`, `_jepa_loss`,
`_jepa_bank_loss_sum`), de modo que el bucle no conozca los detalles del modelo: solo pide una pérdida.

Modos (docs/VLJEPA_AJUSTE_V3.pdf, §4):

  banco        consultas        negativos   simétrica   uso
  captions     el prompt        batch       —           etapa 2 original (InfoNCE del lote + máscara)
  captions     el prompt        bank        no          etapa 2 reformulada (run publicado)
  factorized   una por atributo bank        sí          ajuste v3: InfoNCE bidireccional contra 21 frases

Lado de texto. Con `lora.text_modules: []` los embeddings objetivo son fijos (Y-Encoder congelado) y se
calculan una vez. Con `lora.text_modules: [y_encoder.projector]` la proyección lineal del Y-Encoder es
entrenable: el transformer de texto sigue congelado, su salida promediada se calcula una vez y en cada
paso se vuelve a proyectar, con gradiente, para obtener el banco.

`whole_batch` indica que la pérdida depende del lote completo (negativos del lote, o el término inverso de
la simétrica): ahí no se puede acumular gradiente por micro-lotes ni partir el lote ante un OOM.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .collate import JEPAFeaturizer
from .objectives import bank_infonce_loss, infonce_loss, symmetric_bank_infonce_loss


class JEPAObjective:
    def __init__(self, cfg, describer, device: torch.device, amp, cache_dir: Path):
        o = cfg.objective
        self.cfg, self.describer, self.device, self.amp, self.cache_dir = cfg, describer, device, amp, cache_dir
        self.temperature = float(o.temperature)
        self.bank_negatives = o.get("negatives", "batch") == "bank"
        self.symmetric = bool(o.get("symmetric", False))
        self.mask_duplicates = bool(o.mask_duplicate_captions)
        self.factorized = describer.fbank is not None
        self.text_trainable = bool(cfg.lora.get("text_modules"))
        if self.text_trainable and not self.factorized:
            raise ValueError("lora.text_modules (proyección del Y-Encoder entrenable) requiere "
                             "model.bank.kind=factorized: el banco de captions se cachea con el Y-Encoder congelado.")
        if self.symmetric and not self.bank_negatives:
            raise ValueError("objective.symmetric=true requiere objective.negatives=bank "
                             "(con negatives=batch la pérdida ya es bidireccional).")

    @property
    def whole_batch(self) -> bool:
        return not self.bank_negatives or self.symmetric

    # ------------------------------------------------------------------ datos
    def featurizers(self, prompt_text: str, target_fn) -> tuple[JEPAFeaturizer, JEPAFeaturizer]:
        """(entrenamiento, validación). Solo difieren con el banco factorizado: una consulta por imagen
        al entrenar, las cuatro al validar."""
        kw = {"per_attribute": self.factorized, "seed": int(self.cfg.seed)}
        train = JEPAFeaturizer(self.describer, prompt_text, target_fn, **kw)
        val = JEPAFeaturizer(self.describer, prompt_text, target_fn, expand=True, **kw) if self.factorized else train
        return train, val

    def build_targets(self, captions: list[str]) -> None:
        """Tabla de embeddings objetivo. Banco factorizado: sus 21 frases, en el orden del banco.
        Banco de captions: los captions únicos de train y val (caché en disco, Y-Encoder congelado)."""
        d = self.describer
        if self.factorized:
            texts = list(d.bank_texts)
            self.pooled = d.bank_pooled.to(self.device)
            self.fixed = None if self.text_trainable else F.normalize(d.project_pooled(self.pooled).detach(), dim=-1)
        else:
            texts = sorted(set(captions))
            h = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
            key = f"{d.ckpt_path.name}|{d.ckpt_path.stat().st_size}|{d.max_caption_len}|{h}"
            cache = self.cache_dir / f"vljepa_targets_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pt"
            emb = None
            if cache.exists():
                blob = torch.load(cache, map_location="cpu", weights_only=False)
                if blob["texts"] == texts:
                    emb = blob["emb"]
            if emb is None:
                print(f"[train] codificando {len(texts)} captions únicos con el Y-Encoder (congelado)", flush=True)
                emb = d.encode_texts(texts).cpu()
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"texts": texts, "emb": emb}, cache)
            self.fixed = emb.to(self.device)
        self.target_index = {t: i for i, t in enumerate(texts)}

    def targets(self) -> torch.Tensor:
        """Embeddings del banco objetivo (N, D). Con el lado de texto entrenable llevan gradiente."""
        if self.text_trainable:
            return self.describer.project_pooled(self.pooled)
        return self.fixed

    def before_eval(self) -> None:
        """El banco con que `describe` decodifica en val debe ser el de la proyección actual."""
        if self.text_trainable:
            self.describer.refresh_bank()

    # ---------------------------------------------------------------- pérdidas
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        inner = self.describer.model.get_base_model()
        pv = batch["pixel_values"].to(self.device, dtype=torch.bfloat16)
        qi, qm = batch["query_ids"].to(self.device), batch["query_mask"].to(self.device)
        with self.amp():
            vis = inner.x_encoder(pv)
            pred = inner.predictor(vis, qi, qm)
        idx = torch.tensor([self.target_index[c] for c in batch["captions"]], device=self.device)
        return pred, idx

    def bank_loss_sum(self, batch: dict) -> torch.Tensor:
        """Término directo (predictor -> texto) contra el banco completo: SUMA sobre el micro-lote."""
        pred, idx = self.predict(batch)
        return bank_infonce_loss(pred, self.targets(), idx, self.temperature)

    def batch_loss(self, batch: dict) -> tuple[torch.Tensor, float]:
        """Pérdida que necesita el lote completo: (media para backward, valor que se registra).

        Simétrica: se registra solo el término directo, que es comparable con `val_loss` y con la cota
        InfoNCE (I ≥ log N − L). Negativos del lote: la InfoNCE bidireccional con máscara de duplicados."""
        pred, idx = self.predict(batch)
        if self.symmetric:
            loss, fwd = symmetric_bank_infonce_loss(pred, self.targets(), idx, self.temperature)
            return loss, float(fwd)
        caps = batch["captions"] if self.mask_duplicates else None
        loss = infonce_loss(pred, self.targets()[idx], self.temperature, caps)
        return loss, float(loss.detach())

    def info(self) -> dict:
        return {"negatives": "bank" if self.bank_negatives else "batch", "symmetric": self.symmetric,
                "bank_kind": "factorized" if self.factorized else "captions",
                "text_trainable": self.text_trainable, "n_target_texts": len(self.target_index)}
