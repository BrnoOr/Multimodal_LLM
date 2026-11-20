"""Evaluación en val durante el ajuste de los VLM (docs/metodologia.md §2.3).

El modelo describe cada imagen de val con `describer.describe`, el MISMO camino que `scripts/infer.py`
usa en la etapa 1 y en test (generación greedy con `model.generation`), y el texto pasa por
`vlmfid.eval.attributes.score_texts`: exactitud por atributo y `attr_mean`, que elige el checkpoint.

VL-JEPA no pasa por aquí: sin Y-Decoder no produce texto durante el ajuste y se selecciona por la
InfoNCE en val (P7; `vlmfid.train.jepa`).
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from ..data.manifest import ImageRecordDataset
from ..eval.attributes import score_texts


def _describe_safe(describer, images, prompts):
    try:
        return describer.describe(images, prompts)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(images) == 1:
            raise
        h = len(images) // 2
        return _describe_safe(describer, images[:h], prompts[:h]) + _describe_safe(describer, images[h:], prompts[h:])


def generate_and_score(describer, records, root, prompt_text: str, batch_size: int, amp,
                       json_mode: bool, num_workers: int = 2) -> tuple[dict, list[dict]]:
    """`json_mode` = `attributes.json_mode_for(cfg)`: el mismo extractor que puntúa la etapa 1."""
    loader = DataLoader(ImageRecordDataset(records, root), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=ImageRecordDataset.collate, shuffle=False, generator=torch.Generator())
    texts, preds, t0 = [], [], time.time()
    with amp():
        for recs, imgs in loader:
            outs = _describe_safe(describer, imgs, [prompt_text] * len(imgs))
            for r, o in zip(recs, outs):
                texts.append(o.text)
                preds.append({"id": r["id"], "prediction": o.text, "reference": r["caption"], "extra": o.extra})
    metrics = score_texts(texts, records, json_mode=json_mode)
    metrics["truncated"] = sum(bool(p["extra"].get("truncated")) for p in preds) / max(len(preds), 1)
    metrics["gen_seconds"] = round(time.time() - t0, 1)
    return metrics, preds
