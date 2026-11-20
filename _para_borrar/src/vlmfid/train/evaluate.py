"""Evaluación en validación durante el entrenamiento.

Dos señales en las mismas 500 imágenes de val (idénticas para los 4 modelos):

  * Índice macro y exactitud por atributo: el modelo describe cada imagen con `describer.describe`,
    el MISMO camino que `scripts/infer.py` usará en test (generación greedy con `model.generation`,
    o recuperación sobre el banco de train en VL-JEPA), y el texto pasa por `vlmfid.eval.attributes`.
    Es la métrica con que se elige el checkpoint.
  * Pérdida de validación (teacher forcing / InfoNCE): barata y útil para ver sobreajuste, pero no se
    alinea necesariamente con la exactitud (la plantilla elegida al azar domina la entropía cruzada).

Con el split composicional (`data.holdout: compositional`) las métricas se calculan además por separado
sobre las combinaciones vistas (`seen_*`, las que deciden el checkpoint) y retiradas (`unseen_*`).
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


def generate_and_score(describer, records, root, prompt_text: str, batch_size: int, amp, num_workers: int = 2,
                       target_fn=None, groups: dict | None = None):
    """`groups` = {prefijo: función registro -> bool}: métricas adicionales sobre esos subconjuntos."""
    loader = DataLoader(ImageRecordDataset(records, root), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=ImageRecordDataset.collate, shuffle=False, generator=torch.Generator())
    texts, preds, t0 = [], [], time.time()
    with amp():
        for recs, imgs in loader:
            outs = _describe_safe(describer, imgs, [prompt_text] * len(imgs))
            for r, o in zip(recs, outs):
                texts.append(o.text)
                preds.append({"id": r["id"], "prediction": o.text, "reference": r["caption"],
                              **({"target": target_fn(r)} if target_fn else {}),
                              "extra": {k: v for k, v in o.extra.items() if k != "topk"}})
    metrics = score_texts(texts, records)
    for prefix, keep in (groups or {}).items():
        sel = [i for i, r in enumerate(records) if keep(r)]
        sub = score_texts([texts[i] for i in sel], [records[i] for i in sel])
        metrics |= {f"{prefix}_{k}": v for k, v in sub.items()}
    metrics["gen_seconds"] = round(time.time() - t0, 1)
    return metrics, preds


def score_predictions(pred_path, records, groups: dict | None = None) -> dict:
    """Métricas de un predictions.jsonl de `scripts/infer.py` sobre `records` (mismas que en val)."""
    import json

    by_id = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            by_id[p["id"]] = p["prediction"]
    missing = [r["id"] for r in records if r["id"] not in by_id]
    if missing:
        raise ValueError(f"{pred_path}: faltan {len(missing)} predicciones (p. ej. {missing[:3]})")
    texts = [by_id[r["id"]] for r in records]
    metrics = score_texts(texts, records)
    for prefix, keep in (groups or {}).items():
        sel = [i for i, r in enumerate(records) if keep(r)]
        metrics |= {f"{prefix}_{k}": v for k, v in score_texts([texts[i] for i in sel],
                                                               [records[i] for i in sel]).items()}
    return metrics
