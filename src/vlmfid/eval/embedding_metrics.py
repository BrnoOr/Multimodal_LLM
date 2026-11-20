"""Métricas sobre embeddings de VL-JEPA: alineación (coseno) y discriminación (recuperación, InfoNCE).

Se usan en el diagnóstico de cuantización (scripts/diag_quant_vljepa.py). Distinguen dos efectos:

* **alineación**: qué tan cerca queda cada Ŝ_Y de su propio S_Y (coseno por par). Es lo que ve el
  Y-Decoder, que normaliza su entrada y solo depende de la dirección del embedding.
* **discriminación**: si Ŝ_Y queda más cerca de su S_Y que de los S_Y de las otras imágenes
  (recuperación top-k y InfoNCE bidireccional en lotes fijos).

Un desplazamiento común de todos los embeddings baja el coseno sin cambiar la recuperación; una pérdida
de información baja ambos.

Captions duplicados: en M3DI dos imágenes pueden tener el mismo caption. En la recuperación cuenta
como acierto cualquier candidato con el MISMO TEXTO de caption que el propio; en InfoNCE los duplicados
se excluyen del denominador (misma máscara que el entrenamiento, `vlmfid.train.objectives`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _norm(x) -> torch.Tensor:
    return F.normalize(torch.as_tensor(np.asarray(x), dtype=torch.float32), dim=-1)


def paired_cosine(a, b) -> np.ndarray:
    """Coseno fila a fila entre dos matrices (N, D) → (N,)."""
    a, b = _norm(a), _norm(b)
    if a.shape != b.shape:
        raise ValueError(f"Formas distintas: {tuple(a.shape)} vs {tuple(b.shape)}")
    return (a * b).sum(-1).numpy()


def caption_codes(captions: list[str]) -> np.ndarray:
    """Entero por texto de caption: igual código ⇔ mismo texto."""
    idx: dict[str, int] = {}
    return np.array([idx.setdefault(c, len(idx)) for c in captions])


def retrieval_recall(pred, target, captions: list[str], ks=(1, 5)) -> dict[str, float]:
    """Recuperación sobre el conjunto completo (N candidatos), en las dos direcciones.

    p2t: para cada Ŝ_Y, los k S_Y más cercanos ¿contienen uno con su mismo caption?
    t2p: para cada S_Y, los k Ŝ_Y más cercanos ¿contienen uno de una imagen con su mismo caption?
    """
    p, t = _norm(pred), _norm(target)
    if p.shape != t.shape or p.shape[0] != len(captions):
        raise ValueError("pred, target y captions deben tener el mismo número de filas")
    sim = p @ t.T                                                      # (N, N): fila = Ŝ_Y, columna = S_Y
    codes_np = caption_codes(captions)
    codes = torch.as_tensor(codes_np)
    out = {}
    for k in ks:
        k_eff = min(int(k), sim.shape[0])
        for name, s in (("p2t", sim), ("t2p", sim.T)):
            top = s.topk(k_eff, dim=1).indices                         # (N, k)
            hit = (codes[top] == codes[:, None]).any(dim=1)
            out[f"recall@{k}_{name}"] = float(hit.float().mean())
    out["n_candidates"] = int(sim.shape[0])
    # azar de recall@1: fracción media de candidatos que comparten el caption propio
    _, counts = np.unique(codes_np, return_counts=True)
    out["chance@1"] = float(counts[codes_np].mean() / sim.shape[0])
    return out


def infonce_fixed_batches(pred, target, captions: list[str], batch_size: int,
                          temperature: float) -> dict[str, float]:
    """InfoNCE bidireccional en lotes consecutivos de tamaño fijo (se descarta el resto).

    Mismo cálculo que el entrenamiento: logits = ⟨ŝ_i, s_j⟩ / τ con ambos normalizados, duplicados de
    caption fuera del denominador, promedio de las dos direcciones. Con el orden y el tamaño fijos, el
    valor es comparable entre condiciones (no entre tamaños de lote). Cota de azar: log(batch_size).
    """
    from vlmfid.train.objectives import infonce_loss

    p, t = _norm(pred), _norm(target)
    n_batches = p.shape[0] // int(batch_size)
    if n_batches == 0:
        raise ValueError(f"Hacen falta al menos {batch_size} pares para un lote; hay {p.shape[0]}")
    losses, acc = [], []
    for b in range(n_batches):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        loss, st = infonce_loss(p[sl], t[sl], temperature, captions[sl])
        losses.append(float(loss))
        acc.append((st["acc_p2t"] + st["acc_t2p"]) / 2)
    return {"infonce": float(np.mean(losses)), "infonce_std": float(np.std(losses)),
            "infonce_batch_acc": float(np.mean(acc)), "infonce_n_batches": n_batches,
            "infonce_batch_size": int(batch_size), "infonce_chance": float(np.log(batch_size)),
            "temperature": float(temperature)}
