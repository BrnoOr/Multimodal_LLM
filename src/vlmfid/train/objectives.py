"""Funciones de pérdida de la etapa 2 (docs/metodologia.md §2.3).

VLM autorregresivo — entropía cruzada solo sobre los tokens de la respuesta ([2] ec. 3, Tabla 2),
normalizada por el número de tokens objetivo de TODO el lote efectivo:

    L = (1 / N_lote) · Σ_{micro-lotes} Σ_{tokens objetivo} −log p(y_t | y_<t, imagen, prompt)

Con esa normalización el gradiente acumulado es idéntico sea cual sea el micro-lote: la única diferencia
entre modelos queda en los pesos, no en cómo se partió el lote para caber en memoria.

VL-JEPA — InfoNCE bidireccional del lote entre Ŝ_Y (Predictor) y S_Y (Y-Encoder) en el espacio
compartido, como `openvljepa.losses.bidirectional_infonce` y [5] §2:

    L = ½ [ CE(⟨ŝ_i, s_j⟩/τ, i) + CE(⟨s_j, ŝ_i⟩/τ, j) ],   ambos lados normalizados a norma 1

Los negativos son las demás muestras del lote. Dos filas con el mismo caption son un falso negativo;
D4 decide no enmascararlos con lote físico ≤ 128 y reportar la tasa (`duplicate_stats`). La máscara
queda disponible (`objective.mask_duplicate_captions`) para lotes mayores.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .collate import IGNORE


def lm_loss(model, batch: dict, n_tokens_effective: int) -> tuple[torch.Tensor, float, int]:
    """(pérdida para backward, suma de −log p sin normalizar, tokens objetivo del micro-lote)."""
    keep = batch["max_target_len"] + 1
    out = model(**batch["inputs"], logits_to_keep=keep, use_cache=False)
    logits = out.logits[:, :-1].float()                      # predicen las posiciones L-keep+1 .. L-1
    labels = batch["labels"][:, -(keep - 1):].to(logits.device)
    loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                               ignore_index=IGNORE, reduction="sum")
    n = int((labels != IGNORE).sum())
    if n != batch["n_target_tokens"]:
        raise RuntimeError(f"Desalineación de etiquetas: {n} tokens con pérdida vs {batch['n_target_tokens']} "
                           "esperados. Revisa logits_to_keep / relleno a la izquierda.")
    return loss_sum / n_tokens_effective, float(loss_sum.detach()), n


def duplicate_mask(captions: list[str], device) -> torch.Tensor:
    """M[i, j] = True si i != j y el caption de i es idéntico al de j."""
    idx: dict[str, int] = {}
    codes = torch.tensor([idx.setdefault(c, len(idx)) for c in captions], device=device)
    same = codes[:, None] == codes[None, :]
    return same & ~torch.eye(len(captions), dtype=torch.bool, device=device)


def duplicate_stats(captions: list[str]) -> dict:
    """Pares de captions idénticos en el lote y fracción de muestras afectadas (D4, se reporta)."""
    counts: dict[str, int] = {}
    for c in captions:
        counts[c] = counts.get(c, 0) + 1
    pairs = sum(n * (n - 1) // 2 for n in counts.values())
    affected = sum(n for n in counts.values() if n > 1)
    return {"dup_pairs": pairs, "dup_samples": affected}


def infonce_logits(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    """⟨ŝ_i, s_j⟩ / τ con ambos lados normalizados: (B, B), en fp32."""
    return F.normalize(pred.float(), dim=-1) @ F.normalize(target.float(), dim=-1).T / temperature


def infonce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float,
                 captions: list[str] | None = None) -> tuple[torch.Tensor, dict]:
    """InfoNCE bidireccional (promedio de Predictor -> texto y texto -> Predictor).

    Devuelve (pérdida, estadísticas sin gradiente): las dos direcciones por separado y la exactitud
    top-1 de recuperación dentro del lote en cada dirección. Con `captions`, los pares con el mismo
    caption se excluyen del denominador (máscara de falsos negativos).
    """
    logits = infonce_logits(pred, target, temperature)
    if captions is not None:
        logits = logits.masked_fill(duplicate_mask(captions, logits.device), float("-inf"))
    labels = torch.arange(logits.shape[0], device=logits.device)
    p2t = F.cross_entropy(logits, labels)
    t2p = F.cross_entropy(logits.T, labels)
    with torch.no_grad():
        stats = {"loss_p2t": float(p2t), "loss_t2p": float(t2p),
                 "acc_p2t": float((logits.argmax(1) == labels).float().mean()),
                 "acc_t2p": float((logits.argmax(0) == labels).float().mean())}
    return (p2t + t2p) / 2, stats
