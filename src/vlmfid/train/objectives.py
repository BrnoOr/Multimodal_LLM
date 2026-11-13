"""Funciones de pérdida de la etapa 2.

VLM autorregresivo — entropía cruzada de los tokens del caption (el prompt y la imagen no aportan
pérdida), normalizada por el número de tokens objetivo de TODO el lote efectivo:

    L = (1 / N_lote) · Σ_{micro-lotes} Σ_{tokens objetivo} −log p(y_t | y_<t, imagen, prompt)

Con esa normalización el gradiente acumulado es idéntico sea cual sea el micro-lote (8, 16 o 32): la
única diferencia entre modelos queda en los pesos, no en cómo se partió el lote para caber en memoria.
(Promediar la media de cada micro-lote, en cambio, pondera distinto los captions cortos y largos.)

VL-JEPA — InfoNCE bidireccional entre ŝ_Y (predictor) y s_Y (Y-Encoder), como el preentrenamiento
del checkpoint (`openvljepa.losses.bidirectional_infonce`), con una corrección necesaria en M3DI:
hay solo 10 936 captions distintos entre 125 000 imágenes, así que dos muestras de un lote pueden
tener exactamente el mismo caption. InfoNCE estándar trataría a la otra como negativo y empujaría a
separar dos embeddings que deben coincidir. Se excluyen esos pares del denominador (falsos negativos).

VL-JEPA con `objective.negatives: bank` (reformulación) — los negativos no son las otras 63 muestras del
lote, sino TODOS los textos objetivo distintos (504 captions canónicos o ~11 000 originales):

    L = (1 / B_efectivo) · Σ_i −log softmax_j( ⟨ŝ_i, s_j⟩ / τ )[c_i],     j ∈ textos objetivo únicos

Es una clasificación sobre el mismo banco con que se decodifica por recuperación en inferencia. Cada
muestra aporta un término independiente del resto del lote, así que se puede acumular gradiente en
micro-lotes (como en los VLM) y no hace falta la máscara de duplicados: un caption repetido es la misma
clase. Solo es dirección predictor -> texto: la dirección inversa no tiene sentido con un banco fijo.
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
    idx = {}
    codes = torch.tensor([idx.setdefault(c, len(idx)) for c in captions], device=device)
    same = codes[:, None] == codes[None, :]
    return same & ~torch.eye(len(captions), dtype=torch.bool, device=device)


def infonce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float,
                 captions: list[str] | None = None) -> torch.Tensor:
    """InfoNCE bidireccional (promedio de predictor->texto y texto->predictor)."""
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    logits = p @ t.T / temperature
    if captions is not None:
        logits = logits.masked_fill(duplicate_mask(captions, logits.device), float("-inf"))
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def bank_infonce_loss(pred: torch.Tensor, bank: torch.Tensor, target_idx: torch.Tensor,
                      temperature: float) -> torch.Tensor:
    """Suma (no media) de −log p(texto correcto | ŝ) con softmax sobre todo el banco de objetivos."""
    p = F.normalize(pred.float(), dim=-1)
    b = F.normalize(bank.float(), dim=-1)
    return F.cross_entropy(p @ b.T / temperature, target_idx, reduction="sum")
