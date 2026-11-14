"""Dónde se pierde la información de cada atributo dentro de VL-JEPA (docs/VLJEPA_AJUSTE_V3.pdf, §2 y §6).

La imagen determina el atributo A (forma, color, y, x). El modelo es una cadena de procesamiento

    A  ->  imagen  ->  tokens del X-Encoder  ->  ŝ (predictor)  ->  respuesta (decodificador sobre el banco)

y por la desigualdad de procesamiento de datos la información sobre A solo puede bajar de izquierda a
derecha. Se estima, en nats y sobre imágenes que no se usaron para ajustar el estimador:

  visual    I_V(A; media de tokens del X-Encoder)     sonda lineal
  predictor I_V(A; ŝ)                                  sonda lineal
  decoder   I_NCE(A; ŝ) a través del banco             softmax de ⟨ŝ, referencias⟩/τ, sin parámetros libres
                                                       salvo τ (a la del entrenamiento y calibrada)

Las tres son cotas inferiores de la información mutua de la forma  H(A) − CE,  donde CE es la entropía
cruzada en datos nuevos de un lector q(a | representación):

    I(A; Z) = H(A) − H(A | Z) ≥ H(A) − E[−log q(A | Z)]

  * con q = sonda lineal es la V-información de Xu et al. (2020) para la familia lineal;
  * con q = softmax sobre el banco es la cota InfoNCE (van den Oord et al., 2018; Poole et al., 2019) con
    el crítico fijo f(ŝ, a) = ⟨ŝ, s_a⟩/τ y todos los candidatos como negativos. Con K candidatos su techo
    es log K ≥ H(A): con el banco factorizado no satura antes que la entropía del atributo.

Lectura: `predictor − decoder` es lo que está en ŝ y el banco no deja leer (pérdida en el decodificador);
`visual − predictor` es lo que el predictor no trae a ŝ; `H(A) − visual` es lo que la sonda no encuentra
en el X-Encoder congelado. Las sondas son lineales sobre una media de tokens: una sonda baja es una cota
inferior, no prueba que la información no esté.

Todo aquí opera sobre arreglos ya extraídos (sin cargar modelos), así que se prueba en CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .probe import _fit

TAU_GRID = tuple(float(t) for t in np.geomspace(0.002, 1.0, 60))


def entropy_nats(y: np.ndarray, n_classes: int) -> float:
    p = np.bincount(y, minlength=n_classes) / max(len(y), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def class_logits(sims: torch.Tensor, col_class: torch.Tensor, n_classes: int, tau: float) -> torch.Tensor:
    """Logits por valor del atributo a partir de similitudes con las referencias.

    `sims`: (N, M) cosenos entre ŝ y M referencias; `col_class[m]` = valor del atributo de la referencia m.
    Banco factorizado: M = K y cada referencia es un valor. Banco de captions: varias frases comparten
    valor y se marginaliza, logit_k = logsumexp_{m: clase(m) = k} sims[:, m]/τ (la probabilidad de que el
    caption recuperado tenga ese valor según el softmax del banco).
    """
    z = sims / tau
    out = torch.full((sims.shape[0], n_classes), float("-inf"), dtype=z.dtype, device=z.device)
    for k in range(n_classes):
        m = col_class == k
        if bool(m.any()):
            out[:, k] = torch.logsumexp(z[:, m], dim=1)
    return out


def _ce(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.double(), y))


def decoder_information(sims_cal: np.ndarray, y_cal: np.ndarray, sims_eval: np.ndarray, y_eval: np.ndarray,
                        col_class: np.ndarray, n_classes: int, tau_train: float) -> dict:
    """Información que entrega el decodificador (softmax sobre el banco) sobre un atributo.

    τ es el único parámetro: se reporta con la del entrenamiento y con la que minimiza la entropía cruzada
    en `*_cal` (imágenes distintas de las de evaluación), para no confundir mala calibración con falta de
    información. La exactitud no depende de τ en el banco factorizado (sí, levemente, al marginalizar).
    """
    sc, se = torch.as_tensor(sims_cal, dtype=torch.float32), torch.as_tensor(sims_eval, dtype=torch.float32)
    yc, ye = torch.as_tensor(y_cal), torch.as_tensor(y_eval)
    cc = torch.as_tensor(col_class)
    h = entropy_nats(np.asarray(y_eval), n_classes)
    tau_star = min(TAU_GRID, key=lambda t: _ce(class_logits(sc, cc, n_classes, t), yc))
    lt, ls = class_logits(se, cc, n_classes, tau_train), class_logits(se, cc, n_classes, tau_star)
    return {"h_nats": h, "acc": float((ls.argmax(1) == ye).float().mean()),
            "ce_tau_train": _ce(lt, ye), "info_tau_train": h - _ce(lt, ye),
            "tau_calibrated": tau_star, "ce_calibrated": _ce(ls, ye), "info_calibrated": h - _ce(ls, ye),
            "max_nats": math.log(n_classes)}


def probe_information(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, n_classes: int,
                      weight_decays=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), max_iter: int = 300, seed: int = 0,
                      device: str | None = None) -> dict:
    """V-información lineal: H(A) − entropía cruzada en datos nuevos de una regresión logística.

    Mismo clasificador que `vlmfid.eval.probe.linear_probe`, pero la regularización se elige por entropía
    cruzada en el 20 % retenido (aquí interesa la probabilidad, no solo el argmax).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Xtr = torch.tensor((X_tr - mu) / sd, dtype=torch.float32, device=dev)
    Xte = torch.tensor((X_te - mu) / sd, dtype=torch.float32, device=dev)
    ytr, yte = torch.tensor(y_tr, device=dev), torch.tensor(y_te, device=dev)
    perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(seed)).to(dev)
    cut = int(0.8 * len(Xtr))
    fit_i, hold_i = perm[:cut], perm[cut:]
    best = None
    for wd in weight_decays:
        W, b = _fit(Xtr[fit_i], ytr[fit_i], n_classes, float(wd), max_iter)
        ce = _ce(Xtr[hold_i] @ W + b, ytr[hold_i])
        if best is None or ce < best[0]:
            best = (ce, float(wd))
    W, b = _fit(Xtr, ytr, n_classes, best[1], max_iter)
    logits = Xte @ W + b
    h, ce = entropy_nats(np.asarray(y_te), n_classes), _ce(logits, yte)
    return {"h_nats": h, "acc": float((logits.argmax(1) == yte).float().mean()), "ce": ce, "info": h - ce,
            "weight_decay": best[1], "max_nats": math.log(n_classes)}


def information_chain(attr: str, n_classes: int, visual: dict | None, predictor: dict, decoder: dict) -> dict:
    """Fila resumen de un atributo: información en cada etapa y pérdida entre etapas (nats)."""
    h = decoder["h_nats"]
    row = {"attribute": attr, "k": n_classes, "h_nats": h,
           "info_visual": visual["info"] if visual else None, "acc_visual": visual["acc"] if visual else None,
           "info_predictor": predictor["info"], "acc_predictor": predictor["acc"],
           "info_decoder": decoder["info_calibrated"], "info_decoder_tau_train": decoder["info_tau_train"],
           "acc_decoder": decoder["acc"], "tau_calibrated": decoder["tau_calibrated"]}
    row["loss_decoder"] = row["info_predictor"] - row["info_decoder"]
    row["loss_predictor"] = (row["info_visual"] - row["info_predictor"]) if visual else None
    row["loss_visual"] = (h - row["info_visual"]) if visual else None
    row["frac_decoder"] = row["info_decoder"] / h if h > 0 else float("nan")
    return row
