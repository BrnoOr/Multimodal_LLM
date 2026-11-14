"""Sondas para medir cuánta información sobre cada factor queda en una representación
(docs/metodologia.md §2.5).

Protocolo de Daunhawer et al. (2023, §5.2), que mide exactamente esto sobre M3DI:
  * factores DISCRETOS  -> MLP con UNA capa oculta; se reporta la exactitud;
  * factores CONTINUOS  -> regresión kernel ridge; se reporta R².
La sonda se ajusta con representaciones de un conjunto y se evalúa en otro que no vio.

Lectura en bits (factores discretos). La exactitud es comparable entre puntos, pero no es una cantidad
de información. Con la entropía cruzada CE de la sonda en datos nuevos:

    I(A; Z) = H(A) − H(A | Z)  ≥  H(A) − CE

(cota variacional de Barber y Agakov, 2003; su uso con sondas: Pimentel et al., 2020; es la
V-información de Xu et al., 2020, para la familia de sondas usada). Es una COTA INFERIOR: una sonda
débil subestima la información, y por eso las estimaciones de dos puntos de la cadena pueden violar en
apariencia la desigualdad de procesamiento de datos. La cota puede salir negativa si la sonda está mal
calibrada; se reporta tal cual y, aparte, recortada en 0.

Para el texto decodificado no hay sonda: se compara el valor extraído con el verdadero. Su información
se estima con `discrete_mi_bits`, la información mutua empírica entre ambos, que por la desigualdad de
procesamiento de datos acota por abajo la información del texto.

Hiperparámetros. Daunhawer et al. no detallan los de sus sondas (decisión D8). Aquí NO se fija un
valor: se define una grilla y se elige por punto y factor en una parte retenida del conjunto de ajuste
(20 %). La grilla es la misma en todos los puntos de la cadena, de modo que todos se leen con la
misma familia de sondas. La grilla en sí es una propuesta y debe declararse en el informe.

Todo opera sobre arreglos ya extraídos (sin cargar modelos), así que se prueba en CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)

# Grillas de selección (propuesta, D8). Mismas para todos los puntos de medición.
MLP_GRID = {"hidden": (64, 256, 1024), "weight_decay": (1e-4, 1e-2)}
MLP_TRAIN = {"lr": 1e-3, "batch_size": 256, "max_epochs": 100, "patience": 10}
KRR_GRID = {"gamma_scale": (0.25, 1.0, 4.0), "alpha": (1e-3, 1e-1, 10.0)}
HOLDOUT_FRACTION = 0.2


def _device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _standardize(X_fit: np.ndarray, *others: np.ndarray):
    mu, sd = X_fit.mean(0), X_fit.std(0) + 1e-6
    return [((x - mu) / sd).astype(np.float32) for x in (X_fit, *others)]


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(round((1 - HOLDOUT_FRACTION) * n))
    return perm[:cut], perm[cut:]


def entropy_bits(y: np.ndarray, n_classes: int | None = None) -> float:
    """Entropía empírica de una variable discreta, en bits."""
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes or 0).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def discrete_mi_bits(a: np.ndarray, b: np.ndarray) -> float:
    """Información mutua empírica entre dos variables discretas, en bits.

    Estimador de sustitución (frecuencias): sesgado hacia arriba cuando hay muchas clases y pocas
    muestras; con las ~100 clases de color conviene usar miles de muestras.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("a y b deben tener la misma forma")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.float64)
    np.add.at(joint, (ai, bi), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())


# ── Factores discretos: MLP de una capa oculta ──────────────────────────────────────────────────
def _fit_mlp(X: torch.Tensor, y: torch.Tensor, Xh: torch.Tensor, yh: torch.Tensor, n_classes: int,
             hidden: int, weight_decay: float, seed: int):
    """Entrena Linear-ReLU-Linear con parada temprana por entropía cruzada en el conjunto retenido.
    Devuelve (estado del mejor modelo, CE retenida en nats)."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, n_classes)).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_TRAIN["lr"], weight_decay=weight_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_ce, best_state, bad = float("inf"), None, 0
    bs = MLP_TRAIN["batch_size"]
    for _ in range(MLP_TRAIN["max_epochs"]):
        net.train()
        perm = torch.randperm(X.shape[0], generator=g).to(X.device)
        for i in range(0, X.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(X[idx]), y[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ce = float(F.cross_entropy(net(Xh), yh))
        if ce < best_ce - 1e-5:
            best_ce, bad = ce, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_TRAIN["patience"]:
                break
    net.load_state_dict(best_state)
    return net, best_ce


def mlp_probe(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, n_classes: int,
              seed: int = 0, device: str | None = None) -> dict:
    """Sonda MLP de una capa oculta para un factor discreto.

    Devuelve exactitud y entropía cruzada en `eval`, la entropía del factor y la cota de información
    en bits, más la configuración elegida en la parte retenida de `fit`.
    """
    y_fit, y_eval = np.asarray(y_fit, dtype=np.int64), np.asarray(y_eval, dtype=np.int64)
    if X_fit.shape[0] != y_fit.shape[0] or X_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("X e y deben tener el mismo número de filas")
    if y_fit.max() >= n_classes or y_eval.max() >= n_classes:
        raise ValueError(f"Etiqueta fuera de rango para n_classes={n_classes}")
    dev = _device(device)
    Xf, Xe = _standardize(X_fit, X_eval)
    tr, ho = _split(len(Xf), seed)
    Xt, yt = torch.tensor(Xf[tr], device=dev), torch.tensor(y_fit[tr], device=dev)
    Xh, yh = torch.tensor(Xf[ho], device=dev), torch.tensor(y_fit[ho], device=dev)
    best = None
    for hidden in MLP_GRID["hidden"]:
        for wd in MLP_GRID["weight_decay"]:
            net, ce = _fit_mlp(Xt, yt, Xh, yh, n_classes, hidden, wd, seed)
            if best is None or ce < best[1]:
                best = (net, ce, hidden, wd)
    net, ce_hold, hidden, wd = best
    with torch.no_grad():
        logits = net(torch.tensor(Xe, device=dev))
        ye = torch.tensor(y_eval, device=dev)
        ce = float(F.cross_entropy(logits.double(), ye))
        acc = float((logits.argmax(1) == ye).float().mean())
    h = entropy_bits(y_eval, n_classes)
    info = h - ce / LN2
    return {"probe": "mlp", "accuracy": acc, "ce_bits": ce / LN2, "entropy_bits": h, "info_bits": info,
            "info_bits_clipped": max(0.0, info), "info_fraction": max(0.0, info) / h if h > 0 else float("nan"),
            "hidden": hidden, "weight_decay": wd, "holdout_ce_bits": ce_hold / LN2,
            "n_fit": int(len(Xf)), "n_eval": int(len(Xe)), "n_classes": int(n_classes)}


# ── Factores continuos: regresión kernel ridge ──────────────────────────────────────────────────
def _sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.pow(2).sum(1)[:, None] + B.pow(2).sum(1)[None, :] - 2 * A @ B.T).clamp_min(0)


def _krr_fit_predict(Xt, yt, Xq, gamma: float, alpha: float) -> torch.Tensor:
    K = torch.exp(-gamma * _sqdist(Xt, Xt))
    K.diagonal().add_(alpha)
    coef = torch.cholesky_solve(yt[:, None], torch.linalg.cholesky(K))
    return (torch.exp(-gamma * _sqdist(Xq, Xt)) @ coef)[:, 0]


def _r2(y: torch.Tensor, pred: torch.Tensor) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-12))


def kernel_ridge_probe(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
                       seed: int = 0, device: str | None = None) -> dict:
    """Regresión kernel ridge (núcleo RBF) para un factor continuo; reporta R² en `eval`.

    El ancho del núcleo se expresa en múltiplos de la heurística de la mediana (gamma = escala /
    mediana de las distancias al cuadrado) y se elige, junto con la regularización, por R² en la parte
    retenida de `fit`. El modelo final se reajusta con todo `fit`. Costo O(n³): pensado para n ≤ ~10⁴.
    """
    dev = _device(device)
    Xf, Xe = _standardize(X_fit, X_eval)
    y_fit, y_eval = np.asarray(y_fit, dtype=np.float64), np.asarray(y_eval, dtype=np.float64)
    mu, sd = y_fit.mean(), y_fit.std() + 1e-12
    X = torch.tensor(Xf, device=dev, dtype=torch.float64)
    Xq = torch.tensor(Xe, device=dev, dtype=torch.float64)
    y = torch.tensor((y_fit - mu) / sd, device=dev)
    yq = torch.tensor((y_eval - mu) / sd, device=dev)
    tr, ho = _split(len(Xf), seed)
    tr, ho = torch.tensor(tr, device=dev), torch.tensor(ho, device=dev)
    sample = X[tr[:2000]]
    med = float(_sqdist(sample, sample).flatten().median().clamp_min(1e-12))
    best = None
    for scale in KRR_GRID["gamma_scale"]:
        for alpha in KRR_GRID["alpha"]:
            r2 = _r2(y[ho], _krr_fit_predict(X[tr], y[tr], X[ho], scale / med, alpha))
            if best is None or r2 > best[0]:
                best = (r2, scale, alpha)
    r2_hold, scale, alpha = best
    r2 = _r2(yq, _krr_fit_predict(X, y, Xq, scale / med, alpha))
    return {"probe": "kernel_ridge", "r2": r2, "gamma_scale": scale, "alpha": alpha, "holdout_r2": r2_hold,
            "n_fit": int(len(Xf)), "n_eval": int(len(Xe))}


# ── Lectores sin sonda (texto decodificado) ─────────────────────────────────────────────────────
def reader_information(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Exactitud e información de un lector que entrega un valor por muestra (p. ej. el atributo
    extraído del texto decodificado). `y_pred` puede contener −1 para "no mencionado", que cuenta como
    error y como un valor más al estimar la información mutua."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    h = entropy_bits(y_true, n_classes)
    info = discrete_mi_bits(y_true, y_pred)
    return {"probe": "extractor", "accuracy": float((y_true == y_pred).mean()), "entropy_bits": h,
            "info_bits": info, "info_bits_clipped": info, "info_fraction": info / h if h > 0 else float("nan"),
            "missing": float((y_pred < 0).mean()), "n_eval": int(len(y_true)), "n_classes": int(n_classes)}
