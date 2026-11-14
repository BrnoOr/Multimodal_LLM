"""Sondas de información (vlmfid.eval.probes) sobre datos sintéticos con respuesta conocida. CPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vlmfid.eval.probes import (
    discrete_mi_bits,
    entropy_bits,
    kernel_ridge_probe,
    mlp_probe,
    reader_information,
)


def test_entropia_en_bits():
    assert entropy_bits(np.array([0, 1, 0, 1])) == pytest.approx(1.0)
    assert entropy_bits(np.array([0, 1, 2, 3])) == pytest.approx(2.0)
    assert entropy_bits(np.array([2, 2, 2])) == pytest.approx(0.0)
    assert entropy_bits(np.array([0, 0, 0, 1]), n_classes=5) == pytest.approx(0.8113, abs=1e-4)


def test_informacion_mutua_discreta():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 20_000)
    assert discrete_mi_bits(a, a) == pytest.approx(entropy_bits(a), abs=1e-9)          # copia: I = H
    assert discrete_mi_bits(a, (a + 1) % 4) == pytest.approx(entropy_bits(a), abs=1e-9)  # biyección: igual
    assert discrete_mi_bits(a, rng.integers(0, 4, 20_000)) < 0.01                      # independientes: ~0
    assert discrete_mi_bits(a, a // 2) == pytest.approx(1.0, abs=0.01)                 # se pierde 1 de 2 bits


def _clusters(n, n_classes, dim, noise, seed):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(123).normal(size=(n_classes, dim)) * 3
    y = rng.integers(0, n_classes, n)
    return (centers[y] + noise * rng.normal(size=(n, dim))).astype(np.float32), y


def test_sonda_mlp_recupera_un_factor_presente():
    Xf, yf = _clusters(1500, 4, 16, 0.5, 0)
    Xe, ye = _clusters(500, 4, 16, 0.5, 1)
    r = mlp_probe(Xf, yf, Xe, ye, 4, device="cpu")
    assert r["accuracy"] > 0.97
    assert r["entropy_bits"] == pytest.approx(2.0, abs=0.02)
    assert r["info_bits"] > 1.8 and r["info_fraction"] > 0.9                 # casi toda la entropía
    assert r["hidden"] in (64, 256, 1024) and r["n_classes"] == 4


def test_sonda_mlp_no_inventa_informacion():
    rng = np.random.default_rng(0)
    Xf, Xe = rng.normal(size=(1500, 16)).astype(np.float32), rng.normal(size=(500, 16)).astype(np.float32)
    yf, ye = rng.integers(0, 4, 1500), rng.integers(0, 4, 500)
    r = mlp_probe(Xf, yf, Xe, ye, 4, device="cpu")
    assert r["accuracy"] < 0.35                                             # azar = 0.25
    assert r["info_bits"] < 0.05 and r["info_bits_clipped"] >= 0.0          # la cota no supera lo real
    with pytest.raises(ValueError):
        mlp_probe(Xf, yf, Xe, ye, 3, device="cpu")                           # etiqueta fuera de rango


def test_kernel_ridge_recupera_funcion_no_lineal_y_descarta_ruido():
    rng = np.random.default_rng(0)
    Xf, Xe = rng.uniform(-1, 1, (800, 3)).astype(np.float32), rng.uniform(-1, 1, (300, 3)).astype(np.float32)

    def target(X):
        return np.sin(3 * X[:, 0]) + X[:, 1] ** 2

    r = kernel_ridge_probe(Xf, target(Xf), Xe, target(Xe), device="cpu")
    assert r["r2"] > 0.9
    noise = kernel_ridge_probe(Xf, rng.normal(size=800), Xe, rng.normal(size=300), device="cpu")
    assert noise["r2"] < 0.1


def test_lector_sin_sonda():
    y = np.array([0, 1, 2, 0, 1, 2] * 50)
    r = reader_information(y, y, 3)
    assert r["accuracy"] == 1.0 and r["info_bits"] == pytest.approx(entropy_bits(y), abs=1e-9)
    pred = y.copy()
    pred[:100] = -1                                   # un tercio sin mencionar: cuenta como error
    r = reader_information(y, pred, 3)
    assert r["accuracy"] == pytest.approx(2 / 3) and r["missing"] == pytest.approx(1 / 3)
    assert 0 < r["info_bits"] < entropy_bits(y)
