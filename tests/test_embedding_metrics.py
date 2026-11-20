"""Métricas de embeddings del diagnóstico de cuantización (vlmfid.eval.embedding_metrics).

Casos construidos a mano: vectores base y permutaciones, donde el resultado se conoce sin calcular.
"""

import math

import numpy as np
import pytest

from vlmfid.eval.embedding_metrics import (caption_codes, infonce_fixed_batches, paired_cosine,
                                           retrieval_recall, structure_spearman)


def _eye(n):
    return np.eye(n, dtype=np.float32)


def test_coseno_por_par_ignora_la_norma():
    a = np.array([[1, 0], [0, 2]], dtype=np.float32)
    b = np.array([[3, 0], [1, 1]], dtype=np.float32)
    assert paired_cosine(a, b) == pytest.approx([1.0, 1 / math.sqrt(2)], abs=1e-6)


def test_coseno_formas_distintas_falla():
    with pytest.raises(ValueError):
        paired_cosine(_eye(3), _eye(2))


def test_codigos_de_caption():
    assert caption_codes(["a", "b", "a"]).tolist() == [0, 1, 0]


def test_recuperacion_perfecta():
    r = retrieval_recall(_eye(4), _eye(4), ["a", "b", "c", "d"])
    assert r["recall@1_p2t"] == r["recall@1_t2p"] == 1.0
    assert r["chance@1"] == pytest.approx(0.25)


def test_recuperacion_con_permutacion():
    # Ŝ_Y de la imagen i apunta al S_Y de la imagen (i+1) mod 4: recall@1 = 0, recall@5 = 1 (k > N)
    pred = np.roll(_eye(4), 1, axis=1)
    r = retrieval_recall(pred, _eye(4), ["a", "b", "c", "d"], ks=(1, 5))
    assert r["recall@1_p2t"] == 0.0 and r["recall@1_t2p"] == 0.0
    assert r["recall@5_p2t"] == 1.0


def test_recuperacion_duplicado_cuenta_como_acierto():
    # imágenes 0 y 1 tienen el mismo caption; Ŝ_Y de 0 cae sobre S_Y de 1 → acierto
    pred = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    r = retrieval_recall(pred, _eye(3), ["x", "x", "z"])
    assert r["recall@1_p2t"] == 1.0
    assert r["chance@1"] == pytest.approx((2 + 2 + 1) / 3 / 3)


def test_infonce_lotes_fijos_descarta_el_resto():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(10, 8)).astype(np.float32)
    out = infonce_fixed_batches(p, p, [str(i) for i in range(10)], batch_size=4, temperature=0.07)
    assert out["infonce_n_batches"] == 2
    assert out["infonce_batch_acc"] == 1.0            # pred = target: el positivo siempre gana
    assert out["infonce"] < out["infonce_chance"]


def test_infonce_azar_cerca_de_log_b():
    # todos los pares idénticos en dirección → logits iguales → pérdida = log(B)
    p = np.ones((8, 4), dtype=np.float32)
    out = infonce_fixed_batches(p, p, [str(i) for i in range(8)], batch_size=8, temperature=0.07)
    assert out["infonce"] == pytest.approx(math.log(8), abs=1e-4)


def test_infonce_sin_lote_completo_falla():
    with pytest.raises(ValueError):
        infonce_fixed_batches(_eye(3), _eye(3), ["a", "b", "c"], batch_size=4, temperature=0.07)


def test_estructura_identica_y_rotada_es_1():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))          # rotación: conserva todos los cosenos
    caps = [str(i) for i in range(20)]
    assert structure_spearman(a, a, caps) == pytest.approx(1.0)
    assert structure_spearman(a, a @ q, caps) == pytest.approx(1.0)


def test_estructura_independiente_cerca_de_0():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(60, 16)), rng.normal(size=(60, 16))
    assert abs(structure_spearman(a, b)) < 0.15


def test_estructura_descarta_duplicados():
    a = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    with pytest.raises(ValueError):                         # 2 textos distintos: no alcanza
        structure_spearman(a[:3], a[:3], ["x", "x", "y"])
    assert structure_spearman(a, a, ["x", "x", "y", "z"]) == pytest.approx(1.0)

