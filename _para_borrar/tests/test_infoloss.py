"""Estimadores de pérdida de información por etapa (vlmfid.eval.infoloss) y el script scripts/info_loss.py."""

import math
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vlmfid.eval.infoloss import (class_logits, decoder_information, entropy_nats, information_chain,
                                  probe_information)
from vlmfid.paths import ROOT


def _data(n, k, dim, noise, seed):
    g = np.random.default_rng(seed)
    centers = np.linalg.qr(g.normal(size=(dim, dim)))[0][:k]          # k direcciones ortonormales
    y = g.integers(0, k, size=n)
    return (centers[y] + noise * g.normal(size=(n, dim))).astype(np.float32), y, centers.astype(np.float32)


def test_entropia():
    assert entropy_nats(np.array([0, 1, 2] * 10), 3) == pytest.approx(math.log(3))
    assert entropy_nats(np.zeros(5, dtype=int), 3) == 0.0


def test_logits_marginalizan_sobre_las_referencias_de_cada_valor():
    sims = torch.tensor([[0.9, 0.1, 0.8, 0.0]])
    out = class_logits(sims, torch.tensor([0, 1, 0, 1]), 2, 0.1)
    assert out[0, 0].item() == pytest.approx(torch.logsumexp(torch.tensor([9.0, 8.0]), 0).item())
    assert out.argmax(1).item() == 0
    assert torch.equal(class_logits(sims, torch.arange(4), 4, 1.0), sims)       # banco factorizado: identidad


def test_la_sonda_recupera_la_informacion_y_el_ruido_la_borra():
    Xtr, ytr, _ = _data(600, 3, 12, 0.05, 0)
    Xte, yte, _ = _data(300, 3, 12, 0.05, 0)
    clean = probe_information(Xtr, ytr, Xte, yte, 3, device="cpu")
    assert clean["acc"] > 0.99 and clean["info"] > 0.9 * clean["h_nats"]
    g = np.random.default_rng(1)
    noise = probe_information(g.normal(size=Xtr.shape).astype(np.float32), ytr,
                              g.normal(size=Xte.shape).astype(np.float32), yte, 3, device="cpu")
    assert noise["info"] < 0.1                                               # cota inferior: puede ser < 0


def test_pierde_en_el_decoder_si_el_banco_esta_pegado():
    """ŝ separa perfectamente las clases (la sonda lo ve), pero las referencias del banco son casi
    paralelas: la información cae en el decodificador, no antes."""
    Xtr, ytr, centers = _data(600, 3, 12, 0.02, 0)
    Xte, yte, _ = _data(300, 3, 12, 0.02, 0)
    n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    good = n(centers)
    glued = n(np.ones((3, 12), dtype=np.float32) + 1e-3 * centers)
    pre = probe_information(n(Xtr), ytr, n(Xte), yte, 3, device="cpu")
    cols = np.arange(3)
    d_good = decoder_information(n(Xtr) @ good.T, ytr, n(Xte) @ good.T, yte, cols, 3, 0.07)
    d_glued = decoder_information(n(Xtr) @ glued.T, ytr, n(Xte) @ glued.T, yte, cols, 3, 0.07)
    assert d_good["info_calibrated"] > 0.9 * d_good["h_nats"]
    assert d_glued["info_tau_train"] < 0.05                 # a la τ del entrenamiento no pasa casi nada
    chain_good = information_chain("y", 3, None, pre, d_good)
    chain_glued = information_chain("y", 3, None, pre, d_glued)
    assert chain_good["loss_decoder"] < 0.1 < chain_glued["info_predictor"]
    assert chain_glued["loss_decoder"] >= chain_good["loss_decoder"]
    # calibrar τ nunca empeora la cota, y con τ mal puesta la cota a τ de entrenamiento es peor
    assert d_glued["info_calibrated"] >= d_glued["info_tau_train"] - 1e-6


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "scripts"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from _tiny_vlm import tiny_vljepa_describer, write_m3di

    from vlmfid.data.manifest import to_records

    recs = {"train": to_records(write_m3di(tmp_path / "m3di", 48, "train"))}
    return tmp_path, recs, tiny_vljepa_describer


@pytest.mark.parametrize("model", ["vljepa", "vljepa_fac"])
def test_script_rasgos_columnas_y_geometria(tiny, model):
    import info_loss as il
    from PIL import Image

    from vlmfid.models.bank import ATTRS, VALUES

    tmp, recs, make = tiny
    d = make(tmp, recs, target_format="canonical", model=model)
    imgs = [Image.new("RGB", (64, 64), c) for c in ((200, 30, 30), (30, 30, 200))]
    f = il._batch_features(d, imgs, "Describe the image.")
    keys = {f"pred_{a}" for a in ATTRS} if model == "vljepa_fac" else {"pred"}
    assert set(f) == keys | {"vis_mean"} and all(v.shape[0] == 2 for v in f.values())
    cols, groups = il.bank_columns(d)
    assert set(cols) == set(ATTRS)
    geo = il.geometry(d, groups, 0.07)
    if model == "vljepa_fac":
        assert [len(cols[a]) for a in ATTRS] == [7, 8, 3, 3] and set(geo) == set(ATTRS)
        assert all(0 <= g["capacity_nats"] <= g["max_nats"] + 1e-6 for g in geo.values())
    else:       # banco de captions canónicos: el extractor lee los cuatro atributos de cada frase
        assert all(len(c) == len(d.bank_texts) and (c >= 0).all() for c in cols.values())
    y = il.labels(recs["train"])
    assert all(y[a].max() < len(VALUES[a]) for a in ATTRS)
    sims = F.normalize(f["pred_y" if model == "vljepa_fac" else "pred"], dim=-1).numpy() @ \
        F.normalize(d.bank_emb.float(), dim=-1).numpy()[d.fbank.slices["y"] if d.fbank else slice(None)].T
    out = decoder_information(sims, y["y"][:2], sims, y["y"][:2], cols["y"], 3, 0.07)
    assert np.isfinite(out["info_calibrated"])
