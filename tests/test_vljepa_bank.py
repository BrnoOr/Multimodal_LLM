"""Banco factorizado de VL-JEPA, InfoNCE bidireccional contra un banco cerrado y su configuración."""

import math
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from vlmfid.eval.attributes import score_texts
from vlmfid.models.bank import (ATTRS, QUERIES, VALUES, FactorizedBank, caption_groups, geometry_report,
                                group_capacity, margin_bound, query_rotation, truth_values)
from vlmfid.train.config import compose_train
from vlmfid.train.objectives import (bank_logits, infonce_loss, reverse_bank_infonce_loss,
                                     symmetric_bank_infonce_loss)


def _rec(shape=2, x=0, y=1, hue=0.6, name="blue"):
    return {"id": "r", "caption": "", "latents_image": {"object_shape": shape, "object_xpos": x, "object_ypos": y,
                                                        "object_color": hue},
            "latents_text": {"object_color_name": name}}


def test_banco_tiene_una_frase_por_valor():
    b = FactorizedBank.build()
    assert len(b) == 7 + 8 + 3 + 3 == 21 and len(set(b.texts)) == 21
    assert [b.slices[a].stop - b.slices[a].start for a in ATTRS] == [7, 8, 3, 3]
    for a in ATTRS:
        for v in VALUES[a]:
            assert b.entries[b.index(a, v)] == (a, v) and v in b.texts[b.index(a, v)]
    assert set(QUERIES) == set(ATTRS)
    g = b.group_mask()
    assert g.sum().item() == 7 * 7 + 8 * 8 + 3 * 3 + 3 * 3


def test_decodificacion_compone_el_caption_canonico():
    b, rec = FactorizedBank.build(), _rec()
    truth = truth_values(rec)
    assert truth == {"shape": "dragon", "color": "blue", "y": "mid", "x": "left"}
    emb = F.normalize(torch.randn(21, 16, generator=torch.Generator().manual_seed(0)), dim=-1)
    pred = {a: emb[b.index(a, truth[a])][None] for a in ATTRS}            # predicción perfecta
    out = b.decode(pred, emb, 0.07)[0]
    assert out["values"] == truth
    assert out["text"] == "A blue dragon is at the mid-left of the image."
    assert all(m > 0 for m in out["margin_nats"].values())
    assert all(abs(sum(p.values()) - 1) < 1e-2 for p in out["prob"].values())
    # el caption compuesto pasa por el extractor de la etapa 1 sin pérdida
    m = score_texts([out["text"]], [rec])
    assert m["shape"] == m["pos"] == m["color"] == 1.0


def test_rotacion_de_consultas():
    ids = [f"img{i}" for i in range(400)]
    for i in ids[:50]:        # tras 4 épocas, cada imagen respondió una vez cada atributo
        assert {query_rotation(i, 0, e) for e in range(4)} == set(ATTRS)
    counts = Counter(query_rotation(i, 0, 0) for i in ids)
    assert set(counts) == set(ATTRS) and min(counts.values()) > 60       # ~1/4 por atributo en cada época
    assert query_rotation("img3", 0, 5) == query_rotation("img3", 0, 1)   # determinista, periodo 4


def test_cota_del_margen_y_capacidad():
    assert margin_bound(1.0, 0.07) == 0.0
    assert margin_bound(0.99, 0.07) == pytest.approx(math.sqrt(0.02) / 0.07)
    ortho = torch.eye(3, 8)
    assert group_capacity(ortho, 0.07)["capacity_nats"] == pytest.approx(math.log(3), abs=1e-3)
    # tres referencias casi paralelas: el decodificador no puede transmitir log 3 aunque el predictor sea perfecto
    base = F.normalize(torch.ones(8), dim=-1)
    near = F.normalize(base[None] + 1e-3 * torch.eye(3, 8), dim=-1)
    cap = group_capacity(near, 0.07)
    assert 0 <= cap["capacity_nats"] < 0.05
    rep = geometry_report(torch.cat([ortho, near]), {"sep": [0, 1, 2], "pegado": [3, 4, 5]}, 0.07)
    assert rep["sep"]["capacity_nats"] > rep["pegado"]["capacity_nats"]
    assert rep["pegado"]["cos_max"] > 0.999 and rep["pegado"]["margin_bound_nats_hardest_pair"] < 1.0


def test_grupos_de_captions_que_difieren_en_un_atributo():
    b = FactorizedBank.build()
    texts, attr_of = [], {}
    for s in VALUES["shape"][:2]:
        for c in VALUES["color"][:2]:
            for y in VALUES["y"]:
                for x in VALUES["x"]:
                    t = b.compose({"shape": s, "color": c, "y": y, "x": x})
                    texts.append(t)
                    attr_of[t] = {"shape": s, "color": c, "y": y, "x": x}
    g = caption_groups(texts, attr_of)
    assert len(g["y"]) == 2 * 2 * 3 and all(len(m) == 3 for m in g["y"])
    assert g["shape"] == [] and g["color"] == []          # faltan valores: no hay grupos completos
    i, j, k = g["y"][0]
    assert [attr_of[texts[n]]["y"] for n in (i, j, k)] == VALUES["y"]


def test_inversa_coincide_con_el_paper_si_los_textos_son_distintos():
    g = torch.Generator().manual_seed(0)
    pred, bank = torch.randn(6, 8, generator=g), torch.randn(6, 8, generator=g)
    idx = torch.arange(6)
    logits = bank_logits(pred, bank, 0.07)
    assert reverse_bank_infonce_loss(logits, idx).item() == pytest.approx(F.cross_entropy(logits.T, idx).item(), rel=1e-5)
    sym, fwd = symmetric_bank_infonce_loss(pred, bank, idx, 0.07)
    assert sym.item() == pytest.approx(infonce_loss(pred, bank, 0.07).item(), rel=1e-5)
    assert fwd.item() == pytest.approx(F.cross_entropy(logits, idx).item(), rel=1e-5)


def test_inversa_multipositiva_no_penaliza_textos_repetidos():
    bank = F.normalize(torch.eye(3, 8), dim=-1)
    idx = torch.tensor([0, 0, 1, 2])
    perfect = bank[idx]
    logits = bank_logits(perfect, bank, 0.07)
    assert reverse_bank_infonce_loss(logits, idx).item() < 1e-4      # dos positivos del texto 0: pérdida ~0
    assert F.cross_entropy(logits.T[idx], torch.arange(4)).item() > 0.3   # la forma de un solo positivo sí penaliza
    absent = reverse_bank_infonce_loss(bank_logits(perfect[:2], bank, 0.07), idx[:2])   # textos 1 y 2 ausentes
    assert torch.isfinite(absent)


def test_config_del_ajuste_v3():
    c = compose_train(["model=vljepa_fac"])
    assert c.exp_id == "train_vljepa_open_msrvtt_fac_nf4_qattr_tcan_r16_e1"
    assert c.model.bank.kind == "factorized" and c.model.bank.target_format == "canonical"
    assert c.objective.kind == "infonce" and c.objective.negatives == "bank" and c.objective.symmetric
    assert list(c.lora.text_modules) == ["y_encoder.projector"]
    assert list(c.lora.connector_modules) == ["predictor.vis_proj", "predictor.out_proj"]
    assert c.train.text_lr == pytest.approx(0.05 * c.train.lr)          # multiplicador del paper
    assert c.train.micro_batch_size == c.train.effective_batch_size == 64
    assert c.train.epochs == 1 and c.train.lr == 2e-4 and c.lora.r == 16      # presupuesto común intacto
    assert compose_train(["model=vljepa_fac", "train.epochs=4", "seed=1"]).exp_id.endswith("_qattr_tcan_r16_e4_s1")


def test_config_del_run_publicado_no_cambia():
    c = compose_train(["model=vljepa"])
    assert c.exp_id == "train_vljepa_open_msrvtt_nf4_p1_attributes_tcan_r16_e1"
    assert "kind" not in c.model.bank and "text_modules" not in c.lora and "symmetric" not in c.objective
    assert list(c.lora.connector_modules) == ["vis_proj"]


def test_combinaciones_invalidas():
    with pytest.raises(ValueError, match="micro_batch_size"):
        compose_train(["model=vljepa_fac", "train.micro_batch_size=32"])
    with pytest.raises(ValueError, match="factorized"):
        compose_train(["model=vljepa_fac", "target.format=original"])
    with pytest.raises(ValueError, match="text_modules"):
        compose_train(["model=vljepa", "lora.text_modules=[y_encoder.projector]"])
    with pytest.raises(ValueError, match="symmetric"):
        compose_train(["model=vljepa", "objective.negatives=batch", "objective.symmetric=true"])
    # sin la simétrica se puede acumular gradiente en micro-lotes
    assert compose_train(["model=vljepa_fac", "objective.symmetric=false", "train.micro_batch_size=32"])
