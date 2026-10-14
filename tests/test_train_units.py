"""Etapa 2: piezas sin GPU ni modelos — presupuesto común, orden de datos, etiquetas y pérdidas."""

import pytest
import torch
from omegaconf import OmegaConf

from vlmfid.paths import CONFIGS
from vlmfid.train.config import ALLOWED_MODEL_KEYS, compose_train, load_model_train
from vlmfid.train.data import EpochSampler, steps_per_epoch
from vlmfid.train.objectives import duplicate_mask, infonce_loss

MODELS = ["llava_ov", "internvl3", "qwen35", "vljepa"]


@pytest.mark.parametrize("m", MODELS)
def test_presupuesto_comun(m):
    c = compose_train([f"model={m}"])
    budget = OmegaConf.load(CONFIGS / "train" / "budget.yaml")
    for k in ["epochs", "effective_batch_size", "lr", "connector_lr", "warmup_ratio", "schedule",
              "max_grad_norm", "weight_decay", "selection_metric"]:
        assert c.train[k] == budget.train[k], k
    for k in ["r", "alpha", "dropout"]:
        assert c.lora[k] == budget.lora[k], k
    assert c.data.val_limit == 500 and c.data.val_split == "val"
    assert c.model.quant == "nf4"                               # QLoRA, como la etapa 1
    assert c.lora.connector_modules                              # conector entrenable (decisión)
    assert c.exp_id.startswith(f"train_{c.model.name}_nf4_{c.prompt.id}_r16_e1")


def test_prompts_de_la_etapa_2():
    got = {m: compose_train([f"model={m}"]).prompt.id for m in MODELS}
    assert got == {"llava_ov": "p1_attributes", "internvl3": "p1_attributes", "qwen35": "p1_attributes",
                   "vljepa": "p4m_multiexample"}


def test_archivo_por_modelo_no_puede_cambiar_el_presupuesto(tmp_path, monkeypatch):
    import vlmfid.train.config as tc

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yaml").write_text("prompt: p0_minimal\ntrain:\n  lr: 1.0\n")
    monkeypatch.setattr(tc, "TRAIN_CONFIGS", tmp_path)
    with pytest.raises(ValueError, match="no permitidas"):
        load_model_train("x")
    assert "train.micro_batch_size" in ALLOWED_MODEL_KEYS


def test_infonce_exige_microlote_igual_al_efectivo():
    with pytest.raises(ValueError, match="infonce"):
        compose_train(["model=vljepa", "train.micro_batch_size=32"])
    with pytest.raises(ValueError, match="múltiplo"):
        compose_train(["model=qwen35", "train.micro_batch_size=24"])


def test_smoke_cambia_exp_id():
    assert compose_train(["model=qwen35", "data.train_limit=1280"]).exp_id.endswith("_n1280")


def test_orden_de_datos_determinista_y_reanudable():
    assert steps_per_epoch(125_000, 64) == 1953
    full = list(EpochSampler(1000, 64, seed=0, epoch=0))
    assert len(full) == 15 * 64 and len(set(full)) == len(full)
    assert full == list(EpochSampler(1000, 64, seed=0, epoch=0))            # mismos lotes en los 4 modelos
    assert list(EpochSampler(1000, 64, seed=0, epoch=0, start_step=5)) == full[5 * 64:]
    assert full != list(EpochSampler(1000, 64, seed=0, epoch=1))


def test_mascara_de_duplicados():
    m = duplicate_mask(["a", "b", "a", "c", "b"], "cpu")
    assert m.tolist() == [[False, False, True, False, False], [False, False, False, False, True],
                          [True, False, False, False, False], [False] * 5, [False, True, False, False, False]]


def test_infonce_no_penaliza_captions_repetidos():
    torch.manual_seed(0)
    t = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    t[2] = t[0]                                   # dos imágenes con el mismo caption -> mismo objetivo
    p = t.clone()                                 # predicción perfecta
    sin = infonce_loss(p, t, 0.07)
    con = infonce_loss(p, t, 0.07, ["x", "y", "x", "z"])
    # sin máscara, las filas 0 y 2 reparten su probabilidad entre dos "positivos" idénticos: cada una paga
    # ≈ log 2 en ambas direcciones, es decir ≈ 2·log 2 / 4 ≈ 0,35 de pérdida media extra
    assert sin - con == pytest.approx(0.5 * torch.log(torch.tensor(2.0)).item(), abs=0.05)
    assert con < 0.05
