"""Etapa 2: piezas sin GPU ni modelos — presupuesto común, orden de datos, etiquetas y pérdidas."""

import pytest
import torch
from omegaconf import OmegaConf

from vlmfid.paths import CONFIGS
from vlmfid.train.config import ALLOWED_MODEL_KEYS, compose_train, load_model_train
from vlmfid.train.data import EpochSampler, steps_per_epoch
from vlmfid.train.objectives import bank_infonce_loss, duplicate_mask, infonce_loss

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
    assert c.target.format == "canonical" and c.data.holdout is None
    assert c.train.selection_metric == "shape_pos"
    assert c.exp_id == f"train_{c.model.name}_nf4_{c.prompt.id}_tcan_r16_e1"


def test_prompts_de_la_etapa_2():
    # reformulación: el mismo prompt en los 4 (VL-JEPA dejaba p4m, truncado a 64 tokens)
    assert {compose_train([f"model={m}"]).prompt.id for m in MODELS} == {"p1_attributes"}


def test_exp_id_de_las_variantes():
    assert compose_train(["model=qwen35", "target.format=original"]).exp_id.endswith("_torig_r16_e1")
    c = compose_train(["model=qwen35", "train.epochs=2", "data.holdout=compositional", "seed=1"])
    assert c.exp_id == "train_qwen35_08b_base_nf4_p1_attributes_tcan_r16_e2_hc_s1"
    with pytest.raises(ValueError, match="target.format"):
        compose_train(["model=qwen35", "target.format=json"])
    with pytest.raises(ValueError, match="holdout"):
        compose_train(["model=qwen35", "data.holdout=random"])


def test_vljepa_banco_en_el_formato_del_objetivo():
    c = compose_train(["model=vljepa"])
    assert c.objective.negatives == "bank" and c.model.bank.target_format == "canonical"
    assert compose_train(["model=vljepa", "target.format=original"]).model.bank.target_format == "original"


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
        compose_train(["model=vljepa", "objective.negatives=batch", "train.micro_batch_size=32"])
    # con el banco como negativos cada muestra es independiente: se puede acumular gradiente
    assert compose_train(["model=vljepa", "train.micro_batch_size=32"]).train.micro_batch_size == 32
    with pytest.raises(ValueError, match="múltiplo"):
        compose_train(["model=qwen35", "train.micro_batch_size=24"])


def test_train_completo_sin_limite():
    c = compose_train(["model=qwen35"])
    assert "train_limit" not in c.data and c.train.drop_last is False
    with pytest.raises(ValueError, match="train_limit"):
        compose_train(["model=qwen35", "data.train_limit=1280"])


def test_smoke_cambia_exp_id():
    # un run cortado con max_steps escribe summary.json: no puede compartir exp_id con el run completo
    assert compose_train(["model=qwen35", "train.max_steps=20"]).exp_id.endswith("_tcan_r16_e1_m20")


@pytest.mark.parametrize("m", MODELS)
def test_test_usa_el_adaptador_y_el_subconjunto_del_zeroshot(m):
    from vlmfid.train.config import compose_test

    c = compose_train([f"model={m}"])
    t = compose_test(c)
    # mismos exp_id que configs/queues/stage2b_test.txt
    assert t.exp_id == f"finetuned_{c.exp_id.removeprefix('train_')}_test"
    assert t.model.adapter_path == f"runs/{c.exp_id}/adapter_best"
    assert t.stage == "finetuned" and t.data.split == "test" and t.data.limit == 10000
    assert t.data.subset_seed == 0 and t.prompt.id == c.prompt.id
    if m == "vljepa":
        assert t.model.bank.target_format == "canonical" and t.infer.save_embeddings
    smoke = compose_test(compose_train([f"model={m}", "train.max_steps=20", "test.limit=64"]))
    assert smoke.exp_id.endswith("_m20_test_n64") and smoke.data.limit == 64


def test_orden_de_datos_determinista_y_reanudable():
    assert steps_per_epoch(125_000, 64, drop_last=False) == 1954            # todo train: el último paso con 8
    assert steps_per_epoch(125_000, 64, drop_last=True) == 1953
    every = list(EpochSampler(1000, 64, seed=0, epoch=0, drop_last=False))
    assert sorted(every) == list(range(1000))                               # sin límite: cada imagen una vez
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


def test_infonce_contra_el_banco():
    torch.manual_seed(0)
    bank = torch.nn.functional.normalize(torch.randn(10, 8), dim=-1)
    idx = torch.tensor([3, 3, 7, 1])                             # dos muestras con el mismo objetivo: misma clase
    perfect = bank_infonce_loss(bank[idx], bank, idx, 0.07) / 4
    noisy = bank_infonce_loss(torch.randn(4, 8), bank, idx, 0.07) / 4
    assert perfect < noisy
    # suma por muestra: partir el lote en micro-lotes y sumar da exactamente lo mismo (acumulación válida)
    pred = torch.randn(4, 8)
    whole = bank_infonce_loss(pred, bank, idx, 0.07)
    parts = bank_infonce_loss(pred[:2], bank, idx[:2], 0.07) + bank_infonce_loss(pred[2:], bank, idx[2:], 0.07)
    assert torch.allclose(whole, parts)
