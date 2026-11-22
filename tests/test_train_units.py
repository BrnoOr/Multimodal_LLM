"""Etapa 2: piezas sin GPU ni modelos — presupuesto común, candado de hiperparámetros, orden de datos y
pérdidas (docs/metodologia.md §2.3)."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from vlmfid.paths import CONFIGS
from vlmfid.train.config import ALLOWED_MODEL_KEYS, compose_train, load_model_train
from vlmfid.train.data import EpochSampler, steps_per_epoch
from vlmfid.train.objectives import duplicate_mask, duplicate_stats, infonce_loss

MODELS = ["llava_ov", "internvl3", "qwen35", "vljepa"]
SHORT = ["train.max_steps=10"]          # corrida corta: la única permitida mientras no se confirmen los valores
# Ganadores de la grilla anotados en configs/train/models/ (docs/resultados_etapa2.md §2): (lr, r, multiplicador).
# VL-JEPA aún sin ganador: hereda los de budget.yaml.
WINNERS = {"internvl3": (1e-4, 16, 5.0), "llava_ov": (2e-4, 64, 5.0), "qwen35": (1e-4, 16, 1.0)}


@pytest.mark.parametrize("m", MODELS)
def test_presupuesto_comun(m):
    c = compose_train([f"model={m}", *SHORT])
    budget = OmegaConf.load(CONFIGS / "train" / "budget.yaml")
    for k in ["epochs", "effective_batch_size", "y_encoder_lr_mult", "warmup_ratio",
              "schedule", "max_grad_norm", "weight_decay", "optimizer", "betas"]:
        assert c.train[k] == budget.train[k], k
    for k in ["alpha", "dropout"]:                            # r y conector: por modelo (grilla)
        assert c.lora[k] == budget.lora[k], k
    assert c.data.val_limit == 500 and c.data.val_split == "val" and c.data.train_split == "train"
    assert c.model.quant == "nf4"                               # QLoRA, como la etapa 1
    assert c.lora.connector_modules                              # conector entrenable
    assert c.prompt.id == "p1_attributes"                        # el mismo prompt en los 4 (decisión del usuario)
    lr, r, mult = WINNERS.get(m, (budget.train.lr, budget.lora.r, budget.train.connector_lr_mult))
    assert (c.train.lr, c.lora.r, c.train.connector_lr_mult) == (lr, r, mult)
    tag = "" if mult == 1 else f"_c{mult:g}"
    assert c.exp_id == f"train_{c.model.name}_nf4_p1_attributes_r{r}_lr{lr:g}{tag}_e1_m10"


def test_seleccion_por_familia():
    for m in ["llava_ov", "internvl3", "qwen35"]:
        c = compose_train([f"model={m}", *SHORT])
        assert c.train.selection_metric == "attr_mean" and c.objective.kind == "lm"
    c = compose_train(["model=vljepa", *SHORT])
    assert c.train.selection_metric == "val_loss" and c.objective.kind == "infonce"           # P7
    assert c.train.micro_batch_size == c.train.effective_batch_size                         # lote físico = efectivo
    assert list(c.lora.connector_modules) == ["predictor.vis_proj", "predictor.out_proj", "y_encoder.projector"]


def test_candado_de_hiperparametros():
    with pytest.raises(ValueError, match="sin confirmar"):
        compose_train(["model=qwen35"])
    c = compose_train(["model=qwen35", "train.hparams_confirmed=true"])
    assert c.exp_id == "train_qwen35_08b_base_nf4_p1_attributes_r16_lr0.0001_e1"


def test_control_bf16_y_variantes_en_el_exp_id():
    assert "_bf16_" in compose_train(["model=llava_ov", "model.quant=bf16", *SHORT]).exp_id
    c = compose_train(["model=qwen35", "train.lr=5e-4", "seed=1", *SHORT])
    assert c.exp_id == "train_qwen35_08b_base_nf4_p1_attributes_r16_lr0.0005_e1_s1_m10"


def test_rechaza_lo_que_rompe_el_protocolo():
    with pytest.raises(ValueError, match="lote físico"):
        compose_train(["model=vljepa", "train.micro_batch_size=32", *SHORT])   # ≠ lote efectivo 64 (P8)
    with pytest.raises(ValueError, match="val_loss"):
        compose_train(["model=vljepa", "train.selection_metric=attr_mean", *SHORT])
    with pytest.raises(ValueError, match="exactitud por atributo"):
        compose_train(["model=qwen35", "train.selection_metric=val_loss", *SHORT])
    with pytest.raises(ValueError, match="no corresponde"):
        compose_train(["model=qwen35", "objective.kind=infonce", *SHORT])
    with pytest.raises(ValueError, match="D4"):
        compose_train(["model=vljepa", "train.effective_batch_size=256", "train.micro_batch_size=256", *SHORT])
    with pytest.raises(ValueError, match="train_limit"):
        compose_train(["model=qwen35", "data.train_limit=100", *SHORT])
    with pytest.raises(ValueError, match="test"):
        compose_train(["model=qwen35", "data.val_split=test", *SHORT])
    with pytest.raises(ValueError, match="adapter_path"):
        compose_train(["model=qwen35", "model.adapter_path=runs/x", *SHORT])
    with pytest.raises(ValueError, match="decoder_path"):
        compose_train(["model=vljepa", "model.decoder_path=runs/x", *SHORT])
    with pytest.raises(ValueError, match="sin valor"):
        compose_train(["model=qwen35", "train.lr=null", *SHORT])


def test_archivo_por_modelo_no_puede_cambiar_el_presupuesto(tmp_path, monkeypatch):
    import vlmfid.train.config as tc

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.yaml").write_text("train:\n  epochs: 3\n")
    monkeypatch.setattr(tc, "TRAIN_CONFIGS", tmp_path)
    with pytest.raises(ValueError, match="no permitidas"):
        load_model_train("x")
    assert "train.epochs" not in ALLOWED_MODEL_KEYS and "train.effective_batch_size" not in ALLOWED_MODEL_KEYS


@pytest.mark.parametrize("m", MODELS)
def test_archivos_reales_solo_usan_claves_permitidas(m):
    load_model_train(m)


def test_orden_de_datos_comun_y_reanudable():
    n, b = 1000, 128
    assert steps_per_epoch(n, b, drop_last=False) == 8 and steps_per_epoch(125_000, 128, drop_last=False) == 977
    full = list(EpochSampler(n, b, seed=0, epoch=0, drop_last=False))
    assert sorted(full) == list(range(n))                                   # todo train, sin repetir
    assert list(EpochSampler(n, b, seed=0, epoch=0, start_step=3, drop_last=False)) == full[3 * b:]
    assert list(EpochSampler(n, b, seed=0, epoch=1, drop_last=False)) != full


def test_infonce_bidireccional_como_open_vljepa():
    torch.manual_seed(0)
    p, t = torch.randn(6, 8), torch.randn(6, 8)
    loss, stats = infonce_loss(p, t, 0.07)
    pn, tn = torch.nn.functional.normalize(p, dim=-1), torch.nn.functional.normalize(t, dim=-1)
    logits = pn @ tn.T / 0.07
    lab = torch.arange(6)
    ref = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
    assert torch.allclose(loss, ref)
    assert math.isclose((stats["loss_p2t"] + stats["loss_t2p"]) / 2, float(ref), rel_tol=1e-5)
    # pares perfectos -> pérdida baja y recuperación exacta
    loss_ok, s_ok = infonce_loss(t, t, 0.07)
    assert float(loss_ok) < 0.1 and s_ok["acc_p2t"] == s_ok["acc_t2p"] == 1.0


def test_duplicados_d4():
    caps = ["a", "b", "a", "c", "a"]
    m = duplicate_mask(caps, "cpu")
    assert m[0, 2] and m[2, 4] and not m[0, 0] and not m[0, 1]
    assert duplicate_stats(caps) == {"dup_pairs": 3, "dup_samples": 3}
    t = torch.randn(5, 4)
    masked, _ = infonce_loss(t, t, 0.07, caps)
    assert torch.isfinite(masked)
