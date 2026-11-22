"""Grilla de hiperparámetros de la etapa 2 (configs/train/grid.yaml, scripts/make_grid.py).

No carga modelos: compone configuraciones con compose_train y compara texto.
"""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_grid as mg  # noqa: E402
from vlmfid.train.config import ALLOWED_MODEL_KEYS, GRID_KEYS, compose_train  # noqa: E402


def test_grilla_real_es_el_producto_completo():
    g = mg.load_grid()
    assert len(mg.combinations(g)) == 3 * 2 * 2
    lines = mg.queue_lines(g)
    assert len(lines) == 12 * len(g["models"])
    assert all("train.max_steps=150" in line for _, line in lines)


def test_exp_id_distintos_en_toda_la_grilla():
    g = mg.load_grid()
    ids = mg.validate_lines(mg.queue_lines(g))
    assert len(set(ids)) == len(ids)


def test_cola_versionada_coincide_con_la_grilla():
    # la cola del repo es la que corre en el cluster: no puede desviarse de grid.yaml
    assert mg.main(["--check"]) == 0


def test_colas_por_gpu_coinciden_con_el_reparto(tmp_path):
    g = mg.load_grid()
    parts = mg.split_by_model(g, mg.queue_lines(g), 2)
    for i, part in enumerate(parts):
        name = f"stage2_grid_g{i}.txt"
        assert (mg.QUEUES / name).read_text(encoding="utf-8") == mg.render(g, part, name)
    # cada modelo queda entero en una sola GPU
    for model in g["models"]:
        assert sum(any(m == model for m, _ in part) for part in parts) == 1


def test_eje_no_permitido_falla(tmp_path):
    p = tmp_path / "g.yaml"
    OmegaConf.save(OmegaConf.create({"steps": 10, "eval_every": 5, "quant": "nf4", "models": ["qwen35"],
                                     "axes": {"train.epochs": [1, 2]}}), p)
    with pytest.raises(ValueError, match="no permitidos"):
        mg.load_grid(p)


def test_eje_con_repetidos_falla(tmp_path):
    p = tmp_path / "g.yaml"
    OmegaConf.save(OmegaConf.create({"steps": 10, "eval_every": 5, "quant": "nf4", "models": ["qwen35"],
                                     "axes": {"lora.r": [16, 16]}}), p)
    with pytest.raises(ValueError, match="repetidos"):
        mg.load_grid(p)


def test_ejes_de_la_grilla_se_pueden_fijar_por_modelo():
    assert set(GRID_KEYS) <= ALLOWED_MODEL_KEYS
    assert "lora.alpha" not in ALLOWED_MODEL_KEYS          # α queda común (fijo en 16)


def test_exp_id_marca_el_conector_solo_si_no_es_1():
    base = ["model=qwen35", "train.max_steps=10"]
    c1 = compose_train(base + ["train.connector_lr_mult=1"])
    c5 = compose_train(base + ["train.connector_lr_mult=5", "lora.r=16"])
    assert "_c" not in c1.exp_id.split("_lr")[1]
    assert c5.exp_id == f"train_{c5.model.name}_nf4_p1_attributes_r16_lr0.0001_c5_e1_m10"   # lr: ganador de Qwen3.5 (configs/train/models/qwen35.yaml)
