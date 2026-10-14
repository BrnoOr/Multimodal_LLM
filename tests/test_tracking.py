"""Reanudación de runs: qué cambios de configuración se aceptan y cuáles no."""

import pytest
from omegaconf import OmegaConf

from vlmfid.tracking import RunTracker


def _cfg(**model):
    base = {"exp_id": "x", "seed": 0, "model": {"name": "m", "quant": "nf4"},
            "data": {"split": "test", "num_workers": 2}, "infer": {"batch_size": 8}}
    base["model"].update(model)
    return OmegaConf.create(base)


def _save(tmp_path, cfg, n_pred: int):
    (tmp_path / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (tmp_path / "predictions.jsonl").write_text("".join(f'{{"id": "{i}"}}\n' for i in range(n_pred)))


def test_config_distinta_con_predicciones_falla(tmp_path):
    _save(tmp_path, _cfg(), n_pred=3)
    with pytest.raises(RuntimeError, match="difiere"):
        RunTracker(tmp_path, _cfg(chat_template_from="Qwen/Qwen3.5-0.8B")).check_compatible()


def test_run_sin_predicciones_se_reanuda_con_config_nueva(tmp_path):
    # caso real: el run de Qwen3.5-Base falló al aplicar la plantilla, antes de guardar nada
    _save(tmp_path, _cfg(), n_pred=0)
    RunTracker(tmp_path, _cfg(chat_template_from="Qwen/Qwen3.5-0.8B")).check_compatible()


def test_claves_seguras_no_bloquean_la_reanudacion(tmp_path):
    _save(tmp_path, _cfg(), n_pred=3)
    cfg = _cfg(llama_name="unsloth/Llama-3.2-1B")
    cfg.infer.batch_size = 32
    cfg.data.num_workers = 4
    RunTracker(tmp_path, cfg).check_compatible()
