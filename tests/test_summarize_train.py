"""scripts/summarize_train.py: lectura de runs de la etapa 2 y regla del barrido de LR.

Construye carpetas de run sintéticas con los mismos archivos que escribe vlmfid.train.loop.
"""

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_train as st  # noqa: E402


def _run(tmp: Path, name: str, model: str, lr: float, metric: str, val: list[tuple[int, float]],
         mem=(10.0, 12.5)) -> Path:
    d = tmp / name
    d.mkdir()
    OmegaConf.save(OmegaConf.create({"exp_id": name, "model": {"name": model, "quant": "nf4"},
                                     "train": {"lr": lr, "selection_metric": metric, "max_steps": 150}}),
                   d / "config.yaml")
    with (d / "train_log.jsonl").open("w") as f:
        f.write(json.dumps({"step": 10, "loss": 2.0, "peak_mem_gb": mem[0], "micro_batch": 16}) + "\n")
        f.write("{truncada\n")
        f.write(json.dumps({"step": 150, "loss": 1.0, "peak_mem_gb": mem[1], "micro_batch": 16}) + "\n")
    with (d / "val_log.jsonl").open("w") as f:
        for step, v in val:
            f.write(json.dumps({"step": step, metric: v}) + "\n")
    (d / "summary.json").write_text(json.dumps({"trainable_lora_llm": 123, "lora_layers_llm": 168, "lr": lr}))
    return d


def test_resumen_vlm(tmp_path):
    d = _run(tmp_path, "a", "qwen", 2e-4, "attr_mean", [(0, 0.40), (50, 0.70), (150, 0.65)])
    r = st.summarize_run(d)
    assert r["val_step0"] == 0.40 and r["val_last"] == 0.65
    assert (r["best_step"], r["best_value"]) == (50, 0.70)
    assert r["peak_mem_gb"] == 12.5 and r["loss_first"] == 2.0 and r["loss_last"] == 1.0
    assert r["lora_layers_llm"] == 168 and "lr" not in {k for k in r if k.startswith("trainable")}


def test_resumen_vljepa_minimiza(tmp_path):
    d = _run(tmp_path, "b", "vljepa", 1e-4, "val_loss", [(0, 3.0), (50, 1.0), (150, 1.5)])
    r = st.summarize_run(d)
    assert r["sel_mode"] == "min" and (r["best_step"], r["best_value"]) == (50, 1.0)


def test_ganador_usa_el_ultimo_paso_no_el_mejor(tmp_path):
    rows = [st.summarize_run(_run(tmp_path, "lr1", "qwen", 1e-4, "attr_mean", [(0, 0.4), (50, 0.9), (150, 0.6)])),
            st.summarize_run(_run(tmp_path, "lr2", "qwen", 2e-4, "attr_mean", [(0, 0.4), (50, 0.7), (150, 0.7)])),
            st.summarize_run(_run(tmp_path, "j1", "vljepa", 1e-4, "val_loss", [(0, 3.0), (150, 2.0)])),
            st.summarize_run(_run(tmp_path, "j2", "vljepa", 5e-4, "val_loss", [(0, 3.0), (150, 1.2)]))]
    w = st.sweep_winners(rows)
    assert w[("qwen", "nf4")]["lr"] == 2e-4          # 0.7 en el paso 150 > 0.6, aunque lr1 tuvo 0.9 en el 50
    assert w[("vljepa", "nf4")]["lr"] == 5e-4        # menor InfoNCE


def test_main_escribe_tabla(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RUNS", tmp_path)
    a = _run(tmp_path, "a", "qwen", 1e-4, "attr_mean", [(0, 0.4), (150, 0.5)])
    b = _run(tmp_path, "b", "qwen", 2e-4, "attr_mean", [(0, 0.4), (150, 0.6)])
    assert st.main([str(a), str(b), "--table", "t", "--winners"]) == 0
    assert (tmp_path / "_tablas" / "t.csv").exists()
