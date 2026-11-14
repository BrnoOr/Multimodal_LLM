"""Candados de scripts/final_test.py: el plan incompleto se rechaza y test no se lee sin --confirm."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import final_test  # noqa: E402
from vlmfid.config import compose  # noqa: E402


def _write(tmp_path, text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_el_plan_del_repo_esta_pendiente_o_completo():
    """Mientras queden decisiones por tomar (null), el plan versionado debe rechazarse entero."""
    import yaml

    raw = yaml.safe_load(final_test.PLAN_PATH.read_text(encoding="utf-8"))
    pending = any(e.get("prompt") is None or (e.get("needs_adapter") and not e.get("adapter_path"))
                  or (e.get("needs_decoder") and not e.get("decoder_path")) for e in raw["entries"])
    if pending:
        with pytest.raises(ValueError, match="no está listo"):
            final_test.load_plan(final_test.PLAN_PATH)
    assert len({e["label"] for e in raw["entries"]}) == len(raw["entries"]) == 16      # 4 modelos x 4 condiciones


def test_plan_incompleto_se_rechaza(tmp_path):
    with pytest.raises(ValueError, match="falta `prompt`"):
        final_test.load_plan(_write(tmp_path, "entries:\n  - {label: a, model: llava_ov, quant: nf4, prompt: null}\n"))
    with pytest.raises(ValueError, match="adapter_path"):
        final_test.load_plan(_write(tmp_path, "entries:\n  - {label: a, model: llava_ov, quant: nf4, "
                                              "prompt: p1_attributes, needs_adapter: true, adapter_path: null}\n"))
    with pytest.raises(ValueError, match="no existe"):
        final_test.load_plan(_write(tmp_path, "entries:\n  - {label: a, model: llava_ov, quant: nf4, "
                                              "prompt: p1_attributes, adapter_path: runs/no_existe}\n"))
    with pytest.raises(ValueError, match="label repetido"):
        final_test.load_plan(_write(tmp_path, "entries:\n"
                                              "  - {label: a, model: llava_ov, quant: nf4, prompt: p1_attributes}\n"
                                              "  - {label: a, model: qwen35, quant: nf4, prompt: p1_attributes}\n"))


def test_los_argumentos_abren_el_candado_solo_aqui(tmp_path):
    plan = final_test.load_plan(_write(tmp_path, "limit: 50\nentries:\n"
                                                 "  - {label: a, model: llava_ov, quant: bf16, prompt: p1_attributes}\n"))
    args = final_test.entry_args(plan["entries"][0], plan["limit"])
    cfg = compose(args)
    assert cfg.data.split == "test" and cfg.allow_test is True and cfg.exp_id == "final_a"
    assert cfg.model.quant == "bf16" and cfg.data.limit == 50 and cfg.stage == "zeroshot"


def test_sin_confirm_no_se_lee_test(tmp_path, monkeypatch):
    p = _write(tmp_path, "entries:\n  - {label: a, model: llava_ov, quant: nf4, prompt: p1_attributes}\n")
    monkeypatch.setattr(final_test, "check_lock", lambda plan: pytest.fail("no debió llegar a congelar el plan"))
    assert final_test.main(["--plan", str(p)]) == 2
    assert final_test.main(["--plan", str(p), "--dry-run"]) == 0


def test_el_plan_congelado_no_puede_cambiar(tmp_path, monkeypatch):
    monkeypatch.setattr(final_test, "OUT", tmp_path / "final_test")
    plan = {"limit": None, "entries": [{"label": "a", "model": "llava_ov", "quant": "nf4", "prompt": "p1_attributes"}]}
    final_test.check_lock(plan)                       # primera vez: congela
    final_test.check_lock(plan)                       # idéntico: permite reanudar
    changed = {"limit": None, "entries": [dict(plan["entries"][0], prompt="p0_minimal")]}
    with pytest.raises(RuntimeError, match="difiere del congelado"):
        final_test.check_lock(changed)
