"""reports/: figuras y tablas del informe con procedencia, y copia de runs/_tablas/ por etapa."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from vlmfid import reports  # noqa: E402


@pytest.fixture
def rep(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "REPORTS", tmp_path / "reports")
    return tmp_path


def test_etapa_por_nombre_de_tabla():
    assert reports.stage_of_table("s2_grid_vljepa") == "etapa2"
    assert reports.stage_of_table("etapa2_val_resumen") == "etapa2"
    assert reports.stage_of_table("etapa1_val") == "etapa1"
    assert reports.stage_of_table("vljepa_piloto") == "etapa1"
    assert reports.stage_of_table("final_x") == "final"


def test_save_fig_y_tabla_con_procedencia(rep):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png = reports.save_fig(fig, "etapa2", "curva", source="§4")
    plt.close(fig)
    assert png.exists() and png.with_suffix(".pdf").exists()
    csv = reports.save_table(pd.DataFrame({"a": [1]}), "etapa2", "t", source="§9a")
    assert csv.read_text(encoding="utf-8").startswith(",a")
    prov = json.loads((csv.parent / "PROCEDENCIA.json").read_text(encoding="utf-8"))
    assert prov["t"]["origen"] == "§9a" and "commit" in prov["t"]


def test_etapa_invalida(rep):
    with pytest.raises(ValueError, match="etapa"):
        reports.save_table(pd.DataFrame(), "etapa3", "x")


def test_collect_copia_por_etapa(rep):
    runs = rep / "runs"
    (runs / "_tablas").mkdir(parents=True)
    (runs / "_tablas" / "s2_grid.csv").write_text("a\n1\n")
    (runs / "_tablas" / "etapa1_val.csv").write_text("a\n2\n")
    (runs / "final_test").mkdir()
    (runs / "final_test" / "tabla_final.csv").write_text("a\n3\n")
    out = {p.relative_to(reports.REPORTS).as_posix() for p in reports.collect_run_tables(runs)}
    assert out == {"tables/etapa2/s2_grid.csv", "tables/etapa1/etapa1_val.csv", "tables/final/tabla_final.csv"}
    prov = json.loads((reports.REPORTS / "tables" / "final" / "PROCEDENCIA.json").read_text(encoding="utf-8"))
    assert prov["tabla_final"]["origen"] == "runs/final_test/tabla_final.csv"
