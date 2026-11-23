"""Figuras y tablas del informe en `reports/` (versionadas), separadas de `runs/` (no versionado).

    reports/figures/<etapa>/<nombre>.png y .pdf      figuras de los notebooks 02 (etapa1) y 03 (etapa2)
    reports/tables/<etapa>/<nombre>.csv              tablas de los notebooks y copias de runs/_tablas/
    reports/<figures|tables>/<etapa>/PROCEDENCIA.json  commit, fecha y origen de cada archivo

Solo se guardan resúmenes chicos (decenas de KB): nada de predicciones, embeddings ni pesos. Se generan en el
cluster y se traen al laptop para hacer commit (flujo en `reports/README.md`).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from pathlib import Path

from .paths import ROOT, RUNS
from .tracking import git_info

REPORTS = Path(os.environ.get("VLMFID_REPORTS", ROOT / "reports"))
STAGES = ("etapa1", "etapa2", "final")


def _stage_dir(kind: str, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"etapa {stage!r} no válida; usar una de {STAGES}")
    d = REPORTS / kind / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(folder: Path, name: str, source: str) -> None:
    """Agrega o actualiza la procedencia de `name` en folder/PROCEDENCIA.json."""
    p = folder / "PROCEDENCIA.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    g = git_info()
    data[name] = {"origen": source, "commit": g["commit"], "sin_commitear": g["dirty"],
                  "fecha": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    p.write_text(json.dumps(dict(sorted(data.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def save_fig(fig, stage: str, name: str, source: str = "", pdf: bool = True) -> Path:
    """Guarda la figura en PNG (150 dpi) y, si pdf=True, en PDF (vectorial, para el informe)."""
    d = _stage_dir("figures", stage)
    out = d / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if pdf:
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    _record(d, name, source)
    return out


def save_table(df, stage: str, name: str, source: str = "", index: bool = True) -> Path:
    """Guarda un DataFrame en CSV (UTF-8)."""
    d = _stage_dir("tables", stage)
    out = d / f"{name}.csv"
    df.to_csv(out, index=index, encoding="utf-8")
    _record(d, name, source)
    return out


def stage_of_table(name: str) -> str:
    """Etapa de una tabla de runs/_tablas/ por su nombre (convención de `--table` en scripts y docs)."""
    if name.startswith(("s2_", "etapa2")):
        return "etapa2"
    if name.startswith("final"):
        return "final"
    return "etapa1"                     # etapa1_*, vljepa_*, piloto, diag*


def collect_run_tables(runs: Path = RUNS) -> list[Path]:
    """Copia runs/_tablas/*.csv a reports/tables/<etapa>/ y, si existe, la tabla de test de
    scripts/final_test.py a reports/tables/final/. Sobrescribe y registra la procedencia."""
    pairs = [(src, stage_of_table(src.stem)) for src in sorted((runs / "_tablas").glob("*.csv"))]
    final = runs / "final_test" / "tabla_final.csv"
    if final.exists():
        pairs.append((final, "final"))
    out = []
    for src, stage in pairs:
        d = _stage_dir("tables", stage)
        dst = d / src.name
        shutil.copy2(src, dst)
        _record(d, src.stem, src.relative_to(runs.parent).as_posix() if runs.parent in src.parents else str(src))
        out.append(dst)
    return out
