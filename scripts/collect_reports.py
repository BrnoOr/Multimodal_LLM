"""Copia las tablas de resumen de runs/ (no versionado) a reports/tables/<etapa>/ (versionado), sin GPU.

    uv run --no-sync python scripts/collect_reports.py

Copia runs/_tablas/*.csv (las escriben scripts/summarize_train.py, scripts/score.py y los notebooks) y, si
existe, runs/final_test/tabla_final.csv. La etapa sale del nombre de la tabla (`vlmfid.reports.stage_of_table`):
s2_* y etapa2* -> etapa2; final* -> final; el resto -> etapa1. Las figuras las guardan los notebooks 02 y 03
al ejecutarse (`vlmfid.reports.save_fig`). Flujo completo hasta el commit: reports/README.md.
"""

from __future__ import annotations

import sys

from vlmfid.reports import REPORTS, collect_run_tables


def main() -> int:
    out = collect_run_tables()
    if not out:
        print("[reports] runs/_tablas/ no tiene tablas", file=sys.stderr)
        return 1
    for p in out:
        print(f"[reports] {p.relative_to(REPORTS.parent).as_posix()}")
    print(f"[reports] {len(out)} tablas; procedencia en reports/tables/<etapa>/PROCEDENCIA.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
