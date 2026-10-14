"""Tasa de copia de los ejemplos del prompt, por run (experimento p4 vs p4m).

    uv run --no-sync python scripts/copy_report.py            # runs de prompts con ejemplos
    uv run --no-sync python scripts/copy_report.py --all      # todos los runs (control: ~0 sin ejemplos)

Lee runs/<exp_id>/config.yaml (texto del prompt usado) y predictions.jsonl (respuesta y
referencia). La forma verdadera se toma de la referencia. Escribe
reports/tables/eval/copia_ejemplos.csv. No usa GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

from vlmfid.eval import copy_summary, prompt_examples, reference_shape
from vlmfid.paths import ROOT, RUNS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="incluir runs de prompts sin ejemplos")
    ap.add_argument("--out", default="reports/tables/eval/copia_ejemplos.csv",
                    help="ruta del CSV (relativa a la raíz del repo si no es absoluta)")
    a = ap.parse_args(argv)

    filas = []
    for cfg_path in sorted(RUNS.glob("*/config.yaml")):
        pred_path = cfg_path.with_name("predictions.jsonl")
        if not pred_path.exists() or pred_path.stat().st_size == 0:
            continue
        cfg = OmegaConf.load(cfg_path)
        texto = str(cfg.prompt.text)
        if not a.all and not prompt_examples(texto):
            continue
        rows = [json.loads(l) for l in pred_path.open() if l.strip()]
        s = copy_summary([r["prediction"] for r in rows], texto,
                         [reference_shape(r["reference"]) for r in rows])
        filas.append({"exp_id": cfg.exp_id, "model": cfg.model.name, "prompt": cfg.prompt.id, **s})

    if not filas:
        print("[copy_report] no hay runs con predicciones que cumplan el filtro.")
        return
    df = pl.DataFrame(filas).sort(["prompt", "model"])
    with pl.Config(tbl_rows=-1, tbl_width_chars=160, float_precision=3):
        print(df.select("model", "prompt", "n", "n_examples", "copy_literal", "copy_object",
                        "copy_any", "distinct_ratio"))
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    print(f"[copy_report] {len(filas)} runs -> {out}")


if __name__ == "__main__":
    main()
