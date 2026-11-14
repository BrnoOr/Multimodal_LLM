"""Puntaje de uno o más runs de inferencia: exactitud por atributo y métricas de texto.

    uv run --no-sync python scripts/score.py runs/zeroshot_llava_ov_05b_nf4_p1_attributes_val_n200
    uv run --no-sync python scripts/score.py runs/zeroshot_*_val                 # varios (el shell expande)
    uv run --no-sync python scripts/score.py runs/zeroshot_*_val --no-text       # sin pycocoevalcap (no requiere Java)
    uv run --no-sync python scripts/score.py runs/zeroshot_*_val --baselines     # agrega mayoría y techo del color

Lee `predictions.jsonl` y `config.yaml` de cada run (los escribe scripts/infer.py) y usa la única
implementación de las métricas (vlmfid.eval). Escribe `metrics.json` en cada run y, si se pasan varios,
una tabla comparativa en runs/_tablas/<nombre>.csv ordenada por `attr_mean`.

Es la herramienta para elegir el mejor prompt por modelo en VAL (docs/metodologia.md §2.2): el
criterio es `attr_mean`, la media de forma, x, y y color. Un run incompleto se puntúa igual, pero
se marca con `complete: false`; no compares runs con distinto `n`.

No carga modelos ni usa GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from vlmfid.eval import ATTRIBUTES, METRICS, TEXT_METRICS, color_ceiling, majority_baseline, score_texts, text_metrics
from vlmfid.eval.attributes import extract
from vlmfid.paths import RUNS


def read_predictions(path: Path) -> list[dict]:
    rows, seen = [], set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                               # línea truncada por una caída del proceso
            if r["id"] not in seen:                    # una reanudación puede repetir la última muestra
                seen.add(r["id"])
                rows.append(r)
    return rows


def score_run(run_dir: Path, with_text: bool = True) -> dict:
    pred_path, cfg_path = run_dir / "predictions.jsonl", run_dir / "config.yaml"
    if not pred_path.exists() or not cfg_path.exists():
        raise FileNotFoundError(f"{run_dir}: faltan predictions.jsonl o config.yaml")
    cfg = OmegaConf.load(cfg_path)
    if str(cfg.data.split) == "test" and not bool(cfg.get("allow_test", False)):
        raise ValueError(f"{run_dir}: run sobre test sin `allow_test`; no se puntúa (ver docs/auditoria.md, H1)")
    rows = read_predictions(pred_path)
    if not rows:
        raise ValueError(f"{run_dir}: predictions.jsonl está vacío")
    json_mode = cfg.prompt.get("response_format") == "json"
    texts = [r["prediction"] for r in rows]

    status = {}
    if (run_dir / "status.json").exists():
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    out = {"exp_id": str(cfg.exp_id), "model": str(cfg.model.name), "quant": str(cfg.model.quant),
           "stage": str(cfg.stage), "prompt_id": str(cfg.prompt.id), "split": str(cfg.data.split),
           "json_mode": json_mode, "complete": status.get("status") == "completed"}
    out |= score_texts(texts, rows, json_mode)
    # diagnósticos de formato: cuántas respuestas omiten cada atributo (omitir cuenta como error)
    ext = [extract(t, json_mode) for t in texts]
    out |= {f"missing_{a}": sum(e[a] is None for e in ext) / len(ext) for a in ATTRIBUTES}
    out["mean_words"] = sum(e["n_words"] for e in ext) / len(ext)
    if json_mode:
        out["json_ok"] = sum(bool(e["json_ok"]) for e in ext) / len(ext)
    if with_text:
        tm = text_metrics(texts, [r["reference"] for r in rows])
        out |= {m: tm[m] for m in TEXT_METRICS}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def baselines(split: str) -> dict:
    """Líneas base sin modelo sobre `split`, estimadas con train."""
    from vlmfid.data.manifest import load_manifest

    train, ev = load_manifest("train"), load_manifest(split)
    return {"majority": majority_baseline(train, ev), "color_ceiling": color_ceiling(train, ev)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="carpetas de runs (o exp_id dentro de runs/)")
    ap.add_argument("--no-text", action="store_true", help="omite Bleu/ROUGE/CIDEr (no requiere Java)")
    ap.add_argument("--baselines", action="store_true", help="imprime mayoría y techo del color en val")
    ap.add_argument("--table", default="resumen", help="nombre de la tabla comparativa (runs/_tablas/<nombre>.csv)")
    a = ap.parse_args(argv)

    results = []
    for r in a.runs:
        d = Path(r) if Path(r).exists() else RUNS / r
        try:
            results.append(score_run(d, with_text=not a.no_text))
        except (FileNotFoundError, ValueError) as e:
            print(f"[score] se omite {r}: {e}", file=sys.stderr)
    if not results:
        print("[score] ningún run puntuado", file=sys.stderr)
        return 1

    results.sort(key=lambda m: -m["attr_mean"])
    cols = ["model", "quant", "prompt_id", "split", "n", "complete", *METRICS]
    cols += [m for m in TEXT_METRICS if m in results[0]]
    print("  ".join(f"{c:>14}" for c in cols))
    for m in results:
        print("  ".join(f"{m.get(c):>14.4f}" if isinstance(m.get(c), float) else f"{str(m.get(c)):>14}" for c in cols))
    if len({m["n"] for m in results}) > 1:
        print("[score] AVISO: los runs tienen distinto n; no son comparables entre sí.", file=sys.stderr)

    if len(results) > 1:
        out = RUNS / "_tablas" / f"{a.table}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for m in results for k in m))
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"[score] tabla en {out}")
    if a.baselines:
        split = results[0]["split"]
        print(f"[score] líneas base en {split}: {json.dumps(baselines(split), indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
