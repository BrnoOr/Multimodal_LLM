"""Resumen de corridas de la etapa 2 (humo, grilla de hiperparámetros y ajuste completo), sin GPU.

    uv run --no-sync python scripts/summarize_train.py runs/train_*_m6        # humo: P8 y entrenables
    uv run --no-sync python scripts/summarize_train.py runs/train_*_m150 --table s2_grid --winners   # grilla
    uv run --no-sync python scripts/summarize_train.py runs/train_*_e1 --table s2_train     # ajuste completo

Lee de cada run (los escribe vlmfid.train.loop): config.yaml, train_log.jsonl, val_log.jsonl,
best.json y summary.json. Imprime una fila por run y, si hay varios, escribe runs/_tablas/<tabla>.csv.

Columnas:
    model, quant, lr, lora_r,          identificación y ejes de la grilla (configs/train/grid.yaml)
    connector_lr_mult, steps
    loss_first, loss_last              pérdida de entrenamiento en el primer y el último registro
    peak_mem_gb                        pico de memoria (máximo de train_log): decidió P8 (lote 64; 128 no cabía en VL-JEPA)
    sel_metric, sel_mode               attr_mean (máx., VLM) o val_loss (mín., VL-JEPA; P7)
    val_step0                          métrica de selección en el paso 0 (adaptador en cero = zero-shot)
    val_last                           la del ÚLTIMO paso evaluado: regla de la grilla
    best_step, best_value              mejor paso en val: regla del ajuste completo (adapter_best)
    trainable_*, lora_layers_*         conteos de entrenables (summary.json; comparar con stage2_smoke.txt)

Regla de la grilla (configs/train/grid.yaml): por modelo, gana la combinación (LR, r, multiplicador del
conector) con mejor `val_last` (mayor attr_mean en los VLM, menor InfoNCE en VL-JEPA). Solo compiten los runs
que llegaron al mismo último paso que el resto de su modelo. `--winners` imprime el ganador con las tres claves
que van a configs/train/models/<modelo>.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from vlmfid.paths import RUNS


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue                                   # línea truncada por una caída
    return rows


def summarize_run(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{run_dir}: falta config.yaml")
    cfg = OmegaConf.load(cfg_path)
    metric = str(cfg.train.selection_metric)
    minimize = metric == "val_loss"
    tr, val = read_jsonl(run_dir / "train_log.jsonl"), read_jsonl(run_dir / "val_log.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) \
        if (run_dir / "summary.json").exists() else {}
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8")) \
        if (run_dir / "status.json").exists() else {}
    out = {"exp_id": str(cfg.exp_id), "model": str(cfg.model.name), "quant": str(cfg.model.quant),
           "lr": float(cfg.train.lr), "lora_r": int(cfg.lora.r),
           "connector_lr_mult": float(cfg.train.connector_lr_mult), "max_steps": cfg.train.get("max_steps"),
           "status": status.get("status"), "steps": tr[-1]["step"] if tr else 0,
           "loss_first": tr[0].get("loss") if tr else None, "loss_last": tr[-1].get("loss") if tr else None,
           "peak_mem_gb": max((r.get("peak_mem_gb", 0.0) for r in tr), default=None),
           "micro_batch": tr[-1].get("micro_batch") if tr else None,
           "sel_metric": metric, "sel_mode": "min" if minimize else "max"}
    step0 = [r for r in val if r.get("step") == 0]
    out["val_step0"] = step0[0].get(metric) if step0 else None
    out["val_last_step"] = val[-1]["step"] if val else None
    out["val_last"] = val[-1].get(metric) if val else None
    if val:
        best = (min if minimize else max)(val, key=lambda r: r[metric])
        out["best_step"], out["best_value"] = best["step"], best[metric]
    else:
        out["best_step"] = out["best_value"] = None
    out |= {k: v for k, v in summary.items() if k.startswith(("trainable", "lora_layers"))}
    return out


def sweep_winners(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Por (modelo, quant): el run con mejor `val_last` entre todas sus combinaciones (regla de la grilla).

    Solo compiten los runs evaluados en el mismo último paso que el más avanzado de su modelo: un run
    cortado a medias no se compara con uno completo."""
    last: dict[tuple[str, str], int] = {}
    for r in rows:
        if r["val_last_step"] is not None:
            key = (r["model"], r["quant"])
            last[key] = max(last.get(key, 0), r["val_last_step"])
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r["val_last"] is None or r["val_last_step"] != last[(r["model"], r["quant"])]:
            continue
        key = (r["model"], r["quant"])
        cur = out.get(key)
        better = cur is None or (r["val_last"] < cur["val_last"] if r["sel_mode"] == "min"
                                 else r["val_last"] > cur["val_last"])
        if better:
            out[key] = r
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="carpetas de runs (o exp_id dentro de runs/)")
    ap.add_argument("--table", default="s2_resumen", help="runs/_tablas/<tabla>.csv")
    ap.add_argument("--winners", action="store_true", help="imprime el LR ganador por modelo (barrido)")
    a = ap.parse_args(argv)

    rows = []
    for r in a.runs:
        d = Path(r) if Path(r).exists() else RUNS / r
        try:
            rows.append(summarize_run(d))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[resumen] se omite {r}: {e}", file=sys.stderr)
    if not rows:
        print("[resumen] ningún run", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (r["model"], r["quant"], r["lr"], r["lora_r"], r["connector_lr_mult"]))
    cols = ["model", "quant", "lr", "lora_r", "connector_lr_mult", "status", "steps", "loss_first", "loss_last", "peak_mem_gb", "sel_metric",
            "val_step0", "val_last", "best_step", "best_value"]

    def fmt(v):
        return f"{v:>11.4g}" if isinstance(v, float) else f"{str(v):>11}"

    print("  ".join(f"{c:>11}" for c in cols))
    for r in rows:
        print("  ".join(fmt(r.get(c)) for c in cols))
    if len(rows) > 1:
        out = RUNS / "_tablas" / f"{a.table}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[resumen] tabla en {out}")
    if a.winners:
        print("\n[resumen] ganador por modelo (último paso evaluado) -> configs/train/models/<modelo>.yaml:")
        for (m, q), r in sorted(sweep_winners(rows).items()):
            print(f"  {m:<22} {q:<5} train.lr={r['lr']:g}  lora.r={r['lora_r']}  "
                  f"train.connector_lr_mult={r['connector_lr_mult']:g}  {r['sel_metric']}={r['val_last']:.4f}  "
                  f"({r['exp_id']})")
        incomplete = [r["exp_id"] for r in rows if r["status"] != "completed"]
        if incomplete:
            print(f"[resumen] AVISO: {len(incomplete)} runs sin completar: {incomplete[:4]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
