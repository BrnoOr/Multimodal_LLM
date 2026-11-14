"""Evaluación final en TEST. Es el ÚNICO punto de entrada del proyecto que lee test (CLAUDE.md §3).

    uv run --no-sync python scripts/final_test.py --dry-run          # muestra qué correría; no carga nada
    uv run --no-sync python scripts/final_test.py --confirm          # corre el plan completo
    uv run --no-sync python scripts/final_test.py --confirm --no-text   # sin Bleu/ROUGE/CIDEr (sin Java)

En el cluster:
    GPU=0 scripts/launch.sh final_test uv run --no-sync python scripts/final_test.py --confirm

El plan vive en configs/final_test.yaml: una entrada por (modelo, condición) con el prompt, la
cuantización y, si corresponde, el adaptador y el Y-Decoder YA ELEGIDOS EN VAL. Nada se elige aquí.

Cómo se garantiza "una sola vez":
  * Sin `--confirm` no se lee test.
  * Un plan con valores pendientes (null) se rechaza entero: test no se toca hasta que todas las
    decisiones estén tomadas.
  * Al primer lanzamiento el plan se congela en runs/final_test/plan.lock.yaml. Un lanzamiento
    posterior solo se acepta si el plan es IDÉNTICO al congelado (sirve para reanudar tras una caída;
    los runs completos se saltan). Si el plan cambió, el script se niega: cambiar el plan después de
    ver resultados de test es exactamente lo que este script existe para impedir.
  * Cada lanzamiento queda en runs/final_test/ledger.jsonl con fecha, commit y estado del árbol de git.

Salida:
    runs/final_<label>/            predictions.jsonl, config.yaml, status.json, metrics.json
    runs/final_test/tabla_final.csv  una fila por entrada: exactitud por atributo y métricas de texto
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/: infer.py y score.py

from vlmfid.config import compose  # noqa: E402
from vlmfid.paths import CONFIGS, RUNS, resolve  # noqa: E402
from vlmfid.tracking import git_info  # noqa: E402

PLAN_PATH = CONFIGS / "final_test.yaml"
OUT = RUNS / "final_test"
REQUIRED = ("label", "model", "prompt", "quant")
OPTIONAL = ("adapter_path", "decoder_path")


def load_plan(path: Path) -> dict:
    plan = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    entries = plan.get("entries") or []
    if not entries:
        raise ValueError(f"{path}: el plan no tiene entradas")
    problems, labels = [], set()
    for i, e in enumerate(entries):
        tag = e.get("label") or f"entrada {i}"
        problems += [f"{tag}: falta `{k}`" for k in REQUIRED if e.get(k) in (None, "")]
        unknown = set(e) - set(REQUIRED) - set(OPTIONAL) - {"needs_adapter", "needs_decoder"}
        problems += [f"{tag}: clave desconocida `{k}`" for k in sorted(unknown)]
        if e.get("needs_adapter") and not e.get("adapter_path"):
            problems.append(f"{tag}: es una condición ajustada y falta `adapter_path`")
        if e.get("needs_decoder") and not e.get("decoder_path"):
            problems.append(f"{tag}: VL-JEPA necesita `decoder_path`")
        if e.get("label") in labels:
            problems.append(f"{tag}: label repetido")
        labels.add(e.get("label"))
        for k in OPTIONAL:
            if e.get(k) and not resolve(e[k]).exists():
                problems.append(f"{tag}: `{k}` no existe: {e[k]}")
    if problems:
        raise ValueError("El plan no está listo; test NO se lee hasta completarlo:\n  - " + "\n  - ".join(problems))
    return plan


def entry_args(e: dict, limit) -> list[str]:
    args = [f"model={e['model']}", f"prompt={e['prompt']}", f"model.quant={e['quant']}",
            "data.split=test", "allow_test=true", f"exp_id=final_{e['label']}",
            f"stage={'finetuned' if e.get('adapter_path') else 'zeroshot'}"]
    args += [f"model.{k}={e[k]}" for k in OPTIONAL if e.get(k)]
    if limit:
        args.append(f"data.limit={limit}")
    return args


def check_lock(plan: dict) -> None:
    """Congela el plan la primera vez; después exige que sea idéntico."""
    lock = OUT / "plan.lock.yaml"
    text = OmegaConf.to_yaml(OmegaConf.create(plan), sort_keys=True)
    if lock.exists():
        if lock.read_text(encoding="utf-8") != text:
            raise RuntimeError(
                f"El plan difiere del congelado en {lock}: test ya se usó con otro plan y no se vuelve a "
                "leer con uno distinto. Si el lanzamiento anterior fue un error y NO se miraron sus "
                "resultados, el usuario debe decidirlo y retirar a mano runs/final_test/ y runs/final_*."
            )
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true", help="obligatorio para leer test")
    ap.add_argument("--dry-run", action="store_true", help="valida el plan y muestra los runs; no lee test")
    ap.add_argument("--no-text", action="store_true", help="omite Bleu/ROUGE/CIDEr (no requiere Java)")
    ap.add_argument("--plan", default=str(PLAN_PATH))
    a = ap.parse_args(argv)

    plan = load_plan(Path(a.plan))
    entries, limit = plan["entries"], plan.get("limit")
    runs = [(e, entry_args(e, limit)) for e in entries]
    for e, args in runs:
        compose(args)                                  # falla aquí si un modelo o prompt no existe
        print(f"[final] {e['label']}: {' '.join(args)}")
    if a.dry_run:
        print(f"[final] plan válido: {len(runs)} runs. No se leyó test (--dry-run).")
        return 0
    if not a.confirm:
        print("[final] Falta --confirm: esto lee TEST, que se usa una sola vez. Revisa el plan con --dry-run.",
              file=sys.stderr)
        return 2

    check_lock(plan)
    with (OUT / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "git": git_info(),
                            "n_entries": len(runs), "argv": sys.argv[1:]}, ensure_ascii=False) + "\n")

    import infer                                       # scripts/infer.py
    from score import score_run                        # scripts/score.py

    results, failed = [], []
    for e, args in runs:
        print(f"\n{'=' * 90}\n[final] {e['label']}\n{'=' * 90}", flush=True)
        try:
            rc = infer.run(compose(args))
            if rc != 0:
                raise RuntimeError(f"infer terminó con código {rc}")
            m = score_run(RUNS / f"final_{e['label']}", with_text=not a.no_text)
            results.append({"label": e["label"], "condition": "ajustado" if e.get("adapter_path") else "zero-shot"} | m)
        except Exception as exc:                       # un fallo no detiene el resto; se reanuda relanzando
            print(f"[final] FALLÓ {e['label']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failed.append(e["label"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if results:
        keys = list(dict.fromkeys(k for m in results for k in m))
        with (OUT / "tabla_final.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\n[final] tabla en {OUT / 'tabla_final.csv'} ({len(results)} de {len(runs)} entradas)")
    if failed:
        print(f"[final] entradas con error: {failed}. Corrige y relanza el MISMO plan (los completos se saltan).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
