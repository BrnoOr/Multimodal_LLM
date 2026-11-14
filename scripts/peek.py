"""Revisión manual de runs de inferencia: muestras con su extracción y diagnósticos de formato.

    uv run --no-sync python scripts/peek.py runs/zeroshot_*_val_n200
    uv run --no-sync python scripts/peek.py runs/zeroshot_*_val_n200 --n 12 --only-wrong
    uv run --no-sync python scripts/peek.py runs/zeroshot_llava_ov_05b_nf4_p2_constrained_val_n200 --n 30

Por run imprime:
  * respuestas cortadas por `max_new_tokens` y largo en tokens (mediana y máximo);
  * por atributo: exactitud, fracción omitida y los valores extraídos más frecuentes (una respuesta
    constante, p. ej. por copiar el ejemplo del prompt, se ve aquí);
  * las respuestas más repetidas;
  * `--n` muestras fijas (las mismas imágenes en todos los runs): referencia, respuesta, atributos
    extraídos y acierto.

Sirve para revisar el extractor a mano antes de una corrida completa (docs/plan_reconstruccion.md,
paso 1.2). No carga modelos ni usa GPU; usa la única implementación de la extracción (vlmfid.eval).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/: score.py

from score import read_predictions  # noqa: E402

from vlmfid.eval import ATTRIBUTES  # noqa: E402
from vlmfid.eval.attributes import extract, score, truth_from_record  # noqa: E402
from vlmfid.paths import RUNS  # noqa: E402


def _short(text: str, width: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def peek(run_dir: Path, n: int, width: int, only_wrong: bool) -> None:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    json_mode = cfg.prompt.get("response_format") == "json"
    rows = sorted(read_predictions(run_dir / "predictions.jsonl"), key=lambda r: r["id"])
    exts = [extract(r["prediction"], json_mode) for r in rows]
    truths = [truth_from_record(r) for r in rows]
    scores = [score(e, t) for e, t in zip(exts, truths)]
    total = len(rows)

    print("=" * 110)
    print(f"{run_dir.name}   n={total}" + ("   [json]" if json_mode else ""))
    extra = [r.get("extra") or {} for r in rows]
    toks = sorted(int(e.get("n_new_tokens", 0)) for e in extra)
    cut = sum(bool(e.get("truncated")) for e in extra)
    print(f"  cortadas: {cut}/{total} ({cut / total:.0%})   tokens: mediana={toks[total // 2]} max={toks[-1]}")
    if json_mode:
        ok = sum(bool(e["json_ok"]) for e in exts)
        print(f"  json válido: {ok}/{total} ({ok / total:.0%})")
    for a in ATTRIBUTES:
        acc = sum(s[a] for s in scores) / total
        miss = sum(e[a] is None for e in exts) / total
        top = Counter(str(e[a]) for e in exts).most_common(4)
        vals = ", ".join(f"{v}×{c}" for v, c in top)
        print(f"  {a:5s} acierto={acc:.3f}  omitido={miss:.3f}  extraídos: {vals}")
    rep = Counter(_short(r["prediction"], 90) for r in rows).most_common(3)
    print("  respuestas más repetidas:")
    for text, c in rep:
        print(f"    {c:4d}× {text!r}")

    shown = 0
    for r, e, t, s in zip(rows, exts, truths, scores):
        if shown >= n:
            break
        if only_wrong and all(s[a] for a in ATTRIBUTES):
            continue
        shown += 1
        marks = " ".join(f"{a}={'ok' if s[a] else 'X'}" for a in ATTRIBUTES)
        print(f"  --- {r['id']}   ref: {r['reference']}")
        print(f"      pred: {_short(r['prediction'], width)!r}")
        print(f"      extraído: " + ", ".join(f"{a}={e[a]}" for a in ATTRIBUTES) + f"   |   {marks}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="carpetas de runs (o exp_id dentro de runs/)")
    ap.add_argument("--n", type=int, default=8, help="muestras por run (las primeras por id)")
    ap.add_argument("--width", type=int, default=260, help="caracteres máximos de cada respuesta")
    ap.add_argument("--only-wrong", action="store_true", help="solo muestras con algún atributo errado")
    a = ap.parse_args(argv)
    for r in a.runs:
        d = Path(r) if Path(r).exists() else RUNS / r
        if not (d / "predictions.jsonl").exists() or not (d / "config.yaml").exists():
            print(f"[peek] se omite {r}: faltan predictions.jsonl o config.yaml", file=sys.stderr)
            continue
        peek(d, a.n, a.width, a.only_wrong)
    return 0


if __name__ == "__main__":
    sys.exit(main())
