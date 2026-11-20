"""Expande la grilla de hiperparámetros (configs/train/grid.yaml) en una cola de scripts/train.py.

    uv run --no-sync python scripts/make_grid.py                    # escribe configs/queues/stage2_grid.txt
    uv run --no-sync python scripts/make_grid.py --check            # falla si la cola escrita no coincide con la grilla
    uv run --no-sync python scripts/make_grid.py --gpus 2           # además, stage2_grid_g0.txt y _g1.txt

Una línea por (modelo, combinación): producto cartesiano completo de `axes`, con `train.max_steps` y
`train.eval_every` de la grilla. No carga modelos ni usa GPU. Valida cada línea con `compose_train` (falla si una
combinación rompe el protocolo) y que los exp_id sean distintos (si dos combinaciones compartieran carpeta, la
segunda se saltaría como "completa").

Reparto en GPU (--gpus N): por modelo, para que cada GPU cargue siempre el mismo modelo y las corridas de un
modelo terminen juntas. Los modelos se asignan en orden alternado.

La cola generada se versiona: es la que se ejecuta en el cluster.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from omegaconf import OmegaConf

from vlmfid.paths import CONFIGS
from vlmfid.train.config import GRID_KEYS, compose_train

GRID = CONFIGS / "train" / "grid.yaml"
QUEUES = CONFIGS / "queues"
HEADER = """# GENERADO por scripts/make_grid.py desde configs/train/grid.yaml: no editar a mano.
# Grilla completa de la etapa 2 (docs/metodologia.md §4): {n_comb} combinaciones × {n_models} modelos = {n} corridas, {quant}.
# Regla: por modelo, la mejor métrica de selección en el ÚLTIMO paso evaluado (attr_mean máx. / InfoNCE mín.).
#   SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_grid scripts/run_queue.sh configs/queues/{name}
#   uv run --no-sync python scripts/summarize_train.py runs/train_*_m{steps} --table s2_grid --winners
"""


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def load_grid(path: Path = GRID) -> dict:
    g = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    axes = g.get("axes") or {}
    unknown = set(axes) - set(GRID_KEYS)
    if unknown:
        raise ValueError(f"{path}: ejes {sorted(unknown)} no permitidos; la grilla solo puede variar {GRID_KEYS} "
                         "(las demás claves son comunes a los 4 modelos, configs/train/budget.yaml)")
    for k, vals in axes.items():
        if not isinstance(vals, list) or not vals or len(set(map(str, vals))) != len(vals):
            raise ValueError(f"{path}: el eje {k} necesita una lista no vacía sin repetidos")
    for k in ("steps", "eval_every", "quant", "models"):
        if g.get(k) in (None, "", []):
            raise ValueError(f"{path}: falta '{k}'")
    return g


def combinations(g: dict) -> list[dict]:
    keys = list(g["axes"])
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g["axes"][k] for k in keys))]


def queue_lines(g: dict) -> list[tuple[str, str]]:
    """(modelo, línea de argumentos) por corrida, en orden: modelo, luego combinación."""
    out = []
    for model in g["models"]:
        for comb in combinations(g):
            args = [f"model={model}"]
            if g["quant"] != "nf4":
                args.append(f"model.quant={g['quant']}")
            args += [f"{k}={_fmt(v)}" for k, v in comb.items()]
            args += [f"train.max_steps={g['steps']}", f"train.eval_every={g['eval_every']}"]
            out.append((model, " ".join(args)))
    return out


def validate_lines(lines: list[tuple[str, str]]) -> list[str]:
    """compose_train de cada línea; devuelve los exp_id y falla si se repiten."""
    ids = []
    for _, line in lines:
        cfg = compose_train(line.split())
        ids.append(str(cfg.exp_id))
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"exp_id repetidos (dos combinaciones irían a la misma carpeta): {dup}")
    return ids


def render(g: dict, lines: list[tuple[str, str]], name: str) -> str:
    head = HEADER.format(n_comb=len(combinations(g)), n_models=len(g["models"]), n=len(lines),
                         quant=g["quant"], name=name, steps=g["steps"])
    return head + "".join(f"{line}\n" for _, line in lines)


def split_by_model(g: dict, lines: list[tuple[str, str]], n_gpus: int) -> list[list[tuple[str, str]]]:
    parts: list[list[tuple[str, str]]] = [[] for _ in range(n_gpus)]
    owner = {m: i % n_gpus for i, m in enumerate(g["models"])}
    for model, line in lines:
        parts[owner[model]].append((model, line))
    return parts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--out", default=str(QUEUES / "stage2_grid.txt"))
    ap.add_argument("--gpus", type=int, default=1, help="además, una cola por GPU (reparto por modelo)")
    ap.add_argument("--check", action="store_true", help="no escribe: falla si --out no coincide con la grilla")
    a = ap.parse_args(argv)

    g = load_grid(Path(a.grid))
    lines = queue_lines(g)
    ids = validate_lines(lines)
    out = Path(a.out)
    text = render(g, lines, out.name)
    if a.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"[grid] {out} no coincide con {a.grid}: regenerar con scripts/make_grid.py", file=sys.stderr)
            return 1
        print(f"[grid] {out} coincide con la grilla ({len(lines)} corridas)")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"[grid] {len(lines)} corridas ({len(combinations(g))} combinaciones × {len(g['models'])} modelos) -> {out}")
    for i in ids[:3]:
        print(f"  {i}")
    if a.gpus > 1:
        for i, part in enumerate(split_by_model(g, lines, a.gpus)):
            p = out.with_name(f"{out.stem}_g{i}{out.suffix}")
            p.write_text(render(g, part, p.name), encoding="utf-8")
            print(f"[grid] GPU {i}: {len(part)} corridas -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
