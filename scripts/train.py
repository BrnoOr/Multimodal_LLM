"""Etapa 2: ajuste fino QLoRA con presupuesto común (ver configs/train/budget.yaml) + evaluación en test.

    uv run --no-sync python scripts/train.py model=qwen35
    # prueba de humo: train completo cortado en pocos pasos, val y test chicos (exp_id ..._m10)
    uv run --no-sync python scripts/train.py model=qwen35 train.max_steps=10 train.warmup_ratio=0 \
        data.val_limit=32 train.eval_every=5 train.save_every=5 train.log_every=1 test.limit=64

Datos: train completo (las 125 000 imágenes) para ajustar, val (500 fijas) para elegir el checkpoint
(adapter_best) y, al terminar, test (las 10 000 del zero-shot) para medir el desempeño final. Test se
evalúa con scripts/infer.py, igual que la etapa 1, y nunca interviene en el entrenamiento ni en la
selección. test.enabled=false lo omite (queda disponible configs/queues/stage2b_test.txt).

En el cluster, desacoplado de SSH:
    GPU=1 scripts/launch.sh ft_qwen35 uv run --no-sync python scripts/train.py model=qwen35
    SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2b_main scripts/run_queue.sh configs/queues/stage2b_main.txt

Relanzar el mismo comando reanuda desde el último checkpoint; un run completo salta directo a test, y un
test completo termina de inmediato.
Diseño y cambios: docs/ETAPA2_FINETUNING.md y docs/ETAPA2B_REFORMULACION.md.
"""

from __future__ import annotations

import gc
import json
import sys

import torch

from vlmfid.data.manifest import load_manifest
from vlmfid.data.targets import is_held_out
from vlmfid.paths import RUNS
from vlmfid.train.config import compose_test, compose_train
from vlmfid.train.evaluate import score_predictions
from vlmfid.train.loop import train


def evaluate_test(cfg) -> int:
    """Infiere en test con adapter_best y escribe test_metrics.json en el run de test."""
    from infer import run as infer_run      # scripts/infer.py (scripts/ está en sys.path al correr este script)

    tcfg = compose_test(cfg)
    print(f"\n[test] {cfg.exp_id} -> {tcfg.exp_id}  (adaptador {tcfg.model.adapter_path})", flush=True)
    rc = infer_run(tcfg)
    if rc != 0:
        return rc
    d = tcfg.data
    records = load_manifest(d.split, d.dataset, d.limit, d.subset_seed, variant=d.variant)
    groups = ({"seen": lambda r: not is_held_out(r), "unseen": is_held_out}
              if cfg.data.get("holdout") == "compositional" else {})
    run_dir = RUNS / tcfg.exp_id
    metrics = score_predictions(run_dir / "predictions.jsonl", records, groups)
    best = json.loads((RUNS / cfg.exp_id / "best.json").read_text())
    out = {"exp_id": tcfg.exp_id, "train_exp_id": cfg.exp_id, "model": cfg.model.name, "split": d.split,
           "best_val_step": best["step"], "selection_metric": cfg.train.selection_metric,
           "best_val_value": best["value"], **metrics}
    (run_dir / "test_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    unseen = f" no_vistas[forma+pos]={metrics['unseen_shape_pos']:.3f}" if "unseen_shape_pos" in metrics else ""
    print(f"[test] {cfg.model.name} (n={metrics['n']}): macro={metrics['macro']:.4f} "
          f"forma+pos={metrics['shape_pos']:.4f} forma={metrics['shape']:.3f} pos={metrics['pos']:.3f} "
          f"color={metrics['color']:.3f} color±1={metrics['color_adj']:.3f} nombre={metrics['name']:.3f}{unseen}"
          f"  -> {run_dir / 'test_metrics.json'}", flush=True)
    return 0


def main(argv=None) -> None:
    cfg = compose_train(sys.argv[1:] if argv is None else argv)
    rc = train(cfg)
    if rc != 0 or not cfg.test.enabled:
        sys.exit(rc)
    # libera el modelo de entrenamiento antes de cargar el de test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(evaluate_test(cfg))


if __name__ == "__main__":
    main()
