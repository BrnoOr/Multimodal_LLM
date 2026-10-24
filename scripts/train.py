"""Etapa 2: ajuste fino QLoRA con presupuesto común (ver configs/train/budget.yaml).

    uv run --no-sync python scripts/train.py model=qwen35
    # prueba de humo: train completo cortado en pocos pasos, val chica (exp_id ..._m10)
    uv run --no-sync python scripts/train.py model=qwen35 train.max_steps=10 train.warmup_ratio=0 \
        data.val_limit=32 train.eval_every=5 train.save_every=5 train.log_every=1

Train se usa siempre completo (las 125 000 imágenes): no hay data.train_limit.

En el cluster, desacoplado de SSH:
    GPU=1 scripts/launch.sh ft_qwen35 uv run --no-sync python scripts/train.py model=qwen35
    SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2b_main scripts/run_queue.sh configs/queues/stage2b_main.txt

Relanzar el mismo comando reanuda desde el último checkpoint; un run completo termina de inmediato.
La evaluación en test se hace después con scripts/infer.py y model.adapter_path (ver stage2b_test.txt).
Diseño y cambios: docs/ETAPA2_FINETUNING.md y docs/ETAPA2B_REFORMULACION.md.
"""

from __future__ import annotations

import sys

from vlmfid.train.config import compose_train
from vlmfid.train.loop import train


def main(argv=None) -> None:
    cfg = compose_train(sys.argv[1:] if argv is None else argv)
    sys.exit(train(cfg))


if __name__ == "__main__":
    main()
