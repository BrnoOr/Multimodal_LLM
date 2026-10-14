"""Etapa 2: ajuste fino QLoRA con presupuesto común (ver configs/train/budget.yaml).

    uv run --no-sync python scripts/train.py model=qwen35
    # prueba de humo: pocos pasos, val chica (cambia el exp_id por train_limit)
    uv run --no-sync python scripts/train.py model=qwen35 data.train_limit=640 data.val_limit=32 \
        train.eval_every=5 train.save_every=5 train.log_every=1

En el cluster, desacoplado de SSH:
    GPU=1 scripts/launch.sh ft_qwen35 uv run --no-sync python scripts/train.py model=qwen35
    SCRIPT=scripts/train.py GPU=1 scripts/launch.sh stage2 scripts/run_queue.sh configs/queues/stage2_train.txt

Relanzar el mismo comando reanuda desde el último checkpoint; un run completo termina de inmediato.
La evaluación en test se hace después con scripts/infer.py y model.adapter_path (ver stage2_test.txt).
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
