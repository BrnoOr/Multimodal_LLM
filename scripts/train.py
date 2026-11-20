"""Etapa 2: ajuste QLoRA (LoRA + conector) con presupuesto común; selección del checkpoint en val.

    uv run --no-sync python scripts/train.py model=qwen35
    # control sin cuantizar (D6), idéntico en lo demás:
    uv run --no-sync python scripts/train.py model=qwen35 model.quant=bf16
    # prueba de humo (corrida corta: corta el run y el schedule; exp_id ..._m<N>):
    uv run --no-sync python scripts/train.py model=qwen35 train.max_steps=6 data.val_limit=32 \
        train.eval_every=3 train.save_every=3 train.log_every=1

Datos: train completo (125 000 imágenes) para ajustar y val (500 fijas) para elegir el checkpoint
(`adapter_best`). Test NO se lee aquí: la evaluación final es `scripts/final_test.py`, una sola vez,
con el `adapter_best` de cada modelo anotado en configs/final_test.yaml.

Mientras `train.hparams_confirmed` sea false (configs/train/budget.yaml) solo se aceptan corridas cortas
(`train.max_steps`): la prueba de humo y el barrido de LR (configs/queues/stage2_sweep.txt).

En el cluster, desacoplado de SSH:
    SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_smoke scripts/run_queue.sh configs/queues/stage2_smoke.txt

Relanzar el mismo comando reanuda desde el último checkpoint; un run completo termina de inmediato.
Protocolo: docs/metodologia.md §2.3; hiperparámetros y su fuente: §4.
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
