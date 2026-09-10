#!/usr/bin/env bash
# Lanza un comando en una GPU libre del cluster, con el .env cargado.
# Uso:  ./scripts/run.sh python scripts/train.py experiment=e11_ft_llava
#       GPU=0 ./scripts/run.sh python scripts/infer.py ...
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
GPU=${GPU:-1}
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$USED" -gt 1000 ]; then
  echo "GPU $GPU ocupada (${USED} MiB en uso). Revisa: nvidia-smi" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[run.sh] GPU=$GPU  commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)  $(date -Is)"
exec uv run "$@"
