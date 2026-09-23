#!/usr/bin/env bash
# Etapa 1 completa: un proceso run.sh por modelo, dos GPUs en paralelo.
# Uso:  ./scripts/run_stage1.sh [manifest] [n]
set -u
cd "$(dirname "$0")/.."
MANIFEST=${1:-data/manifests/m3di_test.parquet}
N=${2:-10000}
LOG=logs/stage1_$(date +%Y%m%d-%H%M%S); mkdir -p "$LOG"

chain() {                      # $1 = gpu, resto = modelos en secuencia
  local gpu=$1; shift
  for m in "$@"; do
    echo "[$(date +%T)] GPU$gpu -> $m" >> "$LOG/driver.log"
    GPU=$gpu ./scripts/run.sh python scripts/infer.py \
        --model "$m" --prompt all --manifest "$MANIFEST" -n "$N" \
        > "$LOG/$m.log" 2>&1 \
      || echo "[$(date +%T)] FALLO $m (rc=$?, ver $LOG/$m.log)" >> "$LOG/driver.log"
  done
}

chain 0 llava internvl &
chain 1 qwen  vljepa   &
wait
echo "[$(date +%T)] Etapa 1 completa" >> "$LOG/driver.log"
