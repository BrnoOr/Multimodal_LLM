#!/usr/bin/env bash
# =============================================================================
# run_queue.sh — ejecuta en serie una cola de inferencias (una GPU, un modelo a la vez).
#
#   scripts/launch.sh stage1 scripts/run_queue.sh configs/queues/stage1_pilot.txt
#
# Cada línea no vacía ni comentada son los argumentos de `python scripts/infer.py`.
# Si una línea falla se registra y se sigue con la siguiente. Como infer es reanudable e
# idempotente, relanzar la misma cola retoma el run interrumpido y salta los completos.
# Resumen en logs/<job>/queue_status.tsv (o logs/queue_status.tsv fuera de launch.sh).
# SIGTERM (jobs.sh stop) detiene la cola después de que el run en curso guarde su lote.
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
QUEUE="${1:?uso: run_queue.sh <archivo_cola>}"
if [[ ! -f "$QUEUE" ]]; then
  echo "[queue] ERROR: no existe el archivo de cola '$QUEUE' (ruta relativa a $ROOT)." >&2
  echo "[queue] Colas disponibles:" >&2
  ls -1 "$ROOT"/configs/queues/*.txt 2>/dev/null | sed "s|$ROOT/||" >&2 || echo "  (ninguna: falta configs/queues/)" >&2
  exit 2
fi
if ! sed 's/#.*//' "$QUEUE" | grep -q '[^[:space:]]'; then
  echo "[queue] ERROR: la cola '$QUEUE' no tiene runs (solo líneas vacías o comentadas)." >&2
  exit 3
fi
PY="${PYTHON:-uv run --no-sync python}"
STATUS="$ROOT/logs/${VLMFID_JOB:+$VLMFID_JOB/}queue_status.tsv"
mkdir -p "$(dirname "$STATUS")"

# Verificación previa: si el paquete no se importa, todos los runs fallarían por lo mismo.
if ! $PY -c "import vlmfid.config, vlmfid.data, vlmfid.models, vlmfid.tracking" >/dev/null 2>"$ROOT/logs/.queue_preflight.err"; then
  echo "[queue] ERROR: el entorno no puede importar el paquete vlmfid; no se lanza ningún run." >&2
  cat "$ROOT/logs/.queue_preflight.err" >&2
  echo "[queue] Revisa que src/vlmfid esté actualizado y reinstala en modo editable:" >&2
  echo "        uv sync --extra quant --extra eval --extra dev && uv pip install -e . --no-deps" >&2
  echo "        uv run --no-sync python -c 'import vlmfid; print(vlmfid.__file__)'   # debe apuntar a src/vlmfid" >&2
  rm -f "$ROOT/logs/.queue_preflight.err"
  exit 4
fi
rm -f "$ROOT/logs/.queue_preflight.err"

STOP=0; CHILD=0
trap 'STOP=1; [[ $CHILD -gt 0 ]] && kill -TERM "$CHILD" 2>/dev/null' TERM INT

n=0; fails=0
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line%%#*}"; line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && continue
  n=$((n + 1))
  echo; echo "=================== [$n] $(date -Iseconds) :: $line"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  $PY scripts/infer.py $line &
  CHILD=$!; wait "$CHILD"; rc=$?
  # si llegó una señal durante wait, esperar a que el hijo termine de guardar
  if kill -0 "$CHILD" 2>/dev/null; then wait "$CHILD"; rc=$?; fi
  CHILD=0
  printf "%s\t%s\t%ss\t%s\n" "$(date -Iseconds)" "$rc" "$(( $(date +%s) - t0 ))" "$line" >> "$STATUS"
  [[ $rc -ne 0 ]] && fails=$((fails + 1))
  [[ $STOP -eq 1 ]] && { echo "[queue] detenida por señal tras [$n]"; exit 143; }
done < "$QUEUE"

echo; echo "[queue] fin: $n runs, $fails con error. Resumen: $STATUS"
exit $(( fails > 0 ? 1 : 0 ))
