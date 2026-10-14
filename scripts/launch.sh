#!/usr/bin/env bash
# =============================================================================
# launch.sh — lanza un comando desacoplado de la sesión SSH (sin tmux/screen).
#
#   scripts/launch.sh <nombre_job> <comando> [args...]
#
#   GPU=1 scripts/launch.sh piloto_llava uv run --no-sync python scripts/infer.py model=llava data.limit=500
#   scripts/launch.sh stage1 scripts/run_queue.sh configs/queues/stage1_pilot.txt
#
# Mecanismo: `setsid` crea una sesión nueva (el proceso deja de pertenecer a la terminal, así que
# cerrar SSH no le envía SIGHUP) y `nohup` + redirección de stdin/stdout/stderr eliminan cualquier
# vínculo restante con la TTY. Es POSIX/util-linux estándar: no requiere instalar nada.
#
# Deja en logs/<nombre_job>/:
#   job.log     stdout+stderr completos       job.pid    PID (= PGID) del grupo del job
#   meta.env    comando, GPU, commit, host     exit_code  código de salida al terminar
#
# Variables: GPU (defecto 1, o la de .env), GPU_MAX_USED_MB (1024), FORCE=1 (ignora GPU ocupada)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
  sed -n '2,20p' "$0"; exit 2
fi
NAME="$1"; shift
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "nombre de job inválido: $NAME" >&2; exit 2; }

# .env primero (HF_HOME, UV_CACHE_DIR, HF_TOKEN, GPU...) para que el job herede las variables
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
GPU="${GPU:-1}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1024}"

JOBDIR="$ROOT/logs/$NAME"
if [[ -f "$JOBDIR/job.pid" && ! -f "$JOBDIR/exit_code" ]] && kill -0 "$(cat "$JOBDIR/job.pid")" 2>/dev/null; then
  echo "El job '$NAME' ya está corriendo (PID $(cat "$JOBDIR/job.pid"))." >&2; exit 1
fi
mkdir -p "$JOBDIR"
# conserva el log de intentos anteriores
[[ -f "$JOBDIR/job.log" ]] && mv "$JOBDIR/job.log" "$JOBDIR/job.$(date +%Y%m%d-%H%M%S).log"
rm -f "$JOBDIR/exit_code" "$JOBDIR/job.pid"

# --- verificación de GPU (servidor compartido) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if (( USED > GPU_MAX_USED_MB )) && [[ "${FORCE:-0}" != "1" ]]; then
    echo "GPU $GPU ocupada: ${USED} MiB en uso (> ${GPU_MAX_USED_MB}). Usa otra GPU=… o FORCE=1." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv -i "$GPU" >&2 || true
    exit 1
  fi
else
  echo "[launch] aviso: nvidia-smi no disponible; no se verifica la GPU" >&2
fi

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "sin-git")
DIRTY=$([[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo "+cambios" || echo "")
{
  echo "NAME=$NAME"
  echo "CMD=$(printf '%q ' "$@")"
  echo "GPU=$GPU"
  echo "COMMIT=$COMMIT$DIRTY"
  echo "HOST=$(hostname)"
  echo "USER=$(whoami)"
  echo "START=$(date -Iseconds)"
} > "$JOBDIR/meta.env"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1          # sin esto el log se escribe a trozos
export TOKENIZERS_PARALLELISM=false
export VLMFID_JOB="$NAME"

# El wrapper escribe su propio PID ($$): al ser líder de sesión, PID = PGID, y `jobs.sh stop`
# puede señalizar a todo el grupo (python + workers del DataLoader).
setsid nohup bash -c '
  JOBDIR="$1"; shift
  echo $$ > "$JOBDIR/job.pid"
  echo "[launch] $(date -Iseconds) inicio en $(hostname), GPU=$CUDA_VISIBLE_DEVICES"
  echo "[launch] cmd: $*"
  "$@"
  rc=$?
  echo "$rc" > "$JOBDIR/exit_code"
  echo "[launch] $(date -Iseconds) fin, código $rc"
  exit "$rc"
' _ "$JOBDIR" "$@" > "$JOBDIR/job.log" 2>&1 < /dev/null &

# esperar a que aparezca el PID
for _ in $(seq 1 50); do [[ -s "$JOBDIR/job.pid" ]] && break; sleep 0.1; done
echo "Job '$NAME' lanzado — PID $(cat "$JOBDIR/job.pid" 2>/dev/null || echo '?'), GPU $GPU"
echo "  log:     scripts/jobs.sh logs $NAME -f"
echo "  estado:  scripts/jobs.sh status"
echo "  detener: scripts/jobs.sh stop $NAME"
