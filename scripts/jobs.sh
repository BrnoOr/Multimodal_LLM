#!/usr/bin/env bash
# =============================================================================
# jobs.sh — administración de los jobs lanzados con launch.sh
#
#   scripts/jobs.sh status              tabla de jobs: estado, PID, GPU, inicio, comando
#   scripts/jobs.sh logs <job> [-f]     últimas líneas del log (-f: seguir en vivo; Ctrl-C no mata el job)
#   scripts/jobs.sh stop <job>          SIGTERM al grupo (apagado limpio: se guarda el lote en curso)
#   scripts/jobs.sh kill <job>          SIGKILL al grupo (último recurso)
#   scripts/jobs.sh runs                progreso de cada run en runs/*/status.json
#   scripts/jobs.sh gpu                 uso de las GPU
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$ROOT/logs"

state() {  # imprime RUNNING | DONE(rc) | DIED
  local d="$1"
  if [[ -f "$d/exit_code" ]]; then echo "DONE($(cat "$d/exit_code"))"
  elif [[ -f "$d/job.pid" ]] && kill -0 "$(cat "$d/job.pid")" 2>/dev/null; then echo "RUNNING"
  else echo "DIED"; fi   # sin exit_code: murió sin terminar el wrapper (p. ej. reinicio del nodo o SIGKILL)
}

get() { grep -m1 "^$2=" "$1/meta.env" 2>/dev/null | cut -d= -f2-; }

need_job() {
  [[ -n "${1:-}" && -d "$JOBS/$1" ]] || { echo "job inexistente: ${1:-<vacío>}" >&2; ls "$JOBS" 2>/dev/null; exit 1; }
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  status)
    [[ -d "$JOBS" ]] || { echo "No hay jobs."; exit 0; }
    printf "%-28s %-10s %-8s %-4s %-20s %s\n" JOB ESTADO PID GPU INICIO COMANDO
    for d in "$JOBS"/*/; do
      d="${d%/}"; n="$(basename "$d")"
      [[ -f "$d/meta.env" ]] || continue
      printf "%-28s %-10s %-8s %-4s %-20s %s\n" "$n" "$(state "$d")" "$(cat "$d/job.pid" 2>/dev/null || echo -)" \
        "$(get "$d" GPU)" "$(get "$d" START | cut -c1-19)" "$(get "$d" CMD | cut -c1-70)"
    done ;;
  logs)
    need_job "${1:-}"; f="$JOBS/$1/job.log"
    if [[ "${2:-}" == "-f" ]]; then tail -n 50 -f "$f"; else tail -n "${2:-60}" "$f"; fi ;;
  stop|kill)
    need_job "${1:-}"; d="$JOBS/$1"
    [[ "$(state "$d")" == "RUNNING" ]] || { echo "El job '$1' no está corriendo ($(state "$d"))."; exit 0; }
    pgid="$(cat "$d/job.pid")"
    if [[ "$cmd" == "stop" ]]; then
      kill -TERM -- "-$pgid"; echo "SIGTERM enviado al grupo $pgid; esperando cierre limpio…"
      for _ in $(seq 1 120); do kill -0 "$pgid" 2>/dev/null || { echo "Detenido."; exit 0; }; sleep 1; done
      echo "Sigue vivo tras 120 s: usa 'scripts/jobs.sh kill $1'."
    else
      kill -KILL -- "-$pgid"; echo "SIGKILL enviado al grupo $pgid."
    fi ;;
  runs)
    for s in "$ROOT"/runs/*/status.json; do
      [[ -f "$s" ]] || continue
      python3 - "$s" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); s = json.loads(p.read_text())
n, t = s.get("n_done", 0), s.get("n_total", 0)
print(f"{p.parent.name:<55} {s.get('status','?'):<12} {n:>6}/{t:<6} {s.get('img_per_s','-')} img/s")
PY
    done ;;
  gpu)
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
