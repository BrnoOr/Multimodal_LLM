"""Registro reproducible de cada run: configuración resuelta, entorno, estado y progreso.

runs/<exp_id>/
    config.yaml        configuración resuelta (se compara al reanudar)
    env.jsonl          commit, GPU, versiones, host, semilla (una línea por intento)
    status.json        running | completed | failed | interrupted (+ tiempos y estadísticas)
    predictions.jsonl  una línea por muestra, escrita de forma incremental (reanudable)
    embeddings/        (opcional) embeddings por lote en .npz
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .paths import ROOT

# claves que pueden cambiar entre un intento y su reanudación sin invalidar las predicciones
# Claves que pueden cambiar al reanudar sin afectar las predicciones ya guardadas.
# model.llama_name: réplica sin restricción de Llama-3.2-1B con los mismos pesos y tokenizador
# (además, los pesos del predictor vienen del checkpoint de VL-JEPA).
_RESUME_SAFE = {"infer", "data.num_workers", "model.llama_name"}


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def env_info(seed: int | None = None) -> dict:
    info = {
        "time": _now(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "job_name": os.environ.get("VLMFID_JOB"),
        "seed": seed,
        "git": git_info(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_cc"] = list(torch.cuda.get_device_capability(0))
            info["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    for pkg in ("transformers", "bitsandbytes", "peft", "accelerate"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            info[pkg] = None
    return info


def _strip(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg))
    for key in _RESUME_SAFE:
        node, parts = c, key.split(".")
        for p in parts[:-1]:
            node = node.get(p, {})
        node.pop(parts[-1], None)
    return c


class RunTracker:
    def __init__(self, run_dir: Path, cfg: DictConfig):
        self.dir = Path(run_dir)
        self.cfg = cfg
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dir / "status.json"

    @property
    def predictions_path(self) -> Path:
        return self.dir / "predictions.jsonl"

    def read_status(self) -> dict:
        if self.status_path.exists():
            return json.loads(self.status_path.read_text())
        return {}

    def check_compatible(self) -> None:
        """Al reanudar, la configuración relevante debe coincidir con la guardada."""
        old = self.dir / "config.yaml"
        if not old.exists():
            return
        # Un run que falló antes de guardar predicciones no tiene nada que proteger: se reanuda con
        # la configuración nueva (p. ej. tras corregir un error de carga) en vez de exigir otro exp_id.
        if not self.predictions_path.exists() or self.predictions_path.stat().st_size == 0:
            return
        prev = OmegaConf.to_container(OmegaConf.load(old), resolve=True)
        curr = OmegaConf.to_container(self.cfg, resolve=True)
        if _strip(prev) != _strip(curr):
            raise RuntimeError(
                f"La configuración difiere de la guardada en {old}. "
                "Usa otro exp_id o borra el run si quieres empezar de cero."
            )

    def start(self) -> None:
        self.check_compatible()
        (self.dir / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg, resolve=True))
        envs = self.dir / "env.jsonl"  # un registro por intento (reanudaciones incluidas)
        with envs.open("a") as f:
            f.write(json.dumps(env_info(self.cfg.get("seed"))) + "\n")
        st = self.read_status()
        st.update({"status": "running", "last_start": _now(), "attempts": st.get("attempts", 0) + 1})
        st.setdefault("first_start", st["last_start"])
        self._write_status(st)

    def update(self, **fields) -> None:
        st = self.read_status()
        st.update(fields)
        st["updated"] = _now()
        self._write_status(st)

    def finish(self, status: str, **fields) -> None:
        self.update(status=status, finished=_now(), **fields)

    def _write_status(self, st: dict) -> None:
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
        tmp.replace(self.status_path)  # escritura atómica
