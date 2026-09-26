"""Rutas canónicas del proyecto. Todas se pueden redefinir por variable de entorno."""

from __future__ import annotations

import os
from pathlib import Path

# src/vlmfid/paths.py -> parents[2] = raíz del repo (válido con instalación editable)
ROOT = Path(os.environ.get("VLMFID_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
GENERATED = DATA / "generated"
MANIFESTS = Path(os.environ.get("VLMFID_MANIFESTS", DATA / "manifests"))
CACHE = Path(os.environ.get("VLMFID_CACHE", DATA / "cache"))
M3DI_ROOT = Path(os.environ.get("M3DI_ROOT", RAW / "m3di"))

CONFIGS = ROOT / "configs"
EXTERNAL = ROOT / "external"
RUNS = Path(os.environ.get("VLMFID_RUNS", ROOT / "runs"))   # resultados de experimentos
LOGS = Path(os.environ.get("VLMFID_LOGS", ROOT / "logs"))   # salida de cada ejecución (jobs)


def resolve(p: str | Path) -> Path:
    """Ruta absoluta; las relativas se interpretan respecto de la raíz del repo."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p
