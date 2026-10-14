"""Verificación del entorno (laptop y cluster). Uso: uv run python scripts/check_env.py

Comprueba: build de PyTorch vs GPU, bf16, NF4 real en GPU (bitsandbytes), versiones clave,
submodule open-vljepa, token de HF y espacio en disco de las cachés.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "\033[32mOK\033[0m", "\033[33mAVISO\033[0m", "\033[31mFALLA\033[0m"
failures = 0


def report(status, msg):
    global failures
    failures += status == FAIL
    print(f"[{status}] {msg}")


def version(pkg):
    try:
        return importlib.import_module(pkg).__version__
    except Exception:
        return None


print(f"python {sys.version.split()[0]}  ({sys.executable})")
import torch  # noqa: E402

report(OK, f"torch {torch.__version__}  build CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    report(FAIL, "CUDA no disponible")
else:
    n = torch.cuda.device_count()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(no definido)")
    report(OK, f"{n} GPU visible(s), CUDA_VISIBLE_DEVICES={cvd}")
    cc = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    report(OK, f"GPU 0: {name}  cc={cc}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    archs = torch.cuda.get_arch_list()
    if f"sm_{cc[0]}{cc[1]}" not in archs:
        report(FAIL, f"el build no incluye sm_{cc[0]}{cc[1]} (archs={archs}); ver SETUP §5")
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    report(OK if torch.isfinite(x @ x).all() else FAIL, "matmul bf16 en GPU")

bnb = version("bitsandbytes")
if bnb is None:
    report(FAIL, "bitsandbytes no instalado (uv sync --extra quant)")
elif torch.cuda.is_available():
    try:
        import bitsandbytes as bnbm

        lin = bnbm.nn.Linear4bit(128, 64, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
        y = lin(torch.randn(4, 128, device="cuda", dtype=torch.bfloat16))
        report(OK, f"bitsandbytes {bnb}: capa NF4 en GPU -> {tuple(y.shape)}")
    except Exception as e:
        report(FAIL, f"bitsandbytes {bnb}: NF4 falló: {e}")

tf = version("transformers")


def _ver(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


if tf is None or _ver(tf) < (5, 17):
    report(FAIL, f"transformers {tf}: se requiere >= 5.17 (Qwen3.5; firma de capas usada por open-vljepa)")
else:
    import transformers

    needed = ("AutoModelForImageTextToText", "LlavaOnevisionForConditionalGeneration",
              "InternVLForConditionalGeneration", "Qwen3_5ForConditionalGeneration", "VJEPA2Model")
    missing = [c for c in needed if not hasattr(transformers, c)]
    report(FAIL if missing else OK, f"transformers {tf}" + (f" sin {missing}" if missing else ""))

fla = version("fla")
if fla is None:
    report(OK, "flash-linear-attention no instalado: Qwen3.5 usará Gated DeltaNet en PyTorch (más lento)")
elif fla.startswith("0.5.0"):
    report(FAIL, f"flash-linear-attention {fla}: corrompe salidas multimodales de Qwen3.5; usar 0.4.2")
else:
    report(OK, f"flash-linear-attention {fla}")
for pkg in ("peft", "accelerate", "omegaconf", "polars"):
    v = version(pkg)
    report(OK if v else FAIL, f"{pkg} {v}")

vj = ROOT / "external" / "open-vljepa" / "openvljepa"
report(OK if vj.is_dir() else FAIL, f"submodule open-vljepa {'presente' if vj.is_dir() else 'ausente'} ({vj.parent})")

tok = os.environ.get("HF_TOKEN")
report(OK if tok else WARN, "HF_TOKEN definido" if tok else
       "HF_TOKEN no definido: VL-JEPA necesita acceso a Llama-3.2-1B y EmbeddingGemma")

for var in ("HF_HOME", "UV_CACHE_DIR"):
    p = os.environ.get(var)
    if p:
        free = shutil.disk_usage(Path(p) if Path(p).exists() else Path(p).parent).free / 2**30
        report(OK if free > 60 else WARN, f"{var}={p}  ({free:.0f} GB libres)")
    else:
        report(WARN, f"{var} no definido (¿cargaste .env?)")

man = ROOT / "data" / "manifests" / "m3di_base_test.parquet"
report(OK if man.exists() else WARN, f"manifiesto {'presente' if man.exists() else 'ausente'}: {man}")

print("\nEntorno listo." if not failures else f"\n{failures} falla(s).")
sys.exit(1 if failures else 0)
