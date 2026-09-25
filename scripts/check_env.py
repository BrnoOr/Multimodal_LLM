"""Verificacion del entorno en laptop (5090, cu128) y cluster (4090, cu126).

Uso: uv run python scripts/check_env.py
Sale con codigo 1 si algo critico falla.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys

from dotenv import load_dotenv

load_dotenv()

EXPECTED_CC = {"win32": (12, 0), "linux": (8, 9)}  # laptop Blackwell / cluster Ada
ok = True


def check(cond: bool, msg: str, critical: bool = True) -> None:
    global ok
    tag = "OK " if cond else ("ERR" if critical else "WARN")
    print(f"[{tag}] {msg}")
    if not cond and critical:
        ok = False


print(f"python {platform.python_version()} | {sys.platform} | {platform.node()}")

import torch

print(f"torch {torch.__version__} | build CUDA {torch.version.cuda}")
check(torch.cuda.is_available(), "CUDA disponible")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(
        f"GPU visible: {name} | cc {cc} | {vram:.1f} GiB | CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}"
    )
    exp = EXPECTED_CC.get(sys.platform)
    check(
        cc == exp,
        f"compute capability esperada {exp} para {sys.platform} (obtenida {cc})",
        critical=False,
    )
    if sys.platform == "win32":
        check(
            torch.version.cuda and torch.version.cuda.startswith("12.8"),
            "build cu128 en laptop Blackwell",
        )
    if sys.platform == "linux":
        check(not torch.version.cuda.startswith("12.8"), "build != cu128 en cluster con driver 550")
    check(torch.cuda.is_bf16_supported(), "bf16 soportado")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    check(y == y, "matmul bf16 en GPU ejecuta (kernel real, no fallback)")

    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit

        lin = Linear4bit(2048, 2048, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
        out = lin(x)
        check(out.shape == (2048, 2048), f"bitsandbytes {bnb.__version__}: Linear4bit NF4 ejecuta")
    except Exception as e:  # noqa: BLE001
        check(False, f"bitsandbytes NF4: {e}")

import accelerate
import peft
import transformers

print(
    f"transformers {transformers.__version__} | peft {peft.__version__} | accelerate {accelerate.__version__}"
)

hf_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
os.makedirs(hf_home, exist_ok=True)
free = shutil.disk_usage(hf_home).free / 2**30
check(free > 60, f"HF_HOME={hf_home} con {free:.0f} GiB libres (se necesitan ~50)", critical=False)

for mod, extra in [("qwen_vl_utils", "qwen"), ("sacrebleu", "eval"), ("pycocoevalcap", "eval")]:
    try:
        __import__(mod)
        check(True, f"{mod} importa", critical=False)
    except ImportError:
        check(False, f"{mod} no instalado (extra [{extra}])", critical=False)

try:
    import awq  # noqa: F401

    check(True, "autoawq disponible (opcional)", critical=False)
except ImportError:
    check(True, "autoawq ausente: se usa NF4 para los tres modelos (esperado)", critical=False)

print("\nRESULTADO:", "entorno listo" if ok else "hay errores criticos")
sys.exit(0 if ok else 1)
