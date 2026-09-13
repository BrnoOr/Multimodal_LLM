"""Smoke test de VL-JEPA: inspecciona el checkpoint y describe 4 imagenes.

Uso:
    ./scripts/run.sh python scripts/smoke_vljepa.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl, torch
from PIL import Image

from vlmfid.models.base import ModelSpec
from vlmfid.models.vljepa import VLJepaDescriber
from vlmfid.eval.caption_pool import build_pool, pool_stats

REPO = "external/open-vljepa"
CKPT = f"{REPO}/checkpoints_msrvtt/best.pt"

# 1) que hay en el checkpoint (antes de construir nada)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("claves del checkpoint:", list(ck.keys()))
print("config:")
print(json.dumps({k: v for k, v in ck["config"].items()
                  if isinstance(v, (dict, str, int, float))}, indent=2)[:1500])

# 2) construir el describer
spec = ModelSpec(name="vljepa", hf_id="cun-bjy/open-vljepa", quantization="none",
                 device="cuda:0", extra={"repo": REPO, "ckpt": CKPT,
                                         "quoted_pool": False})
m = VLJepaDescriber(spec)
print(f"parametros: {m.memory_footprint_gib():.2f} GiB | "
      f"frames={m.num_frames} size={m.image_size}")

# 3) pool
print(pool_stats(build_pool(quoted=False)))
m.build_caption_pool()

# 4) describir 4 escenas reales
df = pl.read_parquet("data/manifests/m3di_val.parquet").head(4)
imgs = [Image.open(r["image_path"]).convert("RGB") for r in df.iter_rows(named=True)]
out = m.describe(imgs, "Describe the video.")
top = m.describe_topk(imgs, "Describe the video.", k=3)

for r, o, t in zip(df.iter_rows(named=True), out, top):
    print("\nREF:", r["caption_ref"])
    print("TOP1:", o)
    for txt, s in t:
        print(f"   {s:.4f}  {txt}")
