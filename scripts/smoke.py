"""Smoke test: carga cada modelo en secuencia, describe N imágenes de test y libera la GPU.

    uv run --no-sync python scripts/smoke.py                         # los 4 modelos, 2 imágenes
    uv run --no-sync python scripts/smoke.py --models vljepa --n 4 --set model.bank.max_captions=2000

No escribe en runs/: sirve para verificar descargas, cuantización y plantillas de chat.
"""

from __future__ import annotations

import argparse
import time
import traceback

import torch

from vlmfid.config import compose
from vlmfid.data.manifest import ImageRecordDataset, data_root, load_manifest
from vlmfid.models import build_describer
from vlmfid.paths import CONFIGS
from vlmfid.prompts import render

DEFAULT_ORDER = ["vljepa", "llava_ov", "qwen35", "internvl3"]  # VL-JEPA primero: mayor riesgo
AVAILABLE = sorted(p.stem for p in (CONFIGS / "models").glob("*.yaml"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=AVAILABLE,
                    help="nombres de archivo en configs/models/")
    ap.add_argument("--prompt", default="p0_minimal")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--set", nargs="*", default=[], help="overrides dotlist adicionales")
    a = ap.parse_args(argv)

    recs = load_manifest("test", limit=a.n, subset_seed=0)
    ds = ImageRecordDataset(recs, data_root())
    recs, imgs = ImageRecordDataset.collate([ds[i] for i in range(len(ds))])
    failures = []
    for name in a.models:
        print(f"\n{'=' * 80}\n{name}\n{'=' * 80}", flush=True)
        try:
            cfg = compose([f"model={name}", f"prompt={a.prompt}", *a.set])
            t = time.time()
            d = build_describer(cfg.model)
            d.load()
            print(f"carga {time.time() - t:.0f}s  {d.info()}")
            t = time.time()
            outs = d.describe(imgs, [render(cfg.prompt, r) for r in recs])
            print(f"inferencia {time.time() - t:.1f}s para {len(imgs)} imágenes")
            for r, o in zip(recs, outs):
                print(f"- {r['id']}\n  REF : {r['caption']}\n  PRED: {o.text}\n  extra: {o.extra}")
            d.unload()
            del d
        except Exception:
            traceback.print_exc()
            failures.append(name)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nResumen: OK={[m for m in a.models if m not in failures]}  FALLÓ={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
