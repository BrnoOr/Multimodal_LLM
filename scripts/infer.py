"""Inferencia (etapa 1 zero-shot y, con `model.adapter_path`, modelos ajustados de la etapa 2).

    uv run --no-sync python scripts/infer.py model=llava_ov prompt=p0_minimal data.limit=200
    uv run --no-sync python scripts/infer.py experiment=e01_zeroshot_llava_p0

Diseñado para correr desatendido en el cluster (ver scripts/launch.sh, logs en logs/<job>/):
  * Reanudable: las predicciones se escriben por lote en predictions.jsonl (flush + fsync). Si el
    proceso muere, relanzar el mismo comando continúa desde la última muestra escrita.
  * Idempotente: si el run ya está completo, termina de inmediato (útil en colas).
  * Robusto a OOM: si un lote no cabe (GPU compartida), se parte en mitades recursivamente.
  * Apagado limpio: SIGTERM/SIGINT terminan el lote en curso, guardan y salen con código 143/130.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from vlmfid.config import compose, to_yaml
from vlmfid.data.manifest import ImageRecordDataset, data_root, load_manifest
from vlmfid.models import build_describer
from vlmfid.paths import RUNS
from vlmfid.prompts import render
from vlmfid.tracking import RunTracker

_STOP = {"flag": False, "signum": None}


def _handle(signum, _frame):
    _STOP.update(flag=True, signum=signum)
    print(f"\n[infer] señal {signal.Signals(signum).name}: se termina el lote en curso y se guarda.", flush=True)


def _done_ids(path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass  # línea truncada por una caída: se re-infiere esa muestra
    return done


def describe_safe(describer, images, prompts, depth: int = 0):
    """describe() con partición recursiva del lote ante CUDA OOM."""
    try:
        return describer.describe(images, prompts)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(images) == 1:
            raise
        h = len(images) // 2
        print(f"[infer] OOM con lote {len(images)} -> {h}+{len(images) - h}", flush=True)
        return (describe_safe(describer, images[:h], prompts[:h], depth + 1)
                + describe_safe(describer, images[h:], prompts[h:], depth + 1))


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:d}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def run(cfg) -> int:
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tracker = RunTracker(RUNS / cfg.exp_id, cfg)
    records = load_manifest(cfg.data.split, cfg.data.dataset, cfg.data.limit, cfg.data.subset_seed,
                            variant=cfg.data.variant)
    done = _done_ids(tracker.predictions_path)
    todo = [r for r in records if r["id"] not in done]
    print(f"[infer] exp_id={cfg.exp_id}  total={len(records)}  hechas={len(done)}  pendientes={len(todo)}")
    if not todo:
        tracker.check_compatible()
        tracker.finish("completed", n_done=len(done), n_total=len(records))
        print("[infer] nada pendiente: run completo.")
        return 0

    print(to_yaml(cfg))
    tracker.start()
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    t_load = time.time()
    describer = build_describer(cfg.model)
    describer.load()
    info = describer.info()
    info["load_seconds"] = round(time.time() - t_load, 1)
    tracker.update(model_info=info)
    print(f"[infer] modelo cargado: {json.dumps(info)}", flush=True)

    loader = DataLoader(
        ImageRecordDataset(todo, data_root(cfg.data.variant, cfg.data.get("root"))),
        batch_size=int(cfg.infer.batch_size),
        num_workers=int(cfg.data.num_workers),
        collate_fn=ImageRecordDataset.collate,
        shuffle=False,
    )
    emb_dir = tracker.dir / "embeddings"
    if cfg.infer.get("save_embeddings"):
        emb_dir.mkdir(exist_ok=True)

    n_new, t0 = 0, time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = "completed"
    try:
        with tracker.predictions_path.open("a", encoding="utf-8") as f:
            for b, (recs, imgs) in enumerate(loader):
                prompts = [render(cfg.prompt, r) for r in recs]
                outs = describe_safe(describer, imgs, prompts)
                for r, p, o in zip(recs, prompts, outs):
                    f.write(json.dumps({
                        "id": r["id"], "variant": r["variant"], "image_path": r["image_path"], "exp_id": cfg.exp_id, "model": cfg.model.name,
                        "prompt_id": cfg.prompt.id, "prompt": p,
                        "prediction": o.text, "reference": r["caption"],
                        "latents_text": r["latents_text"], "latents_image": r["latents_image"],
                        "extra": o.extra,
                    }, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                embs = [o.embedding for o in outs if o.embedding is not None]
                if embs and cfg.infer.get("save_embeddings"):
                    np.savez(emb_dir / f"{len(done) + n_new:07d}.npz",
                             ids=np.array([r["id"] for r in recs]), emb=np.stack(embs))
                n_new += len(recs)

                if (b + 1) % int(cfg.infer.log_every) == 0 or n_new == len(todo):
                    el = time.time() - t0
                    rate = n_new / max(el, 1e-6)
                    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
                    print(f"[infer] {len(done) + n_new}/{len(records)}  {rate:.2f} img/s  "
                          f"ETA {_fmt((len(todo) - n_new) / max(rate, 1e-6))}  pico {mem:.1f} GB  "
                          f"| {outs[0].text[:90]!r}", flush=True)
                    tracker.update(n_done=len(done) + n_new, n_total=len(records), img_per_s=round(rate, 3))
                if _STOP["flag"]:
                    status = "interrupted"
                    break
    except Exception as e:  # se registra y se relanza: el log del job conserva el traceback
        tracker.finish("failed", error=f"{type(e).__name__}: {e}", n_done=len(done) + n_new)
        raise

    el = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    tracker.finish(status, n_done=len(done) + n_new, n_total=len(records),
                   seconds_this_attempt=round(el, 1), peak_mem_gb=round(peak, 2),
                   img_per_s=round(n_new / max(el, 1e-6), 3))
    print(f"[infer] {status}: {n_new} nuevas en {_fmt(el)} -> {tracker.predictions_path}")
    if status == "interrupted":
        return 128 + (_STOP["signum"] or signal.SIGTERM)
    return 0


def main(argv=None) -> None:
    cfg = compose(sys.argv[1:] if argv is None else argv)
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
