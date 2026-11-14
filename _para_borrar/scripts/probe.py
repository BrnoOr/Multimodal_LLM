"""Sondas lineales sobre la representación de un modelo (diagnóstico de la etapa 2; ver vlmfid.eval.probe).

    uv run --no-sync python scripts/probe.py model=qwen35 prompt=p1_attributes
    # sobre el modelo ajustado (mismo prompt; el tag distingue el exp_id)
    uv run --no-sync python scripts/probe.py model=qwen35 prompt=p1_attributes \
        model.adapter_path=runs/train_qwen35_08b_base_nf4_p1_attributes_tcan_r16_e1/adapter_best probe.tag=ft_tcan
    # prueba de humo
    uv run --no-sync python scripts/probe.py model=qwen35 prompt=p1_attributes probe.train_limit=256 probe.test_limit=128

En el cluster: SCRIPT=scripts/probe.py GPU=1 scripts/launch.sh probe scripts/run_queue.sh configs/queues/stage2b_probe.txt

runs/probe_<model>_<quant>_<prompt>[_<tag>][_n<train_limit>]/
    config.yaml, env.jsonl, status.json
    features_train.npz, features_test.npz   rasgos por imagen (se reutilizan si se relanza)
    probe_results.jsonl                     una fila por (rasgo, atributo)
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vlmfid.config import compose, to_yaml
from vlmfid.data.manifest import ImageRecordDataset, data_root, load_manifest
from vlmfid.eval.probe import ATTRS, PROBE_DEFAULTS, extract_features, linear_probe, probe_labels
from vlmfid.models import build_describer
from vlmfid.paths import RUNS
from vlmfid.prompts import render
from vlmfid.tracking import RunTracker


def compose_probe(argv: list[str]):
    cfg = compose(argv)
    cfg.probe = OmegaConf.merge(OmegaConf.create(PROBE_DEFAULTS), cfg.get("probe") or {})
    if not any(a.startswith("exp_id=") for a in argv):
        p = cfg.probe
        cfg.exp_id = (f"probe_{cfg.model.name}_{cfg.model.quant}_{cfg.prompt.id}"
                      + (f"_{p.tag}" if p.tag else "")
                      + (f"_n{p.train_limit}" if p.train_limit != PROBE_DEFAULTS["train_limit"] else ""))
    cfg.stage = "probe"
    return cfg


def _features(describer, records, root, prompt, cfg, path, log):
    ids = [r["id"] for r in records]
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        if list(blob["ids"]) == ids:
            log(f"[probe] reutilizando {path.name}")
            return {k: blob[k] for k in blob.files if k != "ids"}
    bs = int(cfg.probe.batch_size or max(1, int(cfg.model.get("batch_size", 16)) // 2))
    loader = DataLoader(ImageRecordDataset(records, root), batch_size=bs, num_workers=int(cfg.data.num_workers),
                        collate_fn=ImageRecordDataset.collate, shuffle=False)
    got_ids, feats = extract_features(describer, loader, prompt, list(cfg.probe.layers), log)
    assert got_ids == ids
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(ids), **feats)
    tmp.replace(path)
    return feats


def run(cfg) -> int:
    p = cfg.probe
    out_dir = RUNS / cfg.exp_id
    results_path = out_dir / "probe_results.jsonl"
    if results_path.exists():
        print(f"[probe] {cfg.exp_id}: completo ({results_path}).")
        return 0
    torch.manual_seed(int(cfg.seed))
    print(to_yaml(cfg))
    tracker = RunTracker(out_dir, cfg)
    tracker.start()
    try:
        tr = load_manifest(p.train_split, cfg.data.dataset, p.train_limit, cfg.data.subset_seed, variant=cfg.data.variant)
        te = load_manifest(p.test_split, cfg.data.dataset, p.test_limit, cfg.data.subset_seed, variant=cfg.data.variant)
        root = data_root(cfg.data.variant, cfg.data.get("root"))
        t0 = time.time()
        describer = build_describer(cfg.model)
        describer.load()
        if describer.family != "vljepa":
            describer.model.eval()
        tracker.update(model_info=describer.info() | {"load_seconds": round(time.time() - t0, 1)})
        prompt = render(cfg.prompt)
        f_tr = _features(describer, tr, root, prompt, cfg, out_dir / "features_train.npz", print)
        f_te = _features(describer, te, root, prompt, cfg, out_dir / "features_test.npz", print)
        describer.unload()
        y_tr, y_te = probe_labels(tr), probe_labels(te)

        rows = []
        for feat in sorted(f_tr):
            for attr, k in ATTRS.items():
                r = linear_probe(f_tr[feat], y_tr[attr], f_te[feat], y_te[attr], k, list(p.weight_decays),
                                 int(p.max_iter), int(cfg.seed))
                rows.append({"exp_id": cfg.exp_id, "model": cfg.model.name, "prompt": cfg.prompt.id,
                             "adapter_path": cfg.model.get("adapter_path"), "tag": p.tag, "feature": feat,
                             "attribute": attr, **r})
                print(f"[probe] {feat:>10s} · {attr:<5s} acc={r['acc']:.3f} (mayoritaria {r['majority_acc']:.3f}, "
                      f"λ={r['weight_decay']:g})", flush=True)
        tmp = results_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(results_path)
    except Exception as e:
        tracker.finish("failed", error=f"{type(e).__name__}: {e}")
        raise
    tracker.finish("completed", n_rows=len(rows), seconds=round(time.time() - t0, 1))
    print(f"[probe] completo -> {results_path}")
    return 0


def main(argv=None) -> None:
    cfg = compose_probe(sys.argv[1:] if argv is None else argv)
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
