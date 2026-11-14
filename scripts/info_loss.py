"""Pérdida de información por etapa en VL-JEPA y geometría del banco (ver vlmfid.eval.infoloss).

    # run publicado (banco de 504 captions canónicos)
    uv run --no-sync python scripts/info_loss.py model=vljepa prompt=p1_attributes model.bank.target_format=canonical \
        model.adapter_path=runs/train_vljepa_open_msrvtt_nf4_p1_attributes_tcan_r16_e1/adapter_best info.tag=ft_tcan
    # ajuste v3 (banco factorizado)
    uv run --no-sync python scripts/info_loss.py model=vljepa_fac prompt=p1_attributes \
        model.adapter_path=runs/train_vljepa_open_msrvtt_fac_nf4_qattr_tcan_r16_e4/adapter_best info.tag=ft_e4
    # sin adaptador: el modelo congelado y la geometría del banco antes de ajustar
    uv run --no-sync python scripts/info_loss.py model=vljepa_fac prompt=p1_attributes
    # prueba de humo
    uv run --no-sync python scripts/info_loss.py model=vljepa_fac prompt=p1_attributes info.train_limit=256 info.test_limit=128

En el cluster: SCRIPT=scripts/info_loss.py GPU=1 scripts/launch.sh infoloss scripts/run_queue.sh configs/queues/stage2c_infoloss.txt

Datos: `info.train_limit` imágenes de train para ajustar las sondas y calibrar τ; `info.test_limit` de val
para medir. Test no se usa.

runs/infoloss_<model>_<quant>[_<tag>][_n<train_limit>]/
    config.yaml, env.jsonl, status.json
    features_{train,test}.npz     vis_mean, ŝ (una por consulta) por imagen (se reutilizan si se relanza)
    bank_geometry.json            cosenos entre referencias que compiten, cota del margen y capacidad en nats
    infoloss.jsonl                una fila por atributo: información en cada etapa y pérdida entre etapas
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vlmfid.config import compose, to_yaml
from vlmfid.data.manifest import ImageRecordDataset, data_root, load_manifest
from vlmfid.eval.attributes import extract
from vlmfid.eval.infoloss import decoder_information, information_chain, probe_information
from vlmfid.models import build_describer
from vlmfid.models.bank import ATTRS, VALUES, caption_groups, geometry_report, truth_values
from vlmfid.paths import RUNS
from vlmfid.prompts import render
from vlmfid.tracking import RunTracker

INFO_DEFAULTS = {"train_split": "train", "train_limit": 10_000, "test_split": "val", "test_limit": 2_000,
                 "batch_size": None, "tag": None, "max_iter": 300}
_EXTRACT_KEY = {"shape": "shape", "color": "color_cat", "y": "ypos", "x": "xpos"}   # claves de attributes.extract


def compose_info(argv: list[str]):
    cfg = compose(argv)
    cfg.info = OmegaConf.merge(OmegaConf.create(INFO_DEFAULTS), cfg.get("info") or {})
    if cfg.model.family != "vljepa":
        raise ValueError("scripts/info_loss.py es para VL-JEPA (model=vljepa | vljepa_fac)")
    if not any(a.startswith("exp_id=") for a in argv):
        i = cfg.info
        cfg.exp_id = (f"infoloss_{cfg.model.name}_{cfg.model.quant}" + (f"_{i.tag}" if i.tag else "")
                      + (f"_n{i.train_limit}" if i.train_limit != INFO_DEFAULTS["train_limit"] else ""))
    cfg.stage = "infoloss"
    return cfg


def labels(records: list[dict]) -> dict[str, np.ndarray]:
    out = {a: [] for a in ATTRS}
    for r in records:
        for a, v in truth_values(r).items():
            out[a].append(VALUES[a].index(v))
    return {a: np.asarray(v, dtype=np.int64) for a, v in out.items()}


@torch.inference_mode()
def _batch_features(d, images, prompt: str) -> dict[str, torch.Tensor]:
    """vis_mean (N, Dv) y ŝ. Banco factorizado: pred_<attr> por consulta. Banco de captions: pred (una)."""
    inner = d._inner()
    frames = torch.stack([d.transform(im) for im in images])
    pv = frames.unsqueeze(1).repeat(1, d.num_frames, 1, 1, 1).to(d.device, dtype=torch.bfloat16)
    B, out = len(images), {}
    with torch.autocast(d.device.type, dtype=torch.bfloat16):
        vis = inner.x_encoder(pv)
        vis_t = vis[0] if isinstance(vis, (tuple, list)) else vis
        out["vis_mean"] = vis_t.float().mean(1)
        if d.fbank is not None:
            for a in ATTRS:
                ids, mask = d.attr_query[a]
                out[f"pred_{a}"] = F.normalize(inner.predictor(vis, ids.expand(B, -1), mask.expand(B, -1)).float(), dim=-1)
        else:
            q = [d.cfg.get("query_override") or prompt] * B
            ids, mask = d._tok(d.query_tok, q, d.max_query_len)
            out["pred"] = F.normalize(inner.predictor(vis, ids, mask).float(), dim=-1)
    return out


def features(d, records, root, prompt, cfg, path, log=print) -> dict[str, np.ndarray]:
    ids = [r["id"] for r in records]
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        if list(blob["ids"]) == ids:
            log(f"[info] reutilizando {path.name}")
            return {k: blob[k] for k in blob.files if k != "ids"}
    bs = int(cfg.info.batch_size or max(1, int(cfg.model.get("batch_size", 16))))
    loader = DataLoader(ImageRecordDataset(records, root), batch_size=bs, num_workers=int(cfg.data.num_workers),
                        collate_fn=ImageRecordDataset.collate, shuffle=False)
    chunks, n = {}, 0
    for _recs, imgs in loader:
        for k, v in _batch_features(d, imgs, prompt).items():
            chunks.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
        n += len(imgs)
        if n % (20 * bs) < bs:
            log(f"[info] rasgos: {n}/{len(records)} imágenes")
    feats = {k: np.concatenate(v) for k, v in chunks.items()}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(ids), **feats)
    tmp.replace(path)
    return feats


def bank_columns(d) -> tuple[dict[str, np.ndarray], dict]:
    """Por atributo: clase (valor) de cada referencia del banco que compite por él, y grupos para la geometría.

    Banco factorizado: las referencias del atributo son su tramo. Banco de captions: todas las frases, cada
    una con el valor que el extractor de la etapa 1 lee en ella (−1 si no lo encuentra: se descarta).
    """
    if d.fbank is not None:
        cols = {a: np.arange(len(VALUES[a])) for a in ATTRS}
        groups = {a: list(range(d.fbank.slices[a].start, d.fbank.slices[a].stop)) for a in ATTRS}
        return cols, groups
    attr_of, cols = {}, {a: [] for a in ATTRS}
    for t in d.bank_texts:
        e = extract(t)
        vals = {a: e.get(_EXTRACT_KEY[a]) for a in ATTRS}
        attr_of[t] = vals
        for a in ATTRS:
            cols[a].append(VALUES[a].index(vals[a]) if vals[a] in VALUES[a] else -1)
    complete = [t for t in d.bank_texts if all(attr_of[t][a] in VALUES[a] for a in ATTRS)]
    index = {t: i for i, t in enumerate(d.bank_texts)}
    groups = {a: [[index[complete[i]] for i in g] for g in gs]
              for a, gs in caption_groups(complete, attr_of).items()}
    return {a: np.asarray(v) for a, v in cols.items()}, groups


def geometry(d, groups: dict, tau: float, max_groups: int = 40) -> dict:
    emb = d.bank_emb.float().cpu()
    if d.fbank is not None:
        return geometry_report(emb, groups, tau)
    out = {}
    for a, gs in groups.items():           # banco de captions: media sobre grupos que solo difieren en `a`
        gs = gs[:: max(1, len(gs) // max_groups)][:max_groups]
        if not gs:
            continue
        reps = [geometry_report(emb, {"g": g}, tau)["g"] for g in gs]
        out[a] = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]} | {"n_groups": len(gs)}
    return out


def run(cfg) -> int:
    i = cfg.info
    out_dir = RUNS / cfg.exp_id
    results_path = out_dir / "infoloss.jsonl"
    if results_path.exists():
        print(f"[info] {cfg.exp_id}: completo ({results_path}).")
        return 0
    torch.manual_seed(int(cfg.seed))
    print(to_yaml(cfg))
    tracker = RunTracker(out_dir, cfg)
    tracker.start()
    try:
        tr = load_manifest(i.train_split, cfg.data.dataset, i.train_limit, cfg.data.subset_seed, variant=cfg.data.variant)
        te = load_manifest(i.test_split, cfg.data.dataset, i.test_limit, cfg.data.subset_seed, variant=cfg.data.variant)
        root = data_root(cfg.data.variant, cfg.data.get("root"))
        t0 = time.time()
        d = build_describer(cfg.model)
        d.load()
        tracker.update(model_info=d.info() | {"load_seconds": round(time.time() - t0, 1)})
        prompt = render(cfg.prompt)
        tau = float(cfg.model.bank.get("temperature") or 0.07)
        cols, groups = bank_columns(d)
        geo = geometry(d, groups, tau)
        (out_dir / "bank_geometry.json").write_text(json.dumps(
            {"bank_kind": d._bank_kind(), "n_references": len(d.bank_texts), "temperature": tau, "attributes": geo},
            indent=2, ensure_ascii=False), encoding="utf-8")
        for a, g in geo.items():
            print(f"[info] banco · {a:<5s} cos máx={g['cos_max']:.4f}  margen ≤ {g['margin_bound_nats_hardest_pair']:.2f} nats"
                  f"  capacidad={g['capacity_nats']:.3f} de {g['max_nats']:.3f} nats", flush=True)
        f_tr = features(d, tr, root, prompt, cfg, out_dir / "features_train.npz")
        f_te = features(d, te, root, prompt, cfg, out_dir / "features_test.npz")
        bank = F.normalize(d.bank_emb.float(), dim=-1).cpu().numpy()
        slices = {a: d.fbank.slices[a] for a in ATTRS} if d.fbank is not None else None
        d.unload()
        y_tr, y_te = labels(tr), labels(te)

        rows = []
        for a in ATTRS:
            k = len(VALUES[a])
            key = f"pred_{a}" if slices else "pred"
            if slices:
                refs, cc = bank[slices[a]], cols[a]
            else:
                keep = cols[a] >= 0
                refs, cc = bank[keep], cols[a][keep]
            vis = probe_information(f_tr["vis_mean"], y_tr[a], f_te["vis_mean"], y_te[a], k, max_iter=int(i.max_iter),
                                    seed=int(cfg.seed))
            pre = probe_information(f_tr[key], y_tr[a], f_te[key], y_te[a], k, max_iter=int(i.max_iter),
                                    seed=int(cfg.seed))
            dec = decoder_information(f_tr[key] @ refs.T, y_tr[a], f_te[key] @ refs.T, y_te[a], cc, k, tau)
            row = information_chain(a, k, vis, pre, dec) | {
                "exp_id": cfg.exp_id, "model": cfg.model.name, "adapter_path": cfg.model.get("adapter_path"),
                "bank_kind": "factorized" if slices else "captions", "n_references": int(len(refs)),
                "bank_capacity_nats": geo.get(a, {}).get("capacity_nats"), "n_train": len(tr), "n_test": len(te)}
            rows.append(row)
            print(f"[info] {a:<5s} H={row['h_nats']:.3f} | visual {row['info_visual']:.3f} -> predictor "
                  f"{row['info_predictor']:.3f} -> decoder {row['info_decoder']:.3f} nats (τ*={row['tau_calibrated']:.3f}; "
                  f"a τ={tau:g}: {row['info_decoder_tau_train']:.3f}) | pérdida en el decoder {row['loss_decoder']:.3f} | "
                  f"acc sonda {row['acc_predictor']:.3f} vs decoder {row['acc_decoder']:.3f}", flush=True)
        tmp = results_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(results_path)
    except Exception as e:
        tracker.finish("failed", error=f"{type(e).__name__}: {e}")
        raise
    tracker.finish("completed", n_rows=len(rows), seconds=round(time.time() - t0, 1))
    print(f"[info] completo -> {results_path}")
    return 0


def main(argv=None) -> None:
    cfg = compose_info(sys.argv[1:] if argv is None else argv)
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
