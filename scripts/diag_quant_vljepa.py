"""Diagnóstico de cuantización de VL-JEPA: qué componente explica la brecha NF4 − bf16.

Contexto (docs/resultados_etapa1.md §7.4): en la inspección, el coseno entre Ŝ_Y y S_Y baja ≈ 0.1 en
NF4 frente a bf16 (n = 4). La cuantización de VL-JEPA es manual (`models/vljepa.py`), así que antes de
atribuir esa brecha a NF4 se localiza: se cuantiza un componente a la vez y se mide alineación y
discriminación sobre las mismas imágenes de VAL.

    uv run --no-sync python scripts/diag_quant_vljepa.py model=vljepa model.quant=bf16
    uv run --no-sync python scripts/diag_quant_vljepa.py model=vljepa
    uv run --no-sync python scripts/diag_quant_vljepa.py model=vljepa model.quant_components=[predictor]
    # humo
    uv run --no-sync python scripts/diag_quant_vljepa.py model=vljepa model.quant=bf16 diag.n=64 diag.batch_size=32 diag.tag=humo

En el cluster: SCRIPT=scripts/diag_quant_vljepa.py GPU=1 scripts/launch.sh diag_quant \
    scripts/run_queue.sh configs/queues/diag_quant_vljepa.txt   (bf16 primero: es la referencia)

Por condición (todas con el prompt p1_attributes, sin Y-Decoder):
    alineación      coseno(Ŝ_Y, S_Y) por imagen: media, desviación y percentiles
    discriminación  recall@{1,5} Ŝ_Y → S_Y y S_Y → Ŝ_Y sobre los N candidatos; InfoNCE bidireccional en
                    lotes fijos de `diag.batch_size` (τ del checkpoint), con su exactitud top-1 en el lote
    error por etapa coseno de cada representación contra la MISMA representación en bf16 (si la referencia
                    ya existe): E1 vis_mean, E2 pred_hidden, E3 pred (Ŝ_Y) y T1 s_y (S_Y)

Datos: `diag.n` imágenes de VAL (subconjunto determinista por data.subset_seed, el mismo en todas las
condiciones). Test no se usa. No entrena nada.

runs/diagquant_<model>_<quant>[_<componentes>][_<tag>]/
    config.yaml, env.jsonl, status.json
    features.npz     ids, vis_mean, pred_hidden, pred, s_y (para comparar contra la referencia)
    metrics.json     todas las métricas de la condición
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
from vlmfid.eval.embedding_metrics import infonce_fixed_batches, paired_cosine, retrieval_recall
from vlmfid.models import build_describer
from vlmfid.models.vljepa import QUANT_COMPONENTS
from vlmfid.paths import RUNS
from vlmfid.prompts import render
from vlmfid.tracking import RunTracker

# n = 1024: 8 lotes fijos de 128 (lote efectivo propuesto para la etapa 2, P8). τ = 0.07: la del
# checkpoint (open-vljepa configs/stage_b_retrain.yaml; configs/train/budget.yaml). Valores del
# diagnóstico, no del protocolo.
DIAG_DEFAULTS = {"n": 1024, "batch_size": 128, "temperature": 0.07, "tag": None}
POINTS = ("vis_mean", "pred_hidden", "pred", "s_y")
POINT_LABEL = {"vis_mean": "E1", "pred_hidden": "E2", "pred": "E3", "s_y": "T1"}


def condition_name(cfg) -> str:
    """bf16 | nf4 (los tres) | nf4_<componentes> (subconjunto)."""
    if cfg.model.quant != "nf4":
        return str(cfg.model.quant)
    comps = cfg.model.get("quant_components")
    if comps is None or set(comps) == set(QUANT_COMPONENTS):
        return "nf4"
    return "nf4_" + "+".join(c for c in QUANT_COMPONENTS if c in set(comps))


def compose_diag(argv: list[str]):
    if not any(a.startswith("prompt=") for a in argv):
        argv = [*argv, "prompt=p1_attributes"]
    cfg = compose(argv)
    cfg.diag = OmegaConf.merge(OmegaConf.create(DIAG_DEFAULTS), cfg.get("diag") or {})
    if cfg.model.family != "vljepa":
        raise ValueError("scripts/diag_quant_vljepa.py es para VL-JEPA (model=vljepa)")
    if cfg.model.get("decoder_path") or cfg.model.get("adapter_path"):
        raise ValueError("El diagnóstico usa el VL-JEPA preentrenado sin Y-Decoder ni adaptador")
    if str(cfg.data.split) != "val":
        raise ValueError("El diagnóstico usa solo VAL")
    tag = f"_{cfg.diag.tag}" if cfg.diag.tag else ""
    if not any(a.startswith("exp_id=") for a in argv):
        cfg.exp_id = f"diagquant_{cfg.model.name}_{condition_name(cfg)}{tag}"
    cfg.diag.reference = f"diagquant_{cfg.model.name}_bf16{tag}"
    cfg.stage = "diagquant"
    return cfg


def extract(d, records, root, prompt: str, batch_size: int, workers: int) -> dict[str, np.ndarray]:
    loader = DataLoader(ImageRecordDataset(records, root), batch_size=batch_size, num_workers=workers,
                        collate_fn=ImageRecordDataset.collate, shuffle=False)
    feats = {k: [] for k in POINTS}
    ids, t0 = [], time.time()
    for b, (recs, imgs) in enumerate(loader):
        st = d.predict_stages(imgs, [prompt] * len(imgs))
        s_y = d.encode_texts([r["caption"] for r in recs])
        for k in ("vis_mean", "pred_hidden", "pred"):
            feats[k].append(st[k].cpu().numpy().astype(np.float32))
        feats["s_y"].append(s_y.cpu().numpy().astype(np.float32))
        ids += [str(r["id"]) for r in recs]
        if (b + 1) % 8 == 0:
            print(f"[diag] {len(ids)}/{len(records)} imágenes ({time.time() - t0:.0f} s)", flush=True)
    if ids != [str(r["id"]) for r in records]:
        raise RuntimeError("El orden de las representaciones no coincide con el de los registros")
    out = {k: np.concatenate(v) for k, v in feats.items()}
    for k, v in out.items():
        if v.ndim != 2 or v.shape[0] != len(records) or not np.isfinite(v).all():
            raise RuntimeError(f"{k}: forma {v.shape} o valores no finitos")
    return out


def summarize(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)), "max": float(x.max())}


def run(cfg) -> int:
    g = cfg.diag
    out_dir = RUNS / cfg.exp_id
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[diag] {cfg.exp_id}: completo ({metrics_path}).")
        return 0
    torch.manual_seed(int(cfg.seed))
    print(to_yaml(cfg))
    tracker = RunTracker(out_dir, cfg)
    tracker.start()
    try:
        recs = load_manifest("val", cfg.data.dataset, int(g.n), cfg.data.subset_seed, variant=cfg.data.variant)
        root = data_root(cfg.data.variant, cfg.data.get("root"))
        prompt = render(cfg.prompt)
        d = build_describer(cfg.model)
        d.load()
        info = d.info()
        tracker.update(model_info=info)
        print(f"[diag] condición {condition_name(cfg)}: NF4 en {info.get('quant_components')}", flush=True)
        feats = extract(d, recs, root, prompt, int(cfg.model.get("batch_size", 16)), int(cfg.data.num_workers))
        d.unload()
        del d
        ids = np.array([str(r["id"]) for r in recs])
        np.savez(out_dir / "features.npz", ids=ids, **feats)

        captions = [r["caption"] for r in recs]
        cos = paired_cosine(feats["pred"], feats["s_y"])
        m = {"exp_id": cfg.exp_id, "condition": condition_name(cfg), "quant": cfg.model.quant,
             "quant_components": info.get("quant_components"), "prompt_id": cfg.prompt.id, "n": len(recs),
             "n_distinct_captions": len(set(captions)),
             "alignment": {"cos_pred_target": summarize(cos)},
             "discrimination": {**retrieval_recall(feats["pred"], feats["s_y"], captions, ks=(1, 5)),
                                **infonce_fixed_batches(feats["pred"], feats["s_y"], captions,
                                                        int(g.batch_size), float(g.temperature))},
             "quant_report": {k: v for k, v in info.items()
                              if k.startswith("quantized_") or k in ("params_quantized", "fraction_quantized")}}

        ref_path = RUNS / g.reference / "features.npz"
        if cfg.exp_id == g.reference:
            m["vs_bf16"] = "esta es la referencia"
        elif ref_path.exists():
            ref = np.load(ref_path, allow_pickle=False)
            if [str(x) for x in ref["ids"]] != ids.tolist():
                raise RuntimeError(f"La referencia {ref_path} tiene otras imágenes; usa el mismo diag.n y subset_seed")
            m["vs_bf16"] = {f"{POINT_LABEL[k]}_{k}": summarize(paired_cosine(feats[k], ref[k])) for k in POINTS}
            m["vs_bf16"]["delta_cos_pred_target"] = float(cos.mean() - paired_cosine(ref["pred"], ref["s_y"]).mean())
        else:
            m["vs_bf16"] = f"sin referencia: falta {ref_path} (correr primero model.quant=bf16)"

        text = json.dumps(m, indent=2, ensure_ascii=False)
        tmp = metrics_path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(metrics_path)
        print(text)
        tracker.finish("completed", n=len(recs))
        return 0
    except BaseException as e:
        tracker.finish("interrupted" if isinstance(e, KeyboardInterrupt) else "failed", error=repr(e))
        raise


def main(argv=None) -> int:
    return run(compose_diag(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
