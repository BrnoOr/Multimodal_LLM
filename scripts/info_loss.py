"""Pérdida de información en cada etapa de VL-JEPA (docs/metodologia.md §2.5; vlmfid.eval.probes).

    # etapa 1: modelo preentrenado, con su Y-Decoder
    uv run --no-sync python scripts/info_loss.py model=vljepa prompt=p1_attributes \
        model.decoder_path=runs/<ydec_..._zs_...>/decoder_best
    # etapa 2: modelo ajustado y su propio decodificador
    uv run --no-sync python scripts/info_loss.py model=vljepa prompt=p1_attributes info.tag=ft \
        model.adapter_path=runs/<train_vljepa...>/adapter_best model.decoder_path=runs/<ydec_..._ft_...>/decoder_best
    # sin decodificador: solo las sondas (E1, E2, E3, T1)
    uv run --no-sync python scripts/info_loss.py model=vljepa prompt=p1_attributes
    # prueba de humo
    uv run --no-sync python scripts/info_loss.py model=vljepa prompt=p1_attributes info.fit_limit=256 info.eval_limit=128 info.tag=humo

En el cluster: SCRIPT=scripts/info_loss.py GPU=0 scripts/launch.sh infoloss scripts/run_queue.sh configs/queues/stage1_infoloss.txt

Cadena:  imagen -> X-Encoder -> Predictor -> proyección -> Y-Decoder -> texto

    E1  vis_mean      salida del X-Encoder (media de tokens)                 sonda
    E2  pred_hidden   estado del Predictor, antes de la proyección de salida  sonda
    E3  pred          embedding predicho Ŝ_Y                                  sonda
    E4  text_pred     texto decodificado desde Ŝ_Y                            extractor de atributos
    T1  s_y           embedding objetivo S_Y = Y-Encoder(caption)             sonda
    T2  text_target   texto decodificado desde S_Y (techo del decodificador)  extractor de atributos

Factores discretos (forma, x, y, nombre de color): MLP de una capa oculta; exactitud y cota de
información en bits. Factores continuos (rotaciones, foco, fondo, matiz del objeto), solo en E1–E3 y T1:
kernel ridge; R². Son factores que el texto no contiene: se espera que el Predictor los descarte.

Datos: `info.fit_limit` imágenes de TRAIN para ajustar las sondas y `info.eval_limit` de VAL para medir.
Test no se usa.

runs/infoloss_<model>_<quant>_<prompt>[_<tag>]/
    config.yaml, env.jsonl, status.json
    features_{fit,eval}.npz    representaciones por imagen (se reutilizan si se relanza)
    texts_eval.jsonl           textos decodificados en val (si hay decodificador)
    infoloss.jsonl             una fila por (punto, factor)
    tabla_discretos.csv        factor × punto: exactitud e información en bits
"""

from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vlmfid.config import compose, to_yaml
from vlmfid.data.manifest import ImageRecordDataset, data_root, load_manifest
from vlmfid.eval.attributes import SHAPE_NAMES, XPOS_NAMES, YPOS_NAMES, extract, truth_from_record
from vlmfid.eval.probes import entropy_bits, kernel_ridge_probe, mlp_probe, reader_information
from vlmfid.models import build_describer
from vlmfid.paths import RUNS
from vlmfid.prompts import render
from vlmfid.tracking import RunTracker

INFO_DEFAULTS = {"fit_limit": 10_000, "eval_limit": 2_000, "batch_size": None, "tag": None}
PROBE_POINTS = ("vis_mean", "pred_hidden", "pred", "s_y")             # E1, E2, E3, T1
POINT_LABEL = {"vis_mean": "E1", "pred_hidden": "E2", "pred": "E3", "text_pred": "E4",
               "s_y": "T1", "text_target": "T2"}
DISCRETE = ("shape", "x", "y", "color")
CONTINUOUS = ("object_color", "object_alpharot", "object_betarot", "object_gammarot", "spotlight_pos",
              "spotlight_color", "background_color")


def compose_info(argv: list[str]):
    cfg = compose(argv)
    cfg.info = OmegaConf.merge(OmegaConf.create(INFO_DEFAULTS), cfg.get("info") or {})
    if cfg.model.family != "vljepa":
        raise ValueError("scripts/info_loss.py es para VL-JEPA (model=vljepa)")
    if not any(a.startswith("exp_id=") for a in argv):
        cfg.exp_id = (f"infoloss_{cfg.model.name}_{cfg.model.quant}_{cfg.prompt.id}"
                      + (f"_{cfg.info.tag}" if cfg.info.tag else ""))
    cfg.stage = "infoloss"
    return cfg


def extract_features(d, records, root, prompt: str, batch_size: int, workers: int, with_text: bool):
    """Representaciones por imagen y, si hay decodificador, los textos decodificados."""
    loader = DataLoader(ImageRecordDataset(records, root), batch_size=batch_size, num_workers=workers,
                        collate_fn=ImageRecordDataset.collate, shuffle=False)
    feats = {k: [] for k in PROBE_POINTS}
    texts = {"text_pred": [], "text_target": []}
    ids, t0 = [], time.time()
    for b, (recs, imgs) in enumerate(loader):
        st = d.predict_stages(imgs, [prompt] * len(imgs))
        s_y = d.encode_texts([r["caption"] for r in recs])
        for k in ("vis_mean", "pred_hidden", "pred"):
            feats[k].append(st[k].cpu().numpy().astype(np.float32))
        feats["s_y"].append(s_y.cpu().numpy().astype(np.float32))
        if with_text:
            texts["text_pred"] += d.decoder.decode(st["pred"])
            texts["text_target"] += d.decoder.decode(s_y)
        ids += [r["id"] for r in recs]
        if (b + 1) % 20 == 0:
            print(f"[info] {len(ids)}/{len(records)} imágenes ({time.time() - t0:.0f} s)", flush=True)
    if ids != [r["id"] for r in records]:
        raise RuntimeError("El orden de las representaciones no coincide con el de los registros")
    return {k: np.concatenate(v) for k, v in feats.items()}, texts


def cached_features(path, d, records, root, prompt, batch_size, workers):
    ids = [str(r["id"]) for r in records]
    if path.exists():
        blob = np.load(path, allow_pickle=False)
        if [str(x) for x in blob["ids"]] == ids:
            print(f"[info] reutilizando {path.name}")
            return {k: blob[k] for k in PROBE_POINTS}
    feats, _ = extract_features(d, records, root, prompt, batch_size, workers, with_text=False)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(ids), **feats)
    tmp.replace(path)
    return feats


def discrete_labels(fit_recs, eval_recs) -> tuple[dict, dict, dict]:
    """Etiquetas enteras por factor. El vocabulario de color sale del conjunto de ajuste."""
    inv = {"shape": {v: k for k, v in SHAPE_NAMES.items()}, "x": {v: k for k, v in XPOS_NAMES.items()},
           "y": {v: k for k, v in YPOS_NAMES.items()}}
    tf, te = [truth_from_record(r) for r in fit_recs], [truth_from_record(r) for r in eval_recs]
    names = sorted({t["color"] for t in tf})
    inv["color"] = {n: i for i, n in enumerate(names)}
    unseen = sorted({t["color"] for t in te} - set(names))
    if unseen:
        raise ValueError(f"Nombres de color de val ausentes en el conjunto de ajuste: {unseen[:5]}. "
                         "Sube info.fit_limit.")
    y_fit = {a: np.array([inv[a][t[a]] for t in tf]) for a in DISCRETE}
    y_eval = {a: np.array([inv[a][t[a]] for t in te]) for a in DISCRETE}
    return y_fit, y_eval, inv


def run(cfg) -> int:
    i = cfg.info
    out_dir = RUNS / cfg.exp_id
    results_path = out_dir / "infoloss.jsonl"
    if results_path.exists():
        print(f"[info] {cfg.exp_id}: completo ({results_path}).")
        return 0
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    print(to_yaml(cfg))
    tracker = RunTracker(out_dir, cfg)
    tracker.start()
    try:
        fit_recs = load_manifest("train", cfg.data.dataset, int(i.fit_limit), cfg.data.subset_seed,
                                 variant=cfg.data.variant)
        eval_recs = load_manifest("val", cfg.data.dataset, int(i.eval_limit), cfg.data.subset_seed,
                                  variant=cfg.data.variant)
        root = data_root(cfg.data.variant, cfg.data.get("root"))
        prompt = render(cfg.prompt)
        d = build_describer(cfg.model)
        d.load()
        tracker.update(model_info=d.info())
        bs = int(i.batch_size or cfg.model.get("batch_size", 16))
        workers = int(cfg.data.num_workers)
        has_dec = d.decoder is not None

        f_fit = cached_features(out_dir / "features_fit.npz", d, fit_recs, root, prompt, bs, workers)
        f_eval, texts = extract_features(d, eval_recs, root, prompt, bs, workers, with_text=has_dec)
        np.savez(out_dir / "features_eval.npz", ids=np.array([str(r["id"]) for r in eval_recs]), **f_eval)
        if has_dec:
            with (out_dir / "texts_eval.jsonl").open("w", encoding="utf-8") as f:
                for r, tp, tt in zip(eval_recs, texts["text_pred"], texts["text_target"]):
                    f.write(json.dumps({"id": r["id"], "reference": r["caption"], "text_pred": tp,
                                        "text_target": tt}, ensure_ascii=False) + "\n")
        d.unload()
        del d

        y_fit, y_eval, inv = discrete_labels(fit_recs, eval_recs)
        rows: list[dict] = []
        common = {"exp_id": cfg.exp_id, "prompt_id": cfg.prompt.id, "adapter": cfg.model.get("adapter_path"),
                  "decoder": cfg.model.get("decoder_path")}

        # ── factores discretos: sondas en E1, E2, E3 y T1 ─────────────────────────────────────────
        for a in DISCRETE:
            n_classes = len(inv[a])
            for p in PROBE_POINTS:
                r = mlp_probe(f_fit[p], y_fit[a], f_eval[p], y_eval[a], n_classes, seed=int(cfg.seed))
                rows.append(common | {"point": POINT_LABEL[p], "feature": p, "factor": a, "kind": "discrete"} | r)
                print(f"[info] {POINT_LABEL[p]} {p:12s} {a:6s} acc={r['accuracy']:.4f} "
                      f"info={r['info_bits']:.3f}/{r['entropy_bits']:.3f} bits (oculta {r['hidden']})", flush=True)
            # ── texto decodificado: E4 y T2, con el extractor común ──────────────────────────────
            if has_dec:
                for p in ("text_pred", "text_target"):
                    pred = np.array([inv[a].get(extract(t)[a], -1) for t in texts[p]])
                    r = reader_information(y_eval[a], pred, n_classes)
                    rows.append(common | {"point": POINT_LABEL[p], "feature": p, "factor": a,
                                          "kind": "discrete"} | r)
                    print(f"[info] {POINT_LABEL[p]} {p:12s} {a:6s} acc={r['accuracy']:.4f} "
                          f"info={r['info_bits']:.3f}/{r['entropy_bits']:.3f} bits", flush=True)

        # ── factores continuos: kernel ridge en E1, E2, E3 y T1 ───────────────────────────────────
        for a in CONTINUOUS:
            yf = np.array([float(r["latents_image"][a]) for r in fit_recs])
            ye = np.array([float(r["latents_image"][a]) for r in eval_recs])
            for p in PROBE_POINTS:
                r = kernel_ridge_probe(f_fit[p], yf, f_eval[p], ye, seed=int(cfg.seed))
                rows.append(common | {"point": POINT_LABEL[p], "feature": p, "factor": a, "kind": "continuous"} | r)
                print(f"[info] {POINT_LABEL[p]} {p:12s} {a:17s} R2={r['r2']:.4f}", flush=True)

        with results_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_table(out_dir / "tabla_discretos.csv", rows, y_eval, inv)
        tracker.finish("completed", n_fit=len(fit_recs), n_eval=len(eval_recs), has_decoder=has_dec)
        return 0
    except Exception as e:
        tracker.finish("failed", error=f"{type(e).__name__}: {e}")
        raise


def write_table(path, rows, y_eval, inv) -> None:
    """Tabla factor × punto con exactitud e información (bits), y las pérdidas entre puntos."""
    order = ["E1", "E2", "E3", "E4", "T1", "T2"]
    by = {(r["factor"], r["point"]): r for r in rows if r["kind"] == "discrete"}
    present = [p for p in order if any((a, p) in by for a in DISCRETE)]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["factor", "H_bits"] + [f"acc_{p}" for p in present] + [f"info_{p}" for p in present]
                   + ["perdida_predictor_E1_E3", "brecha_prediccion_T1_E3", "perdida_decoder_E3_E4",
                      "perdida_decoder_intrinseca_T1_T2"])
        for a in DISCRETE:
            h = entropy_bits(y_eval[a], len(inv[a]))
            info = {p: by[(a, p)]["info_bits_clipped"] for p in present if (a, p) in by}

            def diff(p, q):
                return round(info[p] - info[q], 4) if p in info and q in info else ""

            w.writerow([a, round(h, 4)] + [round(by[(a, p)]["accuracy"], 4) if (a, p) in by else "" for p in present]
                       + [round(info[p], 4) if p in info else "" for p in present]
                       + [diff("E1", "E3"), diff("T1", "E3"), diff("E3", "E4"), diff("T1", "T2")])
    print(f"[info] tabla en {path}")


def main(argv=None) -> int:
    return run(compose_info(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
