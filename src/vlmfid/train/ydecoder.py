"""Entrenamiento del Y-Decoder de VL-JEPA (docs/metodologia.md §2.4).

Protocolo:
  * Se entrena APARTE y DESPUÉS de VL-JEPA, que queda congelado: solo se usa su Y-Encoder para obtener
    S_Y = Y-Encoder(caption).
  * SOLO TEXTO: pares ⟨S_Y(caption), caption⟩ con los captions distintos de train. El decodificador
    nunca ve imágenes ni predicciones del Predictor: lo que falle al decodificar Ŝ_Y es atribuible a la
    predicción, no a que el decodificador haya memorizado imágenes.
  * Un decodificador por etapa: uno sobre el Y-Encoder preentrenado (etapa 1) y otro sobre el ajustado
    (etapa 2, con `model.adapter_path`), porque el ajuste cambia el espacio objetivo.
  * Selección: el mejor paso según el TECHO del decodificador en val, es decir, la exactitud por
    atributo (`attr_mean`) de Y-Decoder(S_Y) sobre captions de val. Test no se usa.

runs/<exp_id>/
    config.yaml, env.jsonl, status.json
    train_log.jsonl      pérdida cada `log_every` pasos
    val_log.jsonl        techo en val en cada evaluación
    decoder_best/        adaptador LoRA + proyección de prefijo + config (lo que carga la inferencia)
    val_ceiling.jsonl    captions de val reconstruidos por el mejor paso (para revisar a mano)
    summary.json         mejor paso, techo, parámetros entrenables y tiempos
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from ..config import to_yaml
from ..data.manifest import load_manifest
from ..eval.attributes import score_texts
from ..models import build_describer
from ..models.ydecoder import YDecoder
from ..paths import RUNS
from ..tracking import RunTracker


def default_exp_id(cfg: DictConfig) -> str:
    d = cfg.decoder
    stage = "ft" if cfg.model.get("adapter_path") else "zs"
    return (f"ydec_{cfg.model.name}_{cfg.model.quant}_{stage}_{d.quant}_k{d.prefix_len}_r{d.lora.r}"
            + (f"_s{cfg.seed}" if int(cfg.seed) != 0 else ""))


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_ceiling(decoder: YDecoder, emb: torch.Tensor, records: list[dict]) -> tuple[dict, list[str]]:
    """Techo del decodificador: reconstruye cada caption desde su propio embedding y mide atributos."""
    texts = decoder.decode(emb)
    metrics = score_texts(texts, records)
    metrics["exact_match"] = sum(t == r["caption"] for t, r in zip(texts, records)) / max(len(records), 1)
    return metrics, texts


def train_ydecoder(cfg: DictConfig) -> int:
    d, t = cfg.decoder, cfg.decoder.train
    run_dir = RUNS / cfg.exp_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and (run_dir / "decoder_best").exists():
        print(f"[ydec] {cfg.exp_id}: completo, nada pendiente ({summary_path}).")
        return 0
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(to_yaml(cfg))
    tracker = RunTracker(run_dir, cfg)
    tracker.start()
    try:
        return _run(cfg, d, t, run_dir, summary_path, tracker)
    except Exception as e:
        tracker.finish("failed", error=f"{type(e).__name__}: {e}")
        raise


def _run(cfg, d, t, run_dir: Path, summary_path: Path, tracker: RunTracker) -> int:
    t0 = time.time()
    # ── 1. Embeddings objetivo con el Y-Encoder congelado de VL-JEPA ─────────────────────────────
    train_recs = load_manifest("train", cfg.data.dataset, None, cfg.data.subset_seed, variant=cfg.data.variant)
    val_recs = load_manifest("val", cfg.data.dataset, int(t.val_limit), cfg.data.subset_seed, variant=cfg.data.variant)
    captions = sorted({r["caption"] for r in train_recs})
    jepa = build_describer(cfg.model)
    if jepa.family != "vljepa":
        raise ValueError("El Y-Decoder se entrena sobre VL-JEPA: usa model=vljepa")
    jepa.load()
    train_emb = jepa.encode_texts(captions).cpu()                                  # (N, 1536)
    val_emb = jepa.encode_texts([r["caption"] for r in val_recs]).cpu()
    if train_emb.shape != (len(captions), int(d.embed_dim)):
        raise RuntimeError(f"Forma inesperada de S_Y: {tuple(train_emb.shape)}")
    jepa_info = jepa.info()
    jepa.unload()                                                                  # libera la GPU para el LM
    del jepa
    print(f"[ydec] S_Y listo: {len(captions)} captions distintos de train, {len(val_recs)} de val "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ── 2. Decodificador ────────────────────────────────────────────────────────────────────────
    dec = YDecoder(d).load(trainable=True)
    counts = dec.count_trainable()
    print(f"[ydec] entrenables: {counts}  cuantización del LM: {dec.quant_report}", flush=True)
    tracker.update(model_info={"vljepa": jepa_info, "decoder_quant": dec.quant_report, **counts})
    if str(t.schedule) != "constant":
        raise ValueError("Solo está implementado schedule=constant (el de QLoRA, Ap. B.2)")
    opt = torch.optim.AdamW(dec.trainable_parameters(), lr=float(t.lr), betas=tuple(t.betas),
                            weight_decay=float(t.weight_decay))

    n, bs = len(captions), int(t.batch_size)
    steps_per_epoch = -(-n // bs)
    total = steps_per_epoch * int(t.epochs)
    if cfg.get("smoke_steps") is not None:          # prueba de humo: corta el entrenamiento
        total = min(total, int(cfg.smoke_steps))
    best, step = None, 0
    meta = {"vljepa_name": str(cfg.model.name), "vljepa_quant": str(cfg.model.quant),
            "vljepa_adapter_path": cfg.model.get("adapter_path"), "exp_id": str(cfg.exp_id)}
    print(f"[ydec] {total} pasos ({steps_per_epoch} por época, lote {bs})", flush=True)

    def run_eval(step: int) -> None:
        nonlocal best
        m, texts = evaluate_ceiling(dec, val_emb, val_recs)
        row = {"step": step, **{k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}}
        better = best is None or m["attr_mean"] > best["attr_mean"]
        row["is_best"] = better
        _append(run_dir / "val_log.jsonl", row)
        print(f"[ydec][val] paso {step}: attr_mean={m['attr_mean']:.4f} forma={m['shape']:.3f} x={m['x']:.3f} "
              f"y={m['y']:.3f} color={m['color']:.3f} exacto={m['exact_match']:.3f}"
              f"{'  *mejor*' if better else ''} | {texts[0][:80]!r}", flush=True)
        if better:
            best = {"step": step, **m}
            dec.save(run_dir / "decoder_best", meta | {"best_step": step})
            with (run_dir / "val_ceiling.jsonl").open("w", encoding="utf-8") as f:
                for r, txt in zip(val_recs, texts):
                    f.write(json.dumps({"id": r["id"], "reference": r["caption"], "decoded": txt},
                                       ensure_ascii=False) + "\n")

    run_eval(0)                                           # control: LoRA en cero y prefijo sin entrenar
    dec.lm.train()
    for epoch in range(int(t.epochs)):
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + epoch)
        perm = torch.randperm(n, generator=g).tolist()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            emb = train_emb[idx]
            if float(t.noise_std) > 0:
                emb = emb + float(t.noise_std) * torch.randn(emb.shape, generator=g)
            loss, n_tok = dec.loss(emb, [captions[j] for j in idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Pérdida no finita en el paso {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.trainable_parameters(), float(t.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % int(t.log_every) == 0 or step == total:
                _append(run_dir / "train_log.jsonl", {"step": step, "epoch": epoch, "loss": round(float(loss), 5),
                                                      "tokens": n_tok, "seconds": round(time.time() - t0, 1)})
                print(f"[ydec] paso {step}/{total}  pérdida={float(loss):.4f}", flush=True)
            if step % int(t.eval_every) == 0 or step == total:
                run_eval(step)
                dec.lm.train()
            if step >= total:
                break
        tracker.update(step=step, total_steps=total)
        if step >= total:
            break

    summary = {"exp_id": str(cfg.exp_id), "best": best, "total_steps": total, **counts,
               "n_train_captions": n, "n_val": len(val_recs), "seconds": round(time.time() - t0, 1), "meta": meta}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tracker.finish("completed", best_step=best["step"], best_attr_mean=best["attr_mean"])
    print(f"[ydec] listo: mejor paso {best['step']} con attr_mean={best['attr_mean']:.4f} -> {run_dir / 'decoder_best'}")
    return 0
