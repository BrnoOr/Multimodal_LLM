"""Entrenamiento del Y-Decoder de VL-JEPA (docs/metodologia.md §2.4).

Protocolo:
  * Se entrena APARTE y DESPUÉS de VL-JEPA, que queda congelado: solo se usa su Y-Encoder para obtener
    S_Y = Y-Encoder(caption).
  * SOLO TEXTO: pares ⟨S_Y(caption), caption⟩ con los captions distintos de train. El decodificador
    nunca ve imágenes ni predicciones del Predictor: lo que falle al decodificar Ŝ_Y es atribuible a la
    predicción, no a que el decodificador haya memorizado imágenes.
  * Fuente de los captions (`decoder.text_source`, decisión del usuario del 16-11-2026):
      msrvtt  decodificador GENÉRICO de la etapa 1: captions de MSR-VTT (train_7k), sin M3DI, para que
              VL-JEPA llegue a la etapa 1 sin conocer el formato del dataset, igual que los VLM.
              Mejor paso por CIDEr de la reconstrucción en MSR-VTT test_1k (vlmfid.data.msrvtt). El techo
              en M3DI se mide en cada evaluación, solo para reportarlo.
      m3di    decodificador AJUSTADO a M3DI (captions de train): etapa 2 y condición intermedia. Mejor
              paso por el techo en val de M3DI (`attr_mean` de Y-Decoder(S_Y)).
  * `decoder.init_from`: continúa desde un decodificador ya entrenado (LoRA y prefijo), p. ej. el genérico
    ajustado después con M3DI. Debe tener la misma arquitectura.
  * Un decodificador por Y-Encoder: el preentrenado (etapa 1) y el ajustado (etapa 2, con
    `model.adapter_path`), porque el ajuste cambia el espacio objetivo. Test de M3DI no se usa.

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
from omegaconf import DictConfig, OmegaConf

from ..config import to_yaml
from ..data.manifest import load_manifest
from ..data.msrvtt import load_msrvtt
from ..eval.attributes import score_texts
from ..eval.text_metrics import text_metrics
from ..models import build_describer
from ..models.ydecoder import CONFIG_FILE, YDecoder
from ..paths import RUNS, resolve
from ..tracking import RunTracker

SOURCES = ("m3di", "msrvtt")
SELECTION = {"m3di": "attr_mean", "msrvtt": "cider"}       # métrica que elige el mejor paso
# lo que debe coincidir entre un decodificador y aquel desde el que continúa (`init_from`)
ARCH_KEYS = ("llama_name", "quant", "embed_dim", "prefix_len", "normalize_input", "lora.r", "lora.alpha")


def text_source(cfg: DictConfig) -> str:
    src = str(cfg.decoder.get("text_source") or "m3di")
    if src not in SOURCES:
        raise ValueError(f"decoder.text_source={src!r}; opciones: {SOURCES}")
    return src


def epochs_for(cfg: DictConfig) -> int:
    d = cfg.decoder
    return int(d.msrvtt.epochs) if text_source(cfg) == "msrvtt" else int(d.train.epochs)


def eval_every_for(cfg: DictConfig) -> int:
    d = cfg.decoder
    return int(d.msrvtt.eval_every) if text_source(cfg) == "msrvtt" else int(d.train.eval_every)


def default_exp_id(cfg: DictConfig) -> str:
    d = cfg.decoder
    stage = "ft" if cfg.model.get("adapter_path") else "zs"
    src = text_source(cfg)
    # Las épocas van en el nombre: sin el sufijo, un run con otras épocas se confundiría con uno ya completo
    # y no se entrenaría. La fuente solo se marca si no es M3DI (los nombres anteriores no la llevan).
    return (f"ydec_{cfg.model.name}_{cfg.model.quant}_{stage}_{d.quant}" + ("" if src == "m3di" else f"_{src}")
            + f"_k{d.prefix_len}_r{d.lora.r}_e{epochs_for(cfg)}" + ("_cont" if d.get("init_from") else "")
            + (f"_s{cfg.seed}" if int(cfg.seed) != 0 else ""))


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_init_compatible(d: DictConfig, init_dir: Path) -> None:
    """El decodificador de partida debe tener la misma arquitectura que el que se va a entrenar."""
    if not (init_dir / CONFIG_FILE).exists():
        raise FileNotFoundError(f"decoder.init_from={init_dir}: no hay {CONFIG_FILE} (¿es un decoder_best?)")
    prev = OmegaConf.load(init_dir / CONFIG_FILE)
    diff = {k: (OmegaConf.select(prev, k), OmegaConf.select(d, k)) for k in ARCH_KEYS
            if OmegaConf.select(prev, k) != OmegaConf.select(d, k)}
    if diff:
        raise ValueError(f"decoder.init_from={init_dir} tiene otra arquitectura (antes, ahora): {diff}")


def evaluate_ceiling(decoder: YDecoder, emb: torch.Tensor, records: list[dict]) -> tuple[dict, list[str]]:
    """Techo en M3DI: reconstruye cada caption desde su propio embedding y mide atributos."""
    texts = decoder.decode(emb)
    metrics = score_texts(texts, records)
    metrics["exact_match"] = sum(t == r["caption"] for t, r in zip(texts, records)) / max(len(records), 1)
    return metrics, texts


def evaluate_text(decoder: YDecoder, emb: torch.Tensor, records: list[dict]) -> tuple[dict, list[str]]:
    """Reconstrucción en un corpus sin atributos (MSR-VTT): métricas de texto contra el caption original."""
    texts = decoder.decode(emb)
    metrics = text_metrics(texts, [r["caption"] for r in records])
    metrics["exact_match"] = sum(t == r["caption"] for t, r in zip(texts, records)) / max(len(records), 1)
    return metrics, texts


def _load_texts(cfg: DictConfig, src: str) -> tuple[list[str], list[dict], list[dict] | None]:
    """(captions de entrenamiento, registros para elegir el paso, registros de M3DI solo para reportar)."""
    d, t = cfg.decoder, cfg.decoder.train
    m3di_val = lambda limit: load_manifest("val", cfg.data.dataset, limit, cfg.data.subset_seed,  # noqa: E731
                                           variant=cfg.data.variant)
    if src == "m3di":
        train_recs = load_manifest("train", cfg.data.dataset, None, cfg.data.subset_seed, variant=cfg.data.variant)
        return sorted({r["caption"] for r in train_recs}), m3di_val(int(t.val_limit)), None
    m = d.msrvtt
    train = load_msrvtt("train_7k", repo=str(m.repo), filename=str(m.train_file))
    val = load_msrvtt("test_1k", repo=str(m.repo), filename=str(m.val_file))
    if m.get("val_limit") is not None:
        val = val[:int(m.val_limit)]
    val_recs = [{"id": f"msrvtt/{v}/{i}", "caption": c} for i, (v, c) in enumerate(val)]
    return sorted({c for _, c in train}), val_recs, m3di_val(int(m.m3di_report_limit))


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
    src = text_source(cfg)
    sel = SELECTION[src]
    init_dir = resolve(d.init_from) if d.get("init_from") else None
    if init_dir is not None:
        check_init_compatible(d, init_dir)
    # ── 1. Embeddings objetivo con el Y-Encoder congelado de VL-JEPA ─────────────────────────────
    captions, val_recs, report_recs = _load_texts(cfg, src)
    jepa = build_describer(cfg.model)
    if jepa.family != "vljepa":
        raise ValueError("El Y-Decoder se entrena sobre VL-JEPA: usa model=vljepa")
    jepa.load()
    train_emb = jepa.encode_texts(captions).cpu()                                  # (N, 1536)
    val_emb = jepa.encode_texts([r["caption"] for r in val_recs]).cpu()
    report_emb = jepa.encode_texts([r["caption"] for r in report_recs]).cpu() if report_recs else None
    if train_emb.shape != (len(captions), int(d.embed_dim)):
        raise RuntimeError(f"Forma inesperada de S_Y: {tuple(train_emb.shape)}")
    jepa_info = jepa.info()
    jepa.unload()                                                                  # libera la GPU para el LM
    del jepa
    print(f"[ydec] fuente={src}  selección={sel}  S_Y listo: {len(captions)} captions distintos de train, "
          f"{len(val_recs)} para elegir el paso"
          + (f", {len(report_recs)} de val de M3DI para reportar" if report_recs else "")
          + f" ({time.time() - t0:.0f} s)", flush=True)

    # ── 2. Decodificador ────────────────────────────────────────────────────────────────────────
    dec = YDecoder(d).load(trainable=True, init_dir=init_dir)
    counts = dec.count_trainable()
    print(f"[ydec] entrenables: {counts}  cuantización del LM: {dec.quant_report}", flush=True)
    tracker.update(model_info={"vljepa": jepa_info, "decoder_quant": dec.quant_report, **counts})
    if str(t.schedule) != "constant":
        raise ValueError("Solo está implementado schedule=constant (el de QLoRA, Ap. B.2)")
    opt = torch.optim.AdamW(dec.trainable_parameters(), lr=float(t.lr), betas=tuple(t.betas),
                            weight_decay=float(t.weight_decay))

    n, bs = len(captions), int(t.batch_size)
    epochs, eval_every = epochs_for(cfg), eval_every_for(cfg)
    steps_per_epoch = -(-n // bs)
    total = steps_per_epoch * epochs
    if cfg.get("smoke_steps") is not None:          # prueba de humo: corta el entrenamiento
        total = min(total, int(cfg.smoke_steps))
    best, step = None, 0
    meta = {"vljepa_name": str(cfg.model.name), "vljepa_quant": str(cfg.model.quant),
            "vljepa_adapter_path": cfg.model.get("adapter_path"), "exp_id": str(cfg.exp_id),
            "text_source": src, "selection": sel, "init_from": str(init_dir) if init_dir else None}
    print(f"[ydec] {total} pasos ({steps_per_epoch} por época, lote {bs})", flush=True)

    def run_eval(step: int) -> None:
        nonlocal best
        if src == "m3di":
            m, texts = evaluate_ceiling(dec, val_emb, val_recs)
            rep_texts = None
        else:
            m, texts = evaluate_text(dec, val_emb, val_recs)
            rm, rep_texts = evaluate_ceiling(dec, report_emb, report_recs)        # solo se reporta
            m |= {f"m3di_{k}": v for k, v in rm.items()}
        row = {"step": step, **{k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}}
        better = best is None or m[sel] > best[sel]
        row["is_best"] = better
        _append(run_dir / "val_log.jsonl", row)
        if src == "m3di":
            detail = (f"attr_mean={m['attr_mean']:.4f} forma={m['shape']:.3f} x={m['x']:.3f} y={m['y']:.3f} "
                      f"color={m['color']:.3f}")
        else:
            detail = (f"cider={m['cider']:.4f} bleu4={m['bleu4']:.4f} rouge_l={m['rouge_l']:.4f} | M3DI (solo "
                      f"reporte): attr_mean={m['m3di_attr_mean']:.4f} forma={m['m3di_shape']:.3f} "
                      f"x={m['m3di_x']:.3f} y={m['m3di_y']:.3f} color={m['m3di_color']:.3f}")
        print(f"[ydec][val] paso {step}: {detail} exacto={m['exact_match']:.3f}"
              f"{'  *mejor*' if better else ''} | {texts[0][:80]!r}", flush=True)
        if better:
            best = {"step": step, **m}
            dec.save(run_dir / "decoder_best", meta | {"best_step": step})
            with (run_dir / "val_ceiling.jsonl").open("w", encoding="utf-8") as f:
                for r, txt in zip(val_recs, texts):
                    f.write(json.dumps({"id": r["id"], "reference": r["caption"], "decoded": txt},
                                       ensure_ascii=False) + "\n")
            if rep_texts is not None:
                with (run_dir / "m3di_ceiling.jsonl").open("w", encoding="utf-8") as f:
                    for r, txt in zip(report_recs, rep_texts):
                        f.write(json.dumps({"id": r["id"], "reference": r["caption"], "decoded": txt},
                                           ensure_ascii=False) + "\n")

    run_eval(0)                                           # control: LoRA en cero y prefijo sin entrenar
    dec.lm.train()
    for epoch in range(epochs):
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
            loss = loss.detach()                          # solo se usa para registrar
            torch.nn.utils.clip_grad_norm_(dec.trainable_parameters(), float(t.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % int(t.log_every) == 0 or step == total:
                _append(run_dir / "train_log.jsonl", {"step": step, "epoch": epoch, "loss": round(float(loss), 5),
                                                      "tokens": n_tok, "seconds": round(time.time() - t0, 1)})
                print(f"[ydec] paso {step}/{total}  pérdida={float(loss):.4f}", flush=True)
            if step % eval_every == 0 or step == total:
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
    tracker.finish("completed", best_step=best["step"], selection=sel, best_value=best[sel])
    print(f"[ydec] listo: mejor paso {best['step']} con {sel}={best[sel]:.4f} -> {run_dir / 'decoder_best'}")
    return 0
