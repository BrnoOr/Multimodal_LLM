"""Verificación de la cuantización del Y-Encoder de VL-JEPA (decisión abierta YQ, docs/metodologia.md §6.2).

Contexto (docs/resultados_etapa1.md §7.4): con el Y-Encoder en NF4, S_Y queda con coseno 0.876 contra bf16,
mientras que X-Encoder y Predictor en NF4 quedan ≥ 0.99. La cuantización de VL-JEPA es manual
(`vlmfid.models.quant.swap_linear_to_nf4`), porque OpenVLJEPA no es un PreTrainedModel. Este script
separa dos explicaciones:

    (1) error de nuestra cuantización manual      -> la carga ESTÁNDAR de transformers da ≈ 1
    (2) sensibilidad real de EmbeddingGemma a NF4  -> la carga estándar también da ≈ 0.88

Mismos pesos en todas las condiciones: el transformer del Y-Encoder del CHECKPOINT (no el de
google/embeddinggemma-300m, que difiere) y su `projector` en bf16. Mismos captions de VAL.

Condiciones (todas contra la referencia bf16):
    fp32           el mismo transformer en float32: cuánto mueve ya la precisión bf16 (calibración)
    nf4_manual     swap_linear_to_nf4 (lo que usa el proyecto); debe reproducir el ≈ 0.876 del diagnóstico
    nf4_hf         from_pretrained(..., quantization_config=BitsAndBytesConfig NF4 + doble cuantización)
    nf4_hf_nodq    ídem sin doble cuantización (¿influye la doble cuantización?)
    nf4_skip_last  NF4 manual salvo el último bloque del transformer (en bf16): ¿el error está ahí?

Métricas por condición:
    cos_s_y        coseno por caption entre S_Y de la condición y S_Y bf16 (media, p10, mínimo)
    per_layer      coseno medio por capa del estado oculto promediado sobre tokens reales (emb + 24 capas):
                   dónde aparece y crece el error
    structure      Spearman entre las matrices de similitud de captions distintos (¿se conserva el orden
                   relativo de distancias aunque los vectores se desplacen?)
    self_recall@1  para cada S_Y de la condición, ¿su vecino más cercano entre los S_Y bf16 es su caption?
    pairwise_mean  similitud media entre captions distintos (qué tan juntos quedan los S_Y)
Además, en bf16: máximo |h| por capa (activaciones atípicas que dañen la cuantización por bloques).

    uv run --no-sync python scripts/diag_yencoder_quant.py model=vljepa
    # humo
    uv run --no-sync python scripts/diag_yencoder_quant.py model=vljepa diag.n=64 diag.tag=humo

Salida: runs/diagyq_<model>[_<tag>]/metrics.json (+ config.yaml, status.json). El transformer se guarda
temporalmente en bf16 (≈ 0.6 GB) dentro de la carpeta del run para cargarlo por la vía estándar, y se borra
al terminar. Solo VAL; no entrena nada.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from vlmfid.config import compose, to_yaml
from vlmfid.data.manifest import load_manifest
from vlmfid.eval.embedding_metrics import caption_codes, paired_cosine, retrieval_recall, structure_spearman
from vlmfid.models import build_describer
from vlmfid.models.quant import hf_quantization_config, quantization_report, swap_linear_to_nf4
from vlmfid.paths import RUNS
from vlmfid.tracking import RunTracker

DIAG_DEFAULTS = {"n": 1024, "batch_size": 64, "tag": None}


def compose_diag(argv: list[str]):
    cfg = compose([*argv, "model.quant=bf16"] if not any(a.startswith("model.quant=") for a in argv) else argv)
    cfg.diag = OmegaConf.merge(OmegaConf.create(DIAG_DEFAULTS), cfg.get("diag") or {})
    if cfg.model.family != "vljepa":
        raise ValueError("scripts/diag_yencoder_quant.py es para VL-JEPA (model=vljepa)")
    if cfg.model.quant != "bf16":
        raise ValueError("Se carga VL-JEPA en bf16 (referencia); las condiciones NF4 se construyen aquí")
    if cfg.model.get("decoder_path") or cfg.model.get("adapter_path"):
        raise ValueError("La verificación usa el VL-JEPA preentrenado, sin Y-Decoder ni adaptador")
    if str(cfg.data.split) != "val":
        raise ValueError("La verificación usa solo VAL")
    tag = f"_{cfg.diag.tag}" if cfg.diag.tag else ""
    if not any(a.startswith("exp_id=") for a in argv):
        cfg.exp_id = f"diagyq_{cfg.model.name}{tag}"
    cfg.stage = "diagyq"
    return cfg


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Promedio sobre tokens reales, igual que `openvljepa.models.y_encoder.YEncoder.forward`."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


@torch.inference_mode()
def encode(base, projector, batches, autocast_dtype) -> dict[str, np.ndarray]:
    """S_Y y estados ocultos promediados por capa. `base` = transformer del Y-Encoder."""
    s_y, layers, max_abs = [], None, None
    dev = next(projector.parameters()).device
    for ids, mask in batches:
        ctx = (torch.autocast(dev.type, dtype=autocast_dtype) if autocast_dtype is not None
               else torch.autocast(dev.type, enabled=False))
        with ctx:
            out = base(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = out.hidden_states                                   # (emb, capa 1, …, capa L)
            pooled = [masked_mean(h, mask).float() for h in hs]
            e = projector(masked_mean(out.last_hidden_state, mask).to(projector.weight.dtype))
        s_y.append(e.float().cpu())
        if layers is None:
            layers, max_abs = [[] for _ in hs], [0.0 for _ in hs]
        for i, (h, p) in enumerate(zip(hs, pooled)):
            layers[i].append(p.cpu())
            real = h[mask.bool()]
            max_abs[i] = max(max_abs[i], float(real.abs().max()))
    out = {"s_y": torch.cat(s_y).numpy()}
    for i, chunks in enumerate(layers):
        out[f"h{i:02d}"] = torch.cat(chunks).numpy()
    out["max_abs"] = np.array(max_abs)
    for k, v in out.items():
        if k != "max_abs" and not np.isfinite(v).all():
            raise RuntimeError(f"{k}: valores no finitos")
    return out


def summarize(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()), "p10": float(np.percentile(x, 10)), "min": float(x.min())}


def compare(cond: dict, ref: dict, captions: list[str]) -> dict:
    n_layers = sum(1 for k in ref if k.startswith("h"))
    cos = paired_cosine(cond["s_y"], ref["s_y"])
    rec = retrieval_recall(cond["s_y"], ref["s_y"], captions, ks=(1,))
    s = torch.nn.functional.normalize(torch.as_tensor(cond["s_y"]), dim=-1)
    codes = caption_codes(captions)
    diff = torch.as_tensor(codes[:, None] != codes[None, :])
    return {"cos_s_y": summarize(cos),
            "per_layer_cos": [float(paired_cosine(cond[f"h{i:02d}"], ref[f"h{i:02d}"]).mean()) for i in range(n_layers)],
            "structure_spearman": structure_spearman(cond["s_y"], ref["s_y"], captions),
            "self_recall@1": rec["recall@1_p2t"], "chance@1": rec["chance@1"],
            "pairwise_mean": float((s @ s.T)[diff].mean())}


def run(cfg) -> int:
    g = cfg.diag
    out_dir = RUNS / cfg.exp_id
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        print(f"[diagyq] {cfg.exp_id}: completo ({metrics_path}).")
        return 0
    torch.manual_seed(int(cfg.seed))
    print(to_yaml(cfg))
    tracker = RunTracker(out_dir, cfg)
    tracker.start()
    try:
        recs = load_manifest("val", cfg.data.dataset, int(g.n), cfg.data.subset_seed, variant=cfg.data.variant)
        captions = [r["caption"] for r in recs]
        d = build_describer(cfg.model)
        d.load()
        yenc = d._inner().y_encoder
        base_bf16, projector = yenc.model.eval(), yenc.projector.eval()
        dev = projector.weight.device
        d._check_fits(captions, "target")
        bs = int(g.batch_size)
        batches = [d._tok(d.target_tok, captions[i:i + bs], d.max_caption_len) for i in range(0, len(captions), bs)]

        # Referencia bf16, con el mismo autocast que VLJEPADescriber.encode_texts; debe coincidir con él.
        t0 = time.time()
        ref = encode(base_bf16, projector, batches, torch.bfloat16)
        k = min(len(captions), 64)
        check = paired_cosine(ref["s_y"][:k], d.encode_texts(captions[:k]).cpu().numpy())
        if check.min() < 0.9999:
            raise RuntimeError(f"La reimplementación del forward no reproduce encode_texts (cos mín {check.min():.5f})")
        print(f"[diagyq] referencia bf16 lista ({time.time() - t0:.0f} s); coincide con encode_texts", flush=True)

        results = {"exp_id": cfg.exp_id, "n": len(captions), "n_distinct_captions": len(set(captions)),
                   "max_caption_len": int(d.max_caption_len),
                   "bf16_max_abs_per_layer": [float(x) for x in ref["max_abs"]], "conditions": {}}

        def add(name: str, cond: dict, base) -> None:
            r = compare(cond, ref, captions)
            r["quant_report"] = quantization_report(base)
            results["conditions"][name] = r
            print(f"[diagyq] {name:<12} cos S_Y {r['cos_s_y']['mean']:.4f} (p10 {r['cos_s_y']['p10']:.4f}) | "
                  f"estructura {r['structure_spearman']:.4f} | self R@1 {r['self_recall@1']:.3f} | "
                  f"lineales 4 bits {r['quant_report']['quantized_linears']}", flush=True)

        # fp32: calibración de cuánto aleja ya bf16
        base_fp32 = copy.deepcopy(base_bf16).float()
        proj_fp32 = copy.deepcopy(projector).float()
        add("fp32", encode(base_fp32, proj_fp32, batches, None), base_fp32)
        del base_fp32, proj_fp32

        # NF4 manual (el del proyecto)
        base_man = copy.deepcopy(base_bf16)
        n_swapped = swap_linear_to_nf4(base_man, dev)
        print(f"[diagyq] NF4 manual: {n_swapped} lineales reemplazadas", flush=True)
        add("nf4_manual", encode(base_man, projector, batches, torch.bfloat16), base_man)
        del base_man

        # NF4 manual salvo el ÚLTIMO bloque (queda en bf16). El humo del 20-11 mostró que el error aparece
        # casi entero en la última posición (coseno por capa ≥ 0.988 hasta la capa 23 y 0.862 al final),
        # donde además están las activaciones más grandes. ¿Basta con dejar ese bloque sin cuantizar?
        base_skip = copy.deepcopy(base_bf16)
        blocks = base_skip.layers
        n_skip = sum(swap_linear_to_nf4(b, dev) for b in list(blocks)[:-1])
        print(f"[diagyq] NF4 salvo el último bloque: {n_skip} lineales reemplazadas "
              f"({len(blocks) - 1} de {len(blocks)} bloques)", flush=True)
        add("nf4_skip_last", encode(base_skip, projector, batches, torch.bfloat16), base_skip)
        del base_skip

        # NF4 por la vía estándar de transformers, con y sin doble cuantización
        from transformers import AutoModel, BitsAndBytesConfig

        # nf4_hf: exactamente la config común del proyecto (vlmfid.models.quant, la de los VLM).
        # nf4_hf_nodq: la misma sin doble cuantización.
        configs = {"nf4_hf": hf_quantization_config("nf4"),
                   "nf4_hf_nodq": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_use_double_quant=False,
                                                     bnb_4bit_compute_dtype=torch.bfloat16)}
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_dir, prefix="_tmp_hf_") as tmp:
            base_bf16.save_pretrained(tmp)
            for name, qc in configs.items():
                base_hf = AutoModel.from_pretrained(tmp, quantization_config=qc, dtype=torch.bfloat16,
                                                    device_map={"": str(dev)}).eval()
                add(name, encode(base_hf, projector, batches, torch.bfloat16), base_hf)
                del base_hf
                torch.cuda.empty_cache()

        text = json.dumps(results, indent=2, ensure_ascii=False)
        tmp_path = metrics_path.with_suffix(".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(metrics_path)

        print("\n[diagyq] coseno por capa contra bf16 (emb, capa 1 … capa L):")
        for name, r in results["conditions"].items():
            print(f"  {name:<12} " + " ".join(f"{x:.3f}" for x in r["per_layer_cos"]))
        print("  max|h| bf16  " + " ".join(f"{x:.0f}" for x in results["bf16_max_abs_per_layer"]))
        tracker.finish("completed", n=len(captions))
        return 0
    except BaseException as e:
        tracker.finish("interrupted" if isinstance(e, KeyboardInterrupt) else "failed", error=repr(e))
        raise


def main(argv=None) -> int:
    return run(compose_diag(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
