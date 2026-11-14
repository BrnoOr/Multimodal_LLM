"""Inspección de un modelo cargado: lo que no se puede afirmar leyendo código (docs/auditoria.md §6).

    uv run --no-sync python scripts/inspect_models.py model=llava_ov
    uv run --no-sync python scripts/inspect_models.py model=qwen35 inspect.n=8
    uv run --no-sync python scripts/inspect_models.py model=vljepa
    uv run --no-sync python scripts/inspect_models.py model=internvl3 model.quant=bf16   # control sin cuantizar

No entrena ni escribe predicciones. Usa unas pocas imágenes de VAL (nunca test). Reporta, por modelo:

  * clase, número de parámetros por componente y `config` resumida;
  * cuantización: capas y fracción de parámetros en NF4, por componente, y capas lineales que quedaron
    SIN cuantizar fuera del conector (para detectar exclusiones no previstas);
  * conector: su estructura completa (¿MLP de 2 capas?);
  * LoRA: módulos que alcanza el regex, agrupados por nombre de hoja, su tipo, y capas lineales del
    transformer de lenguaje que el regex NO alcanza (QLoRA pide todas);
  * tokens visuales por imagen de M3DI y tamaño de las imágenes;
  * la plantilla de chat ya renderizada con el prompt;
  * respuestas del modelo a `inspect.n` imágenes, con los atributos extraídos y su acierto, para
    revisar a mano tanto el modelo como el extractor.
  VL-JEPA: además, tokens de cada prompt del catálogo y de los captions frente a los límites, y la
  forma de cada punto de medición (E1, E2, E3, T1).

Salida: runs/inspect_<model>_<quant>/report.json y lo mismo por pantalla.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import torch
from omegaconf import OmegaConf

from vlmfid.config import compose, load_prompt_catalog, to_yaml
from vlmfid.data.manifest import ImageRecordDataset, data_root, load_manifest
from vlmfid.eval.attributes import ATTRIBUTES, extract, score, truth_from_record
from vlmfid.models import build_describer
from vlmfid.paths import RUNS
from vlmfid.prompts import render

INSPECT_DEFAULTS = {"n": 4, "depth": 2}


def _is_linear(m) -> bool:
    return isinstance(m, torch.nn.Linear)          # bnb.nn.Linear4bit hereda de nn.Linear


def _is_quantized(m) -> bool:
    return type(m).__name__ in ("Linear4bit", "Linear8bitLt")


def params_by_component(model, depth: int) -> dict[str, int]:
    """Parámetros agrupados por los primeros `depth` niveles del nombre (los pesos NF4 van empaquetados,
    así que este conteo subestima los componentes cuantizados; ver `quant` para el tamaño real)."""
    out: Counter = Counter()
    for name, p in model.named_parameters():
        out[".".join(name.split(".")[:depth])] += p.numel()
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def linear_summary(model, connector_keys: list[str]) -> dict:
    """Capas lineales cuantizadas y sin cuantizar, agrupadas por componente."""
    quant, plain = Counter(), Counter()
    plain_examples: dict[str, str] = {}
    for name, m in model.named_modules():
        if not _is_linear(m):
            continue
        comp = ".".join(name.split(".")[:2])
        if _is_quantized(m):
            quant[comp] += 1
        else:
            plain[comp] += 1
            plain_examples.setdefault(comp, name)
    unexpected = {c: n for c, n in plain.items()
                  if not any(k in plain_examples[c] for k in connector_keys) and "lm_head" not in plain_examples[c]}
    return {"quantized_linears_by_component": dict(quant), "plain_linears_by_component": dict(plain),
            "plain_linear_example": plain_examples, "plain_outside_connector_and_head": unexpected}


def lora_coverage(model, pattern, scope_key: str) -> dict:
    """Qué alcanza el regex de LoRA y qué capas lineales del ámbito `scope_key` deja fuera."""
    rx = re.compile(pattern) if isinstance(pattern, str) else None
    names = set(pattern) if rx is None else None

    def hit(name: str) -> bool:
        return bool(rx.fullmatch(name)) if rx is not None else name.split(".")[-1] in names

    matched, matched_types, missed, outside = Counter(), Counter(), Counter(), []
    for name, m in model.named_modules():
        leaf = name.split(".")[-1]
        if hit(name):
            matched[leaf] += 1
            matched_types[type(m).__name__] += 1
            if scope_key not in name:
                outside.append(name)
        elif _is_linear(m) and scope_key in name:
            missed[leaf] += 1
    return {"pattern": pattern if isinstance(pattern, str) else sorted(names), "scope": scope_key,
            "matched_by_leaf": dict(matched), "matched_module_types": dict(matched_types),
            "n_matched": sum(matched.values()), "linear_in_scope_not_matched": dict(missed),
            "matched_outside_scope": outside[:10]}


def connector_structure(model, keys: list[str]) -> dict[str, str]:
    out = {}
    for name, m in model.named_modules():
        if any(name.endswith(k) for k in keys):
            out[name] = repr(m)
    return out


def config_summary(model) -> dict:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    d = cfg.to_dict()
    keep = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
            "image_token_id", "image_token_index", "image_seq_length", "vision_feature_layer",
            "vision_feature_select_strategy", "image_grid_pinpoints", "downsample_ratio", "tie_word_embeddings")
    out = {k: d[k] for k in keep if k in d}
    for sub in ("vision_config", "text_config"):
        if isinstance(d.get(sub), dict):
            out[sub] = {k: v for k, v in d[sub].items()
                        if k in keep + ("image_size", "patch_size", "layer_types", "spatial_merge_size",
                                        "intermediate_size", "out_hidden_size", "depth", "num_heads")}
    return out


def _image_token_id(model) -> int | None:
    cfg = getattr(model, "config", None)
    for k in ("image_token_id", "image_token_index"):
        v = getattr(cfg, k, None)
        if v is not None:
            return int(v)
    return None


def inspect_hf(d, cfg, recs, imgs, prompt_text: str) -> dict:
    model = d.model
    connector = list(cfg.model.get("connector_modules") or [])
    rep: dict = {"class": type(model).__name__, "config": config_summary(model),
                 "params_by_component": params_by_component(model, int(cfg.inspect.depth)),
                 "quant": d.quant_report, "linears": linear_summary(model, connector),
                 "connector": connector_structure(model, connector),
                 "lora": lora_coverage(model, d.lora_target_modules(), "language_model"),
                 "chat_template_origin": d.chat_template_origin}

    text = d.processor.apply_chat_template(d.build_messages(prompt_text), add_generation_prompt=True,
                                           tokenize=False, **d.chat_template_kwargs())
    rep["rendered_prompt"] = text
    inputs = d.processor(text=[text] * len(imgs), images=imgs, padding=True, return_tensors="pt")
    tid = _image_token_id(model)
    rep["image_token_id"] = tid
    if tid is not None:
        rep["visual_tokens_per_image"] = (inputs["input_ids"] == tid).sum(dim=1).tolist()
    rep["input_tokens_per_image"] = inputs["attention_mask"].sum(dim=1).tolist()
    rep["processor_tensors"] = {k: list(v.shape) for k, v in inputs.items() if torch.is_tensor(v)}
    gc = getattr(model, "generation_config", None)
    rep["eos_token_id"] = getattr(gc, "eos_token_id", None)
    return rep


def inspect_vljepa(d, cfg, recs, imgs, prompt_text: str, train_recs) -> dict:
    inner = d._inner()
    rep: dict = {"class": type(inner).__name__, "arch": d.arch,
                 "params_by_component": params_by_component(inner, int(cfg.inspect.depth)),
                 "quant": d.quant_report, "linears": linear_summary(inner, ["vis_proj", "out_proj", "projector"]),
                 "connector": connector_structure(inner, ["vis_proj", "out_proj", "y_encoder.projector"]),
                 "lora_predictor": lora_coverage(inner, d.lora_target_modules(), "predictor.layers"),
                 "y_encoder_linear_leafs": dict(Counter(
                     n.split(".")[-1] for n, m in inner.y_encoder.model.named_modules() if _is_linear(m))),
                 "limits": {"max_query_len": d.max_query_len, "max_caption_len": d.max_caption_len,
                            "num_frames": d.num_frames}}
    catalog = load_prompt_catalog()
    q_len = {k: d.token_lengths([render(catalog[k])], "query")[0] for k in catalog}
    rep["query_tokens_by_prompt"] = q_len
    rep["prompts_over_limit"] = [k for k, n in q_len.items() if n > d.max_query_len]
    caps = sorted({r["caption"] for r in train_recs})
    c_len = d.token_lengths(caps, "target")
    rep["caption_tokens"] = {"n_captions": len(caps), "max": max(c_len), "over_limit": sum(n > d.max_caption_len for n in c_len)}
    if prompt_text and q_len.get(cfg.prompt.id, 0) <= d.max_query_len:
        st = d.predict_stages(imgs, [prompt_text] * len(imgs))
        rep["stages"] = {k: list(v.shape) for k, v in st.items()}
        s_y = d.encode_texts([r["caption"] for r in recs])
        rep["stages"]["s_y"] = list(s_y.shape)
        cos = torch.nn.functional.cosine_similarity(st["pred"], s_y, dim=-1)
        rep["cos_pred_vs_target"] = [round(float(x), 4) for x in cos]
    return rep


def main(argv=None) -> int:
    cfg = compose(sys.argv[1:] if argv is None else argv)
    cfg.inspect = OmegaConf.merge(OmegaConf.create(INSPECT_DEFAULTS), cfg.get("inspect") or {})
    if not any(a.startswith("prompt=") for a in (sys.argv[1:] if argv is None else argv)):
        cfg.prompt = load_prompt_catalog()["p1_attributes"]
    print(to_yaml(cfg))

    n = int(cfg.inspect.n)
    recs = load_manifest("val", cfg.data.dataset, n, cfg.data.subset_seed, variant=cfg.data.variant)
    ds = ImageRecordDataset(recs, data_root(cfg.data.variant, cfg.data.get("root")))
    recs, imgs = ImageRecordDataset.collate([ds[i] for i in range(len(ds))])
    prompt_text = render(cfg.prompt)

    d = build_describer(cfg.model)
    d.load()
    report = {"model": cfg.model.name, "hf_id": cfg.model.get("hf_id"), "quant": cfg.model.quant,
              "prompt_id": cfg.prompt.id, "image_sizes": [list(im.size) for im in imgs], "info": d.info()}
    if d.family == "vljepa":
        train_recs = load_manifest("train", cfg.data.dataset, None, cfg.data.subset_seed, variant=cfg.data.variant)
        report |= inspect_vljepa(d, cfg, recs, imgs, prompt_text, train_recs)
    else:
        report |= inspect_hf(d, cfg, recs, imgs, prompt_text)
        outs = d.describe(imgs, [prompt_text] * len(imgs))
        json_mode = cfg.prompt.get("response_format") == "json"
        samples = []
        for r, o in zip(recs, outs):
            ext, truth = extract(o.text, json_mode), truth_from_record(r)
            samples.append({"id": r["id"], "reference": r["caption"], "prediction": o.text,
                            "truth": truth, "extracted": {a: ext[a] for a in ATTRIBUTES},
                            "correct": {a: bool(v) for a, v in score(ext, truth).items() if a in ATTRIBUTES},
                            "extra": o.extra})
        report["samples"] = samples

    out_dir = RUNS / f"inspect_{cfg.model.name}_{cfg.model.quant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    (out_dir / "report.json").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[inspect] reporte en {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
