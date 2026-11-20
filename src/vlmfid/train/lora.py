"""Qué se entrena: LoRA en las lineales del transformer + conector completo (docs/metodologia.md §2.3).

Base congelada y cuantizada exactamente como se cargó en la etapa 1 (NF4, o bf16 en el control D6).
Encima:

  * LoRA en todas las proyecciones lineales de atención y MLP de los bloques ([8] §4). Los patrones
    salen de `describer.lora_target_modules()`, que nunca alcanzan el encoder visual:
      VLM      el `language_model` (168 lineales en LLaVA-OV e InternVL3, 186 en Qwen3.5)
      VL-JEPA  las 8 capas del Predictor (56) y las del transformer del Y-Encoder (EmbeddingGemma)
  * El conector se entrena completo (`modules_to_save` de PEFT: copia entrenable que se guarda con el
    adaptador). VLM: el proyector o el merger. VL-JEPA: `predictor.vis_proj`, `predictor.out_proj` e
    `y_encoder.projector`.

Grupos de LR (`param_groups`):
    lora            LoRA fuera del Y-Encoder                      train.lr
    lora_y_encoder  LoRA del Y-Encoder (solo VL-JEPA)             train.lr × train.y_encoder_lr_mult  ([5] §3.1)
    connector       conector y proyecciones (incl. y_encoder.projector, P9)   train.lr × train.connector_lr_mult

Precisión: los parámetros entrenables se guardan en fp32 (pesos maestros) y el forward corre en autocast
bf16. Una actualización de AdamW típica queda por debajo de la resolución de bf16, así que entrenar el
conector en bf16 lo dejaría prácticamente congelado.
"""

from __future__ import annotations

import torch

CONNECTOR_TAG = ".modules_to_save."
Y_ENCODER_TAG = "y_encoder."


def _enable_layer_checkpointing(layers) -> None:
    """Gradient checkpointing por capa (no reentrante) para módulos que no son PreTrainedModel,
    igual que `openvljepa/scripts/train.py::enable_gradient_checkpointing`."""
    from torch.utils.checkpoint import checkpoint

    for layer in layers:
        orig = layer.forward

        def ckpt_forward(*args, _orig=orig, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        layer.forward = ckpt_forward


def _component(name: str) -> str:
    if Y_ENCODER_TAG in name:
        return "y_encoder"
    if "predictor." in name:
        return "predictor"
    return "llm"


def attach_lora(describer, cfg) -> dict:
    """Envuelve `describer.model` con PEFT (in situ) y devuelve el conteo de entrenables por componente."""
    from peft import LoraConfig, get_peft_model

    base = describer.model
    base.requires_grad_(False)
    jepa = describer.family == "vljepa"
    if jepa:
        if cfg.model.quant == "nf4":
            # PEFT elige su capa LoRA para bnb.Linear4bit solo si el modelo declara esta bandera;
            # VL-JEPA no es un PreTrainedModel y la cuantización se aplicó a mano (quant.swap_linear_to_nf4).
            base.is_loaded_in_4bit = True
        if cfg.train.gradient_checkpointing:
            _enable_layer_checkpointing(base.predictor.layers)
    elif cfg.train.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    connector = list(cfg.lora.connector_modules)
    lcfg = LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=float(cfg.lora.dropout),
        bias=str(cfg.lora.bias), target_modules=describer.lora_target_modules(), modules_to_save=connector,
    )
    model = get_peft_model(base, lcfg)
    for _, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    describer.model = model

    counts = {"trainable_connector": 0}
    layers: dict[str, int] = {}
    unexpected = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if CONNECTOR_TAG in name:
            # PEFT elige los módulos por sufijo del nombre: se comprueba que solo entren los declarados
            if not any(f"{m}{CONNECTOR_TAG}" in name for m in connector):
                unexpected.append(name)
            counts["trainable_connector"] += p.numel()
        elif "lora_" in name:
            comp = _component(name)
            counts[f"trainable_lora_{comp}"] = counts.get(f"trainable_lora_{comp}", 0) + p.numel()
            if name.endswith("lora_A.default.weight"):
                layers[comp] = layers.get(comp, 0) + 1
        else:
            unexpected.append(name)
    # el encoder visual nunca recibe LoRA (VLM: vision_modules; VL-JEPA: X-Encoder, que tampoco puede quedar
    # alcanzado por un sufijo de modules_to_save como `predictor.out_proj`)
    vision = ["x_encoder"] if jepa else list(cfg.model.get("vision_modules") or [])
    unexpected += [n for n, p in model.named_parameters() if p.requires_grad
                   and ("lora_" in n or jepa) and any(f"{v}." in n for v in vision)]
    if unexpected:
        raise RuntimeError(f"Parámetros entrenables inesperados: {sorted(set(unexpected))[:5]}")
    expected_components = ("predictor", "y_encoder") if jepa else ("llm",)
    for comp in expected_components:
        if not layers.get(comp):
            raise RuntimeError(f"LoRA no se aplicó a ninguna capa de '{comp}': revisa lora_target_modules")
    for c in connector:
        if not any(f"{c}{CONNECTOR_TAG}" in n for n, _ in model.named_parameters()):
            raise RuntimeError(f"El módulo '{c}' no quedó entrenable: revisa lora.connector_modules")
    counts["trainable_total"] = sum(v for k, v in counts.items() if k.startswith("trainable_"))
    counts |= {f"lora_layers_{k}": v for k, v in layers.items()}
    counts |= {"params_total_stored": sum(p.numel() for p in model.parameters()), "connector_modules": connector}
    return counts


def param_groups(model, cfg) -> list[dict]:
    t = cfg.train
    lr = float(t.lr)
    groups = {"lora": [], "lora_y_encoder": [], "connector": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if CONNECTOR_TAG in name:
            groups["connector"].append(p)
        elif Y_ENCODER_TAG in name:
            groups["lora_y_encoder"].append(p)
        else:
            groups["lora"].append(p)
    mult = {"lora": 1.0, "lora_y_encoder": float(t.y_encoder_lr_mult), "connector": float(t.connector_lr_mult)}
    return [{"params": ps, "lr": lr * mult[k], "name": k} for k, ps in groups.items() if ps]


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = [n for n, p in params.items() if p.requires_grad and n not in state]
    if missing:
        raise RuntimeError(f"El checkpoint no trae {len(missing)} parámetros entrenables (p. ej. {missing[:3]})")
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v.to(params[n].device, dtype=params[n].dtype))


def set_train_mode(model, frozen_modules: list[str]) -> None:
    """model.train() salvo los módulos congelados (encoder visual, X-Encoder), que quedan en eval."""
    model.train()
    keys = set(frozen_modules)
    for name, m in model.named_modules():
        if name.split(".")[-1] in keys and CONNECTOR_TAG not in name + ".":
            m.eval()
