"""Qué se entrena: LoRA en el transformer de lenguaje / predictor + conector multimodal completo.

Base congelada y cuantizada (NF4, cómputo bf16), exactamente como se cargó en la etapa 1. Encima:

  * LoRA en las proyecciones lineales de atención y MLP del transformer que produce el texto
    (VLM: `language_model`; VL-JEPA: las 8 capas Llama del predictor). Los patrones son los que ya
    exponía cada adaptador con `lora_target_modules()`, de modo que el encoder visual nunca recibe LoRA.
  * El conector multimodal se entrena completo (`modules_to_save` de PEFT): una copia entrenable que se
    guarda junto al adaptador. VLM: el proyector (LLaVA-OV, InternVL3) o el merger (Qwen3.5);
    VL-JEPA: `predictor.vis_proj`.

Precisión: los parámetros entrenables se guardan en fp32 (pesos maestros) y el forward corre en
autocast bf16. Con AdamW y lr 2e-5, una actualización típica queda por debajo de la resolución de bf16
(≈3 cifras significativas): entrenar el conector en bf16 lo dejaría prácticamente congelado.
"""

from __future__ import annotations

import torch

CONNECTOR_TAG = ".modules_to_save."


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


def attach_lora(describer, cfg) -> dict:
    """Envuelve `describer.model` con PEFT (in situ) y devuelve un resumen de parámetros."""
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

    connector = list(cfg.lora.get("connector_modules") or [])
    lcfg = LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=float(cfg.lora.dropout),
        bias=str(cfg.lora.bias), target_modules=describer.lora_target_modules(),
        modules_to_save=connector or None,
    )
    model = get_peft_model(base, lcfg)
    for _, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    describer.model = model

    n_lora = n_conn = n_lora_layers = 0
    unexpected = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if CONNECTOR_TAG in name:
            n_conn += p.numel()
        elif "lora_" in name:
            n_lora += p.numel()
            n_lora_layers += name.endswith("lora_A.default.weight")
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"Parámetros entrenables inesperados: {unexpected[:5]}")
    if n_lora_layers == 0:
        raise RuntimeError("LoRA no se aplicó a ninguna capa: revisa lora_target_modules")
    for c in connector:
        if not any(CONNECTOR_TAG in n and f"{c}{CONNECTOR_TAG}" in n for n, _ in model.named_parameters()):
            raise RuntimeError(f"El conector '{c}' no quedó entrenable: revisa lora.connector_modules")
    n_total = sum(p.numel() for p in model.parameters())
    return {"trainable_lora": n_lora, "trainable_connector": n_conn, "lora_layers": n_lora_layers,
            "trainable_total": n_lora + n_conn, "params_total_stored": n_total,
            "connector_modules": connector}


def param_groups(model, cfg) -> list[dict]:
    lora, conn = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (conn if CONNECTOR_TAG in name else lora).append(p)
    groups = [{"params": lora, "lr": float(cfg.train.lr), "name": "lora"}]
    if conn:
        groups.append({"params": conn, "lr": float(cfg.train.connector_lr), "name": "connector"})
    return groups


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
    """model.train() salvo los módulos congelados (encoder visual, X/Y-Encoder), que quedan en eval."""
    model.train()
    keys = set(frozen_modules)
    for name, m in model.named_modules():
        if name.split(".")[-1] in keys and CONNECTOR_TAG not in name + ".":
            m.eval()
