"""Registro perezoso de adaptadores: importar un modelo no importa los demás.

Cada config de configs/models/ declara su `family`, que elige el adaptador:
    hf       VLM autorregresivo genérico de transformers (LLaVA-OneVision, InternVL3, ...)
    qwen_vl  Qwen-VL con resolución dinámica (Qwen3.5, Qwen2.5-VL)
    vljepa   open-vljepa (predicción de embeddings + recuperación)
"""

from __future__ import annotations

import importlib

from omegaconf import DictConfig

from .base import Describer

MODELS = {
    "hf": "vlmfid.models.hf_generative:HFGenerativeDescriber",
    "qwen_vl": "vlmfid.models.qwen_vl:QwenVLDescriber",
    "vljepa": "vlmfid.models.vljepa:VLJEPADescriber",
}


def build_describer(model_cfg: DictConfig) -> Describer:
    family = model_cfg.family
    if family not in MODELS:
        raise KeyError(f"Familia '{family}' desconocida. Disponibles: {list(MODELS)}")
    module, cls = MODELS[family].split(":")
    return getattr(importlib.import_module(module), cls)(model_cfg)
