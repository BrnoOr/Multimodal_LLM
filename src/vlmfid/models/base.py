"""Interfaz común de inferencia. Todo el resto del código (inferencia, evaluación, entrenamiento)
solo conoce `Describer`; las diferencias entre modelos quedan encapsuladas en los adaptadores."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig
from PIL import Image


@dataclass
class Description:
    text: str                                        # texto legible (generado o recuperado)
    extra: dict = field(default_factory=dict)        # metadatos serializables en JSON
    embedding: np.ndarray | None = None              # opcional (VL-JEPA): se guarda aparte en .npz


class Describer(ABC):
    """Contrato: `load()` una vez, luego `describe(images, prompts)` por lotes."""

    #: nombre de la familia; se registra en predictions.jsonl
    family: str = "base"
    #: True si el modelo produce texto de forma autorregresiva
    generative: bool = True

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def describe(self, images: list[Image.Image], prompts: list[str]) -> list[Description]: ...

    def lora_target_modules(self) -> str | list[str]:
        """Módulos a adaptar con LoRA en la etapa 2 (regex o lista, formato PEFT)."""
        raise NotImplementedError

    def info(self) -> dict:
        """Resumen para el registro del run: parámetros, dtype, cuantización, memoria."""
        out = {"family": self.family, "hf_id": self.cfg.get("hf_id"), "quant": self.cfg.get("quant")}
        if self.model is not None:
            out["n_params"] = sum(p.numel() for p in self.model.parameters())
        try:
            import torch

            if torch.cuda.is_available():
                out["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        except ImportError:
            pass
        return out

    def unload(self) -> None:
        """Libera la GPU (útil para cargar varios modelos en secuencia en un mismo proceso)."""
        self.model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
