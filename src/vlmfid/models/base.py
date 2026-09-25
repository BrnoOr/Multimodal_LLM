"""Interfaz comun de descripcion. Es el contrato entre los adaptadores y el resto.

Todo lo especifico de cada modelo (formato de chat, tokens visuales, decodificador)
queda dentro de su adaptador. infer.py, train.py y evaluate.py solo ven Describer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass
class GenConfig:
    """Parametros de generacion. Deterministas por defecto: la Etapa 1 debe ser reproducible."""

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0


@runtime_checkable
class Describer(Protocol):
    """Un modelo que produce una descripcion textual de una imagen dado un prompt."""

    name: str

    def describe(
        self, images: list[Image.Image], prompt: str, cfg: GenConfig | None = None
    ) -> list[str]:
        """Una descripcion por imagen. La longitud de la salida iguala la de la entrada."""
        ...

    def memory_footprint_gib(self) -> float: ...


@dataclass
class ModelSpec:
    """Configuracion de carga, comun a los tres modelos."""

    name: str
    hf_id: str
    quantization: str = "nf4"  # nf4 | none
    compute_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda:0"
    double_quant: bool = True
    extra: dict = field(default_factory=dict)


def bnb_config(spec: ModelSpec):
    """BitsAndBytesConfig o None. Mismo esquema NF4 para los tres modelos: al mantener
    la cuantizacion fija se elimina una variable de la comparacion."""
    if spec.quantization == "none":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if spec.quantization != "nf4":
        raise ValueError(f"cuantizacion no soportada: {spec.quantization}")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, spec.compute_dtype),
        bnb_4bit_use_double_quant=spec.double_quant,
    )


def build(spec: ModelSpec) -> Describer:
    """Fabrica: nombre de modelo -> adaptador."""
    key = spec.name.lower()
    if key.startswith("llava"):
        from vlmfid.models.llava import LlavaDescriber

        return LlavaDescriber(spec)
    if key.startswith("qwen"):
        from vlmfid.models.qwen import QwenDescriber

        return QwenDescriber(spec)
    if key.startswith("vljepa") or key.startswith("vl-jepa"):
        from vlmfid.models.vljepa import VLJepaDescriber

        return VLJepaDescriber(spec)
    if key.startswith("internvl"):
        from vlmfid.models.internvl import InternVLDescriber

        return InternVLDescriber(spec)
    raise ValueError(f"modelo desconocido: {spec.name}")
