"""Esquema de cuantización común: NF4 (bitsandbytes) con doble cuantización y cómputo bf16.

Criterio uniforme en los cuatro modelos: se cuantiza el *transformer de lenguaje* (decodificador o
predictor); el encoder visual, el proyector y la cabeza de salida quedan en bf16. Así la
comparación no confunde errores de cuantización del encoder visual con diferencias de paradigma.

Sobre `skip_modules` (transformers >= 5.17, quantizers_utils.should_convert_module): cada patrón se
compara con `re.match` contra el nombre COMPLETO del módulo (anclado al inicio) o con `endswith`.
Un patrón "vision_tower" NO excluye "model.vision_tower.encoder.layers.0.q_proj"; hay que escribir
".*\\.vision_tower". Las configs usan esa forma y `HFGenerativeDescriber` verifica tras la carga
que ninguna capa visual haya quedado cuantizada.
"""

from __future__ import annotations

import torch
import torch.nn as nn

VALID = ("nf4", "int8", "bf16")


def hf_quantization_config(quant: str, skip_modules: list[str] | None = None):
    """`quantization_config` para `from_pretrained` de transformers (None si bf16)."""
    if quant not in VALID:
        raise ValueError(f"quant='{quant}' inválido; opciones {VALID}")
    if quant == "bf16":
        return None
    from transformers import BitsAndBytesConfig

    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=skip_modules,
        )
    return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules)


def swap_linear_to_nf4(module: nn.Module, device: torch.device) -> int:
    """Reemplaza in situ cada nn.Linear de `module` por bnb.nn.Linear4bit (NF4, cómputo bf16).

    Se usa en VL-JEPA, cuyo modelo no es un PreTrainedModel y por tanto no admite
    `quantization_config`. Los pesos se cuantizan al mover el módulo a la GPU.
    Devuelve el número de capas reemplazadas.
    """
    import bitsandbytes as bnb

    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            q = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4",
            )
            q.weight = bnb.nn.Params4bit(
                child.weight.data.to(torch.bfloat16).cpu(), requires_grad=False,
                compress_statistics=True, quant_type="nf4",
            )
            if child.bias is not None:
                q.bias = nn.Parameter(child.bias.data.to(torch.bfloat16), requires_grad=False)
            setattr(module, name, q.to(device))
            n += 1
        else:
            n += swap_linear_to_nf4(child, device)
    return n
