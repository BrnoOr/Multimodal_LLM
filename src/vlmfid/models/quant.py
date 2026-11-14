"""Esquema de cuantización común: NF4 (bitsandbytes) con doble cuantización y cómputo bf16.

Criterio uniforme en los cuatro modelos (docs/metodologia.md §2.1, decisión D5): TODO componente
congelado va en NF4, incluido el encoder visual (en VL-JEPA: X-Encoder, Predictor e Y-Encoder). Quedan
en precisión completa solo los módulos que la etapa 2 entrena completos (el conector o las
proyecciones) y la cabeza de salida `lm_head`. Fundamento: el enunciado pide modelos cuantizados sin
excepciones y QLoRA cuantiza el modelo base completo (Dettmers et al., 2023, §3). El control sin
cuantizar es `quant: bf16`.

bitsandbytes solo reemplaza capas `nn.Linear`: convoluciones (p. ej. la de parches), normalizaciones
y embeddings quedan en bf16. `quantization_report` mide qué fracción de los parámetros quedó
efectivamente en 4 bits; ese número se registra por modelo y se reporta.

Sobre `skip_modules` (transformers >= 5.17, quantizers_utils.should_convert_module): cada patrón se
compara con `re.match` contra el nombre COMPLETO del módulo (anclado al inicio) o con `endswith`.
Un patrón "multi_modal_projector" NO excluye "model.multi_modal_projector.linear_1"; hay que escribir
".*\\.multi_modal_projector". Las configs usan esa forma y `HFGenerativeDescriber` verifica tras la
carga que el conector NO quedó cuantizado y que el encoder visual SÍ.
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


def quantization_report(model: nn.Module, groups: dict[str, list[str]] | None = None) -> dict:
    """Cuántas capas y qué fracción de los parámetros quedaron en 4/8 bits.

    `groups` = {nombre: [subcadenas]}: conteo de capas cuantizadas cuyo nombre contiene alguna de las
    subcadenas (p. ej. {"vision": ["vision_tower"], "connector": ["multi_modal_projector"]}).

    Los pesos de una capa cuantizada están empaquetados, así que su tamaño original se calcula como
    in_features × out_features y no con `numel()`.
    """
    try:
        import bitsandbytes as bnb

        qtypes: tuple = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        qtypes = ()
    groups = groups or {}
    n_layers, q_params, q_param_ids = 0, 0, set()
    per_group = {g: 0 for g in groups}
    for name, m in model.named_modules():
        if qtypes and isinstance(m, qtypes):
            n_layers += 1
            q_params += m.in_features * m.out_features
            q_param_ids.add(id(m.weight))
            for g, keys in groups.items():
                if any(k in name for k in keys):
                    per_group[g] += 1
    other = sum(p.numel() for p in model.parameters() if id(p) not in q_param_ids)
    total = q_params + other
    out = {"quantized_linears": n_layers, "params_quantized": int(q_params), "params_total": int(total),
           "fraction_quantized": round(q_params / total, 4) if total else 0.0}
    out.update({f"quantized_in_{g}": n for g, n in per_group.items()})
    return out
