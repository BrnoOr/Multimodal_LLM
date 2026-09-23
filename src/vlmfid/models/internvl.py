"""Adaptador de InternVL3.5: paradigma ViT-MLP-LLM (InternViT-300M + Qwen3).

Se usan las variantes `-HF` de OpenGVLab (`InternVL3_5-8B-HF`), alineadas con la
API nativa de transformers: AutoProcessor + AutoModelForImageTextToText, con
soporte de BitsAndBytesConfig. Las variantes sin sufijo usan el formato GitHub
(`.chat()`, `use_flash_attn=`, `trust_remote_code`) y NO encajan aqui: ninguno
de esos kwargs se pasa, `from_pretrained` los rechaza en la variante nativa.

Requiere transformers>=4.56 (InternVL3.5 nativo).

TILING DINAMICO. El image processor (familia GotOcr2) implementa Dynamic High
Resolution: recortes de 448x448 mas miniatura global, 256 tokens visuales por
recorte tras pixel shuffle. Se controla con los atributos `crop_to_patches`,
`min_patches`, `max_patches` del image_processor (NO son kwargs de
AutoProcessor.from_pretrained). Para M3DI (escenas sinteticas simples, imagenes
pequenas) el tiling multiplica el coste sin aportar informacion; se acota via
spec.extra.

MODO THINKING. InternVL3.5 lo activa un system prompt especifico. NO se usa:
produciria bloques <think> que contaminarian la descripcion y romperian la
comparacion con LLaVA y Qwen. Sin system prompt el comportamiento es el correcto.

FLASH ATTENTION. El cluster no tiene nvcc y flash-attn no esta compilado: se
usa `sdpa` via spec.attn_implementation.

Tamanos: 1B, 2B, 4B, 8B, 14B, 38B (mas variantes MoE). Con 8B (~6 GiB en NF4)
la escala es comparable a LLaVA-1.5-7B y Qwen2.5-VL-7B, de modo que la
comparacion sea sobre arquitectura y no sobre numero de parametros.
"""

from __future__ import annotations

import torch
from PIL import Image

from vlmfid.models.base import Describer, GenConfig, ModelSpec, bnb_config

_TILING_KEYS = ("crop_to_patches", "min_patches", "max_patches")


class InternVLDescriber(Describer):
    name = "internvl"

    def __init__(self, spec: ModelSpec):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.spec = spec
        self.name = spec.name

        self.processor = AutoProcessor.from_pretrained(spec.hf_id)
        self.processor.tokenizer.padding_side = "left"

        # Acotar el tiling en el image_processor. Se fija como atributo (valor
        # por defecto de cada llamada) y ademas se pasa por llamada en
        # _build_inputs, por si la version instalada solo honra uno de los dos.
        ip = self.processor.image_processor
        self.tiling_kwargs = {k: spec.extra[k] for k in _TILING_KEYS if k in spec.extra}
        for k, v in self.tiling_kwargs.items():
            if hasattr(ip, k):
                setattr(ip, k, v)
            else:
                print(f"  AVISO: image_processor no expone `{k}`; se ignora")
        print(f"  tiling: crop_to_patches={getattr(ip, 'crop_to_patches', '?')} "
              f"min={getattr(ip, 'min_patches', '?')} "
              f"max={getattr(ip, 'max_patches', '?')}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_id,
            quantization_config=bnb_config(spec),
            dtype=getattr(torch, spec.compute_dtype),
            device_map=spec.device,
            attn_implementation=spec.attn_implementation,   # sdpa: no hay flash-attn
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _build_inputs(self, images: list[Image.Image], prompt: str):
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                  tokenize=False)
        kwargs = {k: v for k, v in self.tiling_kwargs.items()}
        try:
            inputs = self.processor(text=[text] * len(images), images=images,
                                    return_tensors="pt", padding=True, **kwargs)
        except TypeError:
            # la version instalada no acepta kwargs de tiling por llamada;
            # quedan los atributos fijados en __init__
            inputs = self.processor(text=[text] * len(images), images=images,
                                    return_tensors="pt", padding=True)
        return inputs.to(self.model.device)

    @torch.inference_mode()
    def describe(self, images: list[Image.Image], prompt: str,
                 cfg: GenConfig | None = None) -> list[str]:
        cfg = cfg or GenConfig()
        inputs = self._build_inputs(images, prompt)
        gen_kwargs = dict(max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample,
                          pad_token_id=self.processor.tokenizer.pad_token_id
                          or self.processor.tokenizer.eos_token_id)
        if cfg.do_sample:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
        out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in
                self.processor.batch_decode(gen, skip_special_tokens=True)]

    def memory_footprint_gib(self) -> float:
        return self.model.get_memory_footprint() / 2**30
