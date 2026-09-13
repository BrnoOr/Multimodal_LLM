"""Adaptador de InternVL3.5: paradigma ViT-MLP-LLM (InternViT-300M + Qwen3).

Se usan las variantes `-HF` de OpenGVLab (con guion bajo en el nombre de version:
`InternVL3_5-8B-HF`), alineadas con la API estandar de transformers: cargan con
AutoProcessor + AutoModelForImageTextToText y admiten BitsAndBytesConfig. Las
variantes sin sufijo usan el formato GitHub con `.chat()` y no encajan aqui.

Requiere transformers>=4.52.1 y `trust_remote_code=True`: el repo esta etiquetado
como custom_code.

TILING DINAMICO. InternVL conserva la estrategia Dynamic High Resolution de
InternVL1.5: la imagen se parte en recortes de 448x448 mas una miniatura global, y
cada recorte produce 1024 tokens visuales que un modulo de pixel shuffle comprime a
256. Para M3DI, escenas sinteticas simples, ese tiling multiplica el coste sin
aportar informacion; se acota con max_patches.

MODO THINKING. InternVL3.5 soporta razonamiento paso a paso activandolo con un
system prompt especifico. NO se usa aqui: produciria bloques <think> que
contaminarian la descripcion y romperian la comparacion con LLaVA y Qwen. El
comportamiento por defecto (sin system prompt) es el correcto para este proyecto.

FLASH ATTENTION. El ejemplo oficial usa use_flash_attn=True, pero el cluster no
tiene nvcc y flash-attn no esta compilado: se fuerza `sdpa`.

Tamanos: 1B, 2B, 4B, 8B, 14B, 38B (mas variantes MoE). Con 8B (8.5B totales,
~6 GiB en NF4) la escala queda comparable a LLaVA-1.5-7B y Qwen2.5-VL-7B, de modo
que la comparacion sea sobre arquitectura y no sobre numero de parametros.
"""

from __future__ import annotations

import torch
from PIL import Image

from vlmfid.models.base import Describer, GenConfig, ModelSpec, bnb_config


class InternVLDescriber(Describer):
    name = "internvl"

    def __init__(self, spec: ModelSpec):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.spec = spec
        self.name = spec.name

        proc_kwargs = {}
        # acota el tiling dinamico. VERIFICAR el nombre del argumento contra la
        # version instalada: si el processor lo rechaza, se carga sin el y se
        # limita redimensionando la imagen antes de pasarla.
        if "max_patches" in spec.extra:
            proc_kwargs["max_patches"] = spec.extra["max_patches"]
        if "crop_to_patches" in spec.extra:
            proc_kwargs["crop_to_patches"] = spec.extra["crop_to_patches"]

        try:
            self.processor = AutoProcessor.from_pretrained(
                spec.hf_id, trust_remote_code=True, **proc_kwargs)
        except TypeError as e:
            print(f"  AVISO: processor rechazo {list(proc_kwargs)} ({e}); "
                  f"cargando sin acotar el tiling")
            self.processor = AutoProcessor.from_pretrained(
                spec.hf_id, trust_remote_code=True)

        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_id,
            quantization_config=bnb_config(spec),
            dtype=getattr(torch, spec.compute_dtype),
            device_map=spec.device,
            attn_implementation=spec.attn_implementation,   # sdpa: no hay flash-attn
            trust_remote_code=True,
            use_flash_attn=False,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _build_inputs(self, images: list[Image.Image], prompt: str):
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                  tokenize=False)
        return self.processor(
            text=[text] * len(images), images=images, return_tensors="pt", padding=True
        ).to(self.model.device)

    @torch.inference_mode()
    def describe(self, images: list[Image.Image], prompt: str,
                 cfg: GenConfig | None = None) -> list[str]:
        cfg = cfg or GenConfig()
        inputs = self._build_inputs(images, prompt)
        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in
                self.processor.batch_decode(gen, skip_special_tokens=True)]

    def memory_footprint_gib(self) -> float:
        return self.model.get_memory_footprint() / 2**30
