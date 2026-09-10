"""Adaptador de Qwen2.5-VL: conector mas expresivo y resolucion dinamica.

Las imagenes de M3DI son pequenas y de contenido simple, asi que se acota el numero
de tokens visuales con min_pixels/max_pixels. Sin ese limite el coste por imagen
crece sin aportar informacion.
"""

from __future__ import annotations

import torch
from PIL import Image

from vlmfid.models.base import Describer, GenConfig, ModelSpec, bnb_config

PATCH = 28  # lado del patch visual de Qwen2.5-VL


class QwenDescriber(Describer):
    name = "qwen"

    def __init__(self, spec: ModelSpec):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.spec = spec
        self.name = spec.name
        min_tok = spec.extra.get("min_visual_tokens", 64)
        max_tok = spec.extra.get("max_visual_tokens", 256)
        self.processor = AutoProcessor.from_pretrained(
            spec.hf_id,
            min_pixels=min_tok * PATCH * PATCH,
            max_pixels=max_tok * PATCH * PATCH,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            spec.hf_id,
            quantization_config=bnb_config(spec),
            dtype=getattr(torch, spec.compute_dtype),
            device_map=spec.device,
            attn_implementation=spec.attn_implementation,
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
