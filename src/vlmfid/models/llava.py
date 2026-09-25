"""Adaptador de LLaVA-1.5: encoder visual congelado, proyector lineal, decodificador."""

from __future__ import annotations

import torch
from PIL import Image

from vlmfid.models.base import Describer, GenConfig, ModelSpec, bnb_config


class LlavaDescriber(Describer):
    name = "llava"

    def __init__(self, spec: ModelSpec):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.spec = spec
        self.name = spec.name
        self.processor = AutoProcessor.from_pretrained(spec.hf_id)
        # el padding a la izquierda es obligatorio para generar en batch con un decoder causal
        self.processor.tokenizer.padding_side = "left"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            spec.hf_id,
            quantization_config=bnb_config(spec),
            dtype=getattr(torch, spec.compute_dtype),
            device_map=spec.device,
            attn_implementation=spec.attn_implementation,
        )
        self.model.eval()

    def _build_inputs(self, images: list[Image.Image], prompt: str):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        return self.processor(
            images=images, text=[text] * len(images), return_tensors="pt", padding=True
        ).to(self.model.device)

    @torch.inference_mode()
    def describe(
        self, images: list[Image.Image], prompt: str, cfg: GenConfig | None = None
    ) -> list[str]:
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
        # recortar el prompt: solo interesa lo generado
        gen = out[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]

    def memory_footprint_gib(self) -> float:
        return self.model.get_memory_footprint() / 2**30
