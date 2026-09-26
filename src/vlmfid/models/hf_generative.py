"""Adaptador genérico para VLM autorregresivos de transformers (>= 5.17).

Todos siguen el mismo patrón:
    plantilla de chat -> processor(text, images) -> model.generate -> decodificar solo lo nuevo.

Lo que cambia entre modelos se declara en configs/models/<modelo>.yaml, no en código:
    model_class            clase de transformers (defecto AutoModelForImageTextToText)
    quant_skip_modules     regex de módulos que quedan en bf16 al cuantizar
    vision_modules         subcadenas que identifican la parte visual (verificación post-carga)
    chat_template_kwargs   argumentos extra de la plantilla (p. ej. enable_thinking: false)
    chat_template_from     repo del que tomar la plantilla si el checkpoint no trae una (modelos base)
    lora_target_modules    regex PEFT para la etapa 2
Solo los modelos con ajustes de processor propios (Qwen: resolución dinámica) tienen subclase.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from PIL import Image

from ..config import as_dict
from .base import Description, Describer
from .quant import hf_quantization_config

# LoRA solo sobre el transformer de lenguaje (los encoders visuales también tienen *_proj)
LM_LORA_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
MODEL_CLASS_FALLBACKS = ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")


class HFGenerativeDescriber(Describer):
    family = "hf"
    generative = True

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.processor = None
        self.quant_report: dict = {}
        self.chat_template_origin: str | None = None

    # ---------------------------------------------------------------- carga
    def processor_kwargs(self) -> dict:
        return {}

    def configure_processor(self) -> None:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"          # imprescindible para generar por lotes
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

    def _model_class(self):
        import transformers

        names = [self.cfg.get("model_class")] + list(MODEL_CLASS_FALLBACKS)
        for name in names:
            if name and hasattr(transformers, name):
                return getattr(transformers, name)
        raise ImportError(f"Ninguna de {names} existe en transformers {transformers.__version__}")

    def load(self) -> None:
        from transformers import AutoProcessor

        c = self.cfg
        rev = c.get("revision")
        self.processor = AutoProcessor.from_pretrained(c.hf_id, revision=rev, **self.processor_kwargs())
        self._ensure_chat_template()
        self.configure_processor()

        skip = list(c.get("quant_skip_modules") or ["lm_head"])
        kwargs = dict(
            revision=rev,
            dtype=torch.bfloat16,
            quantization_config=hf_quantization_config(c.quant, skip),
            device_map={"": 0} if torch.cuda.is_available() else None,
            attn_implementation=c.get("attn_implementation", "sdpa"),
            low_cpu_mem_usage=True,
        )
        if c.get("use_kernels"):
            kwargs["use_kernels"] = True       # kernels precompilados del Hub (paquete `kernels`)
        self.model = self._model_class().from_pretrained(c.hf_id, **kwargs)
        self.model.eval()
        self._verify_quantization()
        # la adaptación LoRA (etapa 2) se carga encima si el config trae un adaptador
        if c.get("adapter_path"):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, c.adapter_path).eval()

    def _ensure_chat_template(self) -> None:
        """Garantiza una plantilla de chat, en este orden de preferencia:

        1. la del processor del checkpoint;
        2. la de su tokenizador (algunos repos la guardan solo ahí);
        3. la del repo `chat_template_from` (p. ej. la versión Instruct de un modelo base).

        Los modelos solo preentrenados (Qwen3.5-0.8B-Base) no traen plantilla. Tomar la de su
        versión Instruct, con el mismo tokenizador, mantiene idéntico el formato de entrada en el
        control Base vs Instruct: la única diferencia entre ambos runs son los pesos.
        """
        if getattr(self.processor, "chat_template", None):
            self.chat_template_origin = "processor"
            return
        tok = getattr(self.processor, "tokenizer", None)
        template, origin = None, None
        donor_id = self.cfg.get("chat_template_from")
        if donor_id:
            from transformers import AutoProcessor

            donor = AutoProcessor.from_pretrained(donor_id)
            template = getattr(donor, "chat_template", None) or getattr(
                getattr(donor, "tokenizer", None), "chat_template", None)
            origin = donor_id
        elif tok is not None and getattr(tok, "chat_template", None):
            template, origin = tok.chat_template, "tokenizer"
        if not template:
            raise ValueError(
                f"{self.cfg.hf_id} no trae plantilla de chat. Define `chat_template_from` en su config "
                "(p. ej. la versión Instruct del mismo modelo)."
            )
        self.processor.chat_template = template
        self.chat_template_origin = origin
        print(f"[hf] {self.cfg.hf_id} sin plantilla de chat: se usa la de {origin}", flush=True)

    def _verify_quantization(self) -> None:
        """Cuenta capas 4/8-bit y falla si alguna quedó en la parte visual.

        El criterio del proyecto es cuantizar solo el transformer de lenguaje; un patrón de
        `quant_skip_modules` mal escrito cuantizaría el encoder visual sin ningún error.
        """
        if self.cfg.quant == "bf16":
            self.quant_report = {"quantized_linears": 0}
            return
        import bitsandbytes as bnb

        qtypes = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
        vision_keys = list(self.cfg.get("vision_modules") or [])
        total, in_vision = 0, []
        for name, m in self.model.named_modules():
            if isinstance(m, qtypes):
                total += 1
                if any(k in name for k in vision_keys):
                    in_vision.append(name)
        self.quant_report = {"quantized_linears": total, "quantized_in_vision": len(in_vision)}
        if total == 0:
            raise RuntimeError(f"quant={self.cfg.quant} pero no hay capas cuantizadas: revisa bitsandbytes")
        if in_vision:
            raise RuntimeError(
                f"{len(in_vision)} capas visuales quedaron cuantizadas (p. ej. {in_vision[:3]}). "
                "Corrige `quant_skip_modules` en la config del modelo."
            )

    def info(self) -> dict:
        out = super().info()
        out.update(self.quant_report)
        out["chat_template_origin"] = self.chat_template_origin
        return out

    # ------------------------------------------------------------ inferencia
    def build_messages(self, prompt: str) -> list[dict]:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        if self.cfg.get("system_prompt"):
            msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": self.cfg.system_prompt}]})
        return msgs

    def chat_template_kwargs(self) -> dict:
        return as_dict(self.cfg.get("chat_template_kwargs"))

    def generation_kwargs(self) -> dict:
        g = as_dict(self.cfg.get("generation"))
        g.setdefault("max_new_tokens", 128)
        g.setdefault("do_sample", False)
        g.setdefault("num_beams", 1)
        return g

    @torch.inference_mode()
    def describe(self, images: list[Image.Image], prompts: list[str]) -> list[Description]:
        tkw = self.chat_template_kwargs()
        texts = [
            self.processor.apply_chat_template(self.build_messages(p), add_generation_prompt=True,
                                               tokenize=False, **tkw)
            for p in prompts
        ]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        dev = self.model.device
        inputs = {
            k: (v.to(dev, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(dev))
            for k, v in inputs.items()
        }
        out = self.model.generate(**inputs, **self.generation_kwargs())
        new = out[:, inputs["input_ids"].shape[1]:]          # con padding a la izquierda
        decoded = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        pad_id = self.processor.tokenizer.pad_token_id
        n_new = (new != pad_id).sum(dim=1).tolist()
        n_in = inputs["attention_mask"].sum(dim=1).tolist()
        return [
            Description(text=t.strip(), extra={"n_new_tokens": int(a), "n_input_tokens": int(b)})
            for t, a, b in zip(decoded, n_new, n_in)
        ]

    def lora_target_modules(self) -> str | list[str]:
        custom = self.cfg.get("lora_target_modules")
        if custom is None:
            return LM_LORA_REGEX
        return custom if isinstance(custom, str) else list(custom)
