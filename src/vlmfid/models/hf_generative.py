"""Adaptador genérico para VLM autorregresivos de transformers (>= 5.17).

Todos siguen el mismo patrón:
    plantilla de chat -> processor(text, images) -> model.generate -> decodificar solo lo nuevo.

Lo que cambia entre modelos se declara en configs/models/<modelo>.yaml, no en código:
    model_class            clase de transformers (defecto AutoModelForImageTextToText)
    quant_skip_modules     regex de módulos que quedan sin cuantizar (conector y lm_head)
    vision_modules         subcadenas que identifican el encoder visual: DEBE quedar cuantizado
    connector_modules      subcadenas que identifican el conector: NO debe quedar cuantizado
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
from .quant import hf_quantization_config, quantization_report

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

            from ..paths import resolve

            self.model = PeftModel.from_pretrained(self.model, str(resolve(c.adapter_path))).eval()
            # el adaptador aprendió a cerrar la respuesta con el fin de turno: generate debe parar ahí
            from ..train.collate import ensure_turn_end_eos

            ensure_turn_end_eos(self)

    def _ensure_chat_template(self) -> None:
        """Garantiza una plantilla de chat en el processor, en este orden de preferencia:

        1. la del processor del checkpoint (si existe, se usa y no se mira nada más);
        2. la del repo `chat_template_from`, si la config lo define;
        3. la del tokenizador del propio checkpoint.

        Qwen3.5-0.8B-Base trae plantilla solo en su tokenizador, no en el processor, y
        `processor.apply_chat_template` no hace fallback al tokenizador. `chat_template_from` va
        antes que el tokenizador a propósito: tomar la plantilla de la versión Instruct (mismo
        tokenizador) mantiene idéntico el formato de entrada en el control Base vs Instruct, así la
        única diferencia entre ambos runs son los pesos. El origen queda en `info()`.
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
        """Comprueba el criterio de cuantización del proyecto y registra cuánto quedó en 4 bits.

        Todo componente congelado va en NF4, incluido el encoder visual; el conector queda en
        precisión completa porque la etapa 2 lo entrena completo (docs/metodologia.md §2.1). Un
        patrón de `quant_skip_modules` mal escrito rompería esto sin ningún error, así que se falla si:
          * no hay ninguna capa cuantizada;
          * el encoder visual no tiene ninguna capa cuantizada;
          * alguna capa del conector quedó cuantizada.
        Con `quant: bf16` (control sin cuantizar) no debe haber ninguna capa cuantizada.
        """
        groups = {"vision": list(self.cfg.get("vision_modules") or []),
                  "connector": list(self.cfg.get("connector_modules") or [])}
        self.quant_report = quantization_report(self.model, groups)
        r = self.quant_report
        if self.cfg.quant == "bf16":
            if r["quantized_linears"]:
                raise RuntimeError(f"quant=bf16 pero hay {r['quantized_linears']} capas cuantizadas")
            return
        if r["quantized_linears"] == 0:
            raise RuntimeError(f"quant={self.cfg.quant} pero no hay capas cuantizadas: revisa bitsandbytes")
        if groups["vision"] and r["quantized_in_vision"] == 0:
            raise RuntimeError(
                f"El encoder visual ({groups['vision']}) no tiene capas cuantizadas. El criterio del proyecto "
                "es NF4 en todo componente congelado: revisa `quant_skip_modules` y `vision_modules`."
            )
        if r["quantized_in_connector"]:
            raise RuntimeError(
                f"{r['quantized_in_connector']} capas del conector ({groups['connector']}) quedaron "
                "cuantizadas; debe quedar en precisión completa. Corrige `quant_skip_modules`."
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
        g.setdefault("max_new_tokens", 256)
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

        # Tokens generados por muestra: hasta el primer EOS inclusive. (Contar los distintos de
        # tokenizer.pad_token_id no sirve: `generate` rellena las filas terminadas con otro id en
        # LLaVA-OV e InternVL3, y todas las filas daban el largo del lote.) Una fila sin EOS llegó a
        # max_new_tokens: su respuesta quedó cortada (`truncated`).
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos_ids = {int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]) if e is not None}
        if self.processor.tokenizer.eos_token_id is not None:
            eos_ids.add(int(self.processor.tokenizer.eos_token_id))
        is_eos = torch.isin(new, torch.tensor(sorted(eos_ids), device=new.device))
        has_eos = is_eos.any(dim=1)
        first = is_eos.int().argmax(dim=1)
        n_new = torch.where(has_eos, first + 1, torch.full_like(first, new.shape[1])).tolist()
        n_in = inputs["attention_mask"].sum(dim=1).tolist()
        return [
            Description(text=t.strip(), extra={"n_new_tokens": int(a), "n_input_tokens": int(b),
                                               "truncated": not bool(e)})
            for t, a, b, e in zip(decoded, n_new, n_in, has_eos.tolist())
        ]

    def lora_target_modules(self) -> str | list[str]:
        custom = self.cfg.get("lora_target_modules")
        if custom is None:
            return LM_LORA_REGEX
        return custom if isinstance(custom, str) else list(custom)
