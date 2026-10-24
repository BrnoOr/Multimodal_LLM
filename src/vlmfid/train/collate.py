"""Construcción de lotes de entrenamiento.

Dos etapas, para que el trabajo pesado corra en los workers del DataLoader y el lote se pueda partir
ante un OOM sin volver a preprocesar:

  1. `*Featurizer.__call__` (en el worker): una muestra -> sus tensores (imagen procesada, ids).
  2. `*Featurizer.pad` (en el proceso principal): lista de muestras -> tensores del lote.

VLM autorregresivo — la muestra se arma igual que en la inferencia:

    ids = processor(plantilla_de_chat(prompt) con add_generation_prompt, imagen)   # prompt + imagen
        + tokenizer(caption + fin_de_turno)                                         # objetivo
    labels = [-100] * len(prompt) + ids_objetivo

  * El prompt sale de `describer.build_messages` y `apply_chat_template` con los mismos kwargs de la
    inferencia (p. ej. `enable_thinking: false` en Qwen3.5): el modelo aprende a continuar exactamente
    el contexto que verá en test.
  * El objetivo se tokeniza por separado y se concatena: así la frontera prompt/objetivo es exacta y
    los tokens objetivo son los mismos que `generate` produce tras el prompt.
  * `fin_de_turno` es el token con que la plantilla cierra un turno del asistente (`<|im_end|>` en los
    tres modelos, que heredan el formato ChatML de Qwen). Se detecta de la plantilla y se exige que
    esté entre los `eos_token_id` de generación: si no, el modelo ajustado no sabría detenerse.
  * Relleno a la IZQUIERDA, como en `describe()`: los tokens objetivo de todas las filas quedan al
    final, y basta calcular logits para las últimas max(len objetivo)+1 posiciones (`logits_to_keep`).
    Con RoPE el desplazamiento constante de posiciones por el relleno no altera la atención.

VL-JEPA — la imagen se repite como video estático igual que en `VLJEPADescriber.predict_embeddings`;
la consulta es el prompt tokenizado con el tokenizador del predictor; el objetivo es el embedding del
caption según el Y-Encoder congelado (se precalcula una vez por caption único).

El texto objetivo sale de `target_fn(registro)` (vlmfid.data.targets): el caption original o el canónico,
según `target.format`. En las muestras queda en la clave "caption" (el texto que se aprende).
"""

from __future__ import annotations

import torch

IGNORE = -100
_SENTINEL = "ZQXJ_RESPUESTA_ZQXJ"
# claves por token que, además de input_ids/attention_mask, devuelven algunos processors
_PER_TOKEN_DEFAULTS = {"mm_token_type_ids": 0, "token_type_ids": 0}


def detect_turn_end(describer, prompt_text: str) -> str:
    """Texto con que la plantilla de chat cierra un turno del asistente (p. ej. '<|im_end|>')."""
    msgs = describer.build_messages(prompt_text) + [
        {"role": "assistant", "content": [{"type": "text", "text": _SENTINEL}]}]
    full = describer.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                                   **describer.chat_template_kwargs())
    i = full.rfind(_SENTINEL)
    if i < 0:
        raise ValueError("La plantilla de chat no reproduce el turno del asistente")
    end = full[i + len(_SENTINEL):].strip()
    if not end:
        raise ValueError("La plantilla de chat no cierra el turno del asistente con ningún token")
    return end


def eos_ids(model) -> set[int]:
    gc = getattr(model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if e is None:
        e = getattr(getattr(model, "config", None), "eos_token_id", None)
    if e is None:
        return set()
    return set(e) if isinstance(e, (list, tuple)) else {int(e)}


def _caption(rec: dict) -> str:
    return rec["caption"]


class GenerativeFeaturizer:
    def __init__(self, describer, prompt_text: str, max_target_tokens: int, target_fn=_caption):
        self.target_fn = target_fn
        self.processor = describer.processor
        self.tok = describer.processor.tokenizer
        self.max_target_tokens = int(max_target_tokens)
        self.prompt_str = self.processor.apply_chat_template(
            describer.build_messages(prompt_text), add_generation_prompt=True, tokenize=False,
            **describer.chat_template_kwargs())
        self.turn_end = detect_turn_end(describer, prompt_text)
        end_ids = self.tok(self.turn_end, add_special_tokens=False)["input_ids"]
        if len(end_ids) != 1:
            raise ValueError(f"El fin de turno {self.turn_end!r} no es un token único: {end_ids}")
        self.turn_end_id = end_ids[0]
        allowed = eos_ids(describer.model)
        if allowed and self.turn_end_id not in allowed:
            raise ValueError(f"El fin de turno {self.turn_end!r} (id {self.turn_end_id}) no está en los "
                             f"eos_token_id de generación {sorted(allowed)}: el modelo no sabría detenerse.")
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def target_ids(self, caption: str) -> list[int]:
        ids = self.tok(caption + self.turn_end, add_special_tokens=False)["input_ids"]
        if ids[-1] != self.turn_end_id:
            raise ValueError(f"La tokenización del objetivo no termina en el fin de turno: {caption!r}")
        if len(ids) > self.max_target_tokens:
            raise ValueError(f"Caption de {len(ids)} tokens > max_target_tokens={self.max_target_tokens}: "
                             f"{caption!r}. Sube train.max_target_tokens (no se trunca en silencio).")
        return ids

    def __call__(self, batch):
        """En el worker: (registro, imagen) -> tensores de cada muestra, sin rellenar."""
        out = []
        for rec, img in batch:
            enc = self.processor(text=[self.prompt_str], images=[img], return_tensors="pt")
            text = self.target_fn(rec)
            out.append({"enc": dict(enc), "target": self.target_ids(text), "id": rec["id"], "caption": text})
        return out

    def pad(self, samples: list[dict]) -> dict:
        """En el proceso principal: relleno a la izquierda + concatenación de tensores de imagen."""
        lens = [s["enc"]["input_ids"].shape[1] + len(s["target"]) for s in samples]
        L, B = max(lens), len(samples)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attention = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), IGNORE, dtype=torch.long)
        per_token: dict[str, torch.Tensor] = {}
        others: dict[str, list] = {}
        for i, s in enumerate(samples):
            enc, tgt = s["enc"], torch.tensor(s["target"], dtype=torch.long)
            p = enc["input_ids"][0]
            n, off = p.shape[0] + tgt.shape[0], L - lens[i]
            input_ids[i, off:off + p.shape[0]] = p
            input_ids[i, off + p.shape[0]:] = tgt
            attention[i, off:] = 1
            labels[i, off + p.shape[0]:] = tgt
            shape = tuple(enc["input_ids"].shape)
            for k, v in enc.items():
                if k in ("input_ids", "attention_mask"):
                    continue
                if torch.is_tensor(v) and tuple(v.shape) == shape:        # tensor por token
                    fill = _PER_TOKEN_DEFAULTS.get(k, 0)
                    if k not in per_token:
                        per_token[k] = torch.full((B, L), fill, dtype=v.dtype)
                    per_token[k][i, off:off + p.shape[0]] = v[0]
                    per_token[k][i, off + p.shape[0]:off + n] = fill          # objetivo = texto
                else:
                    others.setdefault(k, []).append(v)
        inputs = {"input_ids": input_ids, "attention_mask": attention, **per_token}
        for k, vs in others.items():
            if all(torch.is_tensor(v) for v in vs):
                shapes = {tuple(v.shape[1:]) for v in vs}
                if len(shapes) > 1:
                    raise ValueError(f"'{k}' tiene formas distintas entre muestras {shapes}: las imágenes "
                                     "deberían tener todas el mismo tamaño (M3DI: 224×224)")
                inputs[k] = torch.cat(vs, dim=0)
            else:
                inputs[k] = [x for v in vs for x in (v if isinstance(v, (list, tuple)) else [v])]
        n_tgt = [len(s["target"]) for s in samples]
        return {"inputs": inputs, "labels": labels, "n_target_tokens": int(sum(n_tgt)),
                "max_target_len": int(max(n_tgt)), "ids": [s["id"] for s in samples],
                "captions": [s["caption"] for s in samples]}


class JEPAFeaturizer:
    def __init__(self, describer, prompt_text: str, target_fn=_caption):
        self.target_fn = target_fn
        self.transform = describer.transform
        self.num_frames = describer.num_frames
        q = describer.cfg.get("query_override") or prompt_text
        enc = describer.query_tok([q], max_length=describer.max_query_len, padding="max_length",
                                  truncation=True, return_tensors="pt")
        self.query_ids, self.query_mask = enc["input_ids"], enc["attention_mask"]
        n_real = len(describer.query_tok([q], add_special_tokens=True)["input_ids"][0])
        self.query_truncated = n_real > describer.max_query_len
        self.query_tokens = n_real

    def __call__(self, batch):
        return [{"frame": self.transform(img), "id": rec["id"], "caption": self.target_fn(rec)} for rec, img in batch]

    def pad(self, samples: list[dict]) -> dict:
        frames = torch.stack([s["frame"] for s in samples])                       # (B,3,H,W)
        pv = frames.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)               # (B,T,3,H,W)
        B = len(samples)
        return {"pixel_values": pv, "query_ids": self.query_ids.expand(B, -1).clone(),
                "query_mask": self.query_mask.expand(B, -1).clone(),
                "captions": [s["caption"] for s in samples], "ids": [s["id"] for s in samples],
                "n_target_tokens": B, "max_target_len": 0}
