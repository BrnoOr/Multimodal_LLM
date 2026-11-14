"""Copia de los ejemplos del prompt en las respuestas.

Los prompts con ejemplos concretos (p4_dataset_format, p4m_multiexample) incluyen frases del tipo
`A "tab:blue" horse is at the bottom-right of the image.`. Los VLM pequeños tienden a repetirlas
en vez de describir la imagen, y eso infla la exactitud justo cuando la verdad coincide con el
ejemplo. Este módulo detecta esa copia para reportarla junto a la exactitud.

Dos niveles:
  * literal: la respuesta contiene el color entre comillas seguido del objeto del ejemplo
    (`"tab:blue" horse`), aunque agregue o quite texto alrededor;
  * objeto: la respuesta nombra el objeto de un ejemplo que no es la forma verdadera de la
    imagen (LLaVA: "The horse is blue." ante una vaca). Requiere la forma verdadera; sin ella
    se cuenta cualquier mención del objeto, lo que sobrestima la copia.

Los ejemplos se leen del propio texto del prompt, así que no hace falta declararlos aparte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..prompts.parsing import strip_think_tags

_EXAMPLE = re.compile(
    r'\bA\s+"(?P<color>[^"]+)"\s+(?P<obj>[A-Za-z][\w-]*)\s+is\s+at\s+the\s+(?P<pos>[a-z]+-[a-z]+)',
    re.IGNORECASE,
)


def _mentions(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}s?\b", text) is not None


#: formas de M3DI; las descripciones de referencia siempre nombran una
M3DI_SHAPES = ("armadillo", "cow", "dragon", "hare", "head", "horse", "teapot")


def reference_shape(caption: str) -> str | None:
    """Forma verdadera según la descripción de referencia (None si no nombra ninguna)."""
    text = str(caption).lower()
    return next((s for s in M3DI_SHAPES if _mentions(s, text)), None)


@dataclass(frozen=True)
class PromptExample:
    color: str
    obj: str
    position: str

    @property
    def marker(self) -> str:
        """Fragmento cuya presencia en la respuesta cuenta como copia literal."""
        return f'"{self.color}" {self.obj}'.lower()


def prompt_examples(prompt_text: str) -> list[PromptExample]:
    """Ejemplos concretos presentes en el texto de un prompt (lista vacía si no tiene)."""
    text = " ".join(str(prompt_text).split())
    return [PromptExample(m["color"], m["obj"].lower(), m["pos"].lower()) for m in _EXAMPLE.finditer(text)]


def copy_flags(prediction: str, examples: Sequence[PromptExample], true_shape: str | None = None) -> dict:
    """Marca si una respuesta copia algún ejemplo del prompt.

    `true_shape` es la forma verdadera de la imagen (p. ej. "cow"). Si coincide con el objeto de
    un ejemplo, nombrar ese objeto no cuenta como copia de objeto (puede ser un acierto real),
    pero la copia literal sí se cuenta: repetir `"tab:blue" horse` completo sigue siendo copia.
    """
    text = strip_think_tags(prediction).lower()
    literal = any(ex.marker in text for ex in examples)
    truth = true_shape.strip().lower() if true_shape else None
    obj = any(_mentions(ex.obj, text) and ex.obj != truth for ex in examples)
    return {"copy_literal": literal, "copy_object": obj, "copy_any": literal or obj}


def copy_summary(predictions: Iterable[str], prompt_text: str,
                 true_shapes: Iterable[str | None] | None = None) -> dict:
    """Tasas de copia de un run.

    Devuelve n, número de ejemplos del prompt, tasas `copy_literal`, `copy_object`, `copy_any`
    y `distinct_ratio` (respuestas distintas / n: cerca de 0 indica una respuesta fija).
    Para un prompt sin ejemplos las tasas son 0 por construcción.
    """
    preds = list(predictions)
    shapes = list(true_shapes) if true_shapes is not None else [None] * len(preds)
    if len(shapes) != len(preds):
        raise ValueError(f"true_shapes tiene {len(shapes)} elementos y predictions {len(preds)}")
    examples = prompt_examples(prompt_text)
    n = len(preds)
    out = {"n": n, "n_examples": len(examples), "copy_literal": 0.0, "copy_object": 0.0,
           "copy_any": 0.0, "distinct_ratio": 0.0}
    if n == 0:
        return out
    flags = [copy_flags(p, examples, s) for p, s in zip(preds, shapes)]
    for key in ("copy_literal", "copy_object", "copy_any"):
        out[key] = sum(f[key] for f in flags) / n
    out["distinct_ratio"] = len({strip_think_tags(p) for p in preds}) / n
    return out
