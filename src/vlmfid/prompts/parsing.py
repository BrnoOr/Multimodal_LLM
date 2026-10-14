"""Parseo de respuestas estructuradas.

Los VLM rara vez devuelven JSON limpio aunque se les pida "only JSON": suelen envolverlo en
```json ... ```, anteponer una frase o usar comillas simples. Este parser es tolerante a eso,
pero NO repara contenido: si no hay un objeto JSON recuperable devuelve None, y la evaluación
debe contar esa muestra como fallo de formato (no como atributo incorrecto ni omitirla).
"""

from __future__ import annotations

import ast
import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _first_object(text: str) -> str | None:
    """Primer bloque {...} balanceado, respetando llaves dentro de strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc, quote = 0, False, False, ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            elif ch in "\"'":
                in_str, quote = True, ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_json_response(text: str) -> dict | None:
    """Devuelve el primer objeto JSON de la respuesta, con claves en minúscula, o None."""
    if not text:
        return None
    m = _FENCE.search(text)
    candidate = _first_object(m.group(1) if m else text)
    if candidate is None:
        return None
    for loader in (json.loads, ast.literal_eval):   # literal_eval acepta comillas simples
        try:
            obj = loader(candidate)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(obj, dict):
            return {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in obj.items()}
    return None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Quita bloques <think>...</think> y etiquetas sueltas.

    Qwen3.5-Base usa la plantilla de chat del Instruct y a veces emite un `</think>` suelto en
    medio de la respuesta. La evaluación debe limpiar esas etiquetas antes de extraer atributos
    o medir similitud textual.
    """
    if not text:
        return ""
    text = _THINK_TAG.sub(" ", _THINK_BLOCK.sub(" ", text))
    return " ".join(text.split())
