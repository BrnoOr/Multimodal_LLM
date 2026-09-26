"""Prompts: el catálogo vive en configs/prompts.yaml (declarativo, congelado al cerrar la etapa 1).

Este módulo solo expone utilidades:
  * render: texto final del prompt.
  * parse_json_response: parseo de respuestas de prompts con `response_format: json` (p3_json).
  * strip_think_tags: limpia etiquetas <think> que algunos modelos dejan en la respuesta.
"""

from __future__ import annotations

from omegaconf import DictConfig

from .parsing import parse_json_response, strip_think_tags


def render(prompt_cfg: DictConfig, record: dict | None = None) -> str:
    """Texto del prompt, sin plantillas.

    Los prompts son constantes y varios contienen llaves literales ({top, mid, bottom} en p2,
    el esquema JSON en p3), por lo que no se aplica str.format: cualquier llave se envía tal cual.
    `record` queda en la firma por si en el futuro se necesitan prompts dependientes de la muestra.
    """
    return " ".join(str(prompt_cfg.text).split())


__all__ = ["render", "parse_json_response", "strip_think_tags"]
