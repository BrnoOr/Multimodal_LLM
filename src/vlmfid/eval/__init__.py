"""Evaluación. Consume runs/<exp_id>/predictions.jsonl.

Disponible:
  * copying: tasa de copia de los ejemplos del prompt (experimento p4 vs p4m).
  * attributes: extractor de atributos y exactitud por atributo (el del notebook 02, sin cambios de
    lógica); la etapa 2 lo usa para elegir el checkpoint con el mismo "Índice macro" de la etapa 1.

Pendiente: text_metrics (BLEU/ROUGE/CIDEr) en el paquete (hoy en los notebooks).
"""

from .attributes import METRICS, extract, score, score_texts, truth_from_record
from .copying import (
    M3DI_SHAPES,
    PromptExample,
    copy_flags,
    copy_summary,
    prompt_examples,
    reference_shape,
)

__all__ = ["M3DI_SHAPES", "METRICS", "PromptExample", "copy_flags", "copy_summary", "extract", "prompt_examples", "reference_shape", "score", "score_texts", "truth_from_record"]
