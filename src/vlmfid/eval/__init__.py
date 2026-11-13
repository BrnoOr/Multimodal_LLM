"""Evaluación. Consume runs/<exp_id>/predictions.jsonl.

Disponible:
  * copying: tasa de copia de los ejemplos del prompt (experimento p4 vs p4m).
  * attributes: extractor de atributos y exactitud por atributo (el del notebook 02, sin cambios de
    lógica); la etapa 2 lo usa para elegir el checkpoint con el mismo "Índice macro" de la etapa 1.
  * text_metrics: BLEU-4, ROUGE-L y CIDEr-D contra una referencia por imagen (el TextScorer del
    notebook 03), para comparar el texto generado con el caption original o con el canónico.
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
from .text_metrics import TEXT_METRICS, TextScorer, tokenize

__all__ = ["M3DI_SHAPES", "METRICS", "TEXT_METRICS", "TextScorer", "PromptExample", "copy_flags", "copy_summary", "extract", "prompt_examples", "reference_shape", "score", "score_texts", "tokenize", "truth_from_record"]
