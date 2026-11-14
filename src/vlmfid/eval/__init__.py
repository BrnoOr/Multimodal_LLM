"""Evaluación común a los cuatro modelos (docs/metodologia.md §2.1).

  * attributes: extracción de atributos desde texto y exactitud por atributo (métrica principal).
  * text_metrics: Bleu-1, Bleu-4, ROUGE-L y CIDEr con `pycocoevalcap`.
"""

from .attributes import (
    ATTRIBUTES,
    METRICS,
    color_ceiling,
    extract,
    majority_baseline,
    normalize_color_name,
    score,
    score_texts,
    truth_from_record,
)
from .text_metrics import TEXT_METRICS, text_metrics

__all__ = ["ATTRIBUTES", "METRICS", "TEXT_METRICS", "color_ceiling", "extract", "majority_baseline",
           "normalize_color_name", "score", "score_texts", "text_metrics", "truth_from_record"]
