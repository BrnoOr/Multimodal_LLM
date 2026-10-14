"""Evaluación. Consume runs/<exp_id>/predictions.jsonl.

Disponible:
  * copying: tasa de copia de los ejemplos del prompt (experimento p4 vs p4m).

Siguiente hito: text_metrics (BLEU/ROUGE/CIDEr), attribute_extractor (con tests, ≈100 % sobre los
captions de referencia) y attribute_metrics. Antes de extraer atributos, limpiar las respuestas con
vlmfid.prompts.strip_think_tags.
"""

from .copying import M3DI_SHAPES, PromptExample, copy_flags, copy_summary, prompt_examples, reference_shape

__all__ = ["M3DI_SHAPES", "PromptExample", "copy_flags", "copy_summary", "prompt_examples", "reference_shape"]
