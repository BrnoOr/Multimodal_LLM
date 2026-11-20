"""Etapa 2: ajuste QLoRA (LoRA + conector) con presupuesto común. Ver configs/train/budget.yaml y
docs/metodologia.md §2.3.

    config.py      composición budget ⊕ modelo ⊕ prompt ⊕ CLI y validación del protocolo
    data.py        orden de datos determinista y reanudable (mismos lotes para los 4 modelos)
    collate.py     prompt de chat + caption con etiquetas enmascaradas (VLM); lotes de VL-JEPA
    objectives.py  entropía cruzada normalizada por los tokens del lote efectivo; InfoNCE bidireccional
    jepa.py        objetivo de VL-JEPA (opción A): Predictor e Y-Encoder entrenables, X-Encoder congelado
    lora.py        LoRA + conector entrenable sobre la base NF4 (o bf16); grupos de LR
    evaluate.py    exactitud por atributo en val con el mismo camino de inferencia que la etapa 1 (VLM)
    loop.py        bucle reanudable con selección del mejor checkpoint en val
    ydecoder.py    Y-Decoder de VL-JEPA (se entrena aparte, solo con texto)
"""

from .config import compose_train

__all__ = ["compose_train"]
