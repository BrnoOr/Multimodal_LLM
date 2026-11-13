"""Etapa 2: ajuste fino QLoRA (LoRA + conector) con presupuesto común. Ver configs/train/budget.yaml.

    config.py      composición budget ⊕ modelo ⊕ prompt ⊕ CLI, con validación del presupuesto común
    data.py        orden de datos determinista y reanudable (mismos lotes para los 4 modelos)
    collate.py     prompt de chat + caption objetivo con etiquetas enmascaradas; lotes de VL-JEPA
    objectives.py  entropía cruzada normalizada por tokens del lote efectivo; InfoNCE con máscara de duplicados
    lora.py        LoRA + conector entrenable sobre la base NF4; grupos de lr; estado entrenable
    evaluate.py    exactitud por atributo en val con el mismo camino de inferencia que test
    loop.py        bucle reanudable con selección del mejor checkpoint en val
"""

from .config import compose_train

__all__ = ["compose_train"]
