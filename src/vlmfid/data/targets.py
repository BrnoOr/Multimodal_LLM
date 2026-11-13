"""Objetivos de entrenamiento y particiones de la etapa 2 (reformulación, ver docs/ETAPA2B_REFORMULACION.md).

Objetivo (`target.format`)
    original    el caption de M3DI tal cual: 5 plantillas al azar y ~100 nombres de color. La plantilla
                y el nombre exacto no son función de la imagen: ruido irreducible para el modelo.
    canonical   una sola plantilla y solo lo que la imagen determina:
                    A {color} {shape} is at the {vertical}-{horizontal} of the image.
                con el color como categoría básica del tono latente (`hue_to_cat(object_color)`), la
                misma verdad contra la que se mide "color" en la evaluación. Es la plantilla 2 de M3DI
                (y la de p2_constrained) con el nombre de color reemplazado por su categoría.

Split composicional (`data.holdout: compositional`)
    Se retiran de train 14 de las 63 combinaciones forma × posición (2 posiciones por forma, cada
    posición retirada para al menos una forma). En val y test esas combinaciones siguen presentes y
    se reportan aparte: si el modelo las acierta, compuso forma y posición en vez de memorizar pares.
"""

from __future__ import annotations

from functools import partial

from ..eval.attributes import truth_from_record

TARGET_FORMATS = ("original", "canonical")
HOLDOUTS = (None, "compositional")
CANONICAL_TEMPLATE = "A {color} {shape} is at the {y}-{x} of the image."

N_SHAPES, N_POSITIONS = 7, 9
# (forma, posición) retiradas: posiciones 2s y 2s+5 (mód 9) para la forma s. Posición = 3·ypos + xpos.
HELD_OUT_COMBOS = frozenset((s, p) for s in range(N_SHAPES) for p in ((2 * s) % N_POSITIONS,
                                                                    (2 * s + 5) % N_POSITIONS))


def canonical_caption(rec: dict) -> str:
    t = truth_from_record(rec)
    return CANONICAL_TEMPLATE.format(color=t["cat_true"], shape=t["shape_true"], y=t["y_true"], x=t["x_true"])


def target_text(rec: dict, fmt: str = "original") -> str:
    if fmt == "original":
        return rec["caption"]
    if fmt == "canonical":
        return canonical_caption(rec)
    raise ValueError(f"target.format='{fmt}' inválido ({' | '.join(TARGET_FORMATS)})")


def target_fn(fmt: str):
    """Función registro -> texto objetivo, serializable para los workers del DataLoader."""
    if fmt not in TARGET_FORMATS:
        raise ValueError(f"target.format='{fmt}' inválido ({' | '.join(TARGET_FORMATS)})")
    return partial(target_text, fmt=fmt)


def combo(rec: dict) -> tuple[int, int]:
    li = rec["latents_image"]
    return int(li["object_shape"]), 3 * int(li["object_ypos"]) + int(li["object_xpos"])


def is_held_out(rec: dict) -> bool:
    return combo(rec) in HELD_OUT_COMBOS


def apply_holdout(records: list[dict], holdout: str | None) -> list[dict]:
    """Registros de entrenamiento sin las combinaciones retiradas (si `holdout` lo pide)."""
    if holdout in (None, "null", ""):
        return records
    if holdout == "compositional":
        return [r for r in records if not is_held_out(r)]
    raise ValueError(f"data.holdout='{holdout}' inválido (null | compositional)")
