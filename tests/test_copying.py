"""Detección de copia de los ejemplos del prompt. Los casos salen de la prueba piloto real."""

import pytest

from vlmfid.config import load_prompt_catalog
from vlmfid.eval import copy_flags, copy_summary, prompt_examples
from vlmfid.prompts import strip_think_tags

CATALOG = load_prompt_catalog()
P4 = CATALOG.p4_dataset_format.text
P4M = CATALOG.p4m_multiexample.text


def test_ejemplos_se_leen_del_prompt():
    (ex,) = prompt_examples(P4)
    assert (ex.color, ex.obj, ex.position) == ("tab:blue", "horse", "bottom-right")
    assert [e.obj for e in prompt_examples(P4M)] == ["dragon", "teapot", "cow"]
    assert [e.color for e in prompt_examples(P4M)] == ["tab:red", "gold", "xkcd:light blue"]


@pytest.mark.parametrize("name", ["p0_minimal", "p1_attributes", "p2_constrained", "p3_json"])
def test_prompts_sin_ejemplos(name):
    # p2 tiene una plantilla con <color> <shape>, que no es un ejemplo concreto
    assert prompt_examples(CATALOG[name].text) == []


@pytest.mark.parametrize("pred, literal", [
    ('A "tab:blue" horse is at the bottom-right of the image.', True),                 # Qwen Base
    ('"tab:blue" horse at the bottom-right of the image.', True),                      # InternVL
    ('A "tab:blue" horse is at the bottom-right of the image. Put the color name in double quotes.', True),  # Instruct
    ('A "pink" bear is at the bottom-center of the image.', False),
])
def test_copia_literal_p4(pred, literal):
    assert copy_flags(pred, prompt_examples(P4), true_shape="hare")["copy_literal"] is literal


def test_copia_de_objeto_depende_de_la_forma_verdadera():
    ex = prompt_examples(P4)
    # LLaVA: nombra el caballo del ejemplo ante una vaca -> copia de objeto, no literal
    f = copy_flags("The horse is blue.", ex, true_shape="cow")
    assert f["copy_object"] and not f["copy_literal"] and f["copy_any"]
    # si la imagen sí es un caballo, nombrarlo puede ser un acierto
    assert not copy_flags("The horse is blue.", ex, true_shape="horse")["copy_object"]
    # pero repetir el ejemplo completo sigue siendo copia literal
    assert copy_flags('A "tab:blue" horse is here.', ex, true_shape="horse")["copy_literal"]


def test_p4m_copia_parcial_internvl():
    ex = prompt_examples(P4M)
    assert copy_flags('"gold" teapot', ex, true_shape="head")["copy_literal"]
    assert not copy_flags('A "yellow" horn is at the bottom-left of the image.', ex, "head")["copy_any"]


def test_resumen_de_run():
    preds = ['A "tab:blue" horse is at the bottom-right of the image.'] * 3 + ["A pink flower."]
    s = copy_summary(preds, P4, true_shapes=["cow", "hare", "head", "teapot"])
    assert s["n"] == 4 and s["n_examples"] == 1
    assert s["copy_literal"] == pytest.approx(0.75)
    assert s["distinct_ratio"] == pytest.approx(0.5)
    assert copy_summary(preds, CATALOG.p0_minimal.text)["copy_any"] == 0.0
    with pytest.raises(ValueError):
        copy_summary(preds, P4, true_shapes=["cow"])


def test_strip_think_tags():
    raw = 'A "pink" <object> is at the "top" of the image. </think>  A "pink" <object> is at the "top" of the image.'
    assert "</think>" not in strip_think_tags(raw)
    assert strip_think_tags("<think>\nrazonando\n</think>\n\nA red cow.") == "A red cow."
    assert strip_think_tags("") == ""
