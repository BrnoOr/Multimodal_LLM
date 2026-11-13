"""Extractor de atributos (vlmfid.eval.attributes): las 5 plantillas de M3DI, JSON y el Índice macro."""

import pytest

pytest.importorskip("matplotlib")

from vlmfid.eval.attributes import (
    cat_distance,
    extract,
    hue_to_cat,
    score,
    score_texts,
    truth_from_record,
)

REC = {"latents_image": {"object_shape": 3, "object_xpos": 0, "object_ypos": 2, "object_color": 0.62},
       "latents_text": {"object_color_name": "tab:blue"}}


@pytest.mark.parametrize("caption", [
    'A cow of "tab:blue" color is visible, positioned at the bottom-left of the image.',
    'A "tab:blue" cow is at the bottom-left of the image.',
    'The bottom-left of the image shows a "tab:blue" colored cow.',
    'At the bottom-left of the picture is a cow in "tab:blue" color.',
    'At the bottom-left of the image, there is a "tab:blue" object in the form of a cow.',
])
def test_cinco_plantillas(caption):
    s = score(extract(caption), truth_from_record(REC))
    assert s["shape"] and s["pos"] and s["name"] and s["color"] and s["macro"] == 1.0


def test_texto_libre_y_errores():
    s = score(extract("A blue bull standing on the lower left, on a pink background."), truth_from_record(REC))
    assert s["shape"] and s["pos"] and s["color"] and not s["name"]
    s = score(extract("A red horse in the center."), truth_from_record(REC))
    assert not s["shape"] and not s["pos"] and not s["color_adj"] and s["macro"] == 0


def test_json_y_fallo_de_formato():
    j = '```json\n{"shape": "cow", "color": "blue", "vertical": "bottom", "horizontal": "left"}\n```'
    assert score(extract(j, json_mode=True), truth_from_record(REC))["macro"] == 1.0
    assert extract("no json here", json_mode=True)["shape"] is None          # omitido = error


def test_rueda_de_color():
    assert hue_to_cat(0.0) == "red" and hue_to_cat(0.62) == "blue" and hue_to_cat(0.99) == "red"
    assert cat_distance("red", "pink") == 1 and cat_distance("red", "cyan") == 4
    assert score_texts(['A "tab:blue" cow is at the bottom-left of the image.'], [REC])["macro"] == 1.0
