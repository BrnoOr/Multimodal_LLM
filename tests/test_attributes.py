"""Extractor de atributos (vlmfid.eval.attributes) sobre ejemplos verificados a mano.

Los captions de `PLANTILLAS` y de `REALES` están copiados de data/raw/m3di/train (text_raw.txt y
latents_text.csv, primeras filas); la verdad de cada uno se leyó de los latentes, no del extractor.
"""

import pytest

pytest.importorskip("matplotlib")

from vlmfid.eval.attributes import (
    color_ceiling,
    color_mentions,
    extract,
    majority_baseline,
    normalize_color_name,
    parse_position,
    score,
    score_texts,
    truth_from_record,
)


def rec(shape, x, y, name, hue=0.5):
    return {"latents_image": {"object_shape": shape, "object_xpos": x, "object_ypos": y, "object_color": hue},
            "latents_text": {"object_color_name": name}}


# cow (3), left (0), bottom (2), "tab:blue"
REC = rec(3, 0, 2, "tab:blue")

PLANTILLAS = [
    'A cow of "tab:blue" color is visible, positioned at the bottom-left of the image.',
    'A "tab:blue" cow is at the bottom-left of the image.',
    'The bottom-left of the image shows a "tab:blue" colored cow.',
    'At the bottom-left of the picture is a cow in "tab:blue" color.',
    'At the bottom-left of the image, there is a "tab:blue" object in the form of a cow.',
]

# (caption real de train, forma, x, y, nombre de color en latents_text.csv)
REALES = [
    ('A "tab:blue" horse is at the bottom-right of the image.', 5, 2, 2, "tab:blue"),
    ('A "red" dragon is at the bottom-left of the image.', 2, 0, 2, "red"),
    ('A "tab:red" teapot is at the mid-right of the image.', 0, 2, 1, "tab:red"),
    ('At the bottom-right of the picture is a hare in "xkcd:hot pink" color.', 1, 2, 2, "xkcd:hot pink"),
    ('At the bottom-left of the image, there is a "xkcd:poison green" object in the form of a head.',
     6, 0, 2, "xkcd:poison green"),
    ('A head of "tab:olive" color is visible, positioned at the bottom-left of the image.', 6, 0, 2, "tab:olive"),
]


@pytest.mark.parametrize("caption", PLANTILLAS)
def test_cinco_plantillas(caption):
    s = score(extract(caption), truth_from_record(REC))
    assert s["shape"] and s["x"] and s["y"] and s["color"] and s["pos"] and s["attr_mean"] == 1.0


@pytest.mark.parametrize("caption,shape,x,y,name", REALES)
def test_captions_reales_de_train(caption, shape, x, y, name):
    assert score(extract(caption), truth_from_record(rec(shape, x, y, name)))["attr_mean"] == 1.0


def test_el_prefijo_de_paleta_no_cuenta():
    assert normalize_color_name('"tab:blue"') == normalize_color_name("blue") == "blue"
    assert normalize_color_name("xkcd:Hot  Pink") == "hot pink"
    assert normalize_color_name("xkcd:grey") == "gray"
    # sin comillas ni prefijo: acierta
    assert score(extract("A blue cow is at the bottom-left of the image."), truth_from_record(REC))["color"]
    # la verdad sin prefijo y la respuesta con prefijo: también
    assert score(extract('A "tab:blue" cow.'), truth_from_record(rec(3, 0, 2, "blue")))["color"]


def test_nombre_exacto_no_admite_mas_especifico_ni_mas_general():
    truth = truth_from_record(REC)                                    # color = blue
    assert extract("A light blue cow.")["color"] == "light blue"      # coincidencia más larga
    assert not score(extract("A light blue cow."), truth)["color"]
    assert not score(extract("A navy cow."), truth)["color"]
    truth_hp = truth_from_record(rec(1, 2, 2, "xkcd:hot pink"))
    assert score(extract("A hot pink hare at the bottom right."), truth_hp)["color"]
    assert not score(extract("A pink hare at the bottom right."), truth_hp)["color"]


def test_color_del_fondo_no_es_el_del_objeto():
    e = extract("A blue cow in front of a pink background, at the lower left.")
    assert e["color"] == "blue" and e["n_colors"] == 2
    e = extract("On a red background there is a blue cow.")
    assert e["color"] == "blue"
    assert [m["entity"] for m in color_mentions("a green light shines on a blue cow")] == ["light", "object"]


def test_omitir_es_error():
    s = score(extract("An animal."), truth_from_record(REC))
    assert not any(s[a] for a in ("shape", "x", "y", "color")) and s["attr_mean"] == 0.0
    s = score(extract("A blue cow."), truth_from_record(REC))        # sin posición
    assert s["shape"] and s["color"] and not s["x"] and not s["y"] and s["attr_mean"] == 0.5


def test_sinonimos_de_forma_y_posicion():
    s = score(extract("A blue bull standing on the lower left."), truth_from_record(REC))
    assert s["shape"] and s["pos"]
    assert parse_position("mid-center of the image") == ("center", "mid")
    assert parse_position("in the middle of the image") == ("center", "mid")
    assert parse_position("at the top right corner") == ("right", "top")
    assert parse_position("upper-left") == ("left", "top")
    assert parse_position("the teapot is on top of a table") == (None, None)      # "on top of" es ruido
    assert extract("A rabbit.")["shape"] == "hare"
    assert extract("A dragon next to a horse.")["shape"] == "dragon"              # cuenta la primera


def test_json_y_fallo_de_formato():
    j = '```json\n{"shape": "cow", "color": "tab:blue", "vertical": "bottom", "horizontal": "left"}\n```'
    assert score(extract(j, json_mode=True), truth_from_record(REC))["attr_mean"] == 1.0
    e = extract("no json here", json_mode=True)
    assert e["json_ok"] is False and e["shape"] is None and e["color"] is None


def test_score_texts_promedia_por_atributo():
    texts = ['A "tab:blue" cow is at the bottom-left of the image.', "A red horse in the center."]
    m = score_texts(texts, [REC, REC])
    assert m["n"] == 2 and m["shape"] == 0.5 and m["color"] == 0.5 and m["x"] == 0.5 and m["y"] == 0.5
    assert m["attr_mean"] == 0.5
    with pytest.raises(ValueError):
        score_texts(texts, [REC])


def test_lineas_base():
    train = [rec(3, 0, 2, "tab:blue", 0.60)] * 3 + [rec(3, 0, 2, "blue", 0.60)] + [rec(1, 2, 0, "red", 0.01)] * 2
    ev = [rec(3, 0, 2, "tab:blue", 0.60), rec(1, 2, 0, "red", 0.01), rec(1, 2, 0, "xkcd:red", 0.01)]
    b = majority_baseline(train, ev)
    assert b["shape"] == pytest.approx(1 / 3) and b["color"] == pytest.approx(1 / 3)   # mayoría: cow / blue
    # con el prefijo fuera, "tab:blue" y "blue" son el mismo nombre, y "red" y "xkcd:red" también
    assert color_ceiling(train, ev, n_bins=10)["color_ceiling"] == 1.0
