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


# Respuestas zero-shot reales (p1_attributes, inspección del 14-11-2026 en val). La lectura esperada
# de cada una se decidió a mano leyendo el texto, no con el extractor.
OSO_CENTRO = ("The object in the image is a red toy bear. It is positioned in the center of the image, "
              "with its head facing towards the right side of the image.")                    # LLaVA-OV, val/0663
OSO_PARTES = ("The object in the image is a red toy bear. It is a three-dimensional shape with a round head, "
              "a body, and limbs. The bear is positioned in the center of the image, standing upright on its "
              "hind legs.")                                                                    # LLaVA-OV, val/0663
RANA = ("The object in the image is a small, green, cartoon-like frog. It is positioned in the upper right "
        "quadrant of the image, with its body oriented horizontally and its head pointing towards the bottom "
        "left quadrant.")                                                                      # LLaVA-OV, val/4242
ESFERA_INCLINADA = ("The object in the image is a pink, spherical object with a smooth surface. It is positioned "
                    "in the center of the image, slightly tilted to the right.")              # LLaVA-OV, val/6311
ESFERA_TOP = ("The object is a green sphere. It is positioned in the center of the image, with its top facing "
              "the viewer.")                                                                   # LLaVA-OV, val/6890
BALDE = ("The object in the image is a green, rounded, spherical object that appears to be a bucket or a similar "
         "container. It is positioned on a flat, purple surface. The bucket is oriented horizontally, with its "
         "top facing down. The shadow of the bucket is cast on the surface below it.")        # InternVL3, val/6890
OSO_IZQUIERDA = ("The object in the image is a red, three-dimensional, bear-like figure. It is positioned on the "
                 "left side of the image, slightly behind the center. The bear is facing towards the right side "
                 "of the frame.")                                                              # InternVL3, val/0663


def test_la_orientacion_no_es_posicion():
    assert parse_position(OSO_CENTRO) == ("center", "mid")           # "facing towards the right" no es x=right
    assert parse_position(ESFERA_INCLINADA) == ("center", "mid")     # "tilted to the right"
    assert parse_position(ESFERA_TOP) == ("center", "mid")           # "its top facing the viewer" no es y=top
    assert parse_position(BALDE) == (None, None)                     # no dice dónde está
    assert parse_position(RANA) == ("right", "top")                  # la posición va antes que la orientación
    assert parse_position(OSO_IZQUIERDA) == ("left", "mid")
    assert parse_position("It casts a soft shadow to its lower left.") == (None, None)
    # la orientación no borra una posición dicha después de la coma
    assert parse_position("The cow is facing left, positioned at the bottom right.") == ("right", "bottom")


def test_head_como_parte_de_otro_objeto_no_es_la_forma():
    for texto in (OSO_CENTRO, OSO_PARTES, RANA):
        assert extract(texto)["shape"] is None, texto
    assert extract("It has a rounded body, a small head with a pointed snout, and a small tail.")["shape"] is None
    assert extract("The bear faces right.")["shape"] is None                    # "faces" es el verbo
    assert extract("A cow with a large head.")["shape"] == "cow"
    # cuando la cabeza ES el objeto, cuenta (Qwen3.5 e InternVL3 sobre val/6311, cuya forma es head)
    assert extract("The object is a stylized, simplified representation of a human head, specifically a bald "
                   "head with a distinct, dark, rectangular nose bridge.")["shape"] == "head"
    assert extract("The object is a stylized, 3D model resembling a human head.")["shape"] == "head"
    assert extract("a pink, abstract shape that resembles a stylized head or a stylized bird's head")["shape"] == "head"
    assert extract("The image features a pink head in the top right.")["shape"] == "head"
    assert extract("A pink object with the shape of a head.")["shape"] == "head"
    assert extract("It's a pink head.")["shape"] == "head"                      # contracción, no posesivo


def test_etiqueta_de_color_tiene_prioridad():
    # Qwen3.5, val/6311 (fragmento): "dark" describe la nariz; el color declarado es pink
    t = ("**Shape:**\nThe object is a stylized head with a distinct, dark, rectangular nose bridge.\n\n"
         "**Color:**\nThe object is a vibrant, solid pink color.")
    assert extract(t)["color"] == "pink"
    # Qwen3.5, val/0663 (fragmento): declara "bright red"; con nombre exacto no acierta "red"
    t = ("**Shape:**\nThe object is a small, red, 3D-rendered figure resembling a bear.\n\n"
         "**Color:**\nThe object is a solid, bright red color.")
    assert extract(t)["color"] == "bright red"
    assert not score(extract(t), truth_from_record(rec(3, 2, 2, "red")))["color"]


# Piloto del 14-11-2026 (200 imágenes de val). Salvo donde dice "construido", son respuestas reales.
def test_cuenta_la_primera_declaracion_de_posicion():
    # construido: la posición dicha primero manda sobre un par posterior que habla de otra cosa
    assert parse_position("The figure is positioned in the center of the image. Its texture is rough. "
                          "A second light is at the top left.") == ("center", "mid")
    # construido: iluminación y fondo no son la posición del objeto
    assert parse_position("The lighting comes from the top left corner.") == (None, None)
    assert parse_position("It is set against a gradient background that goes from light at the top to dark "
                          "at the bottom.") == (None, None)
    # InternVL3, p1: los matices no cambian la celda declarada...
    assert parse_position("It is positioned in the center of the image, slightly to the right side.") == ("center", "mid")
    assert parse_position("It is positioned in the center of the image, slightly towards the bottom.") == ("center", "mid")
    # ...pero completan el eje que la frase principal no dio
    assert parse_position("It is positioned in the lower part of the image, slightly to the right side.") == ("right", "bottom")
    # una frase sobre un solo eje fija solo ese eje
    assert parse_position("It is positioned horizontally in the center of the image, slightly above the "
                          "center.") == ("center", None)
    # construido, con la estructura de Qwen3.5 en p1: un eje por viñeta
    assert parse_position("**Position:**\n* **Vertical Position:** It is centered vertically.\n"
                          "* **Horizontal Position:** It is on the left side of the frame.") == ("left", "mid")
    assert parse_position("* **Vertical Position:** It is located in the lower right quadrant.\n"
                          "* **Horizontal Po") == ("right", "bottom")           # par explícito: fija los dos
    # p2_constrained: los modelos omiten el eje horizontal (o el vertical)
    assert parse_position("A green object is at the bottom of the image.") == (None, "bottom")
    assert parse_position("A **head** is at the **right** of the image.") == ("right", None)


def test_fuente_de_luz_no_es_posicion():
    # LLaVA-OV, p0_minimal: casi todas sus respuestas cierran con "a light source from the upper left side"
    t = ("The image features a single object, which appears to be a teapot, positioned centrally against a solid "
         "purple background. The teapot is predominantly yellow with a darker rim and handle, and it has a "
         "traditional design with a spout and a lid. The teapot casts a subtle shadow on the purple surface, "
         "indicating a light source from the upper left side of the image.")                    # val/0216
    assert parse_position(t) == ("center", "mid")
    t = ("The image shows a single object, which appears to be a red, irregularly shaped figure, possibly a toy "
         "or a model. The figure is positioned in the center of the image and is set against a gradient "
         "background that transitions from a lighter shade at the top to a darker shade at the bottom. The "
         "figure casts a shadow on the surface, suggesting a light source from the upper left side of the "
         "image.")                                                                               # val/0203
    assert parse_position(t) == ("center", "mid")
    t = ("The image shows a 3D model of a yellow humanoid figure with a round head, two arms, and two legs. The "
         "figure is positioned in a dynamic pose, as if it is in motion, with its arms and legs spread out and "
         "its head tilted back. The figure casts a shadow on the ground, indicating a light source from the "
         "upper left side of the image.")                                                        # val/0230
    assert parse_position(t) == (None, None) and extract(t)["shape"] is None


def test_color_del_fondo_como_sujeto():
    # InternVL3, p1, val/0265: no dice el color del objeto; "purple" es del fondo
    e = extract("The object in the image is a spherical object that appears to be a ball. It is positioned in "
                "the upper right quadrant of the image, slightly towards the center. The background is a solid "
                "purple color.")
    assert e["color"] is None and (e["x"], e["y"]) == ("right", "top")
    assert extract("The background is purple and the object is green.")["color"] == "green"     # construido


def test_json_con_campos_omitidos_o_copiados_del_prompt():
    # Qwen3.5, p3: copia el nombre de la paleta en vez de un color
    e = extract('```json\n{"shape": "teapot", "color": "TABLEAU_COLORS", "vertical": "top", "horizontal": "left"}\n```',
                json_mode=True)
    assert e["json_ok"] and e["shape"] == "teapot" and e["color"] is None and (e["x"], e["y"]) == ("left", "top")
    # InternVL3, p3: omite la clave horizontal
    e = extract('```json\n{"shape": "horse", "color": "purple", "vertical": "bottom"}\n```', json_mode=True)
    assert e["x"] is None and e["y"] == "bottom" and e["color"] == "purple"
    assert extract('{"shape": "head", "color": "YELLOW", "vertical": "top", "horizontal": "left"}',
                   json_mode=True)["color"] == "yellow"


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
