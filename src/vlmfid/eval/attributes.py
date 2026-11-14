"""Extracción de atributos desde texto y exactitud por atributo (docs/metodologia.md §2.1).

Es la ÚNICA implementación: la usan la inferencia, la selección de checkpoint y los notebooks, para
los cuatro modelos (los VLM con su texto generado; VL-JEPA con el texto de su Y-Decoder).

Atributos evaluados y criterio de acierto:

    shape   forma del objeto. Léxico cerrado con sinónimos ("rabbit" -> hare). Si el texto nombra
            varias formas, cuenta la primera. "head"/"face" como PARTE de otro objeto no cuenta
            ("a bear with a round head", "its head pointing left"): ver `_is_body_part`.
    x       posición horizontal: left | center | right.
    y       posición vertical:   top | mid | bottom ("center"/"middle" se aceptan como mid).
            La ORIENTACIÓN del objeto no es su posición: se descartan las frases de orientación
            ("facing towards the right", "tilted to the right"), las partes del objeto ("its top
            facing the viewer") y las sombras ("a shadow to its lower left").
    color   NOMBRE EXACTO del color del caption de referencia (`object_color_name`), sin contar el
            prefijo de paleta: "blue" acierta para "tab:blue" (decisión D1). No se usan categorías
            de color ni tolerancias.

Agregados:

    pos        x e y correctas a la vez (informativo).
    attr_mean  media de shape, x, y y color: es el criterio para comparar prompts y checkpoints.

Reglas generales:
  * Omitir un atributo cuenta como error (exactitud estricta).
  * La verdad de forma y posición sale de los latentes de la imagen, que coinciden con los del texto
    por construcción del dataset (Daunhawer et al., 2023, Tabla 3: factores de contenido).
  * El nombre de color NO es función determinista de la imagen: el dataset elige al azar una de tres
    paletas por muestra. La exactitud de color tiene por eso un techo menor que 1, que se reporta
    aparte (ver `color_ceiling`).

Cómo se encuentra el color en texto libre. El vocabulario son todos los nombres de las tres paletas
que usa M3DI (Tableau, CSS4 y XKCD de matplotlib), sin prefijo. Se busca la coincidencia MÁS LARGA
en cada posición, de modo que "light blue" se lee como "light blue" y no como "blue": decir un
nombre más específico o más general que el de referencia es un error, que es lo que significa
"nombre exacto". Un nombre entre comillas tiene prioridad (es el formato del dataset). Cada mención
se asigna a objeto, fondo o luz según las palabras vecinas, y cuenta la primera mención del objeto.
Si la respuesta trae una etiqueta explícita ("**Color:** ...", "color: ..."), cuenta la primera
mención del objeto DESPUÉS de la etiqueta: es el color que el modelo declara, y evita tomar un
adjetivo anterior de otra sección ("a dark, rectangular nose").

Las reglas de orientación, parte del cuerpo y etiqueta de color salen de respuestas zero-shot reales
(inspección del 14-11-2026, 4 imágenes de val por modelo); están en tests/test_attributes.py.

Requiere matplotlib (tablas de nombres de color).
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..prompts.parsing import parse_json_response, strip_think_tags

# Códigos de M3DI -> palabras del caption (verificado contra data/raw/m3di: filas de
# latents_text.csv frente a text_raw.txt).
SHAPE_NAMES = {0: "teapot", 1: "hare", 2: "dragon", 3: "cow", 4: "armadillo", 5: "horse", 6: "head"}
XPOS_NAMES = {0: "left", 1: "center", 2: "right"}
YPOS_NAMES = {0: "top", 1: "mid", 2: "bottom"}

ATTRIBUTES = ("shape", "x", "y", "color")
METRICS = ATTRIBUTES + ("pos", "attr_mean")

# ── Forma ───────────────────────────────────────────────────────────────────────────────────────
SHAPE_LEXICON = {
    "teapot": ["teapot", "tea pot", "kettle"],
    "hare": ["hare", "rabbit", "bunny", "bunnies"],
    "dragon": ["dragon"],
    "cow": ["cow", "bull", "ox", "oxen", "cattle", "calf"],
    "armadillo": ["armadillo"],
    "horse": ["horse", "pony", "stallion", "mare"],
    "head": ["head", "bust", "face"],
}


def _term_rx(term: str, plural: bool = True) -> str:
    return r"\b" + re.escape(term).replace(r"\ ", r"[\s\-]?") + (r"(?:s|es)?" if plural else "") + r"\b"


# "head" sin plural: hay un solo objeto por imagen y "faces" suele ser el verbo ("the bear faces right").
SHAPE_RE = {k: re.compile("|".join(_term_rx(t, plural=k != "head") for t in v), re.IGNORECASE)
            for k, v in SHAPE_LEXICON.items()}

# "head"/"face" como parte de otro objeto. Dos señales, ambas sobre el texto ANTERIOR a la mención:
#   posesivo cercano      "its head", "the bear's small head"
#   cláusula de partes    "with a round head", "it has a rounded body, a small head" (hasta el fin de frase)
# Excepciones: "the image has/with ...", y "(in) the shape/form of a head", que sí nombran al objeto.
# ("it's a head" / "that's a head" / "there's a head" son contracciones de "is", no posesivos)
_PART_OWNER_RE = re.compile(
    r"(?:\b(?:its|his|her|their)|(?<!\bit)(?<!that)(?<!here)'s|s')\s+(?:[a-z\-]+\s+){0,2}$")
_PART_CLAUSE_RE = re.compile(
    r"\b(?<!image )(?<!picture )(?<!photo )(?<!scene )(?:with|has|have|having)\b[^.;:\n]*$")
_SHAPE_OF_RE = re.compile(
    r"\b(?:shape|form|figure|model|sculpture|statue|representation|likeness|image|picture)\s+of\s+"
    r"(?:(?:a|an|the)\s+)?(?:[a-z\-]+,?\s+){0,3}$")


def _is_body_part(text_lower: str, start: int) -> bool:
    before = text_lower[:start]
    if _SHAPE_OF_RE.search(before):
        return False
    return bool(_PART_OWNER_RE.search(before) or _PART_CLAUSE_RE.search(before))


def find_shapes(text: str) -> list[tuple[int, str]]:
    """(posición, forma) de cada mención, en orden de aparición. Omite "head" como parte de otro objeto."""
    low = text.lower()
    return sorted((m.start(), k) for k, rx in SHAPE_RE.items() for m in rx.finditer(text)
                  if not (k == "head" and _is_body_part(low, m.start())))


# ── Posición ────────────────────────────────────────────────────────────────────────────────────
V_MAP = {"top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom",
         "mid": "mid", "middle": "mid", "center": "mid", "centre": "mid"}
H_MAP = {"left": "left", "right": "right", "center": "center", "centre": "center", "middle": "center",
         "mid": "center"}
PAIR_RE = re.compile(r"\b(top|upper|mid|middle|center|centre|bottom|lower)[\s-]+(left|right|center|centre|middle)\b")
# Una palabra suelta ("top", "left") solo es posición con contexto de lugar: tras una preposición
# ("at the top", "to the right"), tras una etiqueta ("Vertical: top") o seguida de un sustantivo de
# zona ("upper third", "left side", "top of the image"). Así "a dome-like top" o "a rounded bottom",
# que describen la forma del objeto, no cuentan.
_CTX = r"(?:\b(?:at|in|on|to|toward|towards|near|into|along)\s+(?:the\s+)?(?:very\s+|far\s+|extreme\s+)?|:\s*)"
_LOC = (r"(?:side|half|part|portion|third|corner|quadrant|area|section|region|edge|"
        r"of\s+the\s+(?:image|frame|picture|scene|photo|canvas|screen))")
V_RE = re.compile(rf"{_CTX}(top|upper|bottom|lower|mid)\b|\b(top|upper|bottom|lower)[\s-]+{_LOC}\b")
H_RE = re.compile(rf"{_CTX}(left|right)\b|\b(left|right)[\s-]+(?:hand[\s-]+)?{_LOC}\b")
# "mid-air", "mid-gallop", "mid-flight": no son la altura del objeto en la imagen
MID_RE = re.compile(r"\bmid-(?!left|right|cent|middle)[a-z]+|\bmid\s+(?:air|flight|gallop|stride|jump|leap|motion)\b")
# "**Position:**". No "Vertical Position:" ni "Horizontal Position:": esas frases fijan un solo eje.
POS_LABEL_RE = re.compile(r"(?<!vertical )(?<!horizontal )\b(?:position|location)\b\W{0,4}:")


def _word(rx: re.Pattern, t: str) -> str | None:
    m = rx.search(t)
    return (m.group(1) or m.group(2)) if m else None
C_RE = re.compile(r"\b(center|centre|middle|centered|centred|centrally|central)\b")
NOISE_RE = re.compile(r"\b(?:on top of|right[\s-]side[\s-]up|right angles?|all right|top[\s-]down)\b")
# Lo que no es la posición del objeto en la imagen, hasta el siguiente signo de puntuación:
#   orientación        "facing towards the right side", "tilted to the right", "oriented horizontally"
#   partes del objeto  "its top facing the viewer"
#   sombras            "casts a shadow to its lower left"
ORIENT_RE = re.compile(r"\b(?:facing|faces|pointing|points|tilted|tilting|leaning|leans|turned|rotated|angled|"
                       r"oriented|looking|looks)\b[^.,;\n]*")
PART_RE = re.compile(r"\b(?:its|his|her|their)\s+(?:top|bottom|left|right|upper|lower)\b[^.,;\n]*")
SHADOW_RE = re.compile(r"\bshadows?\b[^.,;\n]*")
#   iluminación        "the lighting comes from the top left"
#   fondo              "a gradient background that transitions from a lighter shade at the top"
LIGHT_RE = re.compile(r"\b(?:lighting|illuminat\w*|spotlight|sunlight|light source|"
                      r"light (?:is|comes|coming|from|shines|shining))\b[^.,;\n]*")
BG_POS_RE = re.compile(r"\b(?:background|backdrop|gradient)\b[^.,;\n]*")
# Matices: "in the center, slightly to the right"; "on the left, slightly towards the center". No cambian
# la celda declarada (decisión del usuario, 14-11-2026): cuenta center en el primero y left en el segundo.
# Solo completan, con una dirección explícita, un eje que la frase principal no dio ("in the lower part
# of the image, slightly to the right" -> right). "off-center" declara el centro y lo que sigue es matiz.
SLIGHT_RE = re.compile(r"\b(?:slightly|a bit|a little|somewhat)\b[^.,;\n]*")
OFF_CENTER_RE = re.compile(r"\b(?:(?:slightly|a bit|a little|somewhat)\s+)?off[\s-]cent(?:er|re)(?:ed|d)?\b")
SENT_SPLIT_RE = re.compile(r"[.;!?\n]+")


def _parse_clause(t: str) -> tuple[str | None, str | None, bool]:
    """(horizontal, vertical, es_par) de un trozo de texto ya en minúsculas."""
    m = PAIR_RE.search(t)
    if m:
        return H_MAP[m.group(2)], V_MAP[m.group(1)], True
    v, h = _word(V_RE, t), _word(H_RE, t)
    v, h = (V_MAP[v] if v else None), (H_MAP[h] if h else None)
    if C_RE.search(t):
        v, h = v or "mid", h or "center"
    return h, v, False


def parse_position(text: str) -> tuple[str | None, str | None]:
    """(horizontal, vertical). El formato del dataset es "<vertical>-<horizontal>" (p. ej. mid-left).

    Cuenta la PRIMERA declaración de cada eje, recorriendo las frases en orden (igual que en forma y
    color cuenta la primera mención): "positioned in the center of the image. [...] top left" se lee
    como center/mid. Una frase que habla de un solo eje ("Vertical position: centered") solo fija ese
    eje, salvo que traiga un par explícito ("upper-left"), que fija los dos.

    Si la respuesta trae una etiqueta "Position:", se lee primero lo que sigue a la etiqueta (es la
    posición que el modelo declara) y el resto del texto solo completa un eje que falte.
    """
    t = NOISE_RE.sub(" ", text.lower().replace("*", " "))               # sin énfasis de Markdown
    for rx in (MID_RE, PART_RE, ORIENT_RE, SHADOW_RE, LIGHT_RE, BG_POS_RE):
        t = rx.sub(" ", t)
    t = OFF_CENTER_RE.sub(" center, slightly ", t)
    label = POS_LABEL_RE.search(t)
    if not label:
        return _scan_position(t)
    h, v = _scan_position(t[label.end():])
    if h is None or v is None:
        h2, v2 = _scan_position(t)
        h, v = h or h2, v or v2
    return h, v


def _scan_position(t: str) -> tuple[str | None, str | None]:
    h = v = None
    for sent in SENT_SPLIT_RE.split(t):
        only_v = "vertical" in sent and "horizontal" not in sent
        only_h = "horizontal" in sent and "vertical" not in sent
        ph, pv, pair = _parse_clause(SLIGHT_RE.sub(" ", sent))          # la declaración, sin matices
        if h is None and (pair or not only_v):
            h = ph
        if v is None and (pair or not only_h):
            v = pv
        nuance = " ".join(SLIGHT_RE.findall(sent))                      # los matices: solo left/right/top/bottom
        mh, mv = _word(H_RE, nuance), _word(V_RE, nuance)
        if h is None and mh and not only_v:
            h = H_MAP[mh]
        if v is None and mv and not only_h:
            v = V_MAP[mv]
        if h is not None and v is not None:
            break
    return h, v


# ── Color ───────────────────────────────────────────────────────────────────────────────────────
_PREFIX_RE = re.compile(r"^(?:tab|xkcd)\s*:\s*")
BG_WORDS = {"background", "backdrop", "backgrounds", "wall", "floor", "ground", "surface", "sky"}
LIGHT_WORDS = {"light", "lighting", "lights", "spotlight", "glow", "illumination", "illuminated", "lit",
               "shadow", "shadows"}
WORD_RE = re.compile(r"[a-z]+(?:['\-][a-z]+)*")
QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{2,40})[\"“”']")
COLOR_LABEL_RE = re.compile(r"\bcolou?r\b\W{0,4}:")        # "**Color:**", "color:", '"color":'
# El fondo como sujeto de la frase: el color que sigue es del fondo, salvo que la frase cambie de tema.
BG_SUBJECT_RE = re.compile(r"\b(?:background|backdrop|wall|floor|ground|surface|sky)\s+"
                           r"(?:is|was|appears|looks|seems|has|features|consists)\b"
                           r"(?:(?!\b(?:and|while|with|but)\b)[^.;\n]){0,60}$")


def normalize_color_name(name: str | None) -> str | None:
    """Forma canónica de un nombre de color para compararlo: minúsculas, sin comillas, sin prefijo de
    paleta (`tab:`, `xkcd:`), espacios colapsados y "grey" escrito "gray"."""
    if name is None:
        return None
    s = " ".join(str(name).lower().replace(" ", " ").strip().strip("\"'“”").split())
    s = _PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\bgrey\b", "gray", s)
    return s or None


@lru_cache(maxsize=1)
def color_vocabulary() -> frozenset[str]:
    """Nombres (normalizados) de las tres paletas que usa M3DI: Tableau, CSS4 y XKCD."""
    try:
        import matplotlib.colors as mcolors
    except ImportError as e:  # pragma: no cover
        raise ImportError("vlmfid.eval.attributes necesita matplotlib: `uv sync --extra dev`") from e
    names = set()
    for table in (mcolors.TABLEAU_COLORS, mcolors.CSS4_COLORS, mcolors.XKCD_COLORS):
        names.update(n for n in (normalize_color_name(k) for k in table) if n)
    return frozenset(names)


@lru_cache(maxsize=1)
def _color_regex() -> re.Pattern:
    # alternativas de la más larga a la más corta: gana la coincidencia más específica
    alts = sorted(color_vocabulary(), key=len, reverse=True)
    body = "|".join(re.escape(a).replace(r"\ ", r"[\s\-]+") for a in alts)
    # Un nombre seguido de "-<otro color>" es un compuesto distinto ("neon yellow-green" no es "neon
    # yellow"): no cuenta como ese nombre, igual que "light blue" no cuenta como "blue".
    compound = r"(?!-(?:green|blue|red|yellow|pink|purple|orange|brown|gr[ae]y|black|white|violet)\b)"
    return re.compile(r"(?<![a-z])(?:(?:tab|xkcd)\s*:\s*)?(" + body + r")(?![a-z])" + compound)


def _entity(text: str, start: int, end: int) -> str:
    """A qué se refiere una mención de color: objeto, fondo o luz, según las palabras vecinas."""
    after = WORD_RE.findall(text[end:])[:3]
    before = WORD_RE.findall(text[:start])[-3:]
    if any(w in BG_WORDS for w in after):
        return "background"
    if any(w in LIGHT_WORDS for w in after):
        return "light"
    if any(w in BG_WORDS for w in before):
        return "background"
    if BG_SUBJECT_RE.search(text[:start]):          # "the background is a solid, muted purple color"
        return "background"
    return "object"


def color_mentions(text: str) -> list[dict]:
    """Menciones de color del texto, en orden: {s, e, name, quoted, entity}. `name` va normalizado."""
    t = text.lower()
    vocab = color_vocabulary()
    found, taken = [], []
    for m in QUOTED_RE.finditer(t):
        name = normalize_color_name(m.group(1))
        if name in vocab:
            found.append({"s": m.start(), "e": m.end(), "name": name, "quoted": True})
            taken.append((m.start(), m.end()))
    for m in _color_regex().finditer(t):
        if any(m.start() < e and m.end() > s for s, e in taken):
            continue
        found.append({"s": m.start(), "e": m.end(), "name": normalize_color_name(m.group(1)), "quoted": False})
    found.sort(key=lambda d: d["s"])
    for f in found:
        f["entity"] = _entity(t, f["s"], f["e"])
    return found


# ── Extracción ──────────────────────────────────────────────────────────────────────────────────
def _empty() -> dict:
    return {"shape": None, "n_shapes": 0, "x": None, "y": None, "color": None, "color_quoted": False,
            "n_colors": 0}


def _from_text(text: str) -> dict:
    out = _empty()
    sh = find_shapes(text)
    out["shape"], out["n_shapes"] = (sh[0][1] if sh else None), len({k for _, k in sh})
    out["x"], out["y"] = parse_position(text)
    cm = color_mentions(text)
    out["n_colors"] = len(cm)
    objs = [c for c in cm if c["entity"] == "object"]
    label = COLOR_LABEL_RE.search(text.lower())
    if label:                                   # color declarado tras una etiqueta "Color:"
        objs = [c for c in objs if c["s"] >= label.end()] or objs
    obj = objs[0] if objs else None
    if obj:
        out.update(color=obj["name"], color_quoted=obj["quoted"])
    return out


def extract(raw: str | None, json_mode: bool = False) -> dict:
    """Atributos mencionados en una respuesta.

    `json_mode` para prompts con `response_format: json`: los atributos se leen de las claves
    shape / color / vertical / horizontal. Si no hay un objeto JSON recuperable, todos los atributos
    quedan vacíos (fallo de formato) y `json_ok` es False.
    """
    clean = strip_think_tags(raw or "")
    base = {"n_words": len(WORD_RE.findall(clean.lower())), "n_chars": len(clean), "json_ok": None}
    if not json_mode:
        return base | _from_text(clean)
    obj = parse_json_response(clean)
    base["json_ok"] = obj is not None
    if obj is None:
        return base | _empty()
    f = {k: str(v) for k, v in obj.items() if v is not None}
    out = _empty()
    sh = find_shapes(f.get("shape", ""))
    out["shape"], out["n_shapes"] = (sh[0][1] if sh else None), len({k for _, k in sh})
    out["x"] = H_MAP.get(f.get("horizontal", "").strip().lower())
    out["y"] = V_MAP.get(f.get("vertical", "").strip().lower())
    color = normalize_color_name(f.get("color"))
    if color and color not in color_vocabulary():
        # no es un nombre de color: o trae uno dentro ("light green color") o es texto copiado del
        # prompt ("TABLEAU_COLORS"), que cuenta como omitido
        inner = color_mentions(color)
        color = inner[0]["name"] if inner else None
    if color:
        out.update(color=color, n_colors=1)
    return base | out


# ── Verdad y puntaje ────────────────────────────────────────────────────────────────────────────
def truth_from_record(rec: dict) -> dict:
    """Verdad de una muestra (registro de `load_manifest` o línea de predictions.jsonl)."""
    li, lt = rec["latents_image"], rec["latents_text"]
    return {
        "shape": SHAPE_NAMES[int(li["object_shape"])],
        "x": XPOS_NAMES[int(li["object_xpos"])],
        "y": YPOS_NAMES[int(li["object_ypos"])],
        "color": normalize_color_name(lt["object_color_name"]),
    }


def score(ext: dict, truth: dict) -> dict:
    """Acierto por atributo (bool) y agregados de una muestra."""
    ok = {a: ext[a] is not None and ext[a] == truth[a] for a in ATTRIBUTES}
    return ok | {"pos": ok["x"] and ok["y"], "attr_mean": sum(ok.values()) / len(ATTRIBUTES)}


def score_texts(texts: list[str], records: list[dict], json_mode: bool = False) -> dict:
    """Exactitud media por atributo de una lista de respuestas alineada con sus registros."""
    if len(texts) != len(records):
        raise ValueError(f"{len(texts)} textos para {len(records)} registros")
    if not texts:
        return {m: float("nan") for m in METRICS} | {"n": 0}
    acc = {m: 0.0 for m in METRICS}
    for t, r in zip(texts, records):
        s = score(extract(t, json_mode), truth_from_record(r))
        for m in METRICS:
            acc[m] += float(s[m])
    return {m: acc[m] / len(texts) for m in METRICS} | {"n": len(texts)}


# ── Líneas base sin modelo ──────────────────────────────────────────────────────────────────────
def majority_baseline(train_records: list[dict], eval_records: list[dict]) -> dict:
    """Exactitud de responder siempre el valor más frecuente de train, por atributo."""
    from collections import Counter

    out = {}
    truths_tr = [truth_from_record(r) for r in train_records]
    truths_ev = [truth_from_record(r) for r in eval_records]
    for a in ATTRIBUTES:
        top = Counter(t[a] for t in truths_tr).most_common(1)[0][0]
        out[a] = sum(t[a] == top for t in truths_ev) / max(len(truths_ev), 1)
    out["attr_mean"] = sum(out[a] for a in ATTRIBUTES) / len(ATTRIBUTES)
    return out


def color_ceiling(train_records: list[dict], eval_records: list[dict], n_bins: int = 360) -> dict:
    """Techo empírico de la exactitud de color dada la imagen.

    El nombre de color depende del matiz verdadero (`object_color`) y de una paleta elegida al azar,
    que la imagen no revela. El mejor lector posible responde, para cada matiz, el nombre más
    frecuente. Se estima discretizando el matiz en `n_bins` intervalos con los datos de train y se
    evalúa en `eval_records`. Es una estimación: con más intervalos sube levemente y con pocos baja.
    """
    from collections import Counter, defaultdict

    def hue_bin(rec):
        return min(int(float(rec["latents_image"]["object_color"]) * n_bins), n_bins - 1)

    by_bin: dict[int, Counter] = defaultdict(Counter)
    for r in train_records:
        by_bin[hue_bin(r)][truth_from_record(r)["color"]] += 1
    best = {b: c.most_common(1)[0][0] for b, c in by_bin.items()}
    hits = sum(best.get(hue_bin(r)) == truth_from_record(r)["color"] for r in eval_records)
    return {"color_ceiling": hits / max(len(eval_records), 1), "n_bins": n_bins, "n_eval": len(eval_records)}
