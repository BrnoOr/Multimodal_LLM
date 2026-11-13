"""Extracción de atributos desde texto libre y exactitud por atributo.

Es el extractor del notebook `02_eda_predicciones.ipynb` (celdas 10 y 14) movido al paquete sin
cambiar su lógica, para que la etapa 2 seleccione el checkpoint con la MISMA métrica con que se
reportó la etapa 1. Si se modifica aquí, hay que reemplazar la copia del notebook por un import.

Atributos y criterio de acierto (idénticos a la etapa 1):
    forma            nombre de la forma (léxico con sinónimos: "rabbit" -> hare)
    posición (x,y)   ambas coordenadas correctas
    color categoría  categoría básica del color nombrado == categoría del tono latente `object_color`
    color ±1         categoría a distancia circular <= 1 en la rueda de 8 categorías
    nombre exacto    nombre literal == `text_object_color_name`
    macro            (forma + posición(x,y) + color ±1) / 3        <- "Índice macro" de los informes
    shape_pos        (forma + posición(x,y)) / 2                   <- selección en la etapa 2 (reformulada)
    macro_cat        (forma + posición(x,y) + color categoría) / 3

`shape_pos` y `macro_cat` se agregaron en la reformulación de la etapa 2 (docs/ETAPA2B_REFORMULACION.md):
color ±1 está saturado (≈ 0,98 en los VLM autorregresivos) y diluye el macro. Las métricas de la
etapa 1 no cambian.

Omitir un atributo cuenta como error (exactitud estricta). El color se compara contra el tono de
la imagen y no contra el nombre del caption, porque el nombre no es función determinista del tono.

Requiere matplotlib (tablas de nombres CSS4/XKCD/Tableau); viene con el extra `dev`.
"""

from __future__ import annotations

import colorsys
import re
from functools import lru_cache

from ..prompts.parsing import parse_json_response, strip_think_tags

# ── Categorías de color (celda 10 del notebook) ─────────────────────────────────────────────
COLOR_CATS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]
HUE_UPPER = [15, 40, 70, 160, 195, 255, 285, 345]
CAT_CENTER = {"red": 0, "orange": 27.5, "yellow": 55, "green": 115, "cyan": 177.5, "blue": 225,
              "purple": 270, "pink": 315}

# Códigos de M3DI -> palabras del caption. Verificado en el EDA: mapeo inyectivo y P(w_k|k) = 1.
SHAPE_NAMES = {0: "teapot", 1: "hare", 2: "dragon", 3: "cow", 4: "armadillo", 5: "horse", 6: "head"}
XPOS_NAMES = {0: "left", 1: "center", 2: "right"}
YPOS_NAMES = {0: "top", 1: "mid", 2: "bottom"}


def hue_to_cat(h: float) -> str:
    d = (float(h) * 360) % 360
    for c, u in zip(COLOR_CATS, HUE_UPPER):
        if d < u:
            return c
    return "red"


def cat_distance(a, b) -> int:
    if a not in COLOR_CATS or b not in COLOR_CATS:
        return 99
    d = abs(COLOR_CATS.index(a) - COLOR_CATS.index(b))
    return min(d, len(COLOR_CATS) - d)


# ── Forma ───────────────────────────────────────────────────────────────────────────────────
SHAPE_LEXICON = {
    "teapot": ["teapot", "tea pot", "kettle"], "hare": ["hare", "rabbit", "bunny", "bunnies"],
    "dragon": ["dragon"], "cow": ["cow", "bull", "ox", "oxen", "cattle", "calf"], "armadillo": ["armadillo"],
    "horse": ["horse", "pony", "stallion", "mare"], "head": ["head", "bust", "face"],
}


def _term_rx(t):
    return r"\b" + re.escape(t).replace(r"\ ", r"[\s\-]?") + r"(?:s|es)?\b"


SHAPE_RE = {k: re.compile("|".join(_term_rx(t) for t in v), re.IGNORECASE) for k, v in SHAPE_LEXICON.items()}


def find_shapes(text):
    return sorted((m.start(), k) for k, rx in SHAPE_RE.items() for m in rx.finditer(text))


# ── Posición ────────────────────────────────────────────────────────────────────────────────
V_MAP = {"top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom",
         "mid": "mid", "middle": "mid", "center": "mid", "centre": "mid"}
H_MAP = {"left": "left", "right": "right", "center": "center", "centre": "center", "middle": "center",
         "mid": "center"}
PAIR_RE = re.compile(r"\b(top|upper|mid|middle|center|centre|bottom|lower)[\s-]+(left|right|center|centre|middle)\b")
V_RE = re.compile(r"\b(top|upper|bottom|lower|mid)\b")
H_RE = re.compile(r"\b(left|right)\b")
C_RE = re.compile(r"\b(center|centre|middle|centered|centred)\b")
NOISE_RE = re.compile(r"\b(?:on top of|right[\s-]side[\s-]up|right angles?|all right|top[\s-]down)\b")


def parse_position(text):
    t = NOISE_RE.sub(" ", text.lower())
    m = PAIR_RE.search(t)
    if m:
        return H_MAP[m.group(2)], V_MAP[m.group(1)]
    v, h = V_RE.search(t), H_RE.search(t)
    v, h = (V_MAP[v.group(1)] if v else None), (H_MAP[h.group(1)] if h else None)
    if C_RE.search(t):
        v, h = v or "mid", h or "center"
    return h, v


# ── Color ───────────────────────────────────────────────────────────────────────────────────
COLOR_LEXICON = {
    "red": ["red", "reddish", "crimson", "scarlet", "maroon", "burgundy", "ruby", "cherry"],
    "orange": ["orange", "orangish", "amber", "tangerine", "peach", "rust", "copper", "salmon", "coral"],
    "yellow": ["yellow", "yellowish", "gold", "golden", "mustard", "lemon"],
    "green": ["green", "greenish", "lime", "olive", "chartreuse", "emerald", "mint"],
    "cyan": ["cyan", "teal", "turquoise", "aqua", "aquamarine"],
    "blue": ["blue", "bluish", "navy", "azure", "cobalt", "sapphire", "cerulean"],
    "purple": ["purple", "purplish", "violet", "lavender", "indigo", "lilac", "plum", "mauve"],
    "pink": ["pink", "pinkish", "magenta", "fuchsia", "rose"],
    "achromatic": ["white", "black", "gray", "grey", "silver", "brown", "beige", "tan"],
}
TERM2CAT = {t: c for c, v in COLOR_LEXICON.items() for t in v}
# Nombres CSS sin prefijo que usa M3DI (unión de train/val/test). El notebook los leía del split
# evaluado; la unión solo agrega vocabulario, no cambia ningún acierto.
M3DI_CSS_NAMES = (
    "blue", "blueviolet", "chartreuse", "crimson", "cyan", "darkorange", "darkturquoise", "darkviolet",
    "deeppink", "deepskyblue", "dodgerblue", "gold", "greenyellow", "lawngreen", "lime", "magenta",
    "mediumspringgreen", "orange", "orangered", "red", "springgreen", "yellow",
)
CSS_VOCAB = {n for n in M3DI_CSS_NAMES if n not in TERM2CAT}
TERM_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, set(TERM2CAT) | CSS_VOCAB), key=len, reverse=True)) + r")\b")
QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{2,40})[\"“”']")
PREFIX_RE = re.compile(r"\b(?:xkcd|tab):\s?[a-z]+(?:[ /][a-z]+){0,3}")
BG_WORDS = {"background", "backdrop", "backgrounds", "wall", "floor", "ground", "surface", "sky"}
LIGHT_WORDS = {"light", "lighting", "lights", "spotlight", "glow", "illumination", "illuminated", "lit",
               "shadow", "shadows"}
MODIFIERS = {"light", "dark", "bright", "pale", "deep", "vivid", "neon", "pastel", "soft"}
WORD_RE = re.compile(r"[a-z]+(?:['\-][a-z]+)*")


def _mcolors():
    try:
        import matplotlib.colors as mcolors
    except ImportError as e:  # pragma: no cover
        raise ImportError("vlmfid.eval.attributes necesita matplotlib: `uv sync --extra dev`") from e
    return mcolors


def norm_name(s):
    return " ".join(s.lower().replace(" ", " ").split())


@lru_cache(maxsize=4096)
def resolve_color(name):
    """Nombre -> (categoría, tono en [0,1) o None). None si no es un color."""
    name = norm_name(name)
    if name in TERM2CAT:
        c = TERM2CAT[name]
        return c, (CAT_CENTER[c] / 360 if c in CAT_CENTER else None)
    mcolors = _mcolors()
    if name.startswith("#") or not mcolors.is_color_like(name) or re.fullmatch(r"[\d.\s]+", name):
        return None
    h, s, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(name))
    return ("achromatic", None) if (s < 0.2 or v < 0.2) else (hue_to_cat(h), h)


def color_mentions(text):
    t = text.lower()
    found, taken = [], []
    for m in QUOTED_RE.finditer(t):
        name = norm_name(m.group(1))
        info = resolve_color(name)
        if info:
            found.append({"s": m.start(), "e": m.end(), "name": name, "cat": info[0], "hue": info[1], "quoted": True})
            taken.append((m.start(), m.end()))

    def inside(a, b):
        return any(a < e and b > s for s, e in taken)

    for m in PREFIX_RE.finditer(t):
        if inside(m.start(), m.end()):
            continue
        words = m.group(0).replace(": ", ":").split(" ")
        for k in range(len(words), 0, -1):                  # prefijo más largo que sea un color
            info = resolve_color(" ".join(words[:k]))
            if info:
                end = m.start() + len(" ".join(words[:k]))
                found.append({"s": m.start(), "e": end, "name": norm_name(" ".join(words[:k])), "cat": info[0],
                              "hue": info[1], "quoted": False})
                taken.append((m.start(), end))
                break
    for m in TERM_RE.finditer(t):
        if inside(m.start(), m.end()):
            continue
        info = resolve_color(m.group(1))
        found.append({"s": m.start(), "e": m.end(), "name": m.group(1), "cat": info[0], "hue": info[1],
                      "quoted": False})
    found.sort(key=lambda d: d["s"])
    merged = []                                             # "bluish green", "yellow-green": núcleo = último término
    for f in found:
        if merged and not merged[-1]["quoted"] and not f["quoted"] and re.fullmatch(r"[\s\-]*", t[merged[-1]["e"]:f["s"]]):
            f = f | {"s": merged[-1]["s"], "name": t[merged[-1]["s"]:f["e"]]}
            merged[-1] = f
        else:
            merged.append(f)
    for f in merged:
        after = WORD_RE.findall(t[f["e"]:])[:3]
        before = WORD_RE.findall(t[:f["s"]])[-3:]
        if before and before[-1] in MODIFIERS:
            before = before[:-1]
        if any(w in BG_WORDS for w in after):
            f["entity"] = "background"
        elif any(w in LIGHT_WORDS for w in after):
            f["entity"] = "light"
        elif any(w in BG_WORDS for w in before):
            f["entity"] = "background"
        else:
            f["entity"] = "object"
    return merged


def _empty():
    return {"shape": None, "n_shapes": 0, "xpos": None, "ypos": None, "color_name": None, "color_cat": None,
            "color_hue": None, "color_quoted": False, "n_colors": 0, "bg_cat": None}


def _from_text(text):
    out = _empty()
    sh = find_shapes(text)
    out["shape"], out["n_shapes"] = (sh[0][1] if sh else None), len({k for _, k in sh})
    out["xpos"], out["ypos"] = parse_position(text)
    cm = color_mentions(text)
    obj = next((c for c in cm if c["entity"] == "object"), None)
    bg = next((c for c in cm if c["entity"] == "background"), None)
    out["n_colors"] = len(cm)
    out["bg_cat"] = bg["cat"] if bg else None
    if obj:
        out.update(color_name=obj["name"], color_cat=obj["cat"], color_hue=obj["hue"], color_quoted=obj["quoted"])
    return out


def extract(raw: str | None, json_mode: bool = False, strict_json: bool = True) -> dict:
    """Atributos mencionados en una respuesta. `json_mode` para prompts con response_format: json."""
    raw = raw or ""
    clean = strip_think_tags(raw)
    base = {"think": "think>" in raw.lower(), "n_words": len(WORD_RE.findall(clean.lower())), "n_chars": len(clean),
            "ends_clean": bool(re.search(r"[.!?)\"'\]}`]\s*$", clean)), "json_ok": None,
            "uses_named": bool(re.search(r"\b(?:xkcd|tab):", clean.lower()))}
    if not json_mode:
        return base | _from_text(clean)
    obj = parse_json_response(clean)
    base["json_ok"] = obj is not None
    if obj is None:
        return base | (_empty() if strict_json else _from_text(clean))
    f = {k: str(v) for k, v in obj.items() if v is not None}
    out = _empty()
    sh = find_shapes(f.get("shape", ""))
    out["shape"], out["n_shapes"] = (sh[0][1] if sh else None), len({k for _, k in sh})
    out["xpos"] = H_MAP.get(f.get("horizontal", "").strip().lower())
    out["ypos"] = V_MAP.get(f.get("vertical", "").strip().lower())
    col = f.get("color", "").strip()
    info = resolve_color(col.strip("\"'")) if col else None
    cm = color_mentions(col) if col and not info else []
    if info:
        out.update(color_name=norm_name(col.strip("\"'")), color_cat=info[0], color_hue=info[1], n_colors=1)
    elif cm:
        out.update(color_name=cm[-1]["name"], color_cat=cm[-1]["cat"], color_hue=cm[-1]["hue"], n_colors=len(cm))
    return base | out


# ── Verdad y puntaje ────────────────────────────────────────────────────────────────────────
def truth_from_record(rec: dict) -> dict:
    """Verdad de una muestra (registro de `load_manifest` o línea de predictions.jsonl)."""
    li, lt = rec["latents_image"], rec["latents_text"]
    return {
        "shape_true": SHAPE_NAMES[int(li["object_shape"])],
        "x_true": XPOS_NAMES[int(li["object_xpos"])],
        "y_true": YPOS_NAMES[int(li["object_ypos"])],
        "cat_true": hue_to_cat(li["object_color"]),
        "name_true": str(lt["object_color_name"]).lower(),
    }


def score(ext: dict, truth: dict) -> dict:
    shape_ok = ext["shape"] == truth["shape_true"]
    pos_ok = ext["xpos"] == truth["x_true"] and ext["ypos"] == truth["y_true"]
    color_adj = cat_distance(ext["color_cat"], truth["cat_true"]) <= 1
    color_ok = ext["color_cat"] == truth["cat_true"]
    return {
        "shape": shape_ok,
        "x": ext["xpos"] == truth["x_true"],
        "y": ext["ypos"] == truth["y_true"],
        "pos": pos_ok,
        "color": color_ok,
        "color_adj": color_adj,
        "name": ext["color_name"] == truth["name_true"],
        "macro": (int(shape_ok) + int(pos_ok) + int(color_adj)) / 3,
        "shape_pos": (int(shape_ok) + int(pos_ok)) / 2,
        "macro_cat": (int(shape_ok) + int(pos_ok) + int(color_ok)) / 3,
    }


METRICS = ("shape", "x", "y", "pos", "color", "color_adj", "name", "macro", "shape_pos", "macro_cat")


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
