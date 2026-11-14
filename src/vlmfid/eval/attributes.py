"""Extracción de atributos desde texto y exactitud por atributo (docs/metodologia.md §2.1).

Es la ÚNICA implementación: la usan la inferencia, la selección de checkpoint y los notebooks, para
los cuatro modelos (los VLM con su texto generado; VL-JEPA con el texto de su Y-Decoder).

Atributos evaluados y criterio de acierto:

    shape   forma del objeto. Léxico cerrado con sinónimos ("rabbit" -> hare). Si el texto nombra
            varias formas, cuenta la primera.
    x       posición horizontal: left | center | right.
    y       posición vertical:   top | mid | bottom ("center"/"middle" se aceptan como mid).
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


def _term_rx(term: str) -> str:
    return r"\b" + re.escape(term).replace(r"\ ", r"[\s\-]?") + r"(?:s|es)?\b"


SHAPE_RE = {k: re.compile("|".join(_term_rx(t) for t in v), re.IGNORECASE) for k, v in SHAPE_LEXICON.items()}


def find_shapes(text: str) -> list[tuple[int, str]]:
    """(posición, forma) de cada mención, en orden de aparición."""
    return sorted((m.start(), k) for k, rx in SHAPE_RE.items() for m in rx.finditer(text))


# ── Posición ────────────────────────────────────────────────────────────────────────────────────
V_MAP = {"top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom",
         "mid": "mid", "middle": "mid", "center": "mid", "centre": "mid"}
H_MAP = {"left": "left", "right": "right", "center": "center", "centre": "center", "middle": "center",
         "mid": "center"}
PAIR_RE = re.compile(r"\b(top|upper|mid|middle|center|centre|bottom|lower)[\s-]+(left|right|center|centre|middle)\b")
V_RE = re.compile(r"\b(top|upper|bottom|lower|mid)\b")
H_RE = re.compile(r"\b(left|right)\b")
C_RE = re.compile(r"\b(center|centre|middle|centered|centred)\b")
NOISE_RE = re.compile(r"\b(?:on top of|right[\s-]side[\s-]up|right angles?|all right|top[\s-]down)\b")


def parse_position(text: str) -> tuple[str | None, str | None]:
    """(horizontal, vertical). El formato del dataset es "<vertical>-<horizontal>" (p. ej. mid-left)."""
    t = NOISE_RE.sub(" ", text.lower())
    m = PAIR_RE.search(t)
    if m:
        return H_MAP[m.group(2)], V_MAP[m.group(1)]
    v, h = V_RE.search(t), H_RE.search(t)
    v, h = (V_MAP[v.group(1)] if v else None), (H_MAP[h.group(1)] if h else None)
    if C_RE.search(t):
        v, h = v or "mid", h or "center"
    return h, v


# ── Color ───────────────────────────────────────────────────────────────────────────────────────
_PREFIX_RE = re.compile(r"^(?:tab|xkcd)\s*:\s*")
BG_WORDS = {"background", "backdrop", "backgrounds", "wall", "floor", "ground", "surface", "sky"}
LIGHT_WORDS = {"light", "lighting", "lights", "spotlight", "glow", "illumination", "illuminated", "lit",
               "shadow", "shadows"}
WORD_RE = re.compile(r"[a-z]+(?:['\-][a-z]+)*")
QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{2,40})[\"“”']")


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
    return re.compile(r"(?<![a-z])(?:(?:tab|xkcd)\s*:\s*)?(" + body + r")(?![a-z])")


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
    obj = next((c for c in cm if c["entity"] == "object"), None)
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
