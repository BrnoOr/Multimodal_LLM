"""Pool enumerado de captions para describir por recuperacion (VL-JEPA).

VL-JEPA no decodifica texto: su salida es un embedding en el espacio del
Y-Encoder. Para obtener una "descripcion" se compara ese embedding contra un
pool finito de captions candidatos y se devuelve el mas cercano. El pool fija el
universo de salidas posibles, asi que su construccion es una decision de diseno
que debe declararse en el informe:

  * Plantillas: se extraen del propio manifiesto (columna `caption_ref`)
    sustituyendo color, forma y posicion por marcadores. Asi el pool respeta las
    mismas frases (`text_phrasing`) que la referencia y no se inventa un formato.
  * Vocabulario: por defecto, las formas, posiciones y nombres de color que
    aparecen en el manifiesto (mundo cerrado del dataset). Con
    colors="matplotlib" se amplia a TABLEAU+CSS4+XKCD, el universo del generador
    de M3DI, a costa de un pool ~10x mayor.

Cada PoolEntry lleva sus atributos: la exactitud por atributo de VL-JEPA se
calcula sin extractor, que es la ventaja metodologica de evaluar por
recuperacion (y una asimetria respecto a los modelos generativos, que hay que
declarar).
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

SHAPES = ("teapot", "hare", "dragon", "cow", "armadillo", "head", "horse")
VPOS = ("top", "mid", "bottom")
HPOS = ("left", "center", "right")

CANONICAL_TEMPLATE = 'A "{color}" {shape} is at the {vpos}-{hpos} of the image.'

_POS_RE = re.compile(r"\b(top|mid|middle|bottom)[- ](left|center|centre|right)\b", re.IGNORECASE)
_SHAPE_RE = re.compile(r"\b(" + "|".join(SHAPES) + r")\b", re.IGNORECASE)
_NORM_POS = {"middle": "mid", "centre": "center"}


@dataclass(frozen=True)
class PoolEntry:
    text: str
    shape: str
    vpos: str
    hpos: str
    color: str
    template_id: int = 0

    def attrs(self) -> dict:
        return {"shape": self.shape, "vpos": self.vpos, "hpos": self.hpos, "color": self.color}


# ------------------------------------------------------------------ plantillas


def templatize(caption: str, color_name: str) -> tuple[str, str, str, str] | None:
    """Convierte un caption de referencia en plantilla con marcadores.

    Devuelve (plantilla, shape, vpos, hpos) tal como aparecen en el texto, o None
    si no se localizan los tres atributos. Se sustituye primero el color (cadena
    literal) para que ningun nombre de color interfiera con las regex.
    """
    if not color_name or color_name not in caption:
        return None
    tpl = caption.replace(color_name, "{color}", 1)
    m_pos = _POS_RE.search(tpl)
    m_shape = _SHAPE_RE.search(tpl)
    if not (m_pos and m_shape):
        return None
    vpos = _NORM_POS.get(m_pos.group(1).lower(), m_pos.group(1).lower())
    hpos = _NORM_POS.get(m_pos.group(2).lower(), m_pos.group(2).lower())
    sep = tpl[m_pos.start(2) - 1]  # "-" o " "
    tpl = tpl[: m_pos.start()] + "{vpos}" + sep + "{hpos}" + tpl[m_pos.end() :]
    m_shape = _SHAPE_RE.search(tpl)  # re-localizar tras el corte
    shape = m_shape.group(1).lower()
    tpl = tpl[: m_shape.start()] + "{shape}" + tpl[m_shape.end() :]
    return tpl, shape, vpos, hpos


def _learn(df: pl.DataFrame, min_support: int = 1):
    """Extrae plantillas y mapas valor-crudo -> palabra desde el manifiesto.

    Los mapas resuelven que `object_shape`, `object_ypos` y `object_xpos` puedan
    venir como enteros (indices del generador) y no como palabras.
    """
    cols = ["caption_ref", "text_object_color_name", "object_shape", "object_ypos", "object_xpos"]
    rows = df.select(cols).unique(subset=["caption_ref"]).to_dicts()

    templates: Counter[str] = Counter()
    votes = {k: defaultdict(Counter) for k in ("shape", "vpos", "hpos")}
    failed = 0
    for r in rows:
        out = templatize(r["caption_ref"], str(r["text_object_color_name"]))
        if out is None:
            failed += 1
            continue
        tpl, shape, vpos, hpos = out
        templates[tpl] += 1
        votes["shape"][r["object_shape"]][shape] += 1
        votes["vpos"][r["object_ypos"]][vpos] += 1
        votes["hpos"][r["object_xpos"]][hpos] += 1

    if not templates:
        raise ValueError(
            "no se pudo derivar ninguna plantilla del manifiesto; revisar el formato de caption_ref"
        )
    if failed:
        print(
            f"  AVISO caption_pool: {failed}/{len(rows)} captions unicos no "
            f"se pudieron parametrizar"
        )

    tpls = [t for t, c in templates.most_common() if c >= min_support]
    maps = {k: {raw: cnt.most_common(1)[0][0] for raw, cnt in v.items()} for k, v in votes.items()}
    return tpls, maps


# ----------------------------------------------------------------- vocabulario


def matplotlib_colors() -> list[str]:
    """Universo de nombres de color del generador de M3DI (con prefijos tab:/xkcd:)."""
    import matplotlib.colors as mc

    names = list(mc.TABLEAU_COLORS) + list(mc.CSS4_COLORS) + list(mc.XKCD_COLORS)
    return list(dict.fromkeys(names))  # dedup conservando orden


# ----------------------------------------------------------------------- pool


def build_pool(
    manifest: pl.DataFrame | str | Path | None = None,
    *,
    quoted: bool = True,
    colors: str | Iterable[str] = "manifest",
    templates: list[str] | None = None,
    max_templates: int | None = None,
) -> list[PoolEntry]:
    """Enumera plantillas x formas x posiciones x colores.

    manifest : DataFrame o ruta parquet. Si es None se usa CANONICAL_TEMPLATE y
               el universo matplotlib (modo sin dataset, solo para pruebas).
    quoted   : False elimina las comillas alrededor de {color} en las plantillas.
    colors   : "manifest" (nombres observados), "matplotlib" (universo completo)
               o un iterable explicito.
    """
    if isinstance(manifest, (str, Path)):
        manifest = pl.read_parquet(manifest)

    if manifest is None:
        tpls = templates or [CANONICAL_TEMPLATE]
        shapes, vposs, hposs = list(SHAPES), list(VPOS), list(HPOS)
        color_list = matplotlib_colors() if colors in ("manifest", "matplotlib") else list(colors)
    else:
        learned, maps = _learn(manifest)
        tpls = templates or learned
        shapes = sorted(set(maps["shape"].values()), key=SHAPES.index)
        vposs = sorted(set(maps["vpos"].values()), key=VPOS.index)
        hposs = sorted(set(maps["hpos"].values()), key=HPOS.index)
        if colors == "manifest":
            color_list = sorted(manifest["text_object_color_name"].cast(str).unique().to_list())
        elif colors == "matplotlib":
            color_list = matplotlib_colors()
        else:
            color_list = list(colors)

    if max_templates is not None:
        tpls = tpls[:max_templates]
    if not quoted:
        tpls = [t.replace('"{color}"', "{color}") for t in tpls]

    pool = [
        PoolEntry(
            text=t.format(color=c, shape=s, vpos=v, hpos=h),
            shape=s,
            vpos=v,
            hpos=h,
            color=c,
            template_id=ti,
        )
        for ti, t in enumerate(tpls)
        for s, v, h, c in itertools.product(shapes, vposs, hposs, color_list)
    ]
    print(
        f"  pool: {len(tpls)} plantillas x {len(shapes)} formas x "
        f"{len(vposs)}x{len(hposs)} posiciones x {len(color_list)} colores "
        f"= {len(pool)} captions"
    )
    return pool


def pool_to_frame(pool: list[PoolEntry]) -> pl.DataFrame:
    """Para persistir el pool junto a la corrida (reproducibilidad)."""
    return pl.DataFrame([e.__dict__ for e in pool])
