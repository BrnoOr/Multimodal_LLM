"""Lectura del layout crudo de Multimodal3DIdent (Daunhawer et al., ICLR 2023).

Layout esperado por split (el mismo para el conjunto publicado y para las variantes del generador):

    <raw>/<split>/images/*.png
    <raw>/<split>/latents_image.csv   ground truth de la escena renderizada
    <raw>/<split>/latents_text.csv    latentes usados para generar el caption
    <raw>/<split>/text/*.txt          una descripción por línea, alineada por índice

Convención de columnas en la tabla resultante:
    latentes de imagen   sin prefijo      (object_shape, object_color, ...)
    latentes de texto    prefijo "text_"  (text_object_shape, text_object_color_name, text_text_phrasing)
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

SPLITS = ("train", "val", "test")
TEXT_PREFIX = "text_"

# Atributos que el caption de referencia describe (presentes en latents_text.csv)
DESCRIBED = ["object_shape", "object_xpos", "object_ypos", "object_zpos", "object_color"]
# Atributos presentes solo en la imagen: evaluables únicamente con prompt dirigido
NOT_DESCRIBED = [
    "object_alpharot",
    "object_betarot",
    "object_gammarot",
    "spotlight_pos",
    "spotlight_color",
    "background_color",
]
# Pares imagen/texto directamente comparables (el color de imagen es continuo: se reporta aparte)
DISAGREEMENT_PAIRS = ["object_shape", "object_xpos", "object_ypos", "object_zpos"]


def _read_latents(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    # índice de pandas guardado sin nombre -> columna "" o "Unnamed: 0"
    drop = [c for c in df.columns if c == "" or c.startswith("Unnamed")]
    return df.drop(drop)


def find_caption_file(split_dir: Path) -> Path:
    txts = sorted(split_dir.joinpath("text").glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"sin .txt en {split_dir / 'text'}")
    if len(txts) > 1:
        print(f"  aviso: {len(txts)} archivos en text/, se usa {txts[0].name}")
    return txts[0]


def read_captions(path: Path, n_expected: int) -> list[str]:
    """Una descripción por línea. Si el conteo no calza, intenta segmentar por oraciones
    (el repo original usa nltk.sent_tokenize sobre el archivo completo)."""
    raw = path.read_text(encoding="utf-8")
    lines = [c.strip() for c in raw.splitlines() if c.strip()]
    if len(lines) == n_expected:
        return lines
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw.strip()) if s.strip()]
    if len(sents) == n_expected:
        print(f"  aviso: {path.name} no trae una descripción por línea; se segmentó por oraciones")
        return sents
    return lines  # la validación de conteos reporta el desajuste


def list_images(split_dir: Path, raw: Path) -> pl.DataFrame:
    """Imágenes ordenadas por el entero de su nombre, para alinear con el índice de fila.
    `image_path` queda relativa a `raw`: el manifiesto sirve en laptop y cluster."""
    imgs = sorted(
        split_dir.joinpath("images").glob("*.png"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or -1),
    )
    if not imgs:
        raise FileNotFoundError(f"sin .png en {split_dir / 'images'}")
    return pl.DataFrame({
        "row_idx": list(range(len(imgs))),
        "image_id": [p.stem for p in imgs],
        "image_path": [p.relative_to(raw).as_posix() for p in imgs],
    }, schema_overrides={"row_idx": pl.UInt32})


def build_split(raw: str | Path, split: str, variant: str = "base", verbose: bool = True) -> pl.DataFrame:
    raw = Path(raw)
    d = raw / split
    log = print if verbose else (lambda *a, **k: None)
    log(f"[{split}]")

    lat_img = _read_latents(d / "latents_image.csv").with_row_index("row_idx")
    lat_txt = _read_latents(d / "latents_text.csv").with_row_index("row_idx")
    imgs = list_images(d, raw)
    captions = read_captions(find_caption_file(d), lat_img.height)

    n = {
        "latents_image": lat_img.height,
        "latents_text": lat_txt.height,
        "captions": len(captions),
        "images": imgs.height,
    }
    log(f"  filas: {n}")
    if len(set(n.values())) != 1:
        raise ValueError(f"desalineación en {split}: {n}")

    cap = pl.DataFrame({"row_idx": list(range(len(captions))), "caption_ref": captions},
                       schema_overrides={"row_idx": pl.UInt32})
    # Todas las columnas de latents_text se prefijan, sin excepción, para conservar ambas fuentes
    # sin colisionar. Incluye `text_phrasing` -> `text_text_phrasing`: omitir las que ya empiezan
    # con "text_" haría ambiguo qué columnas vienen de latents_text.
    lat_txt = lat_txt.rename({c: f"{TEXT_PREFIX}{c}" for c in lat_txt.columns if c != "row_idx"})

    df = (
        imgs.join(lat_img, on="row_idx", how="inner")
        .join(lat_txt, on="row_idx", how="inner")
        .join(cap, on="row_idx", how="inner")
        .with_columns(
            pl.lit(split).alias("split"),
            pl.lit(variant).alias("variant"),
            # identificador estable y único entre variantes: clave de reanudación en runs/
            pl.format("{}/{}/{}", pl.lit(variant), pl.lit(split), pl.col("image_id")).alias("id"),
        )
    )
    front = ["id", "image_id", "image_path", "caption_ref", "split", "variant", "row_idx"]
    return df.select(front + [c for c in df.columns if c not in front])


def disagreement(df: pl.DataFrame) -> pl.DataFrame:
    """Tasa de desacuerdo entre latentes de imagen y de texto para los pares comparables.

    Define el techo alcanzable: si el caption dice un valor que la imagen no tiene, ningún modelo
    fiel a la imagen puede acertar a la vez contra la imagen y contra la referencia.
    """
    rows = []
    for a in DISAGREEMENT_PAIRS:
        b = f"{TEXT_PREFIX}{a}"
        if a in df.columns and b in df.columns:
            rows.append({"attribute": a, "rate": df.select((pl.col(a) != pl.col(b)).mean()).item()})
    return pl.DataFrame(rows, schema={"attribute": pl.String, "rate": pl.Float64})


MAX_ATTRIBUTE_CARDINALITY = 256        # el color nominal de M3DI tiene ~106 valores
EXCLUDED_ATTRIBUTES = ("text_phrasing",)  # plantilla del caption: no es un atributo de la escena


def select_discrete_attributes(values: dict[str, list], max_cardinality: int = MAX_ATTRIBUTE_CARDINALITY,
                               exclude: tuple[str, ...] = EXCLUDED_ATTRIBUTES) -> list[str]:
    """Atributos evaluables a partir de {nombre: valores}. Criterio único, compartido por la
    evaluación y por los prototipos de VL-JEPA:

      * enteros o strings (los latentes continuos, como tonos y ángulos, no entran);
      * entre 2 y `max_cardinality` valores distintos (un atributo constante, como object_zpos,
        siempre "acierta" e inflaría la exactitud);
      * sin `exclude`, y sin `<x>_index` cuando existe `<x>_name` (misma información codificada dos veces).
    """
    out = []
    for name, vals in values.items():
        if name in exclude:
            continue
        if name.endswith("_index") and name[: -len("_index")] + "_name" in values:
            continue
        uniq = set(vals)
        if not all(isinstance(v, (int, str)) and not isinstance(v, bool) for v in uniq):
            continue
        if 2 <= len(uniq) <= max_cardinality:
            out.append(name)
    return out


def discrete_text_attributes(df: pl.DataFrame, max_cardinality: int = MAX_ATTRIBUTE_CARDINALITY,
                             exclude: tuple[str, ...] = EXCLUDED_ATTRIBUTES) -> list[str]:
    """Latentes de texto discretos del manifiesto (sin prefijo), según `select_discrete_attributes`.

    Se detectan desde los datos en vez de fijarlos a mano: así el código no depende de si el CSV
    trae `object_color_index`, `object_color` u `object_color_name`.
    """
    values = {}
    for c in df.columns:
        if not c.startswith(TEXT_PREFIX):
            continue
        dtype = df.schema[c]
        if dtype.is_integer() or dtype == pl.String:
            values[c[len(TEXT_PREFIX):]] = df[c].unique().to_list()
    return select_discrete_attributes(values, max_cardinality, exclude)
