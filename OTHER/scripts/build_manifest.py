"""Construye el manifiesto unico de Multimodal3DIdent a partir del layout crudo.

Layout esperado por split:
    <raw>/<split>/images/*.png
    <raw>/<split>/latents_image.csv   ground truth de la escena renderizada
    <raw>/<split>/latents_text.csv    latentes usados para generar el caption
    <raw>/<split>/text/*.txt          una descripcion por linea, alineada por indice

Salida:
    data/manifests/m3di_<split>.parquet

Uso:
    uv run python scripts/build_manifest.py --raw data/raw/m3di --out data/manifests
    uv run python scripts/build_manifest.py --raw data/raw/m3di --report-disagreement
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

SPLITS = ("train", "val", "test")

# Atributos que el caption de referencia describe (presentes en latents_text.csv)
DESCRIBED = ["object_shape", "object_xpos", "object_ypos", "object_zpos", "object_color"]
# Atributos presentes solo en la imagen: evaluables unicamente con prompt dirigido
NOT_DESCRIBED = [
    "object_alpharot",
    "object_betarot",
    "object_gammarot",
    "spotlight_pos",
    "spotlight_color",
    "background_color",
]


def find_caption_file(split_dir: Path) -> Path:
    txts = sorted(split_dir.joinpath("text").glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"sin .txt en {split_dir / 'text'}")
    if len(txts) > 1:
        print(f"  aviso: {len(txts)} archivos en text/, se usa {txts[0].name}")
    return txts[0]


def list_images(split_dir: Path) -> pl.DataFrame:
    """Imagenes ordenadas por el entero de su nombre, para alinear con el indice de fila."""
    imgs = sorted(
        split_dir.joinpath("images").glob("*.png"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or -1),
    )
    if not imgs:
        raise FileNotFoundError(f"sin .png en {split_dir / 'images'}")
    return pl.DataFrame(
        {
            "row_idx": list(range(len(imgs))),
            "image_id": [p.stem for p in imgs],
            "image_path": [str(p.resolve()) for p in imgs],
        }
    )


def build_split(raw: Path, split: str, variant: str) -> pl.DataFrame:
    d = raw / split
    print(f"[{split}]")

    lat_img = pl.read_csv(d / "latents_image.csv").with_row_index("row_idx")
    lat_txt = pl.read_csv(d / "latents_text.csv").with_row_index("row_idx")
    captions = find_caption_file(d).read_text(encoding="utf-8").splitlines()
    captions = [c.strip() for c in captions if c.strip()]
    imgs = list_images(d)

    n = {
        "latents_image": lat_img.height,
        "latents_text": lat_txt.height,
        "captions": len(captions),
        "images": imgs.height,
    }
    print(f"  filas: {n}")
    if len(set(n.values())) != 1:
        raise ValueError(f"desalineacion en {split}: {n}")

    cap = pl.DataFrame({"row_idx": list(range(len(captions))), "caption_ref": captions})

    # latents_text se prefija para conservar ambas fuentes sin colisionar
    lat_txt = lat_txt.rename(
        {c: f"text_{c}" for c in lat_txt.columns if c != "row_idx" and not c.startswith("text_")}
    )

    df = (
        imgs.join(lat_img, on="row_idx", how="inner")
        .join(lat_txt, on="row_idx", how="inner")
        .join(cap, on="row_idx", how="inner")
        .with_columns(
            pl.lit(split).alias("split"),
            pl.lit(variant).alias("variant"),
        )
    )

    front = ["image_id", "image_path", "caption_ref", "split", "variant", "row_idx"]
    return df.select(front + [c for c in df.columns if c not in front])


def report_disagreement(df: pl.DataFrame, split: str) -> None:
    """Cuanto difieren los latentes de imagen y los de texto.

    Define el techo alcanzable: si el caption dice un color que la imagen no tiene,
    ningun modelo fiel a la imagen puede acertar ambas metricas a la vez.
    """
    print(f"[{split}] desacuerdo imagen vs texto")
    pairs = [
        ("object_shape", "text_object_shape"),
        ("object_xpos", "text_object_xpos"),
        ("object_ypos", "text_object_ypos"),
        ("object_zpos", "text_object_zpos"),
    ]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            rate = df.select((pl.col(a) != pl.col(b)).mean()).item()
            print(f"  {a:20s} {rate:6.2%}")
    if "text_object_color_name" in df.columns:
        print(f"  colores nominales: {sorted(df['text_object_color_name'].unique().to_list())}")
    if "text_text_phrasing" in df.columns:
        print(f"  plantillas de caption: {df['text_text_phrasing'].n_unique()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/m3di"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    ap.add_argument(
        "--variant", default="base", help="etiqueta de variante (base, ood_colors, ...)"
    )
    ap.add_argument("--report-disagreement", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        if not (args.raw / split).is_dir():
            print(f"[{split}] ausente, se omite")
            continue
        df = build_split(args.raw, split, args.variant)
        path = args.out / f"m3di_{split}.parquet"
        df.write_parquet(path)
        print(f"  -> {path}  ({df.height} filas, {df.width} columnas)")
        if args.report_disagreement:
            report_disagreement(df, split)

    print("\nAtributos descritos en el caption:", DESCRIBED)
    print("Atributos solo en la imagen     :", NOT_DESCRIBED)


if __name__ == "__main__":
    main()
