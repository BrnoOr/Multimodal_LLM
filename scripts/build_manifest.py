"""Construye el manifiesto único de Multimodal3DIdent a partir del layout crudo.

Layout esperado por split (ver src/vlmfid/data/m3di.py):
    <raw>/<split>/images/*.png
    <raw>/<split>/latents_image.csv   ground truth de la escena renderizada
    <raw>/<split>/latents_text.csv    latentes usados para generar el caption
    <raw>/<split>/text/*.txt          una descripción por línea, alineada por índice

Salida:
    data/manifests/m3di_<variant>_<split>.parquet     (image_path relativa a --raw)

Uso:
    uv run python scripts/build_manifest.py                                    # base, data/raw/m3di
    uv run python scripts/build_manifest.py --report-disagreement
    uv run python scripts/build_manifest.py --variant ood_colors               # data/generated/ood_colors
    uv run python scripts/build_manifest.py --raw /otra/ruta/m3di --splits test

La lógica vive en `vlmfid.data` (importable desde tests, notebooks e infer); este script solo
parsea argumentos, escribe los archivos e imprime el diagnóstico.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vlmfid.data import (
    DESCRIBED,
    NOT_DESCRIBED,
    SPLITS,
    build_split,
    data_root,
    disagreement,
    discrete_text_attributes,
    manifest_path,
)
from vlmfid.paths import MANIFESTS


def report(df, split: str) -> None:
    print(f"[{split}] desacuerdo imagen vs texto")
    for row in disagreement(df).iter_rows(named=True):
        print(f"  {row['attribute']:20s} {row['rate']:6.2%}")
    if "text_object_color_name" in df.columns:
        print(f"  colores nominales: {sorted(df['text_object_color_name'].unique().to_list())}")
    if "text_text_phrasing" in df.columns:
        print(f"  plantillas de caption: {df['text_text_phrasing'].n_unique()}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, default=None,
                    help="raíz del layout crudo (defecto: M3DI_ROOT para base, data/generated/<variant> si no)")
    ap.add_argument("--out", type=Path, default=MANIFESTS)
    ap.add_argument("--variant", default="base", help="etiqueta de variante (base, ood_colors, ...)")
    ap.add_argument("--dataset", default="m3di")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--report-disagreement", action="store_true")
    args = ap.parse_args(argv)

    raw = args.raw or data_root(args.variant)
    print(f"raw={raw}  variant={args.variant}  out={args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    attrs = None
    for split in args.splits:
        if not (raw / split).is_dir():
            print(f"[{split}] ausente, se omite")
            continue
        df = build_split(raw, split, args.variant)
        path = manifest_path(split, args.dataset, args.variant, args.out)
        df.write_parquet(path)
        print(f"  -> {path}  ({df.height} filas, {df.width} columnas)")
        attrs = attrs or discrete_text_attributes(df)
        if args.report_disagreement:
            report(df, split)

    print("\nAtributos descritos en el caption:", DESCRIBED)
    print("Atributos solo en la imagen     :", NOT_DESCRIBED)
    if attrs is not None:
        print("Latentes de texto discretos     :", attrs, " <- base de la exactitud por atributo")


if __name__ == "__main__":
    main()
