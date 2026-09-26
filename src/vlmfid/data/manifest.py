"""Manifiestos: índice único y versionado del dataset.

    data/manifests/<dataset>_<variant>_<split>.parquet      p. ej. m3di_base_test.parquet

Una fila por muestra: id, image_id, image_path (relativa a la raíz de datos de la variante),
caption_ref, split, variant, row_idx, latentes de imagen (sin prefijo) y de texto (prefijo text_).

El resto del código no lee el Parquet directamente: usa `load_manifest`, que devuelve registros
(dicts) con un esquema estable. Así, cambiar el formato en disco solo toca este módulo.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
from PIL import Image

from ..paths import GENERATED, M3DI_ROOT, MANIFESTS
from .m3di import TEXT_PREFIX

META_COLUMNS = {"id", "image_id", "image_path", "caption_ref", "split", "variant", "row_idx"}


def manifest_path(split: str, dataset: str = "m3di", variant: str = "base",
                  manifest_dir: str | Path = MANIFESTS) -> Path:
    return Path(manifest_dir) / f"{dataset}_{variant}_{split}.parquet"


def data_root(variant: str = "base", root: str | Path | None = None) -> Path:
    """Raíz contra la que se resuelven las `image_path` de una variante."""
    if root:
        return Path(root)
    return M3DI_ROOT if variant == "base" else GENERATED / variant


def read_manifest(split: str, dataset: str = "m3di", variant: str = "base",
                  manifest_dir: str | Path = MANIFESTS) -> pl.DataFrame:
    """El Parquet tal cual (útil en notebooks para análisis columnares)."""
    path = manifest_path(split, dataset, variant, manifest_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe: ejecuta `python scripts/build_manifest.py`")
    return pl.read_parquet(path)


def _to_python(v):
    # enteros guardados como float en el CSV -> int, para comparar sin sorpresas
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def to_records(df: pl.DataFrame) -> list[dict]:
    """Filas -> registros con latentes anidados:
    {id, image_id, image_path, caption, split, variant, latents_image{...}, latents_text{...}}"""
    img_cols = [c for c in df.columns if c not in META_COLUMNS and not c.startswith(TEXT_PREFIX)]
    txt_cols = [c for c in df.columns if c.startswith(TEXT_PREFIX)]
    out = []
    for row in df.iter_rows(named=True):
        out.append({
            "id": row["id"], "image_id": row["image_id"], "image_path": row["image_path"],
            "caption": row["caption_ref"], "split": row["split"], "variant": row["variant"],
            "latents_image": {c: _to_python(row[c]) for c in img_cols},
            "latents_text": {c[len(TEXT_PREFIX):]: _to_python(row[c]) for c in txt_cols},
        })
    return out


def load_manifest(split: str, dataset: str = "m3di", limit: int | None = None,
                  subset_seed: int = 0, variant: str = "base",
                  manifest_dir: str | Path = MANIFESTS) -> list[dict]:
    """Registros de un split. Con `limit`, subconjunto aleatorio *determinista* (misma semilla ->
    mismos ids en todos los modelos), devuelto en el orden original del manifiesto."""
    df = read_manifest(split, dataset, variant, manifest_dir)
    if limit and limit < df.height:
        idx = sorted(random.Random(subset_seed).sample(range(df.height), limit))
        df = df[idx]
    return to_records(df)


class ImageRecordDataset:
    """Dataset map-style (compatible con torch DataLoader): devuelve (registro, imagen PIL RGB).
    La colación deja listas: cada modelo hace su propio preprocesado."""

    def __init__(self, records: list[dict], root: str | Path = M3DI_ROOT):
        self.records = records
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        with Image.open(self.root / rec["image_path"]) as im:
            img = im.convert("RGB")
        return rec, img

    @staticmethod
    def collate(batch):
        recs, imgs = zip(*batch)
        return list(recs), list(imgs)
