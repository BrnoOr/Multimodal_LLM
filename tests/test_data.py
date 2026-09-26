"""Capa de datos: layout crudo de M3DI -> Parquet -> registros, sobre un mini-dataset sintético."""

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

from vlmfid.data import (
    ImageRecordDataset,
    build_split,
    disagreement,
    discrete_text_attributes,
    load_manifest,
    manifest_path,
)

ROOT = Path(__file__).resolve().parents[1]
N = 12
COLORS = ["tab:blue", "tab:red", "tab:green"]


def _make_split(d: Path, n: int = N, zero_pad: bool = False):
    (d / "images").mkdir(parents=True)
    (d / "text").mkdir()
    for i in range(n):
        name = f"{i:02d}" if zero_pad else str(i)          # sin ceros: el orden debe ser numérico
        Image.new("RGB", (224, 224), (i * 10, 0, 0)).save(d / "images" / f"{name}.png")
    img = {
        "": list(range(n)),                                  # índice de pandas sin nombre
        "object_shape": [i % 3 for i in range(n)],
        "object_xpos": [i % 3 for i in range(n)],
        "object_ypos": [(i + 1) % 3 for i in range(n)],
        "object_zpos": [0] * n,
        "object_alpharot": [0.1 * i for i in range(n)],
        "object_color": [0.07 * i for i in range(n)],        # tono continuo
    }
    pl.DataFrame(img).write_csv(d / "latents_image.csv")
    txt = {
        "object_shape": [i % 3 for i in range(n)],
        "object_xpos": [i % 3 for i in range(n)],
        "object_ypos": [(i + 1) % 3 if i != 5 else (i + 2) % 3 for i in range(n)],   # 1 desacuerdo
        "object_zpos": [0] * n,
        "object_color_name": [COLORS[i % 3] for i in range(n)],
        "text_phrasing": [i % 5 for i in range(n)],
    }
    pl.DataFrame(txt).write_csv(d / "latents_text.csv")
    (d / "text" / "text_raw.txt").write_text(
        "\n".join(f'A "{COLORS[i % 3]}" teapot number {i} is at the top-left of the image.' for i in range(n)) + "\n")


@pytest.fixture
def m3di(tmp_path):
    root = tmp_path / "m3di"
    _make_split(root / "train")
    _make_split(root / "test")
    return root


def test_build_split_schema(m3di):
    df = build_split(m3di, "test", verbose=False)
    assert df.height == N
    assert df.columns[:7] == ["id", "image_id", "image_path", "caption_ref", "split", "variant", "row_idx"]
    assert "" not in df.columns and "text_" + "" not in df.columns
    # orden numérico (10 va después de 9, no después de 1) y rutas relativas
    assert df["image_id"].to_list()[9:11] == ["9", "10"]
    assert df["image_path"][10] == "test/images/10.png"
    assert df["id"][10] == "base/test/10"
    assert df["caption_ref"][10].startswith('A "tab:red" teapot number 10')
    assert "text_text_phrasing" in df.columns                     # todas las columnas de texto se prefijan
    assert "object_shape" in df.columns and "text_object_shape" in df.columns


def test_count_mismatch_raises(m3di):
    p = m3di / "test" / "text" / "text_raw.txt"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="desalineación"):
        build_split(m3di, "test", verbose=False)


def test_disagreement_and_discrete_attributes(m3di):
    df = build_split(m3di, "test", verbose=False)
    rates = dict(disagreement(df).iter_rows())
    assert rates["object_shape"] == 0 and rates["object_ypos"] == pytest.approx(1 / N)
    attrs = discrete_text_attributes(df)
    assert "object_color_name" in attrs and "object_shape" in attrs
    assert "text_phrasing" not in attrs
    assert "object_zpos" not in attrs                               # constante: se excluye


def test_records_subset_and_dataset(m3di, tmp_path):
    out = tmp_path / "manifests"
    out.mkdir()
    build_split(m3di, "test", verbose=False).write_parquet(manifest_path("test", manifest_dir=out))
    rows = load_manifest("test", manifest_dir=out)
    assert len(rows) == N and rows[3]["id"] == "base/test/3"
    r = rows[4]
    assert r["caption"].startswith('A "tab:red"')
    assert r["latents_text"]["object_color_name"] == "tab:red"
    assert "object_shape" in r["latents_image"] and "text_phrasing" in r["latents_text"]
    assert not any(k.startswith("text_object") for k in r["latents_text"])   # prefijo removido
    a = load_manifest("test", limit=5, subset_seed=1, manifest_dir=out)
    b = load_manifest("test", limit=5, subset_seed=1, manifest_dir=out)
    assert [x["id"] for x in a] == [x["id"] for x in b]                       # determinista
    assert [x["image_id"] for x in a] == sorted((x["image_id"] for x in a), key=int)   # orden original
    json.dumps(a)                                                              # serializable

    ds = ImageRecordDataset(a, m3di)
    recs, imgs = ImageRecordDataset.collate([ds[i] for i in range(len(ds))])
    assert len(imgs) == 5 and imgs[0].mode == "RGB" and imgs[0].size == (224, 224)


def test_build_manifest_script(m3di, tmp_path):
    out = tmp_path / "manifests"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run(
        [sys.executable, "scripts/build_manifest.py", "--raw", str(m3di), "--out", str(out),
         "--variant", "ood_colors", "--report-disagreement"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "val] ausente" in r.stdout and "desacuerdo imagen vs texto" in r.stdout
    assert "object_color_name" in r.stdout                        # atributos detectados
    df = pl.read_parquet(out / "m3di_ood_colors_test.parquet")
    assert df["id"][0] == "ood_colors/test/0" and df["variant"].unique().to_list() == ["ood_colors"]
    assert (out / "m3di_ood_colors_train.parquet").exists()


def test_select_discrete_attributes_rules():
    from vlmfid.data import select_discrete_attributes

    names = [f"c{i}" for i in range(106)]                         # como los ~106 colores de M3DI
    values = {
        "object_shape": [0, 1, 2, 3, 4, 5, 6],
        "object_zpos": [0, 0, 0],                                  # constante
        "object_color_name": names,
        "object_color_index": list(range(106)),                    # redundante con _name
        "text_phrasing": [0, 1, 2],
        "object_color": [0.1, 0.5],                                # continuo
    }
    assert select_discrete_attributes(values) == ["object_shape", "object_color_name"]
