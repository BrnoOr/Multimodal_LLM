# data/

| Carpeta | Contenido | Git |
|---|---|---|
| `raw/m3di/` | M3DI publicado: `{train,val,test}/{images/, text/*.txt, latents_image.csv, latents_text.csv}` | ignorado |
| `generated/<variant>/` | variantes del generador (OOD, nuevas combinaciones, más/menos detalle), mismo layout | ignorado |
| `manifests/` | `m3di_<variant>_<split>.parquet`: id, ruta relativa, caption y latentes | versionado |
| `cache/` | artefactos recalculables (banco de embeddings de VL-JEPA) | ignorado |

Descarga del conjunto publicado (≈ unos GB):

```bash
cd data/raw && wget https://zenodo.org/record/7678231/files/m3di.tar.gz && tar -xzf m3di.tar.gz && rm m3di.tar.gz
cd ../.. && uv run python scripts/build_manifest.py --report-disagreement
```

Variantes: generar en `data/generated/<variant>/` con el mismo layout y luego

```bash
uv run python scripts/build_manifest.py --variant <variant>
```

Si el disco del repo no alcanza, extraer en otro lugar y definir `M3DI_ROOT` en `.env` (o `--raw` al construir y `data.root=` al inferir). Los manifiestos guardan rutas relativas a esa raíz, por lo que el mismo archivo sirve en laptop y cluster.
