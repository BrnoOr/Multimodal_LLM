# Migración desde la estructura anterior

Ejecutar desde la raíz del repo, en una rama nueva (`git switch -c reorg`).

| Antes | Ahora | Nota |
|---|---|---|
| `scripts/build_manifest.py` (Parquet, rutas absolutas) | `scripts/build_manifest.py` + lógica en `src/vlmfid/data/` | rutas relativas, `id` con variante, archivo `m3di_<variant>_<split>.parquet` |
| `scripts/infer.py`, `scripts/run.sh` + tmux | `scripts/infer.py` + `launch.sh` · `jobs.sh` · `run_queue.sh` | jobs desacoplados de SSH, logs en `logs/` |
| `scripts/train.py`, `evaluate.py`, `make_figures.py` | `scripts/` (pendientes) | mismo patrón: lógica en `src/vlmfid/`, script delgado |
| `src/vlmfid/models/{llava,qwen,vljepa}.py` | `models/{hf_generative,qwen_vl,vljepa}.py`; cada modelo es un YAML | LLaVA-OneVision-0.5B, InternVL3-1B, Qwen3.5-0.8B-Base |
| `external/vl-jepa` | `external/open-vljepa` | submodule `dion-jy/open-vljepa` |
| `configs/prompts.yaml` | sin cambios (catálogo p0–p4) | |
| `data/manifests/m3di_<split>.parquet` | `data/manifests/m3di_base_<split>.parquet` | regenerar: las rutas pasan a ser relativas |
| `reports/` | fuera del alcance de esta reorganización | |

## Comandos

```bash
# 1. copiar el contenido de este bundle sobre el repo
# 2. submodule VL-JEPA
git submodule deinit -f external/vl-jepa 2>/dev/null; git rm -f external/vl-jepa 2>/dev/null
rm -rf .git/modules/external/vl-jepa
git submodule add https://github.com/dion-jy/open-vljepa external/open-vljepa
git -C external/open-vljepa checkout f92107fa3d188d69e2fd8ca0222c06483b09a518
# 3. manifiestos con el nuevo esquema
rm -f data/manifests/m3di_train.parquet data/manifests/m3di_val.parquet data/manifests/m3di_test.parquet
uv lock && uv sync --extra quant --extra eval --extra dev
uv run python scripts/build_manifest.py --report-disagreement
# 4. verificación
uv run pytest -q
git add -A && git commit -m "Reorganiza repo: scripts/ como puntos de entrada, logs/, manifiestos Parquet"
```
