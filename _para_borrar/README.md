# _para_borrar/

Archivos retirados durante la reconstrucción. **No están borrados**: conservan su ruta original
bajo esta carpeta y el usuario decide cuándo eliminarlos. Motivo detallado en `docs/auditoria.md`.

| Qué | Motivo |
|---|---|
| `src/vlmfid/models/bank.py`, `configs/**/vljepa_fac.yaml`, `tests/test_vljepa_bank.py`, `docs/VLJEPA_AJUSTE_V3.pdf`, `reports/VLJEPA_finetuning_metodologia.pdf` | Banco factorizado de VL-JEPA: no proviene del paper y la evaluación por candidatos se reemplazó por el Y-Decoder (auditoría H5). |
| `configs/queues/`, `configs/experiments/` | Colas del protocolo anterior: corrían sobre test o dependían del objetivo canónico, el holdout o el banco (H1, H2, H5). |
| `notebooks/02`–`05`, `reports/{figures,tables}/{comparacion_test,eda_finetuning,eda_n10000,eda_top3}` | Resultados obtenidos sobre test, con métrica y objetivo que ya no valen (H1–H3). |
| `src/vlmfid/eval/copying.py`, `tests/test_copying.py`, `scripts/copy_report.py`, `scripts/probe_p4.py`, `configs/models/qwen35_instruct.yaml` | Diagnósticos fuera de alcance: tasa de copia p4/p4m y control Instruct (H9). |
| `docs/ETAPA2_FINETUNING.md`, `docs/ETAPA2B_REFORMULACION.md`, `docs/ESTRUCTURA_REPO.md` | Documentación de la metodología y estructura anteriores, reemplazada por `docs/metodologia.md`. |
| `src/vlmfid/eval/text_metrics.py` | Reimplementación de BLEU/ROUGE/CIDEr sin Java; se reemplazó por `pycocoevalcap` (H10). |
| `tests/test_attributes.py` | Tests de la métrica anterior (categorías de color, índice macro); reescritos. |
| `src/vlmfid/eval/probe.py`, `src/vlmfid/eval/infoloss.py`, `scripts/probe.py`, `scripts/info_loss.py` (versión anterior), `tests/test_infoloss.py` | Sondas lineales y pérdida de información basadas en el banco y en categorías de color. Reemplazadas por `src/vlmfid/eval/probes.py` y el nuevo `scripts/info_loss.py` (docs/metodologia.md §2.5). |
| `src/vlmfid/train/{config,collate,jepa,objectives,evaluate,loop,lora}.py`, `scripts/train.py`, `configs/train_anterior/`, `tests/test_train_{units,e2e}.py` (versiones anteriores) | Etapa 2 del protocolo anterior (objetivo canónico, split composicional, banco de VL-JEPA, métricas `macro`/`shape_pos`, test al final del entrenamiento). Reescritos en su lugar el 16-11-2026 según `docs/metodologia.md` §2.3. |
| `src/vlmfid/data/targets.py`, `tests/test_targets.py`, `notebooks/06_eda_splits.ipynb` | Objetivo canónico y split composicional, que el protocolo actual no usa (el objetivo es el caption original). |
| `configs/queues/stage2_sweep.txt` | Barrido de solo LR de la etapa 2. Reemplazado el 20-11-2026 por la grilla completa LR × r × multiplicador del conector (`configs/train/grid.yaml`, `configs/queues/stage2_grid.txt`), que incluye esas 12 corridas (r = 64, multiplicador 1). |

## Pendiente de mover o reescribir

Nada pendiente: la etapa 2 se reescribió el 16-11-2026 (`docs/plan_reconstruccion.md` §6.0b).
