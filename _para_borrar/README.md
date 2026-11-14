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

## Pendiente de mover o reescribir (fase 5 y análisis)

Siguen en su lugar, pero **no funcionan** con la evaluación nueva hasta que se reescriban:

| Qué | Estado |
|---|---|
| `src/vlmfid/train/` (bucle, objetivos, `jepa.py`, `collate.py`, `evaluate.py`, `config.py`) | Etapa 2. Usa métricas que ya no existen (`macro`, `shape_pos`), el objetivo canónico y el banco de VL-JEPA. Se reescribe en la fase 5. |
| `src/vlmfid/data/targets.py`, `tests/test_targets.py` | Objetivo canónico y split composicional; se mueven aquí al reescribir el entrenamiento. |
| `src/vlmfid/eval/probe.py`, `src/vlmfid/eval/infoloss.py`, `scripts/probe.py`, `scripts/info_loss.py`, `tests/test_infoloss.py` | Importan categorías de color y el banco. Se reemplazan por las sondas de docs/metodologia.md §2.5. |
| `tests/test_train_e2e.py`, `tests/test_train_units.py`, `tests/_tiny_vlm.py` | Tests de la etapa 2; fallarán hasta la fase 5. |
