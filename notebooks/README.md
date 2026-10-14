# notebooks/

Análisis que requieren interpretación e interacción. Reglas:

1. **Solo leen** `data/` y `runs/`; no entrenan ni infieren. Todo lo que corre en GPU se lanza desde `scripts/` (con `scripts/launch.sh` en el cluster).
2. Importan desde el paquete (`from vlmfid.data import read_manifest`); si una función se repite en dos notebooks, pasa a `src/`.
3. Las figuras del informe se generan con código en `src/` a partir de `runs/`, no a mano aquí.
4. Kernel: `uv run python -m ipykernel install --user --name vlmfid`. En el cluster, abrir por VS Code Remote-SSH.

| Notebook | Propósito |
|---|---|
| `00_eda_m3di.ipynb` | distribución de factores, vocabulario por valor de atributo (insumo del extractor) |
| `01_inspeccion_predicciones.ipynb` | comparación cualitativa entre runs, longitud de respuestas, prototipos de VL-JEPA |
| `legacy/` | notebooks de la etapa 1 anterior (EDA de predicciones, evaluación E2, métricas). Usan los manifiestos `m3di_<split>.parquet` y el formato de `exp_id` antiguos: sirven de referencia, no se ejecutan tal cual con los runs nuevos. |
