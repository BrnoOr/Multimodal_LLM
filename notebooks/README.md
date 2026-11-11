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
| `05_comparacion_test.ipynb` | test (10 000 imágenes): zero-shot (p1 y mejor prompt) frente a ajustado, entre modelos; ganancia, ranking, desglose por factor, aciertos por imagen, controles y caption generado vs original y canónico (BLEU-4, ROUGE-L, CIDEr-D) |
| `06_eda_splits.ipynb` | EDA comparativo de train / val / test: integridad, fuga (escenas y captions), homogeneidad de atributos, combinaciones forma × posición y holdout composicional, dependencia matiz ↔ x, textos objetivo y líneas base de azar. Escribe en `reports/{figures,tables}/eda_splits/` |
