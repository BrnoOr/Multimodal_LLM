# notebooks/

Análisis que requieren interpretación e interacción. Reglas:

1. **Solo leen** `data/` y `runs/`; no entrenan ni infieren. Todo lo que corre en GPU se lanza desde `scripts/` (con `scripts/launch.sh` en el cluster).
2. Importan desde el paquete (`from vlmfid.data import read_manifest`); si una función se repite en dos notebooks, pasa a `src/`.
3. Las figuras y tablas del informe se guardan en `reports/` con `vlmfid.reports` (`show(...)` y `keep(...)` en los notebooks 02 y 03; `SAVE = False` para explorar sin escribir). Se generan a partir de `runs/`, nunca a mano. Flujo hasta el commit: `reports/README.md`.
4. Kernel: `uv run python -m ipykernel install --user --name vlmfid`. En el cluster, abrir por VS Code Remote-SSH.

| Notebook | Propósito |
|---|---|
| `01_eda_m3di.ipynb` | distribución de factores, vocabulario por valor de atributo (insumo del extractor) |
| `02_resultados_etapa1.ipynb` | etapa 1 en VAL: líneas base, modelo × prompt, selección de prompt con bootstrap pareado, omisión vs error, confusiones, color, control bf16 (McNemar), texto vs fidelidad, piloto vs val completo, Y-Decoder genérico, VL-JEPA (zero-shot, intermedia, diagnóstico de cuantización, pérdida de información) y tabla resumen. Requiere `scripts/score.py` corrido sobre los runs |
| `03_resultados_etapa2.ipynb` | etapa 2 en VAL: humo (entrenables, memoria pico → P8), grilla LR × r × conector (ganador por modelo con `summarize_train.sweep_winners`, sensibilidad, robustez por bootstrap pareado), curvas del ajuste completo, zero-shot `p1_attributes` frente a ajustado (bootstrap y McNemar), omisiones y confusiones, control bf16 tras el ajuste, VL-JEPA (Y-Decoder de la etapa 2, pérdida de información etapa 1 vs 2), tabla resumen y, solo lectura, la tabla final de test. Las secciones sin runs se saltan |

Los notebooks anteriores a la reconstrucción están en `_para_borrar/notebooks/`.
