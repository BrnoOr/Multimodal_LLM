# notebooks/

Análisis que requieren interpretación e interacción. Reglas:

1. **Solo leen** `data/` y `runs/`; no entrenan ni infieren. Todo lo que corre en GPU se lanza desde `scripts/` (con `scripts/launch.sh` en el cluster).
2. Importan desde el paquete (`from vlmfid.data import read_manifest`); si una función se repite en dos notebooks, pasa a `src/`.
3. Las figuras del informe se generan con código en `src/` a partir de `runs/`, no a mano aquí.
4. Kernel: `uv run python -m ipykernel install --user --name vlmfid`. En el cluster, abrir por VS Code Remote-SSH.

| Notebook | Propósito |
|---|---|
| `01_eda_m3di.ipynb` | distribución de factores, vocabulario por valor de atributo (insumo del extractor) |
| `02_resultados_etapa1.ipynb` | etapa 1 en VAL: líneas base, modelo × prompt, selección de prompt con bootstrap pareado, omisión vs error, confusiones, color, control bf16 (McNemar), texto vs fidelidad, piloto vs val completo, Y-Decoder genérico, VL-JEPA (zero-shot, intermedia, diagnóstico de cuantización, pérdida de información) y tabla resumen. Requiere `scripts/score.py` corrido sobre los runs |

Los notebooks anteriores a la reconstrucción están en `_para_borrar/notebooks/`.
