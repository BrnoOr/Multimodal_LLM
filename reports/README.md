# reports/ — figuras y tablas del informe

Lo que el informe cita: resúmenes chicos, versionados con git. Los resultados completos (predicciones,
embeddings, adaptadores, logs) viven en `runs/` y `logs/`, que **no** se versionan (`.gitignore`) y solo existen
en el cluster.

| Carpeta | Contenido | Lo genera |
|---|---|---|
| `figures/etapa1/`, `tables/etapa1/` | zero-shot (val), Y-Decoder genérico, diagnóstico de VL-JEPA | notebook 02; `scripts/collect_reports.py` |
| `figures/etapa2/`, `tables/etapa2/` | humo, grilla, ajuste completo, val ajustado, pérdida de información | notebook 03; `scripts/collect_reports.py` |
| `tables/final/` | tabla de test (una sola vez) | `scripts/final_test.py` vía `collect_reports.py` |
| `figures/eda*/`, `tables/eda*/` | análisis de datos (notebook 01 y anteriores) | notebook 01 |
| `tables/etapa1_val.csv`, `etapa1_bf16.csv`, `score_*.txt` | copias anteriores, citadas en `docs/resultados_etapa1.md` | a mano |

Cada carpeta de etapa tiene un `PROCEDENCIA.json`: por archivo, el origen (sección del notebook o tabla de
`runs/_tablas/`), el commit con que se generó (y si había cambios sin commitear) y la fecha. Las figuras se guardan
en PNG (150 dpi) y PDF (vectorial, para el informe). Implementación: `src/vlmfid/reports.py`.

> El material anterior a la reconstrucción que pueda quedar aquí es histórico y **no** es resultado del protocolo
> actual (`docs/auditoria.md`).

## Flujo (cluster → laptop → git)

El flujo del repo es laptop → GitHub → cluster, y estos archivos nacen en el cluster. Se traen con `scp` y el
commit se hace en el laptop, como el resto:

1. **Cluster**, después de las corridas y de `score.py` / `summarize_train.py`:
   ```bash
   uv run --no-sync python scripts/collect_reports.py          # runs/_tablas -> reports/tables/<etapa>/
   ```
   y ejecutar los notebooks 02 y/o 03 de principio a fin (kernel `vlmfid`, VS Code Remote-SSH): guardan sus
   figuras en `reports/figures/<etapa>/` y sus tablas resumen en `reports/tables/<etapa>/`.
2. **Laptop** (terminal UCRT64, desde la raíz del repo):
   ```bash
   scp -r t7-vLLM@172.17.28.130:/mnt/home2/t7-vLLM/Multimodal_LLM/reports/figures/etapa2 reports/figures/
   scp -r t7-vLLM@172.17.28.130:/mnt/home2/t7-vLLM/Multimodal_LLM/reports/tables/etapa2 reports/tables/
   git add reports && git commit -m "reports(etapa2): figuras y tablas" && git push
   ```
   (igual con `etapa1` y `final`).
3. **Cluster:** `git pull` cuando no haya colas corriendo. Los archivos de `reports/` que generó el cluster están
   sin commitear allí, y git se niega a sobrescribirlos. Como son los mismos que acaban de llegar, se apartan antes
   del pull (no se borran):
   ```bash
   mkdir -p /tmp/reports_viejo && mv reports/figures/etapa2 reports/tables/etapa2 /tmp/reports_viejo/
   git checkout -- reports/ && git pull
   ```
