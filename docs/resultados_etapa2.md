# Resultados de la etapa 2 (fine-tuning QLoRA, VAL)

> Lo medido en la etapa 2, con su procedencia (run, job, fecha). Protocolo: `docs/metodologia.md` §2.3 y §4;
> decisiones: §6.1. Análisis: `notebooks/03_resultados_etapa2.ipynb`. Comparación válida: entre nuestros 4
> modelos bajo el mismo protocolo, no contra cifras de los papers (modelos de 7B o más) [CLAUDE.md §2].
>
> Estado (20-11-2026): prueba de humo hecha; grilla de hiperparámetros por lanzar.

## 1. Prueba de humo (paso 2.0, 20-11-2026)

6 pasos con el lote efectivo, los 4 modelos, NF4 y bf16 (`configs/queues/stage2_smoke.txt`). Los 3 VLM
corrieron en el job `s2_smoke`, con lote 128. En ese job, VL-JEPA dio OOM en NF4 y en bf16 (pidió 2.05 y 4.10 GiB
con ≈ 22 GB ocupados), y por eso **P8 = 64** para los cuatro (`metodologia.md` §6.1). Después corrió VL-JEPA con
lote 64 (job `s2_smoke_vj`). Tabla de `scripts/summarize_train.py` (`runs/_tablas/s2_smoke.csv`):

| Modelo | Precisión | Pérdida paso 1 → 6 | Pico de memoria (GB) | Métrica de selección | Paso 0 | Paso 6 |
|---|---|---|---|---|---|---|
| LLaVA-OV-0.5B | NF4 | 3.77 → 2.03 | 13.08 | `attr_mean` (32 img.) | 0.273 | 0.281 |
| LLaVA-OV-0.5B | bf16 | 3.78 → 1.96 | 12.15 | `attr_mean` | 0.219 | 0.297 |
| InternVL3-1B | NF4 | 4.71 → 1.78 | 14.78 | `attr_mean` | 0.305 | 0.242 |
| InternVL3-1B | bf16 | 4.66 → 1.79 | 12.71 | `attr_mean` | 0.242 | 0.266 |
| Qwen3.5-0.8B-Base | NF4 | 3.94 → 0.97 | 4.73 | `attr_mean` | 0.453 | 0.578 |
| Qwen3.5-0.8B-Base | bf16 | 3.96 → 0.99 | 5.52 | `attr_mean` | 0.484 | 0.586 |
| VL-JEPA | NF4 | 4.47 → 3.65 | 15.73 | InfoNCE val (128 img.) | 5.119 | 3.568 |
| VL-JEPA | bf16 | 4.33 → 3.64 | 21.09 | InfoNCE val | 5.010 | 3.575 |

- **La pérdida baja en los 8 runs.** El resto de las comprobaciones de la cola (capas con LoRA, grupos de LR)
  está en el log de cada run; los VLM en `s2_smoke` (lote 128) y VL-JEPA en `s2_smoke_vj` (lote 64).
- **Memoria:** con lote 64, VL-JEPA queda en 15.7 GB (NF4) y 21.1 GB (bf16). Cumple el criterio de < 22 GB,
  justo en bf16. Los VLM, con sus micro-lotes, quedan entre 4.7 y 14.8 GB.
- **`attr_mean` sobre 32 imágenes** es solo una comprobación de que el pipeline corre (± 0.08 de ruido binomial):
  no indica nada sobre el ajuste. El paso 0 es coherente con el zero-shot de `p1_attributes` de la etapa 1
  (val completo: Qwen3.5 0.416, LLaVA-OV 0.306, InternVL3 0.274).
- **VL-JEPA:** la InfoNCE en val parte por encima del azar (ln 64 ≈ 4.16; coherente con el diagnóstico,
  `resultados_etapa1.md` §7.4) y en 6 pasos baja a ≈ 3.57, **por debajo del azar**. El ajuste empieza a
  discriminar entre imágenes.

### Norma del gradiente (revisión de `max_grad_norm`, P5′)

Mediana y máximo de `grad_norm` (antes del recorte) en los 6 pasos:

| Modelo | NF4 (mediana / máx.) | bf16 (mediana / máx.) |
|---|---|---|
| LLaVA-OV-0.5B | 1.62 / 4.43 | 1.63 / 4.04 |
| InternVL3-1B | 3.09 / 8.89 | 2.86 / 9.69 |
| Qwen3.5-0.8B-Base | 2.86 / 5.96 | 2.61 / 5.61 |
| VL-JEPA | 4.97 / 9.10 | 4.93 / 8.47 |

Con `max_grad_norm` 0.3, el recorte actúa en todos los pasos de los 4 modelos. **Se mantiene 0.3**
(decisión del usuario, 20-11-2026). Con Adam, recortar siempre no reduce el tamaño del paso, porque el
optimizador divide por la escala del gradiente: lo que hace es normalizar la norma de cada paso. Es el mismo
régimen de QLoRA [8]. Con 1.0 el recorte también actuaría casi siempre, porque todas las medianas superan 1.
Son normas de los primeros 6 pasos, que suelen ser las más altas.
