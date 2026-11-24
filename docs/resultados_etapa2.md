# Resultados de la etapa 2 (fine-tuning QLoRA, VAL)

> Lo medido en la etapa 2, con su procedencia (run, job, fecha). Protocolo: `docs/metodologia.md` §2.3 y §4;
> decisiones: §6.1. Análisis: `notebooks/03_resultados_etapa2.ipynb`. Comparación válida: entre nuestros 4
> modelos bajo el mismo protocolo, no contra cifras de los papers (modelos de 7B o más) [CLAUDE.md §2].
>
> Estado (23-11-2026): humo, grilla y ajuste completo de los 4 hechos; val completo de los 3 VLM ajustados hecho;
> VL-JEPA en val completo (B7) y pérdida de información (B8) por lanzar.

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

## 2. Grilla de hiperparámetros de los VLM (paso B4a, job `s2_grid_vlm`, 20/21-11-2026)

36 corridas NF4 (3 modelos × LR {1e-4, 2e-4, 5e-4} × r {16, 64} × conector {×1, ×5}), 150 pasos con lote 64,
evaluación en las 500 imágenes de val en los pasos 0, 50, 100 y 150 (`configs/train/grid.yaml`). Las 36 terminaron
(`status: completed`). Tabla completa: `runs/_tablas/s2_grid.csv` (`scripts/summarize_train.py --table s2_grid`).
Análisis: notebook 03 §2–3.

`attr_mean` en el paso 150 (regla de la grilla; G2):

| LR · r · conector | InternVL3-1B | LLaVA-OV-0.5B | Qwen3.5-0.8B-Base |
|---|---|---|---|
| 1e-4 · 16 · ×1 | 0.7925 | 0.7825 | **0.7930** |
| 1e-4 · 16 · ×5 | **0.7995** | 0.7890 | 0.7920 |
| 1e-4 · 64 · ×1 | 0.7885 | 0.7825 | 0.7915 |
| 1e-4 · 64 · ×5 | 0.7975 | 0.7910 | 0.7930 |
| 2e-4 · 16 · ×1 | 0.7855 | 0.7795 | 0.7910 |
| 2e-4 · 16 · ×5 | 0.7920 | 0.7915 | 0.7860 |
| 2e-4 · 64 · ×1 | 0.7875 | 0.7830 | 0.7890 |
| 2e-4 · 64 · ×5 | 0.7950 | **0.7935** | 0.7825 |
| 5e-4 · 16 · ×1 | 0.7945 | 0.7880 | 0.7850 |
| 5e-4 · 16 · ×5 | 0.7935 | 0.7845 | 0.7785 |
| 5e-4 · 64 · ×1 | 0.7945 | 0.7895 | 0.7855 |
| 5e-4 · 64 · ×5 | 0.7905 | 0.7885 | 0.7765 |
| paso 0 (zero-shot, p1_attributes) | 0.284 | 0.303 | 0.429 |

**Ganadores** (anotados en `configs/train/models/<modelo>.yaml`):

| Modelo | LR | r | Conector | `attr_mean` (paso 150) | Segunda combinación |
|---|---|---|---|---|---|
| InternVL3-1B | 1e-4 | 16 | ×5 | 0.7995 | 1e-4 · 64 · ×5, 0.7975 |
| LLaVA-OV-0.5B | 2e-4 | 64 | ×5 | 0.7935 | 2e-4 · 16 · ×5, 0.7915 |
| Qwen3.5-0.8B-Base | 1e-4 | 16 | ×1 | 0.7930 | 1e-4 · 64 · ×5, 0.7930 (**empate**; G4) |

Lectura (lo medido y lo que es estimación):

- **Las diferencias entre combinaciones son pequeñas.** Dentro de cada modelo, el rango de las 12 va de 0.014
  (InternVL3, LLaVA-OV) a 0.017 (Qwen3.5), y el ganador supera a la segunda por 0.000–0.002. Como referencia
  (estimación, no medida): con 500 imágenes y p ≈ 0.8, el error estándar binomial de la exactitud de un atributo es
  ≈ 0.018. La comparación entre combinaciones es pareada (mismas imágenes), lo que reduce ese error, pero el
  ganador **no se distingue con seguridad** de varias alternativas. La robustez se cuantifica con el bootstrap
  pareado del notebook 03 §3 (`val_predictions` de cada corrida). Para el ajuste completo esto es poco grave:
  cualquier combinación cercana al máximo daría resultados equivalentes.
- **Qwen3.5:** empate exacto en 0.7930; se resolvió con la regla G4 (parsimonia), decidida por el usuario el
  22-11-2026, después de ver la tabla. Se declara así.
- **El ajuste satura pronto.** En 19 de las 36 corridas el mejor paso evaluado es el 50 (3 200 imágenes) y la pérdida de
  entrenamiento final es ≈ 0.19–0.23 en todas. De 0.28–0.43 (zero-shot) se pasa a ≈ 0.79 en los tres modelos,
  que quedan a ≤ 0.02 entre sí. La época completa (1 954 pasos) dirá si ese techo se mueve.
- **Tendencias débiles:** el conector ×5 gana en InternVL3 y LLaVA-OV, y en Qwen3.5 LR 5e-4 es lo peor. Con el
  ruido anterior son indicios, no conclusiones.
- **Limitación declarada (G2):** cada combinación se compara tras un schedule corto completo (LR recocido a 0 en
  150 pasos), no dentro de una época.

## 3. Grilla de VL-JEPA (paso B4b, job `s2_grid_vljepa`, 21/22-11-2026)

12 corridas NF4, mismo protocolo; métrica: InfoNCE en val (500 imágenes, lote 64; menor es mejor; azar ≈ ln 64 =
4.16). Las 12 terminaron; el paso 0 da 4.382 en todas. Tabla: `runs/_tablas/s2_grid_vljepa.csv`.

| LR · r | ×1 | ×5 |
|---|---|---|
| 1e-4 · 16 | 1.607 | 1.237 |
| 1e-4 · 64 | 1.615 | 1.240 |
| 2e-4 · 16 | 1.030 | 0.927 |
| 2e-4 · 64 | 1.011 | 0.891 |
| 5e-4 · 16 | 0.641 | 0.853 |
| 5e-4 · 64 | **0.633** | 0.781 |

**Ganador:** lr 5e-4 · r64 · ×1 (`configs/train/models/vljepa.yaml`). Lectura:

- **A diferencia de los VLM, aquí la grilla separa con claridad**: la InfoNCE va de 1.6 a 0.63, y el mejor paso es
  el 150 en las 12 corridas (no hay meseta en 150 pasos).
- **El LR domina.** El rango casi no importa (r16 frente a r64: ≤ 0.07). El multiplicador del conector interactúa
  con el LR: ×5 ayuda con LR 1e-4 y 2e-4 y empeora con 5e-4. Lectura posible (no comprobada): el conector
  rinde mejor con un LR efectivo de ≈ 5e-4–1e-3, que se alcanza con 1e-4 · ×5 o con 5e-4 · ×1, y 2.5e-3 (5e-4 · ×5)
  ya es demasiado.
- **El ganador está en el borde del eje de LR** (G5): podría haber un óptimo por encima de 5e-4, o la grilla corta
  podría favorecer el LR alto porque avanza más en 150 pasos. Se acepta y se declara (usuario, 22-11-2026).
- La InfoNCE de val mide discriminación en el espacio de embeddings, no la exactitud por atributo: el número
  comparable con los VLM sale de B7, con el Y-Decoder de la etapa 2.

## 4. Ajuste completo de los VLM (paso B5, job `s2_train_vlm`, 21/22-11-2026)

1 época (1 954 pasos, lote 64) con el ganador de cada modelo; NF4 y control bf16. Las 6 terminaron
(`runs/_tablas/s2_train.csv`). `attr_mean` en las 500 imágenes de val (P1):

| Modelo | Precisión | Paso 0 | Paso 1 954 | Mejor paso | Mejor valor | Pico (GB) |
|---|---|---|---|---|---|---|
| InternVL3-1B | NF4 | 0.284 | 0.8115 | 400 | 0.8175 | 14.35 |
| InternVL3-1B | bf16 | 0.229 | 0.8245 | 1 954 | 0.8245 | 12.27 |
| LLaVA-OV-0.5B | NF4 | 0.303 | 0.8180 | 1 400 | 0.8245 | 13.08 |
| LLaVA-OV-0.5B | bf16 | 0.275 | 0.8155 | 400 | 0.8305 | 12.15 |
| Qwen3.5-0.8B-Base | NF4 | 0.429 | 0.8200 | 1 954 | 0.8200 | 4.24 |
| Qwen3.5-0.8B-Base | bf16 | 0.455 | 0.8270 | 400 | 0.8270 | 5.03 |

- **La época completa sube ≈ 0.02–0.03 sobre la grilla** (≈ 0.79–0.80 en 150 pasos), y los tres modelos quedan
  entre 0.82 y 0.83. La curva se aplana pronto: en 3 de 6 el mejor paso es el 400 (≈ 20 % de la época).
- **El "mejor valor" es optimista**: es el máximo de 11 evaluaciones sobre las mismas 500 imágenes con que se
  elige. El número que se reporta en val es el de las 10 000 (B7, P1).
- **bf16 frente a NF4:** en el mejor valor, bf16 queda 0.006–0.007 arriba en los tres. Es del orden del ruido de 500
  imágenes; B7 (10 000) dirá si hay una diferencia real.
- **El paso 0 difiere entre NF4 y bf16** (p. ej. InternVL3 0.284 frente a 0.229): es el zero-shot de cada precisión
  sobre 500 imágenes, coherente con el control bf16 de la etapa 1.

## 5. Ajuste completo de VL-JEPA (paso B5, job `s2_cadena`, 23-11-2026)

1 época (1 954 pasos, lote 64) con el ganador de la grilla (lr 5e-4 · r64 · ×1). Métrica de selección: InfoNCE
bidireccional en las 500 imágenes de val (P7; azar ≈ ln 64 = 4.16). `runs/_tablas/s2_train_vj.csv`.

| Precisión | Pérdida train (primera → última) | InfoNCE val, paso 0 | InfoNCE val, paso 1 954 (= mejor) | Pico (GB) | Duración |
|---|---|---|---|---|---|
| NF4 | 4.133 → 0.177 | 4.382 | **0.2334** | 15.73 | 8 666 s |
| bf16 | 4.062 → 0.179 | 4.306 | **0.2347** | 21.09 | 8 948 s |

- **El ajuste lleva a VL-JEPA de no discriminar (InfoNCE sobre el azar) a discriminar con claridad** entre las 64
  imágenes de cada lote. La grilla, con 150 pasos, había llegado a 0.63.
- **El mejor paso es el último en ambas precisiones**: a diferencia de los VLM, no hay meseta dentro de la época.
  Más pasos probablemente seguirían bajando la InfoNCE; el presupuesto es el mismo para los cuatro (1 época).
- **Control bf16 (P10′, solo en embeddings): NF4 y bf16 quedan prácticamente iguales** (0.2334 frente a 0.2347).
  Tras el ajuste, la desventaja de NF4 que dejaba el Y-Encoder cuantizado en zero-shot (coseno 0.876 contra bf16,
  `resultados_etapa1.md` §7.4) no se ve en la InfoNCE. Es una comparación sobre 500 imágenes y una sola semilla.
- La InfoNCE mide discriminación en el espacio de embeddings, no exactitud por atributo: la cifra comparable con los
  VLM sale de B7 (val completo con el Y-Decoder de la etapa 2).

Y-Decoder de la etapa 2 (B6, 20 épocas sobre el Y-Encoder ajustado): terminó (código 0, 2 988 s); su curva y su
mejor paso, pendientes de revisar (`runs/ydec_vljepa_open_msrvtt_nf4_ft_nf4_k8_r64_e20_cont`).

## 6. Val completo de los VLM ajustados (paso B7, job `s2_cadena`, 23-11-2026)

`adapter_best` de cada ajuste sobre las 10 000 imágenes de val, prompt `p1_attributes`, mismo `infer.py` y
`score.py` que la etapa 1 (P1). `runs/_tablas/etapa2_val.csv`. 0 respuestas cortadas; mediana de 20–22 tokens.

| Modelo | Precisión | forma | x | y | color | attr_mean | Bleu-4 | ROUGE-L | CIDEr | Zero-shot p1 (NF4) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | NF4 | 0.999 | 1.000 | 0.999 | 0.266 | **0.8159** | 0.467 | 0.560 | 2.485 | 0.416 |
| | bf16 | 0.994 | 1.000 | 0.983 | 0.298 | 0.8186 | 0.420 | 0.543 | 2.716 | |
| InternVL3-1B | NF4 | 0.996 | 1.000 | 0.987 | 0.278 | **0.8152** | 0.421 | 0.543 | 2.718 | 0.274 |
| | bf16 | 1.000 | 1.000 | 1.000 | 0.249 | 0.8120 | 0.468 | 0.560 | 2.442 | |
| LLaVA-OV-0.5B | NF4 | 1.000 | 1.000 | 1.000 | 0.259 | **0.8147** | 0.461 | 0.557 | 2.462 | 0.306 |
| | bf16 | 1.000 | 1.000 | 0.995 | 0.310 | 0.8259 | 0.410 | 0.560 | 2.479 | |

Lectura (lo medido; las pruebas pareadas están en el notebook 03 §5 y §7):

- **Forma y posición quedan resueltas** (≥ 0.983 en las seis condiciones; ≥ 0.987 en NF4). **Todo el error que
  queda es de color** (0.25–0.31).
- **El color está acotado por el dataset.** El nombre depende de una paleta elegida al azar que la imagen no
  muestra; el mejor lector posible acierta 0.410 (techo del color, `resultados_etapa1.md` §2). Con forma y posición
  perfectas, el techo de `attr_mean` es (1 + 1 + 1 + 0.410) / 4 ≈ 0.853. Los VLM ajustados quedan a 0.03–0.04 de ese
  techo. Los `peek` muestran el mecanismo: aciertan el matiz y eligen un nombre de una paleta frecuente (sobre todo
  `tab:`: cyan, olive, red), que coincide con la referencia solo cuando el caption usó esa misma paleta.
- **Los tres VLM quedan empatados en NF4** (0.8147–0.8159, a 0.0012 entre sí). El ajuste borra la diferencia del
  zero-shot (0.27–0.42): +0.40 a +0.54 de `attr_mean`.
- **El mejor valor en 500 imágenes era optimista** en 0.002–0.010 (NF4: 0.8175 / 0.8245 / 0.8200 en el ajuste,
  frente a 0.8152 / 0.8147 / 0.8159 aquí), como se esperaba de elegir el máximo de 11 evaluaciones.
- **bf16 frente a NF4** (attr_mean): LLaVA-OV +0.011, Qwen3.5 +0.003, InternVL3 −0.003. Las diferencias están en el
  color (LLaVA-OV 0.310 frente a 0.259); forma y posición no cambian. Su significancia, con McNemar por atributo, en
  el notebook 03 §7.
- **Las métricas de texto no ordenan igual que la fidelidad**: InternVL3 NF4 tiene el CIDEr más alto (2.718) con la
  misma `attr_mean` que LLaVA-OV NF4 (2.462). Dependen de cuál de las 5 plantillas de M3DI imita cada modelo
  (InternVL3 usa casi siempre "At the … of the picture is a …"), no de si la descripción es correcta.
