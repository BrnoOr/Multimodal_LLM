# Etapa 2 reformulada — cambios al ajuste fino

**Fecha:** 23-10-2026. **Rama:** `fine-tuning`.

Este documento registra **todos** los cambios hechos al ajuste fino respecto del diseño de [`ETAPA2_FINETUNING.md`](ETAPA2_FINETUNING.md), con su justificación. Ese documento sigue siendo válido para todo lo que aquí no se menciona: LoRA + conector, NF4, lote efectivo 64, lr, schedule, reanudación, etc.

Al momento de este cambio **no se había ejecutado ningún entrenamiento**: no había `runs/train_*` ni logs de la etapa 2. Por lo tanto, no se invalida ningún resultado.

> Nada de esto se ejecutó en el laptop: tests, entrenamientos y el notebook se corren en el cluster (§7). La primera verificación son los tests y `stage2b_smoke`.

---

## 1. Punto de partida: dónde está el error de la etapa 1

Zero-shot en test (n = 10 000; `reports/tables/eda_top3/`), con el mejor prompt de cada modelo:

| Atributo | Qwen3.5 Base · p1 | LLaVA-OV · p3 | InternVL3 · p1 | VL-JEPA · p4m |
|---|---|---|---|---|
| Forma | 0,17 | 0,38 | 0,21 | 0,27 |
| Posición (x, y) | 0,43 | 0,11 | 0,13 | 0,16 |
| Color ±1 | 0,98 | 0,91 | 0,96 | 0,45 |
| Nombre exacto | 0,06 | 0,05 | 0,05 | 0,02 |
| Índice macro | 0,527 | 0,468 | 0,433 | 0,291 |

Hay tres lecturas:

1. **Color ±1 está saturado** en los VLM autorregresivos. En el Índice macro pesa 1/3 y casi no varía entre ellos.
2. **El nombre exacto tiene un techo bajo**: el nombre de color no es función del tono (las reglas del EDA lo reproducen en ~27 %).
3. **El margen está en forma y posición.** Ahí tiene que actuar el ajuste.

El diseño original entrenaba con el caption de M3DI, que trae 5 plantillas al azar y ~100 nombres de color. Seleccionaba con el Índice macro y producía un solo número antes/después por modelo. Ese número no separa las dos causas que pide el proyecto (`docs/contexto_proyecto.md`): **falta de conocimiento** frente a **desalineamiento de formato**.

## 2. Resumen de cambios

| # | Aspecto | Antes | Ahora | Por qué |
|---|---|---|---|---|
| 1 | Objetivo de entrenamiento | Caption original | **Caption canónico** (`target.format: canonical`); el original queda como control | La plantilla y el nombre exacto son ruido que no depende de la imagen (§3.1) |
| 2 | Prompt de VL-JEPA | p4m (se truncaba a 64 tokens) | **p1**, igual que los otros tres | Los 6 prompts quedaban dentro del ruido en VL-JEPA; p1 cabe entero (§3.2) |
| 3 | Métrica de selección | Índice macro | **`shape_pos`** = (forma + posición)/2 | Color ±1 saturado diluía la señal (§3.3) |
| 4 | Negativos de VL-JEPA | Las 63 otras muestras del lote | **Todo el banco** de objetivos únicos (`objective.negatives: bank`) | Alinea el entrenamiento con la decodificación por recuperación (§3.4) |
| 5 | Datos de train | `train_limit` para el humo; `drop_last` descartaba 8 imágenes | **Train completo en todos los runs**: sin `train_limit` (se rechaza) y `drop_last: false` (1954 pasos) | Ningún entrenamiento usa menos datos que el split completo (§3.5) |
| 6 | Experimentos | 1 run por modelo | Principal + **split composicional** + **sondas lineales** + controles | Separar formato de percepción y medir generalización (§4) |
| 7 | Métricas | 8 | + `shape_pos`, `macro_cat` | Sin cambiar las 8 de la etapa 1 (§3.3) |
| 8 | exp_id | `train_<m>_<q>_<p>_r16_e1[_n]` | `train_<m>_<q>_<p>_t<can\|orig>_r16_e<E>[_hc][_s<seed>][_m<max_steps>]` | Distingue objetivo, holdout, semilla y humo |
| 9 | Colas | `stage2_{smoke,train,test}` | `stage2b_{smoke,probe,main,compositional,ablations,test,probe_ft}` | Una por experimento |

**No cambia:** los modelos, NF4 + bf16, LoRA r=16/α=32 en el LM o predictor, el conector entrenable en fp32, el lote efectivo 64, los lr (2e-4 / 2e-5), el calentamiento + coseno, las 500 imágenes de val, test intacto hasta el final, la resolución, la plantilla de chat, la generación greedy y el Y-Encoder congelado.

---

## 3. Detalle de cada cambio

### 3.1 Objetivo canónico

```
A {color} {shape} is at the {vertical}-{horizontal} of the image.
```

- `{color}` es la **categoría básica del tono latente** de la imagen: `hue_to_cat(latents_image.object_color)`, una de red, orange, yellow, green, cyan, blue, purple o pink. Es exactamente la verdad contra la que la evaluación mide "color". No se usa el nombre del caption, porque no es función del tono.
- `{shape}`, `{vertical}` y `{horizontal}` salen de los latentes de imagen: teapot…head, top/mid/bottom, left/center/right. El EDA mostró 0 % de desacuerdo imagen↔texto.
- Es la plantilla 2 de M3DI (y la de `p2_constrained`) con el nombre entre comillas reemplazado por la categoría. Hay **504 captions distintos** (7 × 9 × 8), frente a 10 936 en el original.
- El artículo es siempre "A" ("A orange …"). Es intencional: una sola plantilla y ningún token que dependa del color.

**Consecuencias:**
- El extractor de la etapa 1 recupera forma, posición y color del caption canónico con un **100 %** de acierto (`tests/test_targets.py`). El techo de cada métrica es 1, salvo **nombre exacto, que queda en 0**: el modelo canónico nunca dice el nombre literal. Hay que reportarlo así y no como una regresión.
- BLEU/ROUGE/CIDEr frente a la referencia original bajan por construcción: el texto es otro. Las métricas de texto de los runs canónicos no son comparables con las de la etapa 1, y la comparación se apoya en la exactitud por atributo. El control `target.format=original` sí es comparable en texto.
- La entropía cruzada ya no tiene el piso de log 5 por la plantilla (§3.2 del documento original). Así, la `val_loss` vuelve a ser informativa.

**Control:** `stage2b_ablations.txt` entrena los 4 modelos con `target.format=original`. La diferencia canónico − original, con todo lo demás igual, mide **cuánto costaba aprender un formato arbitrario**.

**Código:** `src/vlmfid/data/targets.py` (`canonical_caption`, `target_text`, `target_fn`). `collate.GenerativeFeaturizer` y `collate.JEPAFeaturizer` reciben `target_fn`. `loop.Trainer` lo construye desde `cfg.target.format`.

### 3.2 Prompt p1 en los cuatro modelos

En la etapa 1, los 6 prompts de VL-JEPA quedaron entre 0,285 y 0,291 de Índice macro, dentro del ruido. p4m, el elegido por la regla "mejor zero-shot", tiene ≈ 80-90 tokens de Llama-3 y `max_query_len = 64`, así que se truncaba. p1 cabe entero (`status.json → model_info.query_truncated` lo confirma en el humo), y con él **el prompt es idéntico en los 4 modelos**: una variable menos en la comparación.

**Código:** `configs/train/models/vljepa.yaml` (`prompt: p1_attributes`).

### 3.3 Selección del checkpoint y métricas nuevas

`vlmfid.eval.attributes.score` agrega dos métricas. Las 8 existentes no cambian, y la copia del extractor en el notebook 02 sigue siendo equivalente para ellas.

| Métrica | Definición | Uso |
|---|---|---|
| `shape_pos` | (forma + posición(x,y)) / 2 | **selección del checkpoint** (`train.selection_metric`) |
| `macro_cat` | (forma + posición + color categoría exacta) / 3 | reporte: macro sin el ±1 que satura |

`train.selection_metric` acepta `shape_pos | macro | macro_cat | shape | pos | color | color_adj | name | val_loss`, y `compose_train` rechaza cualquier otro valor. El color se sigue registrando en `val_log.jsonl`: si el ajuste lo degradara, se vería ahí.

### 3.4 VL-JEPA: negativos del banco completo

Antes se usaba InfoNCE bidireccional con los negativos del lote (63) y una máscara de captions duplicados. Ahora, con `objective.negatives: bank`:

$$
\mathcal{L} = \frac{1}{B}\sum_{i}-\log\frac{e^{\langle \hat s_i, s_{c_i}\rangle/\tau}}{\sum_{j\in\mathcal{C}} e^{\langle \hat s_i, s_j\rangle/\tau}},
\qquad \mathcal{C} = \text{textos objetivo únicos (504 canónicos)}
$$

- Es **clasificar sobre el mismo banco con que se decodifica** por recuperación en val y test. Lo que se optimiza es lo que se mide.
- Un caption repetido es la misma clase, así que no hace falta la máscara de duplicados.
- Cada muestra aporta un término independiente del lote. Por eso **se puede acumular gradiente y partir el lote ante un OOM**, igual que en los VLM. `compose_train` solo exige micro-lote = lote efectivo con `negatives: batch`.
- Va en una sola dirección (predictor → texto): con un banco fijo, la dirección inversa no tiene sentido.
- El banco de negativos incluye los textos objetivo de val. Son solo textos, sin imágenes ni etiquetas de val, y con el formato canónico coinciden con los de train.

**Banco de recuperación en el formato aprendido.** `configs/models/vljepa.yaml` tiene la nueva clave `bank.target_format` (original | canonical). `compose_train` la iguala a `target.format`, y `Trainer` falla si no coinciden. En test hay que pasarla explícitamente (`model.bank.target_format=canonical`; ya está en `stage2b_test.txt`). La clave de caché del banco original no cambia, así que los bancos de la etapa 1 siguen sirviendo.

**Código:** `objectives.bank_infonce_loss`, `loop.Trainer._jepa_bank_loss_sum` / `_micro_loss` / `_step`, `models/vljepa.py::_bank_format`, `_build_bank`.

### 3.5 Train completo, sin límite

Todos los entrenamientos usan el split de train completo:

- **No existe `data.train_limit`.** Se quitó de `budget.yaml`, y `compose_train` lanza un error si se pasa por línea de comandos. No hay subconjuntos de train en ninguna cola.
- **`drop_last: false`.** 125 000 = 1953·64 + 8: antes se descartaban esas 8 imágenes, ahora el paso 1954 las usa (lote de 8, con la pérdida normalizada por sus propios tokens o muestras). El bucle arma cada paso con `itertools.islice` sobre el DataLoader, así que el último paso incompleto ya no rompe (`next(it)` lanzaba `StopIteration`). `summary.json → samples_seen` cuenta las muestras reales (125 000).
- **Humo con `train.max_steps`.** El humo recorre el mismo train completo y solo se corta en 20 pasos. `max_steps` va en el exp_id (`_m20`) y salió de las claves seguras para reanudar: un run cortado escribe `summary.json` y `adapter_best`, y con el mismo exp_id que el run completo haría que este se saltara como "completo".
- La verificación de largo de los objetivos (`max_target_tokens`) recorre **todos** los textos objetivo distintos de train y val, no solo las primeras 2 000 imágenes.

La única excepción a "train completo" es el split composicional (§3.6): retira 14 combinaciones forma × posición por diseño y entrena con **todas** las imágenes restantes.

**Código:** `train/config.py` (rechazo de `train_limit`, exp_id con `_m`), `train/data.py` (se eliminó `nested_subset`), `train/loop.py` (`_records`, último paso incompleto, `samples_seen`).

### 3.6 Split composicional

Con `data.holdout: compositional` se retiran de train **14 de las 63 combinaciones forma × posición**: las posiciones 2s y 2s+5 (mód 9) para la forma s, con posición = 3·y + x. Cada forma pierde 2 posiciones y cada posición se retira para al menos una forma, así que el modelo ve todas las formas y todas las posiciones, pero no todas sus combinaciones. Train queda con ≈ 97 000 imágenes.

- **Val:** las mismas 500 imágenes. Las métricas se calculan además sobre `seen_*` y `unseen_*`, y **el checkpoint se elige solo con `seen_<métrica>`**: las combinaciones retiradas no influyen en la selección.
- **Test:** el notebook 04 separa las 10 000 imágenes en vistas y retiradas y compara contra el run principal en las mismas imágenes.
- El banco de VL-JEPA contiene los 504 textos canónicos, también los de combinaciones retiradas. El banco es el "vocabulario" del decodificador, igual que un VLM puede generar cualquier texto. Lo que el modelo no vio son **imágenes** de esas combinaciones.

**Código:** `data/targets.py` (`HELD_OUT_COMBOS`, `is_held_out`, `apply_holdout`), `loop.Trainer._records` / `_val_groups`, `train/evaluate.generate_and_score(groups=…)`.

### 3.7 Sondas lineales (diagnóstico)

`scripts/probe.py` + `src/vlmfid/eval/probe.py`. Hacen una pasada hacia adelante con el mismo prompt y plantilla de chat que la inferencia, sin generar, y ajustan una regresión logística multinomial por atributo sobre la representación congelada:

| Rasgo | Qué es |
|---|---|
| `img_l0` | media de los estados de entrada en las posiciones de imagen = salida del conector |
| `img_l<k>` | media de la capa k del LM en las posiciones de imagen |
| `last_l<k>` | estado de la capa k en la última posición del prompt (de donde sale la primera palabra) |
| `vis_mean`, `pred` | VL-JEPA: media del X-Encoder y ŝ_Y |

con k = round(f · L), f ∈ {0,5; 0,75; 1}. Atributos: forma (7), x (3), y (3), posición conjunta (9) y color básico (8). La sonda se ajusta con 10 000 imágenes de train y se mide en 2 000 de val; test no se usa. La regularización L2 se elige en un 20 % retenido.

**Lectura:**
- Sonda alta y zero-shot bajo → el modelo **sabe** pero no lo dice: error de formato. El ajuste debería cerrarlo rápido, lo que se ve en las curvas de val (pocos pasos hasta el máximo).
- Sonda baja → la información **no está** en la representación. LoRA sobre el LM tiene poco con qué trabajar, y el cuello de botella es el encoder o el conector.
- `img_*` baja en posición y `last_*` alta → la media de tokens pierde la disposición espacial, pero el LM la integra.
- `stage2b_probe_ft.txt` repite las sondas con el adaptador: si la sonda sube, el ajuste cambió la representación; si no, cambió solo cómo se verbaliza.

**Limitación:** es una sonda **lineal** sobre una media o un token. Una sonda baja no prueba que la información no exista: puede estar codificada de forma no lineal o repartida entre tokens. Se reporta como cota inferior de lo decodificable.

### 3.8 Otros cambios menores

- `summary.json` agrega `target_format`, `holdout`, `n_train`, `epochs`, `quant` y `seed`. `status.json → model_info` agrega `target_format`, `holdout`, `n_train`, `n_val` y, en VL-JEPA, `negatives` y `n_target_texts`.
- `val_predictions/step_*.jsonl` agrega el campo `target` (el texto que se aprendía).
- La línea `[val]` del log imprime forma+pos, color y, con holdout, `no_vistas[forma+pos]`.
- La validación de config rechaza `target.format`, `data.holdout`, `objective.negatives` y `train.selection_metric` inválidos.

---

## 4. Experimentos

| Cola | Script | Runs | Pregunta |
|---|---|---|---|
| `stage2b_smoke` | train | 5 × 20 pasos (train completo, `max_steps=20`) | ¿Carga, baja la loss, cabe en memoria? (incluye una línea con holdout) |
| `stage2b_probe` | probe | 4 | ¿Qué sabe cada modelo congelado? (techo lineal antes de ajustar) |
| `stage2b_main` | train | 4 × 1954 pasos (125 000 imágenes) | Resultado principal: zero-shot p1 → ajustado, por modelo |
| `stage2b_test` | infer | hasta 15 | Test (10 000) de cada adaptador `adapter_best` |
| `stage2b_compositional` | train | 4 × ~1520 pasos (todas las no retiradas) | ¿Compone forma y posición o memoriza pares? |
| `stage2b_ablations` | train | 7 × 1954 pasos | Objetivo original, bf16 y semilla 1 |
| `stage2b_probe_ft` | probe | 4 | ¿El ajuste cambió la representación o solo la verbalización? |

Se descartó la **curva de datos** (subconjuntos de 1 000, 4 000 y 16 000 imágenes): todos los entrenamientos usan train completo (§3.5).

**Prioridad sugerida** (con GPU limitada): smoke → probe → main → test (bloque 1) → compositional → test (bloque 2) → ablations → test (bloque 3) → probe_ft. Las colas son reanudables e idempotentes: se pueden relanzar a medida que terminan los entrenamientos.

## 5. Precisión (bf16 frente a NF4)

Se mantiene NF4 por paridad con el zero-shot de la etapa 1, que está todo en NF4. La ablación `model.quant=bf16` (Qwen3.5) mide el costo de cuantizar durante el ajuste. Para compararla de forma justa hace falta el zero-shot bf16 con p1 en las mismas imágenes; si no existe en `runs/`, se corre con:

```bash
GPU=1 scripts/launch.sh zs_qwen_bf16 uv run --no-sync python scripts/infer.py model=qwen35 prompt=p1_attributes data.limit=10000 model.quant=bf16
```

## 6. Archivos

**Nuevos**

| Archivo | Contenido |
|---|---|
| `src/vlmfid/data/targets.py` | objetivo canónico, `target_fn`, split composicional |
| `src/vlmfid/eval/probe.py` | rasgos (VLM y VL-JEPA) y sonda logística |
| `scripts/probe.py` | punto de entrada de las sondas (reanudable: reutiliza `features_*.npz`) |
| `configs/queues/stage2b_*.txt` | 7 colas (§4) |
| `tests/test_targets.py` | caption canónico 100 % extraíble, 504 textos, propiedades del holdout |
| `notebooks/04_eda_finetuning.ipynb` | EDA de los resultados de la etapa 2 (§8) |
| `docs/ETAPA2B_REFORMULACION.md` | este documento |

**Modificados**

| Archivo | Cambio |
|---|---|
| `configs/train/budget.yaml` | `target.format`, `data.holdout`, `objective.negatives`, `selection_metric: shape_pos`, sin `train_limit`, `drop_last: false`, comentarios |
| `configs/train/models/vljepa.yaml` | `prompt: p1_attributes`, `objective.negatives: bank` |
| `configs/models/vljepa.yaml` | `bank.target_format: original` |
| `src/vlmfid/train/config.py` | validación de las claves nuevas, rechazo de `data.train_limit`, exp_id nuevo (con `_m<max_steps>`), propagación de `bank.target_format`, infonce + acumulación permitida con banco |
| `src/vlmfid/train/loop.py` | objetivos vía `target_fn`, train completo + holdout, último paso incompleto, `samples_seen` real, pérdida con banco y micro-lotes en VL-JEPA, métricas seen/unseen, selección en vistas, más campos en summary |
| `src/vlmfid/train/collate.py` | `target_fn` en ambos featurizers |
| `src/vlmfid/train/data.py` | docstring de `drop_last=False` (todo train) |
| `src/vlmfid/train/objectives.py` | `bank_infonce_loss` |
| `src/vlmfid/train/evaluate.py` | `target` en las predicciones de val, métricas por grupo |
| `src/vlmfid/eval/attributes.py` | métricas `shape_pos` y `macro_cat` |
| `src/vlmfid/models/vljepa.py` | banco en formato original o canónico (`_bank_format`) |
| `tests/test_train_units.py`, `tests/test_train_e2e.py`, `tests/_tiny_vlm.py` | expectativas nuevas, banco, holdout, VL-JEPA con ambos negativos |
| `README.md`, `SETUP.md`, `docs/ETAPA2_FINETUNING.md`, `scripts/train.py`, `scripts/run_queue.sh` | referencias a las colas y documentos nuevos |

**Eliminados:** `configs/queues/stage2_{smoke,train,test}.txt` (exp_id y prompt de VL-JEPA del diseño anterior; ver §9 para reproducirlo).

## 7. Ejecución en el cluster

```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM
scripts/jobs.sh status && git pull --ff-only origin fine-tuning
uv run --no-sync pytest -q          # incluye test_targets y los e2e del entrenamiento en CPU

# 1. humo (minutos por línea) y sondas sobre los modelos congelados
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2b_smoke scripts/run_queue.sh configs/queues/stage2b_smoke.txt
scripts/jobs.sh logs s2b_smoke -f
SCRIPT=scripts/probe.py GPU=1 scripts/launch.sh s2b_probe scripts/run_queue.sh configs/queues/stage2b_probe.txt

# 2. entrenamiento principal y test
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2b_main scripts/run_queue.sh configs/queues/stage2b_main.txt
GPU=1 scripts/launch.sh s2b_test scripts/run_queue.sh configs/queues/stage2b_test.txt

# 3. experimentos adicionales (relanzar s2b_test después de cada uno)
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2b_comp  scripts/run_queue.sh configs/queues/stage2b_compositional.txt
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2b_abl   scripts/run_queue.sh configs/queues/stage2b_ablations.txt
SCRIPT=scripts/probe.py GPU=1 scripts/launch.sh s2b_probe_ft scripts/run_queue.sh configs/queues/stage2b_probe_ft.txt
```

**Qué revisar en el humo:**
- `modelo listo` muestra `target_format: canonical` y `n_train: 125000` (≈ 97 000 en la línea con holdout); en VL-JEPA, además, `negatives: bank`, `n_target_texts ≈ 504` y `query_truncated: false`.
- El paso 0 de val se parece al zero-shot con p1.
- La loss baja. **En Qwen3.5, en particular, no debe quedar en NaN ni plana**: `flash-linear-attention==0.4.2` se fijó por un bug de inferencia y su backward con gradient checkpointing no estaba verificado.
- La línea con holdout imprime `no_vistas[forma+pos]`.

Si el **probe** falla en algún modelo por la API de `output_hidden_states` (Qwen3.5 híbrido es el candidato), el traceback queda en `logs/s2b_probe/job.log`. Las demás líneas de la cola siguen.

## 8. Notebook `04_eda_finetuning.ipynb`

Lee `runs/` (solo lectura) y escribe en `reports/{figures,tables}/eda_finetuning/`:

1. Inventario de los runs de entrenamiento, test y sondas, y su integridad.
2. Curvas de entrenamiento: loss, lr, img/s y memoria por paso.
3. Curvas de val: `shape_pos`, forma, posición, color y val_loss por paso, con el paso elegido marcado.
4. Test: zero-shot p1 frente a ajustado, por modelo y atributo, con IC de Wilson y McNemar pareado (Holm).
5. Matrices de confusión de forma y posición antes/después; transiciones acierto↔error.
6. Split composicional: combinaciones vistas frente a retiradas, contra el run principal en las mismas imágenes.
7. Sondas lineales: techo lineal frente a exactitud de las respuestas, congelado frente a ajustado.
8. Controles: canónico frente a original, NF4 frente a bf16, semilla.
9. Costo: horas de GPU, img/s y parámetros entrenables. Formato de las respuestas: copia del objetivo y diversidad.
10. Ejemplos cualitativos y tabla resumen exportada.

Cada sección se salta con un aviso si sus runs todavía no existen, así que el notebook se puede correr en cualquier momento del plan.

## 9. Reproducir el diseño anterior

```bash
# VLM autorregresivos
uv run --no-sync python scripts/train.py model=qwen35 target.format=original train.selection_metric=macro
# VL-JEPA
uv run --no-sync python scripts/train.py model=vljepa target.format=original train.selection_metric=macro \
    prompt=p4m_multiexample objective.negatives=batch
```

Las diferencias que quedan frente al diseño anterior son el exp_id (`_torig_`) y que se usa todo train (1954 pasos en vez de 1953; para el corte anterior, `train.drop_last=true`).

## 10. Qué reportar

- Por modelo: exactitud por atributo y `macro`/`macro_cat`/`shape_pos` en test, zero-shot p1 frente a ajustado, con IC de Wilson y McNemar pareado. El nombre exacto de los runs canónicos es 0 por construcción.
- Sondas: techo lineal congelado y ajustado al lado de la exactitud de las respuestas.
- Split composicional: brecha vistas − retiradas por modelo. Este es el contraste más directo entre paradigmas (autorregresivo frente a VL-JEPA).
- Controles: canónico − original, NF4 − bf16 y la variación entre semillas.
- Costo: horas de GPU, img/s, pico de memoria y parámetros entrenables (LoRA y conector).
