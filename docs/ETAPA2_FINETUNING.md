# Etapa 2 — Ajuste fino QLoRA con presupuesto común

Esta guía documenta qué se entrena, con qué hiperparámetros y por qué, y cómo ejecutarlo en el cluster. Se deja en el repo porque es la base de la sección de metodología del informe.

**Código:** `src/vlmfid/train/`, `scripts/train.py`, `configs/train/`.

**Colas:** `configs/queues/stage2_{smoke,train,test}.txt`.

## 1. Decisiones de diseño

Las tomó el equipo el 14-10-2026. Ninguna está implícita en el código: todas viven en `configs/train/`.

| Decisión | Elección | Consecuencia |
|---|---|---|
| Modelos | Los 4 de la etapa 1: LLaVA-OV 0.5B, InternVL3 1B, Qwen3.5-0.8B **Base** y VL-JEPA | Cada modelo tiene su par antes/después. El control Instruct queda fuera. |
| Objetivo | Caption original de M3DI, con sus 5 plantillas | BLEU/ROUGE/CIDEr comparables con la etapa 1 y con el banco de VL-JEPA. La plantilla es ruido irreducible (§3.2). |
| Prompt | El mejor zero-shot de cada modelo | Qwen3.5 Base e InternVL3 usan p1. VL-JEPA usa p4m. LLaVA-OV usa **p1** (su mejor, p3, pide JSON). |
| Precisión | QLoRA: base NF4 + doble cuantización, cómputo bf16 | Igual que la etapa 1; el antes/después no mezcla el efecto de cuantizar. |
| Presupuesto | **1 época** de train (125 000 imágenes), **mismo lote efectivo (64)** | 1953 pasos para todos; las mismas imágenes en los mismos lotes. Las horas de GPU difieren y se reportan. |
| Entrenables | LoRA en el transformer de lenguaje / predictor **+ conector completo** | Visión, cabeza de salida y el Y-Encoder de VL-JEPA quedan congelados. |
| Checkpoint | El **mejor en val** (500 imágenes fijas) según el Índice macro | Test (10 000) no se toca hasta la evaluación final. |

### 1.1 Prompt de VL-JEPA

La regla "mejor zero-shot" elige p4m (0,2910). Sin embargo, en VL-JEPA los seis prompts quedaron entre 0,285 y 0,291, dentro del ruido: la consulta casi no mueve ŝ_Y.

Además, p4m (329 caracteres) probablemente excede los 64 tokens de consulta del checkpoint y se trunca, igual que en la etapa 1. El conteo exacto queda en `status.json` (`model_info.query_tokens`, `query_truncated`).

Si el equipo prefiere uniformidad, cambiar a p1 es una línea en `configs/train/models/vljepa.yaml`.

## 2. Qué se entrena

Sea $W_0 \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ una proyección lineal congelada (NF4) del transformer. LoRA (Hu et al., 2021) la reemplaza por

$$
W = W_0 + \frac{\alpha}{r}\,BA,\qquad A\in\mathbb{R}^{r\times d_{\text{in}}},\; B\in\mathbb{R}^{d_{\text{out}}\times r},\; B_{t=0}=0,
$$

así que en el paso 0 el modelo es **idéntico** al de la etapa 1. La evaluación en el paso 0 es, por eso, un control de paridad: debe reproducir el zero-shot en val.

| Modelo | LoRA (patrón existente en el adaptador) | Conector entrenable completo | Congelado |
|---|---|---|---|
| LLaVA-OV | `language_model.*.{q,k,v,o,gate,up,down}_proj` | `model.multi_modal_projector` (MLP SigLIP→Qwen2) | SigLIP, `lm_head`, embeddings |
| InternVL3 | ídem | `model.multi_modal_projector` (LN + MLP) | InternViT, `lm_head`, embeddings |
| Qwen3.5 | `language_model.*.{q,k,v,o}_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, MLP | `model.visual.merger` | bloques visuales, `lm_head`, embeddings |
| VL-JEPA | `predictor.layers.*.{q,k,v,o,gate,up,down}_proj` (8 capas Llama) | `predictor.vis_proj` (1024→2048) | X-Encoder V-JEPA 2, Y-Encoder, `out_proj`, embeddings |

Los nombres de módulo se verificaron instanciando las tres arquitecturas de transformers 5.17. El código, además, falla si LoRA o el conector no quedan donde corresponde.

**Precisión de los entrenables.** LoRA y el conector se guardan en fp32 (pesos maestros) y el forward corre en autocast bf16. Con lr = 2·10⁻⁵, una actualización de AdamW sobre un peso del orden de 10⁻² queda bajo la resolución de bf16 (≈ 3 cifras significativas). Entrenado en bf16, el conector quedaría prácticamente congelado.

**Por qué el Y-Encoder de VL-JEPA queda congelado.** Así, el espacio objetivo $s_Y$ no se mueve. El banco de recuperación y los prototipos por atributo de la etapa 1 siguen siendo válidos sin recalcular, y los embeddings objetivo se precalculan una vez por caption único (10 936 en train).

**Parámetros entrenables.** Se registran por run en `summary.json` (`trainable_lora`, `trainable_connector`). Deben reportarse: con el mismo rango, el número de parámetros entrenables difiere entre modelos porque sus dimensiones difieren.

## 3. Funciones de pérdida

### 3.1 VLM autorregresivos

Para un lote efectivo $\mathcal{B}$ con captions $y_i$ de $T_i$ tokens (incluido el fin de turno `<|im_end|>`):

$$
\mathcal{L}_{\text{LM}}(\theta) = -\frac{1}{N}\sum_{i\in\mathcal{B}}\sum_{t=1}^{T_i}\log p_\theta\!\left(y_{i,t}\mid y_{i,<t},\,x_i,\,q\right),
\qquad N=\sum_{i\in\mathcal{B}} T_i .
$$

- **Sin pérdida en el prompt ni en la imagen.** Las etiquetas son −100 en esas posiciones.
- **Normalización por los tokens del lote efectivo, no del micro-lote.** Así, el gradiente acumulado es idéntico con micro-lotes de 8, 16 o 32: está verificado en `tests/test_train_e2e.py::test_micro_lote_no_cambia_el_gradiente`. Promediar la media de cada micro-lote ponderaría distinto los captions cortos y largos, y la diferencia dependería del modelo.
- **Mismo contexto que en inferencia.** El prompt se arma con `describer.build_messages` y la plantilla de chat con los mismos kwargs (`enable_thinking: false` en Qwen3.5). El objetivo se tokeniza aparte y se concatena: la frontera prompt/objetivo es exacta y los tokens objetivo son los que `generate` produce tras el prompt.
- **Fin de turno.** Se exige que `<|im_end|>` esté entre los `eos_token_id` de generación; si no, el modelo ajustado no sabría detenerse.
- **Relleno a la izquierda,** como en `describe()`. Los objetivos quedan al final de todas las filas y basta con `logits_to_keep = max T_i + 1`: no se materializan logits de vocabulario (~150 k) para los ~1 500 tokens visuales de LLaVA-OV. La pérdida coincide exactamente con la calculada sobre todos los logits (verificado).

### 3.2 Piso irreducible

La plantilla (`text_phrasing`) se eligió al azar, independiente de la imagen. Por eso, ningún modelo puede predecirla mejor que con probabilidad 1/5, y

$$
\mathcal{L}_{\text{LM}} \;\gtrsim\; \frac{\mathbb{E}[\#\text{decisiones de plantilla}]\cdot \log 5}{\mathbb{E}[T]} > 0 .
$$

Lo mismo ocurre con el nombre exacto de color: hay 100 nombres de 3 paletas, y el nombre no es función determinista del tono (las reglas del EDA lo reproducen en ~27 %).

Por eso **la pérdida no es la métrica de selección** y la exactitud de *nombre exacto* tiene techo bajo incluso tras ajustar. Las principales son forma, posición (x, y) y color ±1, que forman el Índice macro.

Como referencia, el extractor da 100 % de forma, posición y nombre sobre los propios captions de referencia. En color por categoría da 86 %: ese es el techo de esa columna.

### 3.3 VL-JEPA

Se usa InfoNCE bidireccional, como en el preentrenamiento del checkpoint, con similitud coseno $s_{ij}=\langle \hat s_i, s_j\rangle$ entre la predicción del predictor y el embedding del Y-Encoder, y temperatura $\tau=0{,}07$:

$$
\mathcal{L}_{\text{NCE}} = \frac{1}{2|\mathcal{B}|}\sum_{i\in\mathcal{B}}\left[
-\log\frac{e^{s_{ii}/\tau}}{\sum_{j\notin D_i} e^{s_{ij}/\tau}}
-\log\frac{e^{s_{ii}/\tau}}{\sum_{j\notin D_i} e^{s_{ji}/\tau}}\right],
\qquad D_i=\{\,j\neq i : c_j=c_i\,\}.
$$

**Corrección necesaria en M3DI.** Hay solo 10 936 captions distintos entre 125 000 imágenes, así que dos muestras de un lote pueden tener el mismo caption $c$. InfoNCE estándar trataría a la otra como negativo y empujaría a separar dos embeddings que deben coincidir. $D_i$ excluye esos falsos negativos.

Sin la máscara, una predicción perfecta con un par repetido en un lote de 4 paga ≈ ½·log 2 de pérdida extra (verificado en `tests/test_train_units.py`).

**Por qué no hay acumulación de gradientes en VL-JEPA.** Los negativos son las otras muestras del lote, y acumular gradientes cambia la pérdida. El micro-lote es el lote efectivo (64, el del preentrenamiento del checkpoint: 63 negativos).

## 4. Hiperparámetros

Todos están en `configs/train/budget.yaml` y son comunes a los 4 modelos. El archivo por modelo solo puede fijar el prompt, el micro-lote, el conector y el objetivo; cualquier otra clave se rechaza.

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| Épocas | 1 | Decisión de presupuesto. Son 125 000 ejemplos de una tarea de baja complejidad (7 formas × 9 posiciones × 100 nombres × 5 plantillas); una época es amplia. |
| Lote efectivo | 64 | Igual al del preentrenamiento de VL-JEPA, para no cambiar su número de negativos. Da 1953 pasos, suficientes para calentamiento + coseno. Es el único valor que sirve a la vez a InfoNCE y a los VLM sin tratarlos distinto. |
| `drop_last` | sí | 125 000 = 1953·64 + 8: el último lote incompleto se descarta para que todos los pasos tengan el mismo tamaño. |
| Micro-lote | 8 / 16 / 32 / 64 | Solo memoria (LLaVA-OV / InternVL3 / Qwen3.5 / VL-JEPA), según los tokens visuales por imagen (≈1 500 / 256 / 64). No cambia la matemática (§3.1). Ante un OOM se parte solo. |
| lr LoRA | 2·10⁻⁴ | Valor estándar de QLoRA para modelos ≤ 13 B y del script LoRA de LLaVA-1.5. Con B = 0 al inicio, LoRA tolera lr altos. |
| lr conector | 2·10⁻⁵ (= lr / 10) | El conector son pesos preentrenados completos que alinean visión y lenguaje. Con el lr de LoRA se destruiría esa alineación al inicio. Es el mismo criterio de LLaVA-1.5 LoRA (`mm_projector_lr = 2e-5` con `learning_rate = 2e-4`). |
| Rango r, α | 16, 32 (α/r = 2) | La tarea tiene baja dimensión intrínseca. QLoRA (Dettmers et al., 2023) muestra que lo decisivo es aplicar LoRA a **todas** las capas lineales y que r influye poco. r = 16 deja margen sin sobredimensionar. Se fija igual en los 4 por paridad. α = 2r es la convención que mantiene la escala del update al variar r. |
| Dropout LoRA | 0,05 | Regularización leve. QLoRA usó 0,1 hasta 13 B y 0,05 en 33-65 B. Aquí cada ejemplo se ve una sola vez (1 época de 125 k), así que el riesgo de memorizar es bajo y basta el valor menor. |
| Weight decay | 0 | Habitual en LoRA (QLoRA, LLaVA). Con una época y B inicializado en 0, el decaimiento solo acercaría el adaptador a cero sin beneficio medible. |
| Optimizador | AdamW, β = (0,9; 0,999), ε = 10⁻⁸ | Valores por defecto. Solo hay unos millones de parámetros entrenables, así que no hace falta AdamW paginado ni de 8 bits. |
| Schedule | calentamiento lineal 3 % (≈ 59 pasos) + coseno a 0 | `warmup_ratio = 0.03` y coseno son los de LLaVA. El calentamiento evita pasos grandes cuando los momentos de Adam aún están sesgados. El coseno estabiliza los checkpoints finales, que son candidatos a "mejor". |
| Recorte de gradiente | norma 1,0 | Estándar; protege de picos con NF4/bf16. |
| `max_target_tokens` | 64 | El caption más largo tiene 21 palabras (≈ 30-40 tokens). Si uno excede, el run **falla** en vez de truncar en silencio. |
| Gradient checkpointing | sí | Memoria, a costa de ≈ 20-30 % más de tiempo por paso. Necesario para LLaVA-OV en 24 GB compartidos. |
| Evaluación | paso 0, cada 200 y el último (11 puntos) | El paso 0 es el control de paridad. 11 puntos bastan para ver la curva sin encarecer el run (500 generaciones cada uno). |
| Val | 500 imágenes fijas | El Índice macro por imagen toma valores en {0, ⅓, ⅔, 1}, con varianza ≤ ¼. Con n = 500 su error estándar es ≤ 0,022: alcanza para elegir el checkpoint, pero no para comparar modelos. **La comparación se hace en test** (n = 10 000, EE ≤ 0,005, McNemar pareado como en la etapa 1). |
| Temperatura InfoNCE | 0,07 | La del preentrenamiento. Cambiarla altera la geometría del espacio respecto del banco y los prototipos. |
| Semilla | 0 | Fija el orden de los datos (mismo para los 4), la inicialización de LoRA y el dropout. |

**Qué NO se modifica respecto de la etapa 1:** la cuantización, la resolución de imagen (`min_pixels` nulo en Qwen3.5), la plantilla de chat, los parámetros de generación (greedy, 128 tokens nuevos), los prompts (congelados), los 8 fotogramas, `max_query_len = 64` y el banco de 20 000 captions de train de VL-JEPA.

## 5. Consideraciones

1. **Sin aumentación de datos.** Un volteo horizontal invierte izquierda/derecha y un jitter de color cambia el nombre del color: ambos contradicen la etiqueta. En M3DI el texto describe la geometría exacta de la imagen.
2. **Test intacto.** La selección usa val y test se evalúa una sola vez por modelo, con el adaptador ya elegido.
3. **Mismo camino de inferencia.** La evaluación en val usa `describer.describe`, el mismo código que `scripts/infer.py` en test. Lo que se selecciona es lo que se reporta.
4. **Costo desigual.** Con el mismo número de imágenes, LLaVA-OV procesa ~20× más tokens visuales que Qwen3.5. Hay que reportar horas de GPU (`summary.json`: `train_hours`, `eval_hours`) junto a la exactitud.
5. **Una semilla.** No hay estimación de la varianza entre corridas de entrenamiento: las pruebas pareadas en test capturan la variación entre imágenes, no entre semillas. Si hay tiempo, repetir el mejor y el peor modelo con `seed=1` acota cuánto de una diferencia es ruido de entrenamiento. *Decisión pendiente del equipo.*
6. **GPU compartida.** `launch.sh` aborta si la GPU elegida está ocupada. Si otro usuario entra a mitad de camino y hay OOM, el loop parte el micro-lote y repite el paso con las mismas muestras (en VL-JEPA, falla con mensaje).
7. **Reanudación exacta.** Con SIGTERM (`jobs.sh stop`) o una caída, relanzar el mismo comando continúa desde el último checkpoint. Esto queda bit a bit igual a no haber interrumpido (verificado en CPU), porque el orden de datos depende solo de (semilla, época, paso) y cada DataLoader usa su propio generador. Cambiar hiperparámetros con un checkpoint a medias se rechaza; cambiar el micro-lote no.
8. **Disco (~80 % usado).** Un adaptador ocupa decenas de MB y `checkpoint/` agrega el estado de AdamW (≈ 2× los entrenables). `checkpoint/` se puede borrar al terminar; lo que se usa después es `adapter_best/`.

## 6. Ejecución en el cluster

```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM
scripts/jobs.sh status && git pull --ff-only origin modelos
uv run --no-sync pytest -q                                    # incluye un entrenamiento diminuto en CPU

# 1. Prueba de humo: 20 pasos por modelo con val de 64 imágenes. Mirar en el log que carguen, el
#    paso 0 de val (≈ zero-shot), img/s, el pico de memoria y que la loss baje.
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh stage2_smoke scripts/run_queue.sh configs/queues/stage2_smoke.txt
scripts/jobs.sh logs stage2_smoke -f

# 2. Entrenamiento (ETA ≈ 125 000 / img/s del paso 1, más ~11 evaluaciones)
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh stage2_train scripts/run_queue.sh configs/queues/stage2_train.txt

# 3. Evaluación en test con los adaptadores elegidos (scripts/infer.py, mismo formato que la etapa 1)
GPU=1 scripts/launch.sh stage2_test scripts/run_queue.sh configs/queues/stage2_test.txt
```

Un modelo suelto: `GPU=1 scripts/launch.sh ft_qwen35 uv run --no-sync python scripts/train.py model=qwen35`.

**Qué revisar en el paso 0 de val.** Debe parecerse al zero-shot de la etapa 1 con el mismo prompt (en val, no en test). Si difiere mucho, algo cambió en la carga o el preprocesado: detener antes de gastar la época.

**Análisis.** Los notebooks de EDA leen las predicciones de test con `STAGE = "finetuned"`. El extractor de atributos ahora está en `vlmfid.eval.attributes` (idéntico al del notebook), así que conviene importar desde ahí. La comparación central del proyecto es, por modelo, zero-shot (mismo prompt, `data.limit=10000`) frente a ajustado, con McNemar pareado por atributo.

## 7. Qué reportar

- Por modelo: exactitud por atributo y Índice macro en test, antes y después, con IC de Wilson y McNemar pareado.
- Curvas de val (`val_log.jsonl`): macro y pérdida por paso, y el paso elegido.
- Costo: horas de GPU, img/s, pico de memoria, parámetros entrenables (LoRA y conector por separado).
- BLEU/ROUGE/CIDEr antes/después. El objetivo es el caption original, así que deberían subir mucho por puro formato: por eso la conclusión se apoya en la exactitud por atributo.
