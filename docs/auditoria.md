# Auditoría del repositorio — Proyecto T7 (fase 2 de la reconstrucción)

> Clasifica cada módulo, script y config en **reutilizable tal cual**, **reutilizable con cambios**
> o **descartar**, con el motivo referido a `docs/metodologia.md` (abreviado **M**) o a `CLAUDE.md`.
> No se modificó ni movió ningún archivo.

## 0. Alcance y método

- **Qué se auditó.** El árbol de trabajo de la rama `vl-jepa`, incluidos los cambios sin commit y
  los archivos sin seguimiento, más el historial de ramas (`git log`, `git diff --stat`).
- **Cómo.** Lectura de código. **No se ejecutó nada**: ni tests, ni scripts, ni modelos. Toda
  afirmación sobre el comportamiento en ejecución está marcada como pendiente de verificar en el
  cluster (§6).
- **Profundidad.** Leídos completos: `models/` (salvo la segunda mitad de `bank.py`), `train/lora.py`,
  `train/objectives.py`, `train/collate.py`, `train/data.py`, `train/evaluate.py`, la mayor parte de
  `train/loop.py`, `eval/attributes.py`, `eval/text_metrics.py`, y todos los configs de modelos y de
  entrenamiento. Leídos en cabecera, docstring y firmas: `train/jepa.py`, `train/config.py`,
  `eval/probe.py`, `eval/infoloss.py`, `eval/copying.py`, `data/`, `config.py`, `tracking.py`,
  `prompts/`, `scripts/`. **No abiertos**: los 7 notebooks, `tests/` (solo nombres), `reports/`,
  `docs/ETAPA2*.md`, `docs/VLJEPA_AJUSTE_V3.pdf`, `README.md`, `SETUP.md`, `uv.lock`.
- **Sobre "descartar".** Nada se borra: lo clasificado así se mueve a una carpeta de descarte, con su
  motivo, en un commit explícito de la rama `reconstruccion`, y la decisión final es del usuario.

## 1. Ramas e historial

| Rama | Commit | Relación con `vl-jepa` |
|---|---|---|
| `vl-jepa` (actual) = `origin/vl-jepa` | `8f5e2ea` | — |
| `fine-tuning` = `origin/vljepa` | `08ed345` | ancestro; `vl-jepa` va 3 commits adelante |
| `main` | `e1a39f7` | ancestro; `vl-jepa` va 15 commits adelante |
| `origin/modelos` | `a91b5a5` | ancestro; 16 commits atrás |
| `origin/main` | `1f718e0` | merge de `fine-tuning` en `main` (PR #2); no contiene los 3 últimos commits de `vl-jepa` |

- `vl-jepa` contiene todo lo que hay en las demás ramas: **no hay código exclusivo de otra rama que
  auditar por separado**. Hay además un stash sobre `main` (commits `3dde752` y `8d9d2a7`) cuyo
  contenido no se inspeccionó.
- Entre `main` y `vl-jepa` se agregó toda la etapa 2 (`src/vlmfid/train/`, `eval/attributes.py`,
  `eval/text_metrics.py`, `eval/probe.py`, `data/targets.py`, configs de entrenamiento y colas
  `stage2b_*`): 46 archivos, +3 755 líneas fuera de `reports/` y `notebooks/`.
- **Trabajo sin commit** sobre `vl-jepa`: 11 archivos modificados (+545 / −118) y 9 archivos nuevos
  de código (`train/jepa.py`, `models/bank.py`, `eval/infoloss.py`, `scripts/info_loss.py`, configs
  `vljepa_fac`, colas `stage2c_*`, dos tests). Es el "ajuste v3" de VL-JEPA (banco factorizado).
  **Recomendación**: antes de crear `reconstruccion`, guardar este trabajo en un commit en `vl-jepa`
  para que no se pierda y quede en el historial, aunque luego se descarte.
- **Rama base para `reconstruccion`**: debe indicarla el usuario [CLAUDE.md §5]. Como `vl-jepa`
  contiene a todas las demás, es la candidata natural.

## 2. Hallazgos transversales

Ordenados por impacto. Los tres primeros invalidan resultados ya obtenidos, no solo código.

**H1. Test se usó para seleccionar y se evaluó muchas veces.** `configs/defaults.yaml` fija
`data.split: test`; `configs/queues/stage1_full.txt` corre *"Etapa 1 completa sobre test, 4 modelos x
6 prompts"*; ninguna cola usa `data.split=val`. Los configs de entrenamiento eligen el prompt citando
el "mejor zero-shot de la etapa 1", es decir, medido en test. Además `scripts/train.py` evalúa en test
al terminar **cada** entrenamiento (`test.enabled: true`), incluidas ablaciones y variantes. Contradice
M §2.1–2.3 y CLAUDE.md §3 (prompt y checkpoint se eligen en val; test una sola vez). Consecuencia: el
test publicado ya no es un conjunto no visto para las decisiones tomadas hasta ahora. Ver decisión A1.

**H2. El objetivo de entrenamiento no es el acordado.** `configs/train/budget.yaml` fija
`target.format: canonical` (una plantilla, color en 8 categorías) y deja el caption original como
control. CLAUDE.md §3 y M §2.3 fijan el **caption original con sus 5 plantillas**.

**H3. La métrica de color y la de selección no son las acordadas.** `eval/attributes.py` puntúa el
color por categoría básica del matiz verdadero (y "±1 categoría"), y selecciona checkpoints con
`shape_pos` o `macro`. M §2.1 fija **nombre exacto sin prefijo de paleta** para el color. El criterio
`name` del repo compara con prefijo, así que tampoco coincide.

**H4. La cuantización deja fuera el encoder visual.** `models/quant.py` y los configs
(`quant_skip_modules`) dejan en bf16 el encoder visual, el conector y `lm_head`, y
`_verify_quantization` **falla** si alguna capa visual queda cuantizada. En VL-JEPA solo se
cuantizan las capas del Predictor; X-Encoder e Y-Encoder quedan en bf16. M §2.1 (D5) fija todo
componente congelado en NF4.

**H5. VL-JEPA no sigue el protocolo en ningún punto central.**
- Texto por recuperación sobre un banco de captions y prototipos por atributo, en vez del Y-Decoder
  generativo de M §2.4.
- Objetivo de entrenamiento por defecto `negatives: bank` (clasificación sobre un banco cerrado), que
  no es el InfoNCE bidireccional del paper; el InfoNCE del lote existe, pero como modo secundario y
  con máscara de duplicados activada (M §2.3, D4: sin máscara con lote ≤ 128).
- El transformer del Y-Encoder queda congelado y solo se entrena su proyección, y solo en el ajuste
  v3. M §2.3 fija la opción A: Y-Encoder con LoRA y LR ×0.05.
- El banco factorizado (una consulta por atributo, 21 frases) no proviene del paper.

**H6. Prompt de la etapa 2.** Los cuatro modelos usan `p1_attributes` por una reformulación posterior.
CLAUDE.md §3 fija el mejor zero-shot de cada modelo, con la sola excepción de LLaVA-OV. Además el
"mejor" se determinó en test (H1) y con la métrica antigua (H3), así que hay que rehacer la selección.

**H7. Hiperparámetros sin justificación registrada.** `budget.yaml` fija r = 16, α = 32, dropout 0.05,
LR 2e-4, LR del conector 2e-5, coseno, warmup 3 %, max grad norm 1.0, lote efectivo 64 y 64 tokens
máximos. Ninguno tiene fuente en un paper para modelos de 0.5–1B; en M §4 todos están PENDIENTE.
El código que los aplica es reutilizable; los valores no se dan por buenos.

**H8. Selección en un subconjunto de val.** `val_limit: 500`. M §2.3 dice "val" sin más. Usar 500
imágenes es defendible por costo, pero es una decisión no registrada. Ver decisión A3.

**H9. Extensiones fuera de alcance mezcladas en el flujo principal.** Split composicional
(`data.holdout`), control Qwen Instruct, experimentos de copia p4/p4m, sondas sobre los VLM. No
contradicen el protocolo, pero CLAUDE.md §1 deja las variantes OOD fuera de alcance y §3 excluye las
variantes Instruct.

**H10. Métricas de texto reimplementadas.** `eval/text_metrics.py` reimplementa BLEU-4, ROUGE-L y
CIDEr-D "sin Java", con tokenización propia (`[a-z0-9]+`), y no calcula Bleu-1. M §2.1 fija el código
de evaluación de MS COCO. El módulo afirma haber sido verificado numéricamente contra
`pycocoevalcap` en un notebook; no se comprobó. Ver decisión A2.

## 3. Clasificación por módulo

### 3.1 `src/vlmfid/` — raíz, datos y prompts

| Archivo | Clasificación | Motivo |
|---|---|---|
| `paths.py`, `config.py`, `tracking.py` | reutilizable tal cual | Infraestructura (rutas con `pathlib`, composición de config, registro de config, semilla y commit por run), acorde a CLAUDE.md §6. Revisión superficial: confirmar en la fase 3. |
| `data/m3di.py`, `data/manifest.py` | reutilizable tal cual | Lectura del layout publicado y manifiestos; el subconjunto determinista común a todos los modelos apoya la comparabilidad. El layout coincide con lo observado en `data/raw/m3di`. |
| `data/targets.py` | **descartar** | Define el objetivo canónico y el split composicional (H2, H9). Con el caption original como único objetivo no queda nada que conservar. |
| `prompts/__init__.py`, `prompts/parsing.py` | reutilizable tal cual | Render de prompts y parseo tolerante de JSON; independiente del protocolo. |
| `configs/prompts.yaml` | reutilizable con cambios | El catálogo es válido como conjunto de prompts candidatos. `p3_json` pide `mid/center` y paletas, y `p4`/`p4m` incluyen ejemplos con prefijo: revisar que sigan siendo coherentes con la métrica de color sin prefijo. La nota "se congelan al terminar la etapa 1" deja de valer: la etapa 1 se rehace. |

### 3.2 `src/vlmfid/models/`

| Archivo | Clasificación | Motivo |
|---|---|---|
| `base.py`, `registry.py` | reutilizable tal cual | Interfaz común y registro por familia. |
| `quant.py` | reutilizable con cambios | La configuración NF4 + double quantization + bf16 coincide con [8] (M §1.6). Cambiar el criterio: cuantizar también los componentes visuales congelados (H4). `swap_linear_to_nf4` sirve para aplicar NF4 al X-Encoder y al Y-Encoder de VL-JEPA. |
| `hf_generative.py` | reutilizable con cambios | Carga, plantilla de chat, generación determinista y decodificación de solo lo nuevo son correctas en lectura. Cambios: (i) `_verify_quantization` debe verificar lo contrario de hoy (H4) y reportar la fracción de parámetros en NF4; (ii) el regex de LoRA cubre atención y MLP del LLM, acorde a [8], pero hay que confirmar los nombres de módulo por modelo (§4). |
| `qwen_vl.py` | reutilizable tal cual | Solo ajusta la resolución dinámica y la registra. Los valores quedan como PENDIENTE de M §4. |
| `vljepa.py` | reutilizable con cambios | **Conservar**: carga del checkpoint y de `open-vljepa`, transformación de imagen, duplicación en cuadros, `predict_embeddings`, `encode_texts`. **Quitar**: banco de captions, prototipos, banco factorizado y `describe` por recuperación (H5); se reemplazan por el Y-Decoder. **Agregar**: NF4 en X-Encoder e Y-Encoder; LoRA también en el Y-Encoder (el regex actual solo cubre `predictor.layers`). |
| `bank.py` (sin seguimiento) | **descartar** | Banco factorizado; no proviene del paper (H5) y la evaluación por candidatos quedó fuera (M §5, C6). |

### 3.3 `src/vlmfid/train/`

| Archivo | Clasificación | Motivo |
|---|---|---|
| `data.py` | reutilizable tal cual | Orden de datos determinista por (semilla, época), igual para los 4 modelos y agrupado en los mismos lotes efectivos: es justo lo que exige la comparabilidad. |
| `collate.py` | reutilizable con cambios | `GenerativeFeaturizer` arma la muestra como en inferencia y enmascara prompt e imagen en las etiquetas, acorde a [2] (M §1.2). Quitar la rama por atributo de `JEPAFeaturizer` (banco factorizado) y el `target_fn` configurable. |
| `objectives.py` | reutilizable con cambios | **Conservar** `lm_loss` (entropía cruzada solo sobre tokens objetivo, normalizada por el lote efectivo, con verificación de alineación) e `infonce_loss` (bidireccional, la del paper). **Descartar** `bank_infonce_loss`, `reverse_bank_infonce_loss`, `symmetric_bank_infonce_loss` (H5). La máscara de duplicados queda desactivada por defecto (D4). |
| `lora.py` | reutilizable con cambios | LoRA + conector completo vía `modules_to_save`, pesos entrenables en fp32, conteo de entrenables y **fallo si aparece un parámetro entrenable inesperado**: cumple la validación de CLAUDE.md §5. Cambios: grupo de parámetros para LoRA del Y-Encoder con LR ×0.05; revisar `set_train_mode`, que pone en `eval()` todo `visual` en Qwen3.5, incluida la copia entrenable del `merger` (sin efecto si no tiene dropout ni normalización por lote; confirmar). |
| `loop.py` | reutilizable con cambios | Reanudable, idempotente, con registro por run. Cambios: quitar holdout, formato de objetivo y dependencias del banco; métrica de selección nueva (H3); la evaluación de VL-JEPA pasa por el Y-Decoder. La partición del micro-lote ante OOM es válida para los VLM y no para InfoNCE del lote, como ya distingue el código. |
| `evaluate.py` | reutilizable con cambios | Usa el mismo camino de inferencia que la evaluación final, lo que es correcto. Quitar grupos de holdout; adaptar a la métrica nueva. |
| `config.py` | reutilizable con cambios | La lista blanca de claves por modelo (`ALLOWED_MODEL_KEYS`) impide que un modelo cambie LR o rango por su cuenta: protege la comparabilidad. Ajustar las claves a lo que M permite variar por modelo. |
| `jepa.py` (sin seguimiento) | reutilizable con cambios | Encapsula el objetivo de VL-JEPA. Conservar solo el modo de InfoNCE bidireccional del lote; los modos de banco se descartan. Reescribir el lado de texto para la opción A. |

### 3.4 `src/vlmfid/eval/`

| Archivo | Clasificación | Motivo |
|---|---|---|
| `attributes.py` | reutilizable con cambios | **Conservar**: extracción de forma (léxico con sinónimos), de posición (acepta `mid`/`center`/`middle`, cubre M §5 C16) y de menciones de color con entidad (objeto, fondo, luz); mapeos de códigos, que coinciden con lo observado en los datos. **Cambiar**: el acierto de color pasa a nombre exacto sin prefijo (H3); se eliminan categorías, "±1" y los índices `macro`, `shape_pos`, `macro_cat`. El docstring dice que la lógica se copió de un notebook: la copia del notebook debe pasar a importar este módulo [CLAUDE.md §6, una sola implementación]. Sus tests deben rehacerse con ejemplos verificados a mano. |
| `text_metrics.py` | **descartar o conservar como respaldo** (decisión A2) | H10. |
| `probe.py` | reutilizable con cambios | Extracción de rasgos del X-Encoder y de `ŝ_Y` aprovechable para M §2.5. La sonda es **lineal** (regresión logística con L-BFGS); M §2.5 fija MLP de una capa oculta para discretos y kernel ridge para continuos, según [1]. Falta el punto E2 (estado oculto del Predictor) y los factores continuos. Las sondas sobre los VLM quedan fuera de alcance. |
| `infoloss.py` (sin seguimiento) | reutilizable con cambios | Implementa la cota `H(A) − CE` de M §2.5 y el encadenamiento por etapas. Su etapa "decoder" mide el softmax sobre el banco: se reemplaza por el texto del Y-Decoder más el extractor. Usa nats; fijar unidad. |
| `copying.py` | fuera de alcance | Tasa de copia de ejemplos del prompt. No es parte del protocolo; puede conservarse como diagnóstico de la etapa 1 si se mantienen `p4`/`p4m`. |

### 3.5 `configs/`

| Archivo | Clasificación | Motivo |
|---|---|---|
| `defaults.yaml` | reutilizable con cambios | `data.split: test` por defecto es la raíz de H1: el valor por defecto debe ser `val`, y test debe exigir una acción explícita. |
| `models/llava_ov.yaml`, `internvl3.yaml`, `qwen35.yaml` | reutilizable con cambios | IDs y clases coherentes con M §3. Cambiar `quant_skip_modules` (H4). Los comentarios sobre número de tokens visuales son afirmaciones sin verificar (§4). |
| `models/qwen35_instruct.yaml` | **descartar** | Variante Instruct, excluida por CLAUDE.md §3 (H9). |
| `models/vljepa.yaml` | reutilizable con cambios | Conservar ruta del submódulo, checkpoint, `llama_name`, cuadros y tamaños. Quitar la sección `bank`. `max_caption_len: 96` y `max_query_len` quedan como PENDIENTE de M §4. |
| `models/vljepa_fac.yaml`, `train/models/vljepa_fac.yaml` (sin seguimiento) | **descartar** | Banco factorizado (H5). |
| `train/budget.yaml` | reutilizable con cambios | La estructura (un presupuesto común y solo unas claves por modelo) es la correcta. Los valores no (H2, H3, H7, H8), y la evaluación automática en test debe quitarse (H1). |
| `train/models/*.yaml` | reutilizable con cambios | Los `connector_modules` identifican bien el conector de cada modelo en lectura: `multi_modal_projector` (LLaVA-OV, InternVL3), `merger` (Qwen3.5), `vis_proj` (VL-JEPA; M §2.3 agrega las proyecciones de salida y del Y-Encoder). Rehacer la elección de prompt (H6). |
| `experiments/`, `queues/` | **descartar** | Todas las colas corren sobre test o dependen de la reformulación canónica, el holdout o el banco (H1, H2, H5). Las colas nuevas se escriben para el protocolo de M. El formato de cola en sí es reutilizable. |

### 3.6 `scripts/`

| Archivo | Clasificación | Motivo |
|---|---|---|
| `launch.sh`, `run_queue.sh`, `jobs.sh` | reutilizable tal cual | Lanzamiento con `setsid nohup`, sin `tmux`, como exige CLAUDE.md §4. |
| `check_env.py`, `build_manifest.py` | reutilizable tal cual | Verificación de entorno y construcción del manifiesto. |
| `infer.py` | reutilizable con cambios | Reanudable e idempotente. Depende de `defaults.yaml` para el split (H1). |
| `train.py` | reutilizable con cambios | Quitar la evaluación automática en test (H1). |
| `smoke.py` | reutilizable con cambios | Describe imágenes **de test**; debe usar val o train. |
| `probe.py`, `info_loss.py` | reutilizable con cambios | Se unifican en el análisis de M §2.5; quitan la dependencia del banco. |
| `copy_report.py`, `probe_p4.py` | fuera de alcance | Experimento de copia p4/p4m (H9). |

### 3.7 Tests, notebooks, reportes y documentación

| Ruta | Clasificación | Motivo |
|---|---|---|
| `tests/test_config.py`, `test_data.py`, `test_jobs.py`, `test_prompts.py`, `test_tracking.py`, `test_models_config.py` | reutilizable con cambios | No leídos; por nombre cubren infraestructura que se conserva. Revisar cuando se adapten los módulos. |
| `tests/test_attributes.py` | reescribir | La métrica cambia (H3). Debe incluir casos verificados a mano de las 5 plantillas y de respuestas zero-shot reales. |
| `tests/test_train_units.py`, `test_train_e2e.py`, `_tiny_vlm.py` | reutilizable con cambios | Un VLM diminuto en CPU para probar el bucle es valioso. `test_train_e2e.py` usa `from_pretrained` o el Hub: revisar que no descargue modelos. |
| `tests/test_targets.py`, `test_vljepa_bank.py`, `test_infoloss.py`, `test_copying.py` | descartar o adaptar | Siguen a sus módulos. |
| `notebooks/01_eda_m3di.ipynb`, `06_eda_splits.ipynb` | reutilizable con cambios | EDA de los datos, independiente del protocolo. No abiertos. `06` usa categorías de color y objetivo canónico en parte de sus conclusiones. |
| `notebooks/02`–`05` | **descartar** | Analizan resultados obtenidos sobre test, con métricas y objetivo que ya no valen (H1–H3). El docstring de `attributes.py` indica que `02` contiene una copia del extractor. |
| `reports/` | **descartar como resultados**; conservar como histórico | Figuras y tablas de las corridas anteriores. `reports/tables/eda_splits/hallazgos.md` contiene mediciones de datos útiles (integridad, fugas, dependencia x–color), **sin verificar**; una de ellas, 10 936 captions distintos en train, sí se confirmó de forma independiente (M §2.3). |
| `docs/ETAPA2_FINETUNING.md`, `ETAPA2B_REFORMULACION.md`, `VLJEPA_AJUSTE_V3.pdf`, `reports/VLJEPA_finetuning_metodologia.pdf` | **descartar** (histórico) | Documentan la metodología anterior, reemplazada por M. No leídos. |
| `docs/contexto_proyecto.md`, `ESTRUCTURA_REPO.md`, `README.md`, `SETUP.md`, `GUIA_LAPTOP_CLUSTER.md` | reutilizable con cambios | Describen el repo actual; se actualizan en la fase 3. `SETUP.md` y la guía del cluster probablemente sigan valiendo. |
| `pyproject.toml`, `uv.lock` | reutilizable con cambios | `uv` con marcadores de plataforma (cu128 en Windows, cu126 en Linux), como exige CLAUDE.md §4. `pycocoevalcap` ya está en el extra `eval`. El extra `linattn` (kernels de Gated DeltaNet) fija una versión por un fallo observado: verificar. |
| `external/open-vljepa` | reutilizable tal cual | Submódulo fijado a un commit, sin modificar. |

## 4. Los modelos frente a los papers

Contraste de la lista de M §3 con lo que se puede afirmar leyendo el código. "Coherente" significa que
el config y el adaptador no contradicen el paper; **no** que se haya comprobado sobre el modelo
cargado.

| Modelo | Punto de M §3 | Estado en el repo |
|---|---|---|
| LLaVA-OV-0.5B | conector = MLP de 2 capas, fuera de la cuantización | Coherente: `multi_modal_projector` excluido de NF4 y entrenable. Sin verificar que sea de 2 capas. |
| | tokens visuales con AnyRes | El config afirma ≈ 2×729 por imagen de 224 px. **Sin verificar.** |
| | pérdida solo sobre la respuesta | Coherente en lectura (`collate.py`, `lm_loss`, con comprobación de alineación en ejecución). |
| InternVL3-1B | pixel unshuffle, 256 tokens por tesela | El config afirma una tesela y 256 tokens. **Sin verificar.** |
| | LoRA solo en el LLM | El regex exige `language_model` en el nombre. **Sin verificar** que el checkpoint `-hf` use ese nombre; si no coincide, `attach_lora` falla en vez de entrenar otra cosa, lo que es el comportamiento seguro. |
| Qwen3.5-0.8B-Base | ID y variante | `Qwen/Qwen3.5-0.8B-Base`. **Sin verificar en HF.** |
| | dónde vive el conector | Resuelto en lectura: `visual.merger`, entrenable vía `modules_to_save` aunque `visual` esté excluido de NF4 y congelado. La duda planteada en M §3 queda aclarada, a confirmar en ejecución. |
| | LoRA en las capas Gated DeltaNet | El regex incluye `in_proj_qkv`, `in_proj_z`, `out_proj`. El comentario del config dice que `in_proj_a` e `in_proj_b` quedan fuera: **desviación de [8]** (todas las capas lineales) sin justificación registrada. |
| | plantilla de chat | El checkpoint Base no trae plantilla en el processor; se toma la de la versión Instruct. Es una decisión de diseño razonable pero **no documentada en M**: un modelo sin ajuste por instrucciones recibe un formato de chat que no vio en preentrenamiento. Debe constar y, si es posible, contrastarse con el model card. |
| | resolución | 256² mínimo → 64 tokens visuales, según el config. **Sin verificar.** |
| VL-JEPA | no causal, promedio sobre no-PAD | Implementado en `external/open-vljepa/openvljepa/models/predictor.py` (leído en la fase 1). |
| | X-Encoder congelado | `attach_lora` falla si algún parámetro de `x_encoder` queda entrenable. |
| | duplicación de la imagen en cuadros | Implementada (8 cuadros). |
| | longitud de consulta | `max_query_len` 64; los configs documentan que `p4m` se truncaba. El prompt elegido debe caber o hay que subir el límite. |
| | prefijo de tarea de EmbeddingGemma | El adaptador tokeniza el caption sin prefijo. Hay que confirmar que el checkpoint se entrenó igual (en `open-vljepa`). |
| | temperatura | 0.07, la del checkpoint. Coherente con M §4. |

## 5. Decisiones que surgen de la auditoría

| # | Decisión | Opciones | Propuesta |
|---|---|---|---|
| A1 | Test ya se usó para elegir prompts y se evaluó en cada corrida (H1). | (a) Rehacer todo con selección en val y usar el test publicado una vez, declarando en el informe que hubo exposición previa. (b) Generar un test nuevo con el generador de M3DI. | **Resuelta por el usuario: (a).** Se rehace todo seleccionando en val; el test publicado se usa una sola vez al final y el informe declara la exposición previa. Ningún resultado ni elección de prompt anterior se hereda. |
| A2 | Métricas de texto (H10). | (a) `pycocoevalcap` como implementación oficial; requiere Java en el cluster para el tokenizador. (b) Conservar la reimplementación sin Java, con un test que la compare contra `pycocoevalcap`. | (a) si hay Java en el cluster; si no, (b) con el test de equivalencia y Bleu-1 agregado. Hay que comprobar primero si el cluster tiene Java. |
| A3 | Tamaño del subconjunto de val para seleccionar checkpoint (H8). | 500 imágenes (actual), val completo (10 000), o un intermedio. | Sin propuesta: depende del costo de generar en val durante el entrenamiento. |
| A4 | Agregado de la exactitud por atributo para elegir prompt y checkpoint. M dice "exactitud por atributo" pero no define cómo se combinan los 4 atributos. | Media de los 4 atributos; o solo forma, x, y (el color tiene techo por diseño). | Media de forma, x, y y color, reportando cada uno por separado. |
| A5 | LoRA en `in_proj_a` / `in_proj_b` de Qwen3.5. | Incluirlas ([8]: todas las lineales) o mantenerlas fuera. | Incluirlas, salvo que haya una razón técnica; no encontré ninguna registrada. |
| A6 | Rama base de `reconstruccion` y destino del trabajo sin commit. | — | Commit del trabajo pendiente en `vl-jepa` y `reconstruccion` desde ahí. |
| A7 | Diagnósticos fuera de alcance (copia p4/p4m, sondas sobre VLM, split composicional). | Conservar como opcionales o descartar. | Descartar el split composicional y el control Instruct; conservar la tasa de copia solo si `p4`/`p4m` siguen entre los prompts. |

## 6. Pendiente de verificar en el cluster

Nada de esto se puede afirmar leyendo código. Debe comprobarse sobre los modelos cargados antes de
dar por buena la implementación de la etapa 1 [CLAUDE.md §5]:

1. Existencia y variante de `Qwen/Qwen3.5-0.8B-Base`, y su `config.json` (capas, proporción Gated
   DeltaNet / atención, tamaño de parche).
2. Número de tokens visuales por imagen de M3DI en cada VLM, y tamaño real de las imágenes.
3. `named_modules` de cada modelo: nombres del conector y de las capas lineales del LLM, y qué
   módulos alcanza cada regex de LoRA.
4. Conteo de parámetros entrenables por componente con el esquema nuevo, y fracción de parámetros
   efectivamente en NF4 (incluido el encoder visual).
5. Si bitsandbytes cuantiza correctamente los encoders visuales de cada modelo y el X-Encoder y el
   Y-Encoder de VL-JEPA, y cuánto cambia la salida respecto de bf16.
6. Que la plantilla de chat usada con Qwen3.5-Base produzca la secuencia esperada.
7. Estado de los tests actuales (`pytest`), que no se ejecutaron en esta auditoría.
8. Disponibilidad de Java para `pycocoevalcap` (A2).
