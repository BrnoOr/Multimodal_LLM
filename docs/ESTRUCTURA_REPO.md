# Estructura del repositorio: qué hace cada carpeta y archivo

Inventario del repositorio al 11-11-2026 (rama `vl-jepa`, 384 archivos versionados).

**Cómo leer este documento**

- Cada descripción sale de leer el archivo: su docstring o cabecera, sus comentarios y, en los notebooks, la introducción y las secciones.
- **⚠ No lo tengo claro** marca lo que no pude confirmar. Ahí digo qué vi y qué falta saber.
- Las figuras y tablas de `reports/` no las abrí una por una. Su descripción se deduce del nombre del archivo y de la sección del notebook que las genera; trátalas como orientación, no como verificación.
- Todos los puntos dudosos están reunidos al final, en [Puntos que no tengo claros](#puntos-que-no-tengo-claros).

**De qué trata el proyecto.** Paquete `vlmfid` (EL7006, U. de Chile): mide qué tan fiel es la descripción que un modelo multimodal da de una imagen, usando el dataset sintético Multimodal3DIdent (M3DI), donde se conocen los atributos verdaderos de cada escena (forma, posición, color). Compara tres VLM autorregresivos (LLaVA-OneVision 0.5B, InternVL3-1B, Qwen3.5-0.8B) con VL-JEPA, en dos etapas: zero-shot (etapa 1) y ajuste fino QLoRA (etapa 2).

---

## Mapa general

| Ruta | Qué es | En git |
|---|---|---|
| `src/vlmfid/` | Paquete Python con toda la lógica (datos, modelos, evaluación, entrenamiento) | sí |
| `scripts/` | Puntos de entrada: inferencia, entrenamiento, sondas y lanzamiento de jobs en el cluster | sí |
| `configs/` | Configuración declarativa: modelos, prompts, presupuesto de entrenamiento y colas de experimentos | sí |
| `tests/` | Pruebas con pytest, todas ejecutables en CPU | sí |
| `notebooks/` | Análisis interactivo; leen `data/` y `runs/` y escriben en `reports/` | sí |
| `reports/` | Figuras y tablas generadas por los notebooks, más un PDF de metodología | sí |
| `docs/` | Documentación de diseño y guías de trabajo | sí |
| `data/` | Dataset, manifiestos y cachés | solo manifiestos y README |
| `external/` | Código de terceros como submódulo (open-vljepa) | sí (submódulo) |
| `logs/` | Salida de los jobs lanzados con `launch.sh` | solo `.gitkeep` |
| `runs/` | Resultados de cada experimento. No existe en este equipo: vive solo en el cluster | no |
| `.venv/` | Entorno virtual local creado por `uv` | no |

---

## Archivos de la raíz

| Archivo | Qué hace |
|---|---|
| `README.md` | Presentación del proyecto: modelos, estructura, uso, configuración, salidas, prompts y etapas. |
| `SETUP.md` | Preparación del entorno en el cluster y en el laptop (Windows), modelos y licencias, hitos y problemas frecuentes. |
| `pyproject.toml` | Define el paquete `vlmfid` 0.2.0, Python 3.12, dependencias y extras (`quant`, `eval`, `track`, entre otros). Fija los índices de PyTorch: cu128 en Windows y cu126 en Linux. Configura pytest y ruff. |
| `uv.lock` | Versiones exactas de todas las dependencias, generado por `uv`. No se edita a mano. |
| `.python-version` | Fija Python 3.12 para `uv`. |
| `.gitignore` | Excluye el entorno, cachés, `data/raw`, `data/generated`, `data/cache`, pesos (`*.pt`, `*.safetensors`, `*.bin`), `runs/`, `logs/` y archivos comprimidos. |
| `.gitattributes` | Fuerza saltos de línea LF en `.sh`, `.py`, `.yaml` y `.txt` (los scripts corren en Linux) y marca como binarios `.parquet`, `.png`, `.pdf` y `.zip`. |
| `.gitmodules` | Declara el submódulo `external/open-vljepa`. |

---

## `src/vlmfid/` — el paquete

### Raíz del paquete

| Archivo | Qué hace |
|---|---|
| `__init__.py` | Docstring del paquete y versión (`0.2.0`). |
| `paths.py` | Rutas canónicas del proyecto (`ROOT`, `DATA`, `MANIFESTS`, `RUNS`, `LOGS`, etc.), todas redefinibles por variable de entorno (`VLMFID_ROOT`, `M3DI_ROOT`, `VLMFID_RUNS`…). |
| `config.py` | Compone la configuración de inferencia con OmegaConf: `defaults.yaml` + modelo + prompt + experimento opcional + overrides de línea de comandos. También genera el `exp_id` por defecto. |
| `tracking.py` | `RunTracker`: registra cada run en `runs/<exp_id>/` (configuración resuelta, entorno, estado, predicciones incrementales) y decide si un run se puede reanudar. |

### `data/` — lectura del dataset

| Archivo | Qué hace |
|---|---|
| `__init__.py` | Reexporta las funciones públicas de `m3di.py` y `manifest.py`. |
| `m3di.py` | Lee el layout crudo de M3DI por split (imágenes, `latents_image.csv`, `latents_text.csv`, captions) y arma la tabla con latentes de imagen (sin prefijo) y de texto (prefijo `text_`). Incluye el diagnóstico de desacuerdo imagen/texto. |
| `manifest.py` | Lee y escribe los manifiestos Parquet y los entrega como registros con esquema estable (`load_manifest`). Define `ImageRecordDataset` y el subconjunto aleatorio determinista. |
| `targets.py` | Etapa 2: construye el texto objetivo (`original` o `canonical`) y aplica el split composicional, que retira de train 14 de las 63 combinaciones forma × posición. |

### `models/` — adaptadores de modelos

| Archivo | Qué hace |
|---|---|
| `__init__.py` | Reexporta `Describer`, `Description`, `MODELS` y `build_describer`. |
| `base.py` | Interfaz común `Describer` y el resultado `Description` (texto, metadatos, embedding opcional). El resto del código solo conoce esta interfaz. |
| `registry.py` | Registro perezoso: según el campo `family` de la config (`hf`, `qwen_vl`, `vljepa`) carga el adaptador correspondiente. |
| `hf_generative.py` | Adaptador genérico para VLM autorregresivos de transformers: plantilla de chat, `generate` y decodificación. Lo usan LLaVA-OneVision e InternVL3. |
| `qwen_vl.py` | Subclase para Qwen-VL: ajusta la resolución dinámica (`min_pixels`, `max_pixels`), que determina cuántos tokens visuales recibe el modelo. |
| `vljepa.py` | Adaptador de VL-JEPA (open-vljepa). El modelo predice un embedding, no texto; el adaptador devuelve la descripción más cercana de un banco de captions de train y construye prototipos por atributo. |
| `quant.py` | Esquema de cuantización común (NF4 con cómputo bf16): se cuantiza el transformer de lenguaje; el encoder visual, el proyector y la cabeza quedan en bf16. |

### `prompts/`

| Archivo | Qué hace |
|---|---|
| `__init__.py` | `render`: devuelve el texto final del prompt, sin aplicar `str.format` porque varios prompts tienen llaves literales. |
| `parsing.py` | Parser tolerante de respuestas JSON (bloques con ```` ``` ````, texto antes del objeto, comillas simples) y limpieza de etiquetas `<think>`. Si no hay JSON recuperable devuelve `None`. |

### `eval/` — evaluación

| Archivo | Qué hace |
|---|---|
| `__init__.py` | Reexporta las funciones de evaluación. |
| `attributes.py` | Extractor léxico de atributos desde texto libre (forma, posición, color) y cálculo de exactitud por atributo, incluido el "Índice macro" y las métricas `shape_pos` y `macro_cat` de la etapa 2. |
| `copying.py` | Detecta si una respuesta copia los ejemplos del prompt (p4 y p4m), en dos niveles: copia literal y copia del objeto del ejemplo. |
| `text_metrics.py` | `TextScorer`: BLEU-4, ROUGE-L y CIDEr-D contra una referencia por imagen, implementados sin Java. |
| `probe.py` | Sondas lineales sobre las representaciones internas del modelo, para saber si un atributo es decodificable aunque el modelo no lo verbalice. |

### `train/` — ajuste fino (etapa 2)

| Archivo | Qué hace |
|---|---|
| `__init__.py` | Docstring que resume el subpaquete; expone `compose_train`. |
| `config.py` | Compone la configuración de entrenamiento (`budget.yaml` + archivo del modelo + config de inferencia + prompt + overrides) y rechaza claves por modelo que romperían el presupuesto común. |
| `data.py` | Orden de datos determinista y reanudable: los cuatro modelos ven las mismas imágenes, en el mismo orden y en los mismos lotes. |
| `collate.py` | Arma los lotes de entrenamiento: prompt de chat + caption objetivo con las etiquetas del prompt enmascaradas; versión aparte para VL-JEPA. |
| `objectives.py` | Funciones de pérdida: entropía cruzada normalizada por tokens del lote efectivo (VLM) e InfoNCE con máscara de duplicados o con el banco completo como negativos (VL-JEPA). |
| `lora.py` | Define qué se entrena: LoRA en el transformer de lenguaje o predictor, más el conector multimodal completo; grupos de tasa de aprendizaje. |
| `evaluate.py` | Evaluación en validación durante el entrenamiento, por el mismo camino de inferencia que se usa en test. |
| `loop.py` | Bucle de entrenamiento (512 líneas, el archivo más largo): reanudable, con checkpoints atómicos, apagado limpio, manejo de falta de memoria y selección del mejor checkpoint en val. |

---

## `scripts/` — puntos de entrada

| Archivo | Qué hace |
|---|---|
| `build_manifest.py` | Construye los manifiestos `data/manifests/m3di_<variant>_<split>.parquet` desde el dataset crudo. |
| `check_env.py` | Verifica el entorno: build de PyTorch frente a la GPU, bf16, NF4, versiones, submódulo, token de Hugging Face y espacio en disco. |
| `smoke.py` | Prueba rápida: carga cada modelo, describe unas pocas imágenes de test y libera la GPU. No escribe en `runs/`. |
| `infer.py` | Inferencia de la etapa 1 (zero-shot) y, con `model.adapter_path`, de los modelos ajustados. Reanudable, idempotente y robusta a falta de memoria. |
| `train.py` | Etapa 2: ajuste fino QLoRA con el presupuesto común, seguido de la evaluación en test. |
| `probe.py` | Ejecuta las sondas lineales sobre un modelo congelado o ajustado y guarda `probe_results.jsonl`. |
| `copy_report.py` | Calcula la tasa de copia de ejemplos por run y la escribe en un CSV. **⚠** Su salida por defecto es `reports/tables/eval/copia_ejemplos.csv`, carpeta que se eliminó en la limpieza; el script la vuelve a crear si se ejecuta. |
| `probe_p4.py` | **⚠ No lo tengo claro.** Prueba puntual de variantes del prompt p4 (8 imágenes × 4 modelos × 2 variantes). Parece un diagnóstico de la fase piloto: ningún documento lo menciona y su docstring indica ejecutarlo como `python probe_p4.py` desde la raíz aunque vive en `scripts/`. No sé si sigue en uso. |
| `launch.sh` | Lanza un comando desacoplado de la sesión SSH (`setsid` + `nohup`) y deja log, PID y código de salida en `logs/<job>/`. |
| `jobs.sh` | Administra los jobs lanzados: estado, ver logs, detener, matar, progreso de los runs y uso de GPU. |
| `run_queue.sh` | Ejecuta en serie una cola de experimentos (un archivo de `configs/queues/`). Si una línea falla, sigue con la siguiente. |

---

## `configs/` — configuración

### Raíz

| Archivo | Qué hace |
|---|---|
| `defaults.yaml` | Valores base de inferencia: etapa, semilla, modelo y prompt por defecto, datos (split, límite, workers) y opciones de inferencia. |
| `prompts.yaml` | Catálogo único de los seis prompts de la etapa 1, cada uno con su texto y su justificación. Se congeló al cerrar la etapa 1. |

Los seis prompts:

| Clave | Qué pide |
|---|---|
| `p0_minimal` | "Describe the image." Línea base sin guía. |
| `p1_attributes` | Nombra los atributos a describir (forma, color, posición) sin imponer formato. Es el prompt usado en la etapa 2. |
| `p2_constrained` | Impone la plantilla del dataset y el vocabulario cerrado de forma y posición; el color queda libre. |
| `p3_json` | Pide la respuesta solo en JSON. |
| `p4_dataset_format` | Replica el formato exacto de la referencia con un ejemplo concreto. |
| `p4m_multiexample` | Igual que p4 pero con tres ejemplos distintos, para detectar copia. |

### `experiments/`

| Archivo | Qué hace |
|---|---|
| `e01_zeroshot_llava_p0.yaml` | Ejemplo de experimento declarativo (LLaVA-OV con p0 en test). El propio archivo dice que es un ejemplo; lo usa `tests/test_config.py`. |

### `models/` — un archivo por modelo (config de inferencia)

| Archivo | Modelo |
|---|---|
| `llava_ov.yaml` | LLaVA-OneVision-0.5B (SigLIP + proyector MLP + Qwen2-0.5B). |
| `internvl3.yaml` | InternVL3-1B (InternViT-300M + MLP + Qwen2.5-0.5B). |
| `qwen35.yaml` | Qwen3.5-0.8B-Base, solo preentrenado. Toma la plantilla de chat de la versión Instruct. |
| `qwen35_instruct.yaml` | Qwen3.5-0.8B con ajuste por instrucciones. Es un control para separar percepción de capacidad de seguir instrucciones; no participa en la etapa 2. |
| `vljepa.yaml` | VL-JEPA (open-vljepa, checkpoint MSRVTT), con la configuración del banco de captions usado para recuperar texto. |

Cada archivo fija el identificador de Hugging Face, la cuantización, los módulos que quedan sin cuantizar, el tamaño de lote y los parámetros de generación.

### `train/` — etapa 2

| Archivo | Qué hace |
|---|---|
| `budget.yaml` | Presupuesto y optimización comunes a los cuatro modelos: 1 época, lote efectivo 64, tasas de aprendizaje, LoRA, selección de checkpoint y evaluación en test. Documenta las decisiones de diseño y la reformulación del 23-10-2026. |
| `models/llava_ov.yaml` | Lo específico de LLaVA-OV: prompt p1, micro-lote 8, conector `multi_modal_projector`. |
| `models/internvl3.yaml` | Lo específico de InternVL3: prompt p1, micro-lote 16, conector `multi_modal_projector`. |
| `models/qwen35.yaml` | Lo específico de Qwen3.5: prompt p1, micro-lote 32, conector `merger`. |
| `models/vljepa.yaml` | Lo específico de VL-JEPA: prompt p1, micro-lote 64, conector `vis_proj`, pérdida InfoNCE con el banco completo como negativos. |

### `queues/` — colas de experimentos

Cada línea de una cola son los argumentos de un run; las ejecuta `scripts/run_queue.sh`.

| Archivo | Qué corre |
|---|---|
| `stage1_pilot.txt` | Piloto de la etapa 1: 4 modelos × 6 prompts sobre 500 imágenes de test. |
| `qwen_p0_p3.txt` | Qwen3.5 Base con los prompts sin ejemplos (p0 a p3), 500 imágenes. Se separó del piloto mientras se diagnosticaba la copia en p4. |
| `p4_sensibilidad.txt` | Experimento de sensibilidad a ejemplos: p4 frente a p4m, 10 000 imágenes. |
| `stage1_full.txt` | Etapa 1 sobre todo test, 4 modelos × 6 prompts. |
| `todos_prompts_n10000.txt` | Etapa 1 sobre 10 000 imágenes de test: 5 modelos (incluye Qwen Instruct) × 6 prompts. |
| `stage2b_smoke.txt` | Prueba de humo de la etapa 2: 20 pasos por modelo, para correr antes que cualquier otra cola `stage2b`. |
| `stage2b_probe.txt` | Sondas lineales sobre los modelos congelados. |
| `stage2b_main.txt` | Entrenamiento principal: objetivo canónico, 1 época, los 4 modelos. |
| `stage2b_main_gpu0.txt` | Mitad A del entrenamiento principal repartido en 2 GPU (LLaVA-OV). |
| `stage2b_main_gpu1.txt` | Mitad B (VL-JEPA, Qwen3.5, InternVL3). No se mezcla con `stage2b_main.txt`. |
| `stage2b_compositional.txt` | Entrenamiento con el split composicional. |
| `stage2b_ablations.txt` | Controles: objetivo original, base en bf16 y segunda semilla. |
| `stage2b_probe_ft.txt` | Las mismas sondas sobre los modelos ya ajustados. |
| `stage2b_test.txt` | Reevaluación en test con los adaptadores elegidos. Normalmente no hace falta, porque `train.py` ya evalúa en test. |

**⚠ No lo tengo claro:** cuáles de estas colas ya se ejecutaron. Los resultados viven en `runs/`, que solo existe en el cluster. Por las figuras de `reports/` se deduce que al menos corrieron `todos_prompts_n10000` y `stage2b_main`.

---

## `tests/` — pruebas

| Archivo | Qué prueba |
|---|---|
| `conftest.py` | Agrega `src/` al path para importar `vlmfid`. |
| `_tiny_vlm.py` | Utilidad de pruebas: un Qwen3.5 diminuto con pesos aleatorios y un processor simulado, para probar entrenamiento e inferencia en CPU sin descargar nada. |
| `test_config.py` | Que la config de cada modelo se componga bien y que la precedencia experimento / línea de comandos sea correcta. |
| `test_models_config.py` | Coherencia de los YAML de modelos: registro, nombres únicos y patrones de exclusión de NF4. |
| `test_prompts.py` | Catálogo de prompts y parseo de respuestas JSON. |
| `test_data.py` | Capa de datos, de layout crudo a Parquet y a registros, sobre un mini-dataset sintético. |
| `test_attributes.py` | Extractor de atributos sobre las 5 plantillas de M3DI, JSON e Índice macro. |
| `test_copying.py` | Detección de copia de los ejemplos del prompt, con casos del piloto real. |
| `test_targets.py` | Caption canónico y split composicional. |
| `test_tracking.py` | Reanudación de runs: qué cambios de configuración se aceptan. |
| `test_jobs.py` | `launch.sh` y `jobs.sh`: el job sobrevive desacoplado y se detiene limpio. Se salta si no hay `setsid`, es decir, en Windows. |
| `test_train_units.py` | Piezas de la etapa 2 sin GPU: presupuesto común, orden de datos, etiquetas y pérdidas. |
| `test_train_e2e.py` | Etapa 2 de punta a punta en CPU con modelos diminutos: LoRA, selección en val, reanudación tras interrupción y carga del adaptador. |

No ejecuté las pruebas al preparar este documento.

---

## `notebooks/` — análisis

| Archivo | Qué hace | Escribe en |
|---|---|---|
| `README.md` | Reglas de uso de los notebooks y tabla de propósitos. **⚠** Está desactualizado: ver puntos dudosos. | — |
| `01_eda_m3di.ipynb` | EDA del split de test de M3DI: integridad del manifiesto, latentes de imagen frente a texto, mapeo código → palabra, color, verificación a nivel de píxel, líneas base y plantillas. | `reports/{figures,tables}/eda/` |
| `02_eda_predicciones.ipynb` | Etapa 1 zero-shot sobre 10 000 imágenes: las 30 corridas modelo × prompt, validación del extractor, copia de ejemplos, fidelidad por atributo, sensibilidad al prompt y control Qwen Base frente a Instruct. | `reports/{figures,tables}/eda_n10000/` |
| `03_eda_top3.ipynb` | Selecciona las tres mejores configuraciones de la etapa 1 y analiza qué perciben, qué confunden y cuándo fallan. | `reports/{figures,tables}/eda_top3/` |
| `03_eda_top3_vljepa.ipynb` | El mismo análisis que el anterior, agregando VL-JEPA a la comparación. | `reports/{figures,tables}/eda_top3/` (misma carpeta) |
| `04_eda_finetuning.ipynb` | Etapa 2: curvas de entrenamiento y validación, test zero-shot frente a ajustado, split composicional, sondas, controles, costo y ejemplos. Cada sección se salta si sus runs no existen. | `reports/{figures,tables}/eda_finetuning/` |
| `05_comparacion_test.ipynb` | Comparación entre modelos en test, zero-shot frente a ajustado: tabla general, ganancia, ranking, desglose por factor, aciertos por imagen y métricas de texto. | `reports/{figures,tables}/comparacion_test/` |
| `06_eda_splits.ipynb` | Compara train, val y test: integridad, fuga entre splits, homogeneidad de atributos, combinaciones forma × posición, dependencias y líneas base de azar. | `reports/{figures,tables}/eda_splits/` |

**⚠ No lo tengo claro:** `03_eda_top3.ipynb` y `03_eda_top3_vljepa.ipynb` comparten número, etiqueta y carpeta de salida. El segundo parece una versión ampliada del primero. No sé si el primero sigue siendo necesario.

---

## `reports/` — resultados

| Ruta | Qué es |
|---|---|
| `VLJEPA_finetuning_metodologia.pdf` | Documento de 8 páginas titulado "Ajuste fino de VL-JEPA: metodología e implementación" (autor en metadatos: Proyecto T7, EL7006). **⚠ No lo tengo claro:** solo pude leer sus metadatos, no su contenido. Tampoco sé qué lo genera: no encontré un script ni un notebook que lo produzca. |
| `figures/.gitkeep` | Mantiene la carpeta en git. |
| `figures/<carpeta>/` | Figuras; casi todas en dos formatos, `.png` y `.pdf`. |
| `tables/<carpeta>/` | Tablas; casi todas en `.csv` y `.tex`, algunas en `.parquet` o `.md`. |

### `eda/` — notebook 01 (dataset, split de test)

Figuras (16, solo `.png`): `test_discretos`, `test_latentes_continuos`, `test_grilla_posiciones`, `test_posicion_pixel`, `test_pixel_color_area`, `test_galeria_formas`, `test_galeria_colores`, `test_color_latente_vs_nombre`, `test_color_terminos_basicos`, `test_color_fondo_foco`, `test_vocab_colores_top`, `test_vocab_colores_espectro`, `test_caption_longitud`, `test_caption_codigo_palabra`, `test_informacion_mutua`, `test_casos_dificiles`.

Tablas (8): `hallazgos_test`, `test_color_nombre_vs_matiz`, `test_desacuerdo_imagen_texto`, `test_entropia_lineas_base`, `test_latentes_continuos`, `test_pixeles_muestra`, `test_plantillas`, `test_vocab_colores`.

### `eda_n10000/` — notebook 02 (etapa 1, las 30 corridas)

| Figura | Contenido (deducido del nombre) |
|---|---|
| `00_diseno_experimental` | Esquema del diseño modelo × prompt. |
| `01_factores_verdaderos` | Distribución de los factores reales. |
| `02_validacion_tono_referencia` | Validación del tono frente al color de la referencia. |
| `03_indicadores_texto`, `04_longitud` | Indicadores y longitud de las respuestas. |
| `05_copia_ejemplos` | Tasa de copia de los ejemplos del prompt. |
| `06_exactitud_modelo_prompt` | Exactitud por modelo y prompt. |
| `07_descomposicion_resultado` | Correcto, incorrecto y omitido. |
| `08_sensibilidad_prompt` | Sensibilidad al prompt. |
| `09_cobertura_vs_condicional` | Cobertura frente a exactitud condicional. |
| `10_qwen_base_vs_instruct` | Control Qwen Base frente a Instruct. |
| `11_confusion_forma`, `12_posicion_grilla`, `13_color_vs_tono` | Errores de forma, posición y color. |
| `15_dificultad_por_imagen` | Dificultad por imagen. |
| `16_posicion_por_eje_metricas`, `17_posicion_confusion_por_eje`, `18_posicion_x_vs_y` | Posición separada por eje horizontal y vertical. |

No existe una figura `14_`.

Tablas (26 archivos): `atributos_extraidos`, `contaminacion_fondo`, `copia_ejemplos`, `dificultad_por_forma`, `exactitud_por_corrida`, `fidelidad_largo`, `mcnemar_modelos`, `posicion_por_eje`, `posicion_x_vs_y_mcnemar`, `qwen_base_vs_instruct`, `resumen_metricas`, `robustez_por_modelo`, `tabla_informe`, `texto_por_corrida`.

El notebook 02 también produce `integridad_corridas`, que se eliminó por ser idéntica a la de `eda_top3`; reaparecerá si se vuelve a ejecutar.

### `eda_top3/` — notebooks 03 (mejores configuraciones y VL-JEPA)

Figuras (34 archivos): `01_panorama_seleccion`, `02_largo_respuestas`, `03_exactitud_atributos`, `04_descomposicion`, `05_confusion_forma`, `06_forma_por_clase`, `07_posicion`, `08_color`, `09_confusion_color`, `10_dificultad`, `11_errores_compartidos`, `12_texto_vs_fidelidad`, `13_ejemplos`, `13_puntos_sobre_azar`, `14_aciertos_por_imagen`, `15_similitud_vs_fidelidad`, `16_ejemplos`.

Abrí `01_panorama_seleccion`: muestra el índice macro por modelo × prompt y el mejor prompt de cada modelo con su intervalo de confianza.

**⚠ No lo tengo claro:** hay dos figuras con el número 13 (`13_ejemplos` y `13_puntos_sobre_azar`) y dos de ejemplos (`13_ejemplos` y `16_ejemplos`). Lo más probable es que `13_ejemplos` venga de `03_eda_top3.ipynb` y que el resto de la numeración 13 a 16 venga de `03_eda_top3_vljepa.ipynb`, pero no lo confirmé ejecutándolos.

Tablas (25 archivos): `aciertos_por_imagen`, `fidelidad_top_largo`, `integridad_corridas`, `panorama_30_corridas`, `pareadas_top`, `resumen_metricas_top`, `resumen_metricas_top_ic`, `seleccion_top`, `similitud_vs_fidelidad`, `texto_top`, `texto_vs_referencia_top`, `top_atributos` (`.parquet`), `vljepa_puntos_sobre_azar`.

### `eda_finetuning/` — notebook 04 (etapa 2)

Figuras (12 archivos): `01_curvas_entrenamiento`, `02_curvas_val`, `03_test_antes_despues`, `04_confusion_forma`, `05_posicion_por_celda`, `09_ejemplos`.

Tablas (14 archivos): `costo`, `formato_respuestas`, `sintesis`, `test_pareadas_zeroshot_vs_ajustado`, `test_zeroshot_vs_ajustado`, `transiciones`, `val_paso_elegido`.

**⚠ No lo tengo claro:** faltan las figuras 06, 07 y 08. El notebook salta las secciones cuyos runs no existen, así que lo más probable es que correspondan al split composicional, las sondas y los controles, y que esos experimentos no se hayan corrido o no estuvieran disponibles al generar las figuras. No pude comprobarlo sin acceso a `runs/`.

### `comparacion_test/` — notebook 05 (zero-shot frente a ajustado)

Figuras (28 archivos): `01_tabla_general`, `02_panorama`, `03_ganancia_ajuste`, `04_matrices_pareadas`, `05_pendientes_ranking`, `06_desglose_factores`, `07_desglose_ganancia`, `08_aciertos_por_imagen`, `09_texto_vs_referencia`, `10_texto_vs_fidelidad`, `11_texto_distribucion`, `12_texto_por_aciertos`, `13_texto_ganancia`, `14_texto_por_plantilla`.

Tablas (22 archivos): `tabla_general`, `ranking`, `ganancia_ajuste`, `pareadas_entre_modelos`, `desglose_factores`, `oraculo`, `sintesis`, `texto_ganancia`, `texto_por_plantilla`, `texto_vs_fidelidad`, `texto_vs_referencia`.

### `eda_splits/` — notebook 06 (train, val y test)

Figuras (18 archivos): `02_vecino_mas_cercano`, `04_discretos_por_split`, `04_matiz_categorias`, `05_combinaciones_forma_posicion`, `06_latentes_continuos`, `07_nombres_categoria`, `08_captions`, `09_dependencias`, `09_matiz_vs_x`.

Tablas (31 archivos): `01_integridad`, `02_captions_compartidos`, `02_fuga_exacta`, `02_vecino_mas_cercano`, `03_desacuerdo_imagen_texto`, `04_color_arcos`, `04_homogeneidad_discretos`, `05_combinaciones`, `06_latentes_continuos`, `07_nombres_color`, `07_nombres_por_categoria`, `08_plantillas`, `09_bases_condicionales`, `09_rango_matiz_por_x`, `10_lineas_base`, `hallazgos`.

En esta carpeta el número del archivo coincide con la sección del notebook.

---

## `docs/` — documentación

| Archivo | Qué contiene |
|---|---|
| `contexto_proyecto.md` | Enunciado del proyecto: descripción, etapas, modelos a comparar, métricas y datos. |
| `ETAPA2_FINETUNING.md` | Diseño original de la etapa 2: decisiones, qué se entrena, funciones de pérdida, hiperparámetros, ejecución en el cluster y qué reportar. |
| `ETAPA2B_REFORMULACION.md` | Cambios a la etapa 2: objetivo canónico, prompt p1 en los cuatro modelos, nueva métrica de selección, negativos del banco en VL-JEPA, train completo, split composicional y sondas. Es el diseño vigente. |
| `GUIA_LAPTOP_CLUSTER.md` | Flujo de trabajo laptop → GitHub → cluster. La parte A describe una puesta al día puntual del 26-09-2026; la parte B, el ciclo diario. |
| `ESTRUCTURA_REPO.md` | Este documento. |

---

## `data/`

| Ruta | Qué es | En git |
|---|---|---|
| `README.md` | Explica las subcarpetas, cómo descargar M3DI y cómo generar manifiestos. | sí |
| `raw/m3di/` | Dataset M3DI publicado, con `train/`, `val/` y `test/` (4,3 GB en este equipo). | no |
| `raw/.gitkeep` | Mantiene la carpeta en git. | sí |
| `generated/` | Destino previsto para variantes del generador (fuera de distribución, nuevas combinaciones). Hoy está vacía. | solo `.gitkeep` |
| `cache/` | Artefactos recalculables, como el banco de embeddings de VL-JEPA. Hoy está vacía. | solo `.gitkeep` |
| `manifests/m3di_base_train.parquet` | Manifiesto de train (8 MB): una fila por muestra con id, ruta de imagen, caption y latentes. | sí |
| `manifests/m3di_base_val.parquet` | Manifiesto de val. | sí |
| `manifests/m3di_base_test.parquet` | Manifiesto de test. | sí |
| `manifests/.gitkeep` | Mantiene la carpeta en git. | sí |

No abrí los Parquet (el Python del sistema no tiene `polars`); el esquema descrito es el que documenta `src/vlmfid/data/manifest.py`.

---

## `external/`

| Ruta | Qué es |
|---|---|
| `README.md` | Explica que el código de terceros entra como submódulo fijado a un commit, sin copiar ni modificar. |
| `open-vljepa/` | Submódulo de `https://github.com/dion-jy/open-vljepa`, fijado en el commit `f92107f`. Contiene `openvljepa/` (el paquete), `configs/`, `demos/`, `scripts/` y su README. Lo usa `src/vlmfid/models/vljepa.py`. No revisé su contenido interno por ser código de terceros. |

---

## `logs/`

Solo contiene `.gitkeep`. En el cluster, `launch.sh` crea aquí `logs/<job>/` con `job.log`, `job.pid`, `meta.env` y `exit_code`.

---

## Puntos que no tengo claros

1. **`README.md` y `notebooks/README.md` están desactualizados.** Ambos nombran `00_eda_m3di.ipynb` y `01_inspeccion_predicciones.ipynb`, que no existen. Los notebooks reales van de `01_eda_m3di` a `06_eda_splits`, y los notebooks 02, 03 y 04 no aparecen en la tabla de `notebooks/README.md`.
2. **`.env.example` no existe.** `docs/GUIA_LAPTOP_CLUSTER.md` lo menciona como plantilla de los secretos, pero no está en el repositorio.
3. **`.gitignore` espera `runs/.gitkeep`**, pero la carpeta `runs/` no existe en el repositorio ni en este equipo.
4. **`notebooks/01_eda_m3di.ipynb` cita el manifiesto `m3di_test.parquet`**; el archivo real se llama `m3di_base_test.parquet`. No sé si el notebook corre tal como está o si solo el texto quedó con el nombre antiguo.
5. **`scripts/probe_p4.py`**: parece un diagnóstico puntual del piloto; no sé si sigue en uso.
6. **`scripts/copy_report.py`** escribe por defecto en `reports/tables/eval/`, carpeta eliminada. Convendría decidir un destino nuevo, por ejemplo `eda_n10000`.
7. **Los dos notebooks 03** comparten número y carpeta de salida; no sé si `03_eda_top3.ipynb` sigue siendo necesario. De ahí viene también la numeración repetida en `reports/figures/eda_top3/`.
8. **Figuras 06 a 08 ausentes en `eda_finetuning/`**: no sé si los experimentos de split composicional, sondas y controles se llegaron a correr.
9. **Qué colas de `configs/queues/` ya se ejecutaron**: solo se puede saber mirando `runs/` en el cluster.
10. **`reports/VLJEPA_finetuning_metodologia.pdf`**: no pude leer su contenido ni identificar qué lo genera.
11. **`configs/train/budget.yaml` conserva en su cabecera las decisiones del 14-10-2026** (objetivo original, mejor prompt por modelo) junto a la reformulación del 23-10-2026 que las reemplaza. No es un error, pero se presta a confusión al leerlo.
