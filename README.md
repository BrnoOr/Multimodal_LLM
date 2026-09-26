# Descripción de escenas controladas con modelos multimodales: VLMs autorregresivos frente a VL-JEPA

Proyecto T7 del curso **EL7006 — Redes Neuronales y Teoría de la Información para el Aprendizaje** (Universidad de Chile).

Un VLM recibe una imagen y un prompt y produce texto. ¿Qué tan fiel es ese texto a lo que realmente hay en la imagen? Se usa **Multimodal3DIdent** (M3DI), un conjunto sintético donde cada imagen se generó a partir de atributos conocidos (forma, color, posición y ángulo del objeto, color del foco y del fondo), lo que permite comparar la descripción generada contra la verdad **atributo por atributo** y no solo por similitud textual.

## Modelos

| Clave (`model=`) | Modelo | Arquitectura | Tokens visuales por imagen de 224 px (aprox.) | Salida |
|---|---|---|---|---|
| `llava_ov` | LLaVA-OneVision-0.5B (`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`) | SigLIP-SO400M (AnyRes) + MLP + Qwen2-0.5B | ≈ 1 500 | texto autorregresivo |
| `internvl3` | InternVL3-1B (`OpenGVLab/InternVL3-1B-hf`) | InternViT-300M (teselas de 448) + MLP + Qwen2.5-0.5B | 256 | texto autorregresivo |
| `qwen35` | Qwen3.5-0.8B-Base (`Qwen/Qwen3.5-0.8B-Base`) | VLM nativo; LM híbrido Gated DeltaNet + atención (3:1) | ≈ 50–64 | texto autorregresivo |
| `vljepa` | open-vljepa (`cun-bjy/open-vljepa`, MSRVTT Stage B) | V-JEPA 2 + predictor Llama-3.2 (8 capas) + EmbeddingGemma | — | embedding ŝ_Y ∈ ℝ¹⁵³⁶; texto por recuperación |

Control opcional: `qwen35_instruct` (`Qwen/Qwen3.5-0.8B`). `qwen35` es un modelo **solo preentrenado**: su resultado zero-shot mezcla percepción con capacidad de seguir instrucciones, y la versión Instruct, derivada del mismo preentrenamiento, permite separarlas.

Los cuatro modelos tienen entre ~0,8 y 1,1 B parámetros en total, por lo que la comparación no está dominada por el tamaño. Las tres arquitecturas autorregresivas difieren sobre todo en el conector y en cuántos tokens dedican a la imagen (hasta un factor ~30), lo que conviene reportar junto a los resultados: el número real queda registrado por muestra en `extra.n_input_tokens`.

Esquema común: **NF4 + doble cuantización, cómputo bf16**, aplicado al transformer de lenguaje (decodificador o predictor); encoder visual, proyector y cabeza de salida quedan en bf16. Tras la carga se verifica que ninguna capa visual haya quedado cuantizada. Con modelos de este tamaño NF4 ahorra poca memoria y degrada relativamente más, así que el costo de NF4 (runs con `model.quant=bf16`) debe reportarse. Adaptación con **LoRA** (etapa 2). Nada se entrena desde cero.

**VL-JEPA no genera tokens.** Su "decodificador ligero" es de recuperación: el Y-Encoder codifica un banco de descripciones del split de *entrenamiento* y se devuelve la más cercana a ŝ_Y. El mismo banco define prototipos por valor de atributo, que dan una predicción por atributo sin pasar por texto (`extra.attr_pred_proto` en las predicciones). El checkpoint procesa video: la imagen se trata como video estático de 8 fotogramas.

## Estructura

```
Multimodal_LLM/
├── configs/                  QUÉ se corre (declarativo)
│   ├── defaults.yaml
│   ├── models/               llava_ov · internvl3 · qwen35 · qwen35_instruct · vljepa
│   ├── prompts.yaml          catálogo de prompts de la etapa 1
│   ├── experiments/          runs con nombre fijo
│   └── queues/               listas de runs para ejecutar en serie
├── data/                     DATOS
│   ├── raw/m3di/             M3DI publicado {train,val,test}             (git-ignored)
│   ├── generated/<variant>/  variantes del generador, mismo layout        (git-ignored)
│   ├── manifests/            m3di_<variant>_<split>.parquet               (versionado)
│   └── cache/                banco de embeddings de VL-JEPA               (git-ignored)
├── external/open-vljepa/     submodule fijado a un commit
├── notebooks/                ANÁLISIS INTERACTIVO (solo lee data/ y runs/)
│   ├── 00_eda_m3di.ipynb
│   └── 01_inspeccion_predicciones.ipynb
├── scripts/                  PUNTOS DE ENTRADA (cómo se corre)
│   ├── build_manifest.py     layout crudo -> Parquet (+ reporte de desacuerdo)
│   ├── infer.py              inferencia reanudable (etapa 1 y modelos ajustados)
│   ├── smoke.py              carga y prueba rápida de los 4 modelos
│   ├── check_env.py          verificación del entorno
│   ├── launch.sh             lanza un job desacoplado de SSH (setsid + nohup)
│   ├── jobs.sh               status · logs · stop · runs · gpu
│   └── run_queue.sh          ejecuta una cola de configs/queues en serie
├── src/vlmfid/               LÓGICA REUTILIZABLE (paquete instalable)
│   ├── paths.py  config.py  tracking.py
│   ├── data/                 m3di.py (layout crudo) · manifest.py (Parquet -> registros)
│   ├── models/               base.py (Describer) · registry.py · quant.py
│   │                         hf_generative.py (genérico por config) · qwen_vl.py
│   │                         vljepa.py
│   ├── prompts/              render y parseo de respuestas JSON
│   └── eval/                 métricas de texto y por atributo            (siguiente hito)
├── tests/
├── logs/<job>/               salida de cada ejecución                      (git-ignored)
└── runs/<exp_id>/            resultados de cada experimento                (git-ignored)
```

La regla entre `scripts/` y `src/`: **la lógica vive en `src/vlmfid/`** (importable desde tests, notebooks y otros scripts) y **cada script solo parsea argumentos y llama a esa lógica**. Por ejemplo, `build_split` y `disagreement` están en `vlmfid.data`; `scripts/build_manifest.py` los invoca y escribe los archivos, y el notebook de EDA usa las mismas funciones.

`logs/` y `runs/` separan dos cosas distintas: `logs/<job>/` es lo que produjo *una ejecución* (salida de consola, código de salida, estado de una cola), mientras que `runs/<exp_id>/` es el *resultado de un experimento* (predicciones y métricas). Una cola genera un solo directorio en `logs/` y varios en `runs/`; relanzar un run interrumpido genera un log nuevo pero continúa el mismo directorio en `runs/`.

Principios: una **interfaz común** (`Describer.describe(images, prompts)`) que oculta las diferencias entre modelos; **predicciones persistidas como datos** (JSONL, incluyen referencia y latentes), de modo que las métricas se recalculan sin re-inferir; **configuración declarativa y reanudable**.

## Uso

Todos los comandos se ejecutan desde la raíz del repo. `uv sync` instala `vlmfid` en modo editable, por lo que los scripts importan el paquete sin ajustes de rutas.

```bash
uv sync --extra quant --extra eval --extra dev            # ver SETUP.md
uv run python scripts/check_env.py
uv run python scripts/build_manifest.py --report-disagreement   # data/manifests/m3di_base_{train,val,test}.parquet
```

### Ejecuciones largas en el cluster (sin tmux)

`scripts/launch.sh` desacopla el proceso de la sesión SSH con `setsid nohup` (estándar en cualquier Linux, no requiere instalar nada). Se puede cerrar la terminal o perder la conexión; el job sigue.

```bash
# smoke test de los 4 modelos (VL-JEPA primero: es el de mayor riesgo)
scripts/launch.sh smoke uv run --no-sync python scripts/smoke.py --set model.bank.max_captions=2000

# un run
scripts/launch.sh llava_p0 uv run --no-sync python scripts/infer.py model=llava_ov prompt=p0_minimal data.limit=500

# una cola completa (en serie, una GPU)
scripts/launch.sh stage1_pilot scripts/run_queue.sh configs/queues/stage1_pilot.txt

scripts/jobs.sh status                 # RUNNING / DONE(rc) / DIED
scripts/jobs.sh logs stage1_pilot -f   # Ctrl-C solo deja de mirar
scripts/jobs.sh runs                   # progreso por run (n_done/n_total, img/s)
scripts/jobs.sh stop stage1_pilot      # SIGTERM: termina el lote en curso y guarda
```

Garantías: `launch.sh` aborta si la GPU elegida (por defecto la 1) ya tiene > 1 GB en uso; `infer` escribe cada lote con `fsync`, así que **relanzar el mismo comando reanuda** donde quedó y un run completo se salta de inmediato; si un lote no cabe en memoria (GPU compartida) se parte en mitades automáticamente.

### Configuración

```bash
uv run python scripts/infer.py model=qwen35 prompt=p2_constrained model.min_pixels=200704 data.limit=500
uv run python scripts/infer.py model=internvl3 model.quant=bf16      # costo de NF4
uv run python scripts/infer.py model=llava_ov data.variant=ood_colors   # variante del generador
uv run python scripts/infer.py experiment=e01_zeroshot_llava_p0
```

`exp_id` por defecto: `<stage>_<model>_<quant>_<prompt>_<split>[_<variant>][_n<limit>]`. Con `data.limit`, el subconjunto es aleatorio pero determinista: **las mismas imágenes para los cuatro modelos**.

## Salidas y reproducibilidad

```
runs/<exp_id>/config.yaml        configuración resuelta (se verifica al reanudar)
runs/<exp_id>/env.jsonl          commit, GPU, versiones, host (un registro por intento)
runs/<exp_id>/status.json        estado, progreso, img/s, memoria pico
runs/<exp_id>/predictions.jsonl  id, variant, prompt, prediction, reference, latentes, extra
runs/<exp_id>/embeddings/*.npz   (VL-JEPA, opcional) ŝ_Y por muestra

logs/<job>/job.log               salida completa de la ejecución (y job.<fecha>.log de intentos previos)
logs/<job>/meta.env              comando, GPU, commit, host, hora de inicio
logs/<job>/exit_code             código de salida al terminar
logs/<job>/queue_status.tsv      (colas) una línea por run: hora, código, duración, argumentos
```

## Agregar otro VLM de transformers

Para un modelo cargable con `AutoModelForImageTextToText` basta un YAML en `configs/models/` con `family: hf` (o `qwen_vl` si es de la familia Qwen-VL), su `hf_id`, los patrones `quant_skip_modules` de su parte visual y `vision_modules`. No hace falta código nuevo: la verificación post-carga detecta patrones mal escritos.

## Manifiestos

`scripts/build_manifest.py` lee el layout crudo y escribe un Parquet por split y variante. Cada fila tiene `id` (`<variant>/<split>/<image_id>`, único entre variantes y clave de reanudación), `image_path` **relativa a la raíz de datos** (el mismo archivo sirve en laptop y cluster), `caption_ref`, los latentes de imagen sin prefijo y los de texto con prefijo `text_`. El script valida que coincidan los cuatro conteos (dos CSV, captions, imágenes) y, con `--report-disagreement`, reporta cuánto difieren los latentes de imagen y de texto: ese desacuerdo acota la exactitud alcanzable.

Los latentes de texto discretos usados para la exactitud por atributo (y para los prototipos de VL-JEPA) se detectan desde los datos, así que el código no depende del nombre exacto de la columna de color.

## Prompts

`configs/prompts.yaml` es un catálogo único de cinco prompts, ordenados de menor a mayor especificación. Cada entrada tiene `id`, `text` y `rationale`, y se congela al cerrar la etapa 1.

| Clave | Qué fija | Qué permite aislar |
|---|---|---|
| `p0_minimal` | nada | línea base; desajuste de dominio |
| `p1_attributes` | qué atributos mencionar | "no lo menciona" frente a "no lo percibe" |
| `p2_constrained` | vocabulario cerrado de forma y posición + plantilla; color libre | efecto del vocabulario de color |
| `p3_json` | salida estructurada (`response_format: json`) | error de extracción ≈ 0 |
| `p4_dataset_format` | formato literal de la referencia, con un ejemplo | exactitud sobre el nombre literal de color |

Las respuestas de `p3_json` se leen con `vlmfid.prompts.parse_json_response`, tolerante a bloques ```` ```json ```` y texto previo; si no hay un objeto recuperable devuelve `None`, que la evaluación debe contar como fallo de formato.

## Etapas y siguiente hito

1. **Inferencia sin ajuste** (este código): modelo × prompt, NF4 y bf16.
2. **Ajuste fino LoRA** con presupuesto común: los adaptadores ya exponen `lora_target_modules()` y cargan `model.adapter_path`; falta el loop de entrenamiento (`src/vlmfid/train/`).

Antes de la etapa 2: `src/vlmfid/eval/` con el extractor de atributos y sus tests (≈100 % sobre los captions de referencia; si no, la métrica mide el parser y no el modelo).

## Datos y licencias

M3DI: Daunhawer et al., *Identifiability Results for Multimodal Contrastive Learning*, ICLR 2023. VL-JEPA: Chen et al., arXiv:2512.10942; reimplementación de J. Baek (Llama 3.2 y Gemma: requieren aceptar sus licencias en Hugging Face). Datos y pesos no se versionan; solo los manifiestos.
