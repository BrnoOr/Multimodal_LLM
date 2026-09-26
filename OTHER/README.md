# Descripción de escenas controladas con modelos multimodales: VLMs autorregresivos frente a VL-JEPA

Proyecto del curso **EL7006 — Redes Neuronales y Teoría de la Información para el Aprendizaje** (Universidad de Chile)

## Contexto

Un VLM recibe una imagen y un prompt y produce texto. ¿Qué tan fiel es ese texto a lo que realmente hay en la imagen? En datos naturales la pregunta es difícil de responder porque no existe una descripción correcta única. Aquí se usa Multimodal3DIdent, un conjunto sintético donde cada imagen se generó a partir de atributos conocidos (forma, color, posición y ángulo del objeto, color del foco y del fondo) y cuyo texto de referencia los describe explícitamente. Eso permite comparar la descripción generada contra la verdad atributo por atributo, y no solo mediante similitud textual.

## Modelos comparados

| Modelo | Rol | Característica |
|---|---|---|
| **LLaVA-1.5-7B** | línea base | encoder visual congelado, proyector lineal, decodificador de lenguaje |
| **Qwen2.5-VL-7B** (alt.: InternVL 3) | VLM moderno | conector más expresivo, resolución dinámica |
| **VL-JEPA** | contraste central | predice el *embedding* continuo del texto objetivo; decodificador ligero solo cuando se necesita texto legible |

Los tres se usan preentrenados, cuantizados en **NF4** (mismo esquema para todos, cómputo en bf16) y adaptados con **LoRA**. No se entrena nada desde cero.

## Etapas

1. **Inferencia sin ajuste.** Cada modelo tal como viene preentrenado, con distintos prompts. Establece el punto de partida y expone el desajuste entre el dominio de preentrenamiento (imágenes web) y el dominio sintético.
2. **Ajuste fino y comparación.** Fine-tuning de los tres sobre el mismo conjunto con el **mismo presupuesto de cómputo**, y nueva medición. La diferencia entre etapas indica cuánto del error inicial era falta de conocimiento y cuánto simple desalineamiento de formato.


Adicionalmente se evalúan variantes generadas con el generador de M3DI (nuevas combinaciones de atributos, conjuntos fuera de distribución, descripciones con más o menos detalle) para probar generalización.

## Métricas

- Generación: **BLEU, ROUGE, CIDEr**.
- **Exactitud por atributo**: se extrae de la descripción generada el valor de cada atributo y se compara con el verdadero. Es la medida que responde la pregunta del proyecto — un texto puede parecerse mucho al de referencia y equivocarse justo en el color o la posición.

El extractor de atributos es un componente de primera clase, con tests: aplicado sobre los *captions* de referencia debe alcanzar ≈100 % de exactitud; si no, la métrica mide el parser y no el modelo.

## Estructura del repositorio

```
configs/          modelo × prompt × variante de datos × presupuesto de entrenamiento
data/
  raw/            Multimodal3DIdent publicado          (ignorado por git)
  generated/      variantes construidas con el generador (ignorado por git)
external/
  vl-jepa/        submodule, fijado a un commit
src/vlmfid/
  data/           loader unificado y wrapper del generador
  models/         base.py (Protocol Describer) + adaptadores llava/qwen/vljepa + quant.py
  prompts/        plantillas y parseo de la respuesta
  eval/           text_metrics, attribute_extractor, attribute_metrics
  train/          loop LoRA/QLoRA único, parametrizado por config
  tracking.py     registro de config, commit, GPU y semilla
scripts/          build_manifest, generate_variant, infer, train, evaluate, make_figures,
                  check_env.py, run.sh
runs/             <exp_id>/{config.yaml, predictions.jsonl, metrics.json, adapter/}  (ignorado)
tests/            extractor de atributos, loaders, smoke tests de adaptadores
reports/          figures/, informe/ (LaTeX), presentacion/
```

Tres principios de diseño: una **interfaz común de inferencia** (`Describer`) que oculta las diferencias entre modelos; **predicciones persistidas como datos** (JSONL), de modo que las métricas se recalculen sin re-inferir; y **configuración declarativa**, porque el proyecto es esencialmente una grilla de experimentos.

## Entorno

Ver **[SETUP.md](SETUP.md)** para la instalación completa. Resumen:

| | Laptop | Cluster `deepmind` |
|---|---|---|
| GPU | RTX 5090 (sm_120) | 2× RTX 4090 24 GB (sm_89) — usar la **GPU 1** |
| Driver / build PyTorch | ≥ 570 → **cu128** | 550.120 → **cu126** |
| Rol | desarrollo, tests, iteración | entrenamiento y evaluación |

```bash
uv sync --extra quant --extra qwen --extra eval --extra dev   # laptop
uv sync --extra quant --extra qwen --extra eval --extra track --extra dev   # cluster
uv run python scripts/check_env.py
```

Un solo `pyproject.toml` + `uv.lock` sirve para ambas máquinas: el build de PyTorch se selecciona por plataforma.

## Uso

```bash
# Cluster: run.sh verifica que la GPU esté libre, carga .env y registra GPU + commit
./scripts/run.sh python scripts/build_manifest.py
./scripts/run.sh python scripts/infer.py    experiment=e01_zeroshot_llava_p1
./scripts/run.sh python scripts/train.py    experiment=e11_ft_llava
./scripts/run.sh python scripts/evaluate.py run=runs/e11_ft_llava
uv run python scripts/make_figures.py       # figuras del informe, desde runs/
```

Sesiones largas siempre bajo `tmux`. Las figuras del informe se generan desde `runs/`; nunca a mano en notebooks.

## Reproducibilidad

Cada run guarda su configuración resuelta, el hash del commit, la semilla, `predictions.jsonl` y `metrics.json`. El presupuesto de cómputo común está en `configs/train/budget.yaml` (pasos × batch efectivo, schedule, LR) y se registra junto con las GPU-horas reales, que difieren entre modelos. Cuando un run se ejecuta en el laptop y otro en el cluster, el hardware queda registrado y se reporta.

## Extensión

Si alguno de los modelos resulta suficientemente bueno, el framework puede adaptarse a datos clínicos reales del Instituto de Neurocirugía.

## Datos

Multimodal3DIdent (imágenes renderizadas con atributos generativos conocidos). Se usa tanto el conjunto publicado como su generador. Los datos y los pesos de los modelos no se versionan; solo los manifiestos.
