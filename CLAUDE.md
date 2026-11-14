# Multimodal_LLM — Proyecto T7 (EL7006, Universidad de Chile)

> **Estado del repo: en reconstrucción.** El código existente (rama `vl-jepa`)
> NO es confiable: no sigue la metodología acordada. Trátalo como referencia a auditar,
> nunca como base correcta. La fuente de verdad es, en este orden:
> 1. Los papers en `docs/papers/`
> 2. Las decisiones de diseño de este archivo
> 3. El código existente (solo para reutilizar lo que se verifique correcto)
>
> Si un paper contradice una decisión de este archivo, **no elijas tú**: detente, muestra la
> contradicción citando la sección del paper, y pregunta.

---

## 1. Objetivo del proyecto

Comparar **VLMs autorregresivos** vs **VL-JEPA** en la descripción de escenas controladas,
usando **Multimodal3DIdent (M3DI)**, un dataset sintético con atributos conocidos
(posición, color, forma, etc.), lo que permite medir **fidelidad por atributo** además de
métricas de texto.

- Métricas de texto: BLEU, ROUGE, CIDEr.
- Métrica principal de fidelidad: **exactitud por atributo**.
- Posible extensión (fuera de alcance ahora): variantes OOD generadas con el generador de M3DI.

## 2. Modelos

| Modelo | Identificador | Tipo |
|---|---|---|
| LLaVA-OneVision-0.5B | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | autorregresivo |
| InternVL3-1B | `OpenGVLab/InternVL3-1B-hf` | autorregresivo |
| Qwen3.5-0.8B-Base | (verificar ID exacto en HF) | autorregresivo |
| VL-JEPA | reimplementación `cun-bjy/open-vljepa`, checkpoint MSRVTT | predictivo en espacio de embeddings |

**VL-JEPA no genera texto**: produce embeddings. Su fine-tuning sigue el paper (`docs/metodologia.md`
§2.3). Para evaluarlo, el embedding predicho pasa por un **Y-Decoder** (prefijo sobre Llama-3.2-1B,
NF4 + LoRA, entrenado aparte solo con texto; `docs/metodologia.md` §2.4) y el caption resultante se
mide igual que el de los VLM. El paper no especifica ese decodificador: es diseño propio y así debe
declararse. No se entrena desde cero: se parte del
checkpoint preentrenado. (Entrenar VL-JEPA desde cero fue el enfoque de la rama `vl-jepa` y se descartó.)

### ⚠️ Diferencia de escala respecto de los papers
Los papers de `docs/papers/` usan **versiones más grandes** de los modelos (típicamente 7B–8B o más),
mientras que este proyecto usa versiones de **0.5B–1B** por restricciones de hardware. Consecuencias:

- **Qué se toma de los papers:** el *protocolo*. Esto incluye qué componentes se entrenan, la
  función objetivo, el formato de entrada y salida, el procedimiento de evaluación y las métricas.
- **Qué NO se copia directamente:** los hiperparámetros que dependen de la escala, como learning
  rate, rango y alpha de LoRA, tamaño de lote, número de pasos, warmup, longitud máxima de
  secuencia y resolución o número de tiles de imagen. Cada uno se adapta a nuestros modelos.
- **Registro de adaptaciones:** en `docs/metodologia.md` incluye una tabla
  `parámetro | valor en el paper (modelo) | valor nuestro | justificación`. Si no hay una
  justificación clara (otro paper, documentación oficial del modelo pequeño o un barrido corto
  en val), pregunta antes de fijar el valor.
- **Prioridad de fuentes para un modelo pequeño:** si el model card o el repo oficial de la
  versión pequeña recomienda una configuración (p. ej. resolución o plantilla de chat), esa
  manda sobre la del paper del modelo grande en lo que es específico de la arquitectura.
- **Resultados:** no compares números absolutos con los de los papers como si fueran
  equivalentes. La comparación válida es **entre nuestros 4 modelos bajo el mismo protocolo**.
  Si mencionas cifras de un paper, indica siempre el tamaño de modelo al que corresponden.
- **Arquitectura:** verifica que la versión pequeña tenga los mismos componentes que el paper
  (p. ej. tipo de conector o proyector, encoder visual, número de capas sobre las que se aplica
  LoRA). Si difieren, documéntalo y no asumas que los `target_modules` del paper existen con el
  mismo nombre.

## 3. Etapas

### Etapa 1 — Evaluación zero-shot
- Los 4 modelos, preentrenados, sin entrenamiento adicional.
- Varios prompts por modelo; se selecciona el mejor prompt por modelo según exactitud por atributo en **val**
  (media de forma, x, y y color). Esta selección se reporta, pero no decide el prompt de la etapa 2.
- Atributos evaluados: forma, posición x, posición y, color (nombre exacto, sin contar el prefijo de paleta).
- Extracción de atributos desde el texto generado: método explícito, documentado y testeado.
- Control sin cuantizar: cada modelo se evalúa también en bf16 con el prompt seleccionado.
- Métricas de texto con `pycocoevalcap`: Bleu-1, Bleu-4, ROUGE-L, CIDEr.

### Etapa 2 — Fine-tuning
- Modelos: los 4 de la etapa 1 (sin variantes Instruct de Qwen).
- Objetivo: caption original de M3DI con sus 5 plantillas.
- Prompt: **`p1_attributes` en los 4 modelos**.
- Método: **QLoRA NF4**; se entrenan **LoRA + conector multimodal**. Todo componente congelado va en NF4,
  incluido el encoder visual. En VL-JEPA: LoRA en Predictor e Y-Encoder (este con LR ×0.05), proyecciones
  lineales en precisión completa, X-Encoder congelado, InfoNCE bidireccional.
- Control sin cuantizar: una corrida adicional de LoRA sobre base bf16 por modelo, idéntica en lo demás.
- Presupuesto igual para todos: **1 época completa de train con el mismo lote efectivo**
  (ajustar con gradient accumulation según la VRAM de cada modelo).
- Selección de checkpoint: mejor en **val** según exactitud por atributo.
- Reporte final sobre **test**, usado una sola vez. (Los resultados anteriores a la reconstrucción usaron
  test para elegir prompts; se descartan y el informe declara esa exposición previa.)
- Además: pérdida de información por etapa de VL-JEPA (`docs/metodologia.md` §2.5).

### Comparabilidad (no negociable)
Mismos splits, mismas semillas, mismo lote efectivo y mismo protocolo de evaluación para todos
los modelos. Cualquier diferencia entre modelos debe estar justificada y registrada.

## 4. Entornos de ejecución

**Flujo:** editar en el laptop → push a GitHub (`BrnoOr/Multimodal_LLM`) → pull en el cluster → ejecutar.

| | Laptop (desarrollo) | Cluster CILab (ejecución) |
|---|---|---|
| SO | Windows | Linux |
| GPU | RTX 5090 laptop, 24 GB, Blackwell sm_120 | 2× RTX 4090, 24 GB, Ada sm_89 |
| CUDA / PyTorch | cu128 obligatorio | driver CUDA 12.4 → **cu124/cu126, NUNCA cu128** |
| Notas | límites térmicos: nada de corridas largas | sin `nvcc` (solo wheels precompiladas), sin `tmux` (usar `setsid nohup` o `screen`), servidor compartido, disco ~80 % |

- Dependencias con **`uv`** (`pyproject.toml`, markers de plataforma para el build de PyTorch). No usar `pip install` suelto.
- Antes de agregar una dependencia con extensiones CUDA (p. ej. `bitsandbytes`, `flash-attn`), verificar que existan wheels para ambos entornos.
- Por el disco del cluster: guardar solo adaptadores LoRA y el mejor checkpoint, nunca modelos completos ni checkpoints intermedios innecesarios.

## 5. Cómo trabajar en esta reconstrucción

Sigue estas fases **en orden**. Al terminar cada fase, detente y espera aprobación antes de pasar a la siguiente.

1. **Leer los papers** de `docs/papers/`. Escribir `docs/metodologia.md` con: qué toma el proyecto
   de cada paper (citando sección o ecuación), el protocolo resultante para la etapa 1 y la etapa 2,
   y las contradicciones o vacíos encontrados.
2. **Auditar el repo**, tanto la rama actual como `vl-jepa` (`git diff`, `git log`). Escribir
   `docs/auditoria.md` clasificando cada módulo o script en: *reutilizable tal cual* /
   *reutilizable con cambios* / *descartar*, con el motivo referido a `docs/metodologia.md`
   o a este archivo.
3. **Proponer la nueva estructura** de directorios y módulos, y el plan de implementación
   (`docs/plan_reconstruccion.md`).
4. **Implementar la etapa 1** y validarla con una corrida pequeña (pocos ejemplos) antes de la completa.
5. **Implementar la etapa 2** de la misma forma.

Reglas durante la reconstrucción:
- No borres ramas ni historial. Trabaja en la rama `reconstruccion` (creada desde `vl-jepa`). Lo
  descartado se mueve a `_para_borrar/` con su motivo, en commits explícitos; el usuario decide cuándo
  se elimina.
- Nada se ejecuta en el laptop: toda corrida (incluidos tests) se deja preparada para el cluster.
- El split por defecto es `val`. Solo `scripts/final_test.py` lee test.
- No inventes hiperparámetros: cada valor viene de un paper, de este archivo o de una decisión
  que el usuario confirma. Si falta, pregunta.
- No asumas que algo funciona porque corre: valida formas de tensores, que solo los parámetros
  esperados tengan `requires_grad=True` (imprimir el conteo de entrenables) y las métricas
  sobre ejemplos verificados a mano.

## 6. Convenciones de código

- Python + PyTorch + Hugging Face (`transformers`, `peft`, `bitsandbytes`).
- Configuración por archivos (YAML/dataclasses), no constantes en el código. Cada run guarda su
  config, semilla, commit de git y métricas en su carpeta de salida.
- Lógica en módulos `.py` importables; los notebooks solo para análisis y figuras.
- Una sola implementación de las métricas y de la extracción de atributos, compartida por todos los modelos.
- Comentarios y docstrings en español; nombres de variables en inglés.
- Rutas con `pathlib`, sin comandos de shell dentro de Python (portabilidad Windows/Linux).

## 7. Contexto académico

Este código respalda un **informe académico de postgrado**. Al explicar decisiones, sé preciso y
citable: usa la terminología correcta de los papers y distingue entre resultados medidos y supuestos.
