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

Enunciado oficial: **T7 — "Descripción de escenas controladas con modelos multimodales: VLMs
autoregresivos frente a VL-JEPA"**. El PDF está en
`docs/enunciado_T7.pdf`. Ante cualquier duda de alcance, el enunciado manda.

**Pregunta central:** ¿qué tan *fiel* es el texto que genera un VLM a lo que realmente hay en la
imagen? En imágenes naturales no existe una descripción correcta única. **Multimodal3DIdent (M3DI)**
en cambio es sintético: cada imagen se genera a partir de atributos conocidos y el texto de
referencia los describe explícitamente. Esto permite evaluar **atributo por atributo**.

**Atributos generativos:** forma del objeto, color del objeto, posición del objeto, ángulo
(rotación) del objeto, color del foco y color del fondo.

**Métricas:**
- Generación: BLEU, ROUGE, CIDEr (SPICE es opcional; aparece en la bibliografía).
- **Exactitud por atributo** (métrica principal): extraer de la descripción generada el valor de
  cada atributo y compararlo con el verdadero. Un texto puede parecerse mucho a la referencia y
  aun así equivocarse justo en el color o la posición, así que **las conclusiones se basan en esta
  métrica, no en BLEU**.

**Restricciones del enunciado:** todos los modelos son preentrenados y cuantizados (4-8 bit, NF4 o
AWQ) y se adaptan con LoRA. **No se entrena nada desde cero.** Software: PyTorch, HF Transformers,
PEFT y bitsandbytes.

### Preguntas de investigación (el código debe producir la evidencia para responderlas)
1. Sin fine-tuning, ¿qué tan lejos está el texto generado de la referencia y en qué atributos
   falla cada modelo? → métricas **desglosadas por atributo** en la etapa 1.
2. ¿Qué atributos son sistemáticamente más difíciles (posición y ángulo frente a color y forma)?
3. ¿Cuánto mejora cada modelo con fine-tuning, y cuál converge con menos datos o pasos? →
   **curvas de aprendizaje**: evaluar en val (exactitud por atributo) a intervalos regulares
   durante el entrenamiento, no solo al final.
4. ¿La ventaja del VLM moderno sobre LLaVA se traduce en mejores descripciones o solo en texto
   mejor formado? → contrastar las métricas textuales con la exactitud por atributo.
5. ¿Predecir en espacio de embeddings (VL-JEPA) da ventaja frente a generar tokens? ¿A qué costo
   en calidad del texto final?
6. ¿Cómo evaluar de forma justa a VL-JEPA, cuyo texto pasa por un decodificador de lectura
   adicional? → documentar el protocolo en `docs/metodologia.md`.
7. ¿Cómo afecta la cuantización a la calidad de las descripciones y al ajuste? → la cuantización
   es una **variable de estudio**, no solo un recurso para ahorrar memoria.
8. ¿Qué revela la sensibilidad al prompt? → guardar resultados de **todos** los prompts de la
   etapa 1, no solo del mejor.

**Comparación entre etapas:** la diferencia entre la etapa 1 y la etapa 2 es tan informativa como
la diferencia entre modelos. Indica cuánto del error inicial era falta de conocimiento y cuánto era
desalineamiento de formato. Por eso conviene separar, por ejemplo, errores de formato (atributo no
mencionado o imposible de extraer) de errores de contenido (atributo mencionado con un valor
incorrecto).

### Bibliografía del enunciado (los PDFs deberían estar en `docs/papers/`)
| # | Paper | Uso en el proyecto |
|---|---|---|
| [1] | Daunhawer et al., *Identifiability Results for Multimodal Contrastive Learning*, ICLR 2023 (arXiv:2303.09166) | Dataset M3DI y su generador |
| [2] | Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023 (arXiv:2304.08485) | Línea base |
| [3] | Bai et al., *Qwen2.5-VL Technical Report*, 2025 (arXiv:2502.13923) | VLM moderno |
| [4] | Chen et al., *InternVL*, CVPR 2024 (arXiv:2312.14238) | VLM moderno |
| [5] | Chen et al., *VL-JEPA*, 2025 (arXiv:2512.10942) | Modelo de contraste central |
| [6] | Radford et al., *CLIP*, ICML 2021 (arXiv:2103.00020) | Encoders y embeddings |
| [7] | Hu et al., *LoRA*, ICLR 2022 (arXiv:2106.09685) | Adaptación |
| [8] | Dettmers et al., *QLoRA*, NeurIPS 2023 (arXiv:2305.14314) | Cuantización y ajuste |
| [9] | Anderson et al., *SPICE*, ECCV 2016 (arXiv:1607.08822) | Métrica opcional |

Si falta alguno en `docs/papers/`, avísalo en la fase 1 en vez de suplirlo con conocimiento previo.

## 2. Modelos

| Modelo | Identificador | Tipo |
|---|---|---|
| LLaVA-OneVision-0.5B | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | autorregresivo |
| InternVL3-1B | `OpenGVLab/InternVL3-1B-hf` | autorregresivo |
| Qwen3.5-0.8B-Base | (verificar ID exacto en HF) | autorregresivo |
| VL-JEPA | reimplementación `cun-bjy/open-vljepa`, checkpoint MSRVTT | predictivo en espacio de embeddings |

**Desviación respecto del enunciado:** el enunciado plantea 3 modelos (LLaVA, un VLM moderno
entre Qwen2.5-VL e InternVL 3, y VL-JEPA). El equipo usa **4 modelos y versiones pequeñas** para comparar con VL-JEPA
LLaVA-OV como línea base, dos VLMs modernos (InternVL3 y Qwen3.5) y
VL-JEPA. Esta decisión ya está tomada; documenta su justificación en `docs/metodologia.md`.

**VL-JEPA** no genera tokens de forma autorregresiva. Predice el **embedding continuo del texto
objetivo** a partir de la imagen y la consulta, y solo usa un **decodificador ligero** cuando se
necesita texto legible. Por eso:
- Su función objetivo de fine-tuning se toma del paper [5] (pérdida en espacio de embeddings), no
  de la entropía cruzada sobre tokens que usan los VLMs autorregresivos.
- Hay que evaluarlo de **dos formas** y reportar ambas: (a) en espacio de embeddings, y (b) sobre
  el texto que produce el decodificador, con las mismas métricas que los otros modelos. Así se
  separa el error del predictor del error del decodificador (pregunta 6).
- Se usa la reimplementación `cun-bjy/open-vljepa` con el checkpoint MSRVTT. Verifica en la
  fase 1 qué partes del paper implementa realmente (en particular, si trae el decodificador) y
  documenta las diferencias.
- No se entrena desde cero; se parte del checkpoint preentrenado. Entrenar VL-JEPA desde cero fue
  el enfoque de la rama `vl-jepa` y está **descartado**, porque además contradice el enunciado.

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
- Varios prompts por modelo; se selecciona el mejor prompt por modelo según exactitud por atributo en **val**.
- Extracción de atributos desde el texto generado: método explícito, documentado y testeado.

### Etapa 2 — Fine-tuning
- Modelos: los 4 de la etapa 1 (sin variantes Instruct de Qwen).
- Objetivo: caption original de M3DI con sus 5 plantillas.
- Prompt: el mejor zero-shot de cada modelo, **excepto LLaVA-OV, que usa `p1`** (su mejor, `p3`, pide JSON).
- Método: **QLoRA NF4**; se entrenan **LoRA + conector multimodal**.
- Presupuesto igual para todos: **1 época completa de train con el mismo lote efectivo**
  (ajustar con gradient accumulation según la VRAM de cada modelo).
- Evaluación periódica en val (subconjunto fijo) durante el entrenamiento, para construir
  curvas de aprendizaje comparables entre modelos (pregunta 3).
- Selección de checkpoint: mejor en **val** según exactitud por atributo.
- Reporte final sobre **test**, usado una sola vez.

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
3. **Proponer la nueva estructura** de directorios y módulos, y el plan de implementación.
4. **Implementar la etapa 1** y validarla con una corrida pequeña (pocos ejemplos) antes de la completa.
5. **Implementar la etapa 2** de la misma forma.

Reglas durante la reconstrucción:
- No borres ramas ni historial. Trabaja en una rama nueva (`reconstruccion`) creada desde la rama
  que indique el usuario. Lo descartado se elimina en commits explícitos, no se sobrescribe.
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
