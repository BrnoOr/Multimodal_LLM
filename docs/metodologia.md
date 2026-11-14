# Metodología — Proyecto T7 (fase 1 de la reconstrucción)

> **Alcance.** Este documento fija el protocolo de las etapas 1 y 2 a partir de los papers de
> `docs/papers/`, del enunciado (`docs/enunciado_T7.pdf`) y de las decisiones de `CLAUDE.md`.
> No describe ni valida el código existente; eso corresponde a `docs/auditoria.md` (fase 2).
>
> **Convención de procedencia.** Cada afirmación lleva una de estas marcas:
> - **[paper]**: dicho por el paper, con sección, ecuación o tabla.
> - **[CLAUDE.md]** / **[enunciado]**: decisión del proyecto.
> - **[datos]**: observado directamente en `data/raw/m3di/` al escribir este documento (lectura de
>   las primeras filas, sin ejecutar código del repo).
> - **[repo, sin verificar]**: afirmado por configs, README o reportes del repo. El repo no es
>   confiable, así que estos puntos son hipótesis a comprobar en la auditoría.
> - **PENDIENTE**: no hay fuente; requiere decisión del usuario. No se fija ningún valor.

## 0. Papers leídos

| Ref. | arXiv | Paper | Uso en el proyecto |
|---|---|---|---|
| [1] | 2303.09166 | Daunhawer et al., *Identifiability Results for Multimodal Contrastive Learning* (M3DI) | dataset, atributos, splits |
| [2] | 2304.08485 | Liu et al., *Visual Instruction Tuning* (LLaVA) | paradigma de la línea base, objetivo autorregresivo |
| [3] | 2502.13923 | Bai et al., *Qwen2.5-VL Technical Report* | referencia del conector de la familia Qwen-VL |
| [4] | 2312.14238 | Chen et al., *InternVL* | origen de la familia InternVL (arquitectura distinta a la que usamos) |
| [5] | 2512.10942v2 | Chen et al., *VL-JEPA* | objetivo, evaluación y rol del Y-Decoder |
| [6] | 2103.00020 | Radford et al., CLIP | InfoNCE simétrica, evaluación zero-shot por candidatos |
| [7] | 2106.09685 | Hu et al., LoRA | reparametrización de bajo rango |
| [8] | 2305.14314 | Dettmers et al., QLoRA | NF4, double quantization, paged optimizers |
| [9] | 1607.08822 | Anderson et al., SPICE | antecedente de la evaluación por atributos |
| — | 2408.03326 | Li et al., *LLaVA-OneVision* | arquitectura real de nuestro modelo LLaVA |
| — | 2504.10479 | Zhu et al., *InternVL3* | arquitectura real de nuestro modelo InternVL |
| — | 2407.10671 / 2412.15115 | *Qwen2* / *Qwen2.5 Technical Report* | LLM de LLaVA-OV-0.5B / de InternVL3-1B |
| — | 2604.15804 | *Qwen3.5-Omni Technical Report* | única fuente en la carpeta sobre la arquitectura Qwen3.5 |
| — | 2412.06464 | Yang et al., *Gated Delta Networks* | capas Gated DeltaNet del LLM de Qwen3.5 |
| — | 2303.15343 | Zhai et al., SigLIP | encoder visual de LLaVA-OV |
| — | 2506.09985 | Assran et al., V-JEPA 2 | X-Encoder de VL-JEPA |
| — | 2509.20354 | Vera et al., EmbeddingGemma | Y-Encoder de VL-JEPA |

Profundidad de lectura: [1], [2], [5], [8], LLaVA-OneVision e InternVL3 se leyeron en las secciones
de método, implementación y apéndices de hiperparámetros. El resto se consultó de forma dirigida
(arquitectura y configuración). No se revisaron sus secciones de resultados.

---

## 1. Qué toma el proyecto de cada paper

### 1.1 [1] Multimodal3DIdent

**Atributos (Tabla 3, §5.2, Ap. B.1).** La imagen se genera con 10 factores (más `z`, constante):

| Factor | Tipo | En la imagen | En el texto |
|---|---|---|---|
| forma del objeto | categórico, 7 valores | sí | sí (contenido) |
| posición x | categórico, 3 valores | sí | sí (contenido) |
| posición y | categórico, 3 valores | sí | sí (contenido) |
| color del objeto | matiz continuo en [0, 1] | sí | sí, como **nombre de color** (estilo) |
| rotación del objeto (α, β, γ) | uniforme en [0, 1] | sí | **no** |
| posición del foco | uniforme en [0, 1] | sí | **no** |
| color del foco | matiz en [0, 1] | sí | **no** |
| color del fondo | matiz en [0, 1] | sí | **no** |
| fraseo | categórico, 5 valores | no | sí (específico del texto) |

- Forma y posición se comparten **siempre** entre imagen y texto (factores de contenido) [paper, §5.2].
- El color se comparte **estocásticamente**: el matiz continuo se convierte a RGB y se asigna al
  nombre más cercano de **una de tres paletas elegida al azar por muestra** (Tableau 10 valores,
  CSS4 148, XKCD 949, nota 7 del Ap. B.1), con prefijo de paleta cuando hay colisión
  (p. ej. `xkcd:black`). El paper lo diseña así a propósito: *"the object color depicted in the image
  does not uniquely determine the color described in the text"* [paper, §5.2].
- En la versión usada en §5.2 hay una dependencia causal contenido → estilo: el rango de matiz se
  divide en tres intervalos iguales y cada uno se asocia a una posición x (p. ej. x = left ⇒ matiz en
  [0, 1/3]) [paper, Ap. B.1]. Existe además una versión con factores mutuamente independientes
  [paper, Ap. B.2, Fig. 11].

**Formato de los captions.** El paper solo dice que adapta el render de texto de CLEVR y usa 5 tipos
de frase [paper, §5.2, Ap. B.1]; **no lista las plantillas**. Las plantillas observadas en
`data/raw/m3di/train/text/text_raw.txt`, cruzadas con `latents_text.csv` [datos]:

| `text_phrasing` | Plantilla |
|---|---|
| 0 | `A {shape} of "{color}" color is visible, positioned at the {y}-{x} of the image.` |
| 1 | `A "{color}" {shape} is at the {y}-{x} of the image.` |
| 2 | `The {y}-{x} of the image shows a "{color}" colored {shape}.` (asignación por descarte, a confirmar) |
| 3 | `At the {y}-{x} of the picture is a {shape} in "{color}" color.` |
| 4 | `At the {y}-{x} of the image, there is a "{color}" object in the form of a {shape}.` |

Detalles observados [datos]: el nombre de color va entre comillas dobles y conserva el prefijo de
paleta (`tab:blue`, `xkcd:hot pink`, `red`); la posición es `{vertical}-{horizontal}` con vertical en
{top, mid, bottom} y horizontal en {left, center, right}; el artículo es siempre `A` (aparece
`A armadillo`). El paper habla de *top/center/bottom*, los datos usan **`mid`** para el eje vertical.
El mapeo índice → forma debe leerse del generador, no inferirse.

**Splits.** 125 000 / 10 000 / 10 000 (train / val / test) [paper, Tabla 2b]. Se usan los splits
publicados, sin re-particionar [CLAUDE.md, comparabilidad].

**Generador.** Blender + código derivado de Causal3DIdent y CLEVR [paper, Ap. B.1]. El enunciado
permite usarlo para variantes (nuevas combinaciones, OOD, más o menos detalle). **Fuera de alcance
ahora** [CLAUDE.md §1]. Lo único que se necesita del generador en esta fase es de lectura: el mapeo
índice → forma, las 5 plantillas literales y la regla matiz → nombre de color.

**Lo que NO se toma de [1]:** sus encoders (ResNet-18 y ConvNet), su entrenamiento contrastivo y su
evaluación de identificabilidad (R², MLP sobre representaciones).

### 1.2 [2] LLaVA y LLaVA-OneVision

**LLaVA original [paper [2], §4.1–4.2].**
- Arquitectura: CLIP ViT-L/14 → **proyección lineal** `H_v = W·Z_v` (ec. 1) → Vicuna.
- Objetivo: verosimilitud autorregresiva de la respuesta (ec. 3). La pérdida se calcula **solo sobre
  los tokens de la respuesta del asistente y el token de parada** (Tabla 2).
- Qué se entrena: etapa 1, solo `W` con encoder visual y LLM congelados; etapa 2, `W` + LLM, con el
  **encoder visual siempre congelado** (§4.2).
- Formato: `X_system <STOP> Human: X_instruct <STOP> Assistant: X_a <STOP>`, con `<STOP> = ###`
  (Vicuna-v0). En la etapa 1 la instrucción es una petición de descripción breve y la respuesta es
  el caption original.
- Hiperparámetros (Vicuna 7B/13B, 8×A100; §5 y Ap.): etapa 1, 1 época, LR 2e-3, lote 128; etapa 2,
  3 épocas, LR 2e-5, lote 32; Adam sin weight decay, coseno, warmup 3 %.

**LLaVA-OneVision [paper 2408.03326, §3, Tabla 1, Ap. C].** Es el modelo que usamos; difiere de [2]:
- LLM **Qwen-2**; encoder visual **SigLIP SO400M** a 384 px (729 tokens por vista); proyector
  **MLP de 2 capas**, no lineal (§3).
- Representación visual **AnyRes**: vista base + recortes; hasta 729 × (1 + 9) tokens por imagen
  (Ap. C.1).
- Plantilla **ChatML** de Qwen-2 y marcador `<image>` (Ap. C.2).
- Qué se entrena: etapa 1, solo el proyector; etapas 1.5 y 2, **modelo completo incluido el encoder
  visual**, con LR del encoder 5 veces menor (§5, Tabla 1).
- Hiperparámetros de la variante **0.5B** (Tabla 1): lote global 512; LR 1e-3 (etapa 1, proyector);
  LR 1e-5 para {proyector, LLM} y 2e-6 para visión (etapas 1.5 y 2); 1 época por etapa.
- LLM Qwen2-0.5B: 24 capas, dimensión oculta 896 [paper 2407.10671, Tabla 1].

**Qué toma el proyecto:** el objetivo autorregresivo con pérdida solo sobre la respuesta; el formato
instrucción → caption original de la etapa 1 de [2]; el encoder visual congelado de [2] §4.2; la
plantilla ChatML y el preprocesado propios de LLaVA-OV-0.5B.

### 1.3 [4] InternVL e InternVL3

**InternVL [paper [4], §3.2–3.3, §4.1].** InternViT-6B + QLLaMA (middleware de 8B parámetros con
96 queries y cross-attention) + LLM (Vicuna-13B) unido por una capa MLP. En la etapa 3 (SFT) el LLM
puede quedar congelado y se entrena solo el MLP, o el MLP y QLLaMA.

**InternVL3 [paper 2504.10479, §2.1–2.3, Tabla 1].** Es el modelo que usamos:
- Paradigma **ViT-MLP-LLM**. InternVL3-1B = **InternViT-300M-448px-V2.5 + Qwen2.5-0.5B** (0.9B
  parámetros). No hay QLLaMA.
- Conector: **MLP de 2 capas** precedido de **pixel unshuffle**, que reduce los tokens a un cuarto:
  **256 tokens por tesela de 448×448** (§2.1).
- Resolución dinámica por teselas (heredada de InternVL2.5) y V2PE para posiciones visuales (§2.1).
- El LLM se inicializa desde el modelo **base** de Qwen2.5, no desde la variante instruct (§2.1).
- Objetivo autorregresivo estándar (ec. 5), con pérdida restringida a los tokens de texto en el
  preentrenamiento nativo (§2.2). En SFT se usa *square loss re-weighting* (§2.3).
- Qué se entrena: **todas las capas conjuntamente** (§2.2), en contraste explícito con los esquemas
  que congelan partes.

**Qué toma el proyecto:** la identificación correcta de los componentes (encoder, conector, LLM)
para decidir qué se congela y dónde va LoRA; el objetivo autorregresivo. El formato de prompt se
toma del model card y de la plantilla de chat del checkpoint `-hf` [CLAUDE.md §2, prioridad de
fuentes], porque el paper no la detalla.

### 1.4 Qwen3.5-0.8B-Base

No hay en `docs/papers/` un reporte técnico de Qwen3.5. Lo disponible:
- *Qwen3.5-Omni* [paper 2604.15804, §2–3]: su encoder visual es **el de Qwen3.5**, listado como
  **SigLIP2** en la tabla de módulos; el backbone Qwen3.5 es una arquitectura híbrida que incluye
  módulos **Gated Delta Net (GDN)**. El reporte describe el modelo Omni (Thinker-Talker, MoE), no el
  0.8B.
- *Gated Delta Networks* [paper 2412.06464]: capas de atención lineal con regla delta con compuerta,
  y arquitecturas híbridas que las combinan con atención.
- *Qwen2.5-VL* [3], §2.1: conector de la familia Qwen-VL = agrupar 4 parches adyacentes (2×2),
  concatenarlos y pasarlos por un **MLP de 2 capas** (*merger*); entrada redimensionada a múltiplos
  de 28 con parches de 14. Sirve como referencia de familia, **no** como especificación de Qwen3.5.

Consecuencia: la arquitectura exacta del 0.8B (número de capas, proporción GDN/atención, tamaño de
parche, ubicación del merger, plantilla de chat) debe leerse del **model card y del `config.json`**
del checkpoint. El ID `Qwen/Qwen3.5-0.8B-Base` aparece en `configs/models/qwen35.yaml` [repo, sin
verificar] y sigue pendiente de verificación en HF [CLAUDE.md §2].

### 1.5 [5] VL-JEPA

**Componentes [paper, §2, §3.1].**
- **X-Encoder**: V-JEPA 2 ViT-L (304M), **congelado**, 256² px; una imagen se **duplica** para
  igualar la forma de entrada de video.
- **Predictor**: últimas 8 capas de Llama-3.2-1B (490M entrenables), con tokenizador y embeddings de
  Llama; **sin máscara causal**; hasta 512 tokens de consulta con `[PAD]`; proyecciones lineales de
  entrada y salida; **promedio sobre tokens no-PAD** para obtener el embedding predicho.
- **Y-Encoder**: EmbeddingGemma-300M, contexto máximo 512, con **multiplicador de LR ×0.05**.
  Cabezas de proyección lineales en Predictor e Y-Encoder hacia un espacio compartido de **1 536
  dimensiones**, donde se calcula la pérdida.
- **Y-Decoder**: *"is not involved during the main training phase"*; en inferencia traduce el
  embedding predicho a texto *"when necessary"* (§2).

**Función objetivo [paper, §2 "Training Objective"].** **InfoNCE bidireccional** entre el embedding
predicho `Ŝ_Y` y el objetivo `S_Y`, entrenando conjuntamente Predictor e Y-Encoder. InfoNCE aporta el
término de alineación y el de uniformidad (anti-colapso). El paper admite alternativas (EMA o
Y-Encoder congelado). Ablación (Tabla 7c, con Y-Encoder congelado y sin cabeza de proyección):
InfoNCE supera a coseno, L1 y L2 en clasificación y recuperación; coseno gana solo en VQA, y *"only
InfoNCE has the anti-collapse regularization and can be applied with unfrozen Y-Encoder"* (§4.8).
El paper **no indica la temperatura**.

**Entrenamiento en dos etapas [paper, §3.2].** (i) Preentrenamiento **sin consulta** (*query-free*)
sobre captions, LR constante 5e-5, lote 24k en la fase de imagen; (ii) SFT **condicionado por
consulta**, 83k pasos, lote 3 072, coseno.

**Evaluación [paper, §2 "Multi-tasking", §4.1, §4.2, §4.5].**
- **Clasificación de vocabulario abierto y VQA discriminativo**: se codifican los textos candidatos
  con el Y-Encoder y se elige el **más cercano al embedding predicho** (protocolo tipo CLIP).
- **Recuperación texto → video**: se ordena por similitud entre el embedding predicho y la consulta
  codificada.
- **Generación (captioning)**: se decodifica el embedding con el Y-Decoder y se reporta CIDEr (§4.5).

**Rol del Y-Decoder: vacío del paper.** La versión v2 no describe su arquitectura, sus datos ni su
entrenamiento; solo lo llama *lightweight* y *readout text decoder* (§1, §2, §5). No hay apéndice.

**Reimplementación `open-vljepa`** [repo: `external/open-vljepa/README.md` y código, sin ejecutar]:
- Misma arquitectura declarada: `facebook/vjepa2-vitl-fpc64-256` congelado, últimas 8 capas de
  Llama-3.2-1B sin máscara causal y con promedio enmascarado, EmbeddingGemma + proyección, espacio
  de 1 536-D, InfoNCE bidireccional.
- Diferencias respecto del paper: τ = 0.07 fijado por el autor; consulta fija `"Describe the
  image."` en el preentrenamiento de imagen (el paper lo hace sin consulta); `max_query_len` 64 en
  vez de 512; 8 cuadros; lote efectivo 512; LR 1e-4 (predictor) y 5e-6 (Y-Encoder).
- Datos: CC3M (~1M) y luego MSRVTT (140K), frente a ~3.3B muestras del paper. El README reporta
  R@1 = 23.9 en MSRVTT (pool de 500) y declara que `best.pt` se selecciona por pérdida de
  entrenamiento, no por R@1.
- **No incluye Y-Decoder.**

### 1.6 [8] QLoRA y [7] LoRA

**LoRA [paper [7], §4.1–4.2].** `W₀ + ΔW = W₀ + BA`, con `B ∈ R^{d×r}`, `A ∈ R^{r×k}`; `A` gaussiana y
`B = 0` al inicio; `ΔW·x` se escala por `α/r`; `W₀` congelado. En [7] se adaptan solo `W_q` y `W_v`.

**QLoRA [paper [8], §3, Ap. B.2].**
- **NF4** (4-bit NormalFloat): tipo de dato óptimo en teoría de la información para pesos con
  distribución normal; se usa para `W`.
- **Double Quantization**: cuantiza las constantes de cuantización (FP32 → FP8, bloque de 256);
  reduce de 0.5 a 0.127 bits por parámetro (bloque de 64 para `W`).
- **Tipo de cómputo BFloat16**: los pesos se descuantizan a bf16 para el forward y el backward
  (ec. 5–6). Solo los adaptadores reciben gradiente.
- **Paged Optimizers**: memoria unificada de NVIDIA para paginar el estado del optimizador a RAM
  cuando la GPU se queda sin memoria. El paper los declara críticos para 33B y 65B y no da mediciones
  para modelos pequeños (§4).
- **Dónde va LoRA**: *"LoRA on all linear transformer block layers are required to match full
  finetuning performance"*; el rango `r` no afecta el resultado si LoRA está en todas las capas
  (§4, Ap. A.2).
- Hiperparámetros (LLaMA 7B/13B, Ap. B.2, Tabla 9): r = 64, α = 16, dropout 0.1 (hasta 13B), LR 2e-4,
  lote 16, LR constante, Adam β₂ = 0.999, max grad norm 0.3, *group-by-length*.

**Qué toma el proyecto:** NF4 + double quantization + cómputo en bf16; LoRA en todas las capas
lineales de los bloques del LLM. Los valores numéricos dependen de la escala (§4 de este documento).

### 1.7 [6] CLIP, SigLIP, V-JEPA 2, EmbeddingGemma y [9] SPICE

- **CLIP [6], §2.3 y §3.1.4**: pérdida de entropía cruzada simétrica con temperatura aprendida (la
  InfoNCE de VL-JEPA); evaluación zero-shot por similitud con textos candidatos; el *prompt
  engineering* de los textos candidatos cambia el resultado.
- **SigLIP [2303.15343]**: pérdida sigmoide por pares; es el preentrenamiento del encoder de LLaVA-OV.
- **V-JEPA 2 [2506.09985], §2**: parchea en *tubelets* de 2×16×16 y se evalúa con encoder congelado.
- **EmbeddingGemma [2509.20354], §2–3**: 308M parámetros, atención bidireccional, *mean pooling*,
  768-D, y **prompts de tarea** antepuestos al texto. Si el Y-Encoder usa o no ese prefijo depende
  del checkpoint; se verifica en la auditoría.
- **SPICE [9], §3**: F-score sobre tuplas (objetos, atributos, relaciones) de un grafo de escena;
  motiva que las métricas de n-gramas no miden corrección de atributos. Es el antecedente conceptual
  de la exactitud por atributo. SPICE **no** está entre las métricas exigidas por el enunciado.

---

## 2. Protocolo resultante

### 2.1 Común a ambas etapas

- **Datos**: M3DI publicado, splits 125k / 10k / 10k [paper [1]]. Train para ajustar, val para
  seleccionar (prompt y checkpoint), test **una sola vez** al final [CLAUDE.md §3].
- **Atributos evaluados**: forma, posición x, posición y, color del objeto [decisión del usuario,
  C2]. Son los únicos que aparecen en el texto de referencia [paper [1], Tabla 3].
- **Verdad de referencia por atributo**: forma, x, y y nombre de color desde `latents_text.csv`.
- **Color**: acierto por **nombre exacto** (`object_color_name`) [decisión del usuario, C3]. Como la
  paleta se elige al azar [paper [1], §5.2], el techo desde la imagen es menor que 1; se reporta
  junto al resultado el techo empírico (mejor predicción del nombre dado el matiz verdadero,
  estimada en train). **El prefijo de paleta no cuenta**: predicción y referencia se comparan tras
  quitar `tab:` y `xkcd:`, de modo que `blue` acierta para `tab:blue` [decisión del usuario, D1].
  Consecuencia a declarar: nombres homónimos de paletas distintas (p. ej. `blue`, `tab:blue` y
  `xkcd:blue`, que son RGB distintos) se tratan como la misma clase. El objetivo de entrenamiento
  de la etapa 2 sigue siendo el caption original, con prefijo.
- **Métricas de texto**: **Bleu-1, Bleu-4, ROUGE-L y CIDEr**, calculadas con el **código de
  evaluación de MS COCO**, contra el caption original (una referencia por imagen). Implementación:
  **`pycocoevalcap`** [decisión del usuario]; su tokenizador necesita Java, a comprobar en el cluster. Es lo que usan
  los papers de la carpeta: [9] reporta exactamente esas cuatro y declara *"we use the MS COCO
  evaluation code"* (§5.1–5.3); [5] reporta CIDEr para captioning (§4.5–4.6) sin dar implementación.
  Ningún paper de la carpeta especifica variante de CIDEr ni tokenización más allá de eso; al
  adoptar el código de MS COCO se heredan su tokenizador y su CIDEr (según la documentación de ese
  código, tokenizador PTB y variante CIDEr-D: **a confirmar en la fase 3**, no proviene de un paper
  de la carpeta). El idf de CIDEr se calcula sobre las referencias del split evaluado; [9] §3.2
  advierte que CIDEr depende de estadísticas del corpus, por lo que sus valores en val y test no son
  comparables entre sí. Una sola implementación para todos los modelos [CLAUDE.md §6].
- **Métrica principal**: exactitud por atributo [enunciado, CLAUDE.md §1]. Ningún paper prescribe el
  extractor texto → atributos; se especifica, documenta y testea en la fase 4 [CLAUDE.md §3].
- **Cuantización**: NF4 + double quantization + cómputo bf16 [paper [8]; enunciado].
- **Qué se cuantiza** [D5, resuelto con los criterios del proyecto]: en los 4 modelos, **todo
  componente congelado va en NF4, incluido el encoder visual** (también el X-Encoder de VL-JEPA);
  quedan fuera solo los módulos que se entrenan en precisión completa (conector o proyecciones) y
  los adaptadores LoRA. Fundamento: el enunciado pide modelos *"preentrenados y cuantizados"* sin
  excepciones; en [8] se cuantiza el modelo base completo (§3); y una regla única es la que hace
  comparables los 4 modelos [CLAUDE.md, comparabilidad]. Dejar la visión en bf16 haría además que el
  control sin cuantizar midiera solo la cuantización del LLM. A verificar por modelo: la
  cuantización de 4 bits se aplica a capas lineales, así que capas de otro tipo (p. ej. la
  convolución de parches) pueden quedar sin cuantizar; se registra qué fracción de parámetros queda
  efectivamente en NF4 en cada modelo.
- **Control sin cuantizar** [decisión del usuario, C14]: cada modelo se evalúa además en **bf16**
  (el tipo de cómputo de [8], sin cuantizar los pesos), con el mismo prompt, semilla y protocolo.
  La diferencia NF4 − bf16 responde la pregunta del enunciado sobre la cuantización. Alcance en
  §2.2 (inferencia) y §2.3 (ajuste).
- **Decodificación de los VLM**: determinista, con la misma configuración para los tres. La longitud
  máxima es dependiente de escala y se registra en §4.

### 2.2 Etapa 1 — zero-shot

1. Los 4 modelos preentrenados, sin entrenamiento [CLAUDE.md §3].
2. **VLMs autorregresivos**: imagen + prompt con la plantilla de chat propia de cada checkpoint
   (ChatML en LLaVA-OV [paper 2408.03326, Ap. C.2]; model card en InternVL3 y Qwen3.5). Se genera
   texto, se extraen atributos y se calculan las métricas.
3. **VL-JEPA**: la consulta `X_Q` es el prompt [paper [5], §2]. El embedding predicho `Ŝ_Y` se pasa
   por un **Y-Decoder** que produce el caption predicho; ese texto se compara con el original con
   las mismas métricas y el mismo extractor que los VLM [decisión del usuario, C5; paper [5], §2 y
   §4.5: *"VL-JEPA decodes the predicted embeddings while VLM generates the tokens directly"*].
   El Y-Decoder se define en §2.4.
4. Varios prompts por modelo; se elige el mejor por exactitud por atributo en **val** [CLAUDE.md §3].
   El agregado para comparar prompts y checkpoints es la **media de forma, x, y y color**, con
   cada atributo reportado por separado (valor por defecto, revisable).
5. **Control bf16**: con el prompt seleccionado, cada modelo se evalúa en val también sin cuantizar.
   Se usa solo para medir el efecto de la cuantización, no para seleccionar prompts.
6. Test no se toca en esta etapa. En el reporte final sobre test se incluyen, para cada modelo, las
   condiciones zero-shot NF4, zero-shot bf16 y ajustado NF4.

### 2.3 Etapa 2 — fine-tuning

**VLMs autorregresivos (LLaVA-OV, InternVL3, Qwen3.5-Base).**
- Objetivo: caption original de M3DI con sus 5 plantillas [CLAUDE.md §3]. Entropía cruzada
  autorregresiva **solo sobre los tokens de la respuesta** [paper [2], ec. 3 y Tabla 2].
- Entrada: imagen + prompt en el turno de usuario, caption en el turno del asistente, con la
  plantilla de chat del modelo. Prompt: **`p1_attributes` en los 4 modelos** [decisión del
  usuario; reemplaza la regla "mejor zero-shot de cada modelo" de CLAUDE.md §3, que debe
  actualizarse]. La selección de prompts de la etapa 1 se reporta, pero no decide el del ajuste.
- Entrenables: **LoRA en el LLM + conector multimodal** [CLAUDE.md §3]. Encoder visual congelado
  [paper [2], §4.2]. LoRA en todas las capas lineales de los bloques del LLM [paper [8], §4].
- Base cuantizada en NF4 [paper [8]]. El conector se entrena en precisión completa, por lo que no
  puede cuantizarse (§5, C8).

**VL-JEPA.** Se parte del checkpoint MSRVTT de `open-vljepa`; no se entrena desde cero
[CLAUDE.md §2]. El fine-tuning sigue al paper [decisión del usuario, C7].

Lo que el paper fija sin alternativas:
- **X-Encoder congelado** (§3.1: *"we use a frozen V-JEPA 2 ViT-L"*). No hay ningún experimento del
  paper con el X-Encoder entrenable.
- Tripletas ⟨imagen, consulta, objetivo⟩; la consulta es el prompt de captioning y el objetivo es el
  caption (§2, "Multi-tasking").
- El Y-Decoder no participa del entrenamiento principal (§2).
- El fine-tuning a una tarea nueva existe en el paper (§4.4, EPIC-KITCHENS-100 y COIN), pero **sin
  detalles**: no indica módulos, pérdida ni hiperparámetros. Se asume la receta de SFT de §3.2.

Opciones que el paper sí documenta para Predictor, Y-Encoder y pérdida:

| Opción | Predictor | Y-Encoder | Pérdida | Evidencia en el paper |
|---|---|---|---|---|
| **A (configuración por defecto)** | entrenable | entrenable, LR ×0.05 | InfoNCE bidireccional | §2, §3.1; Tabla 7b: ×0.05–0.10 es el óptimo |
| B | entrenable | **congelado**, con proyección lineal entrenable encima | InfoNCE | Tabla 7e |
| C | entrenable | congelado, sin cabeza de proyección | InfoNCE, coseno, L1 o L2 | Tabla 7c; §2 menciona congelar el Y-Encoder como estrategia anti-colapso |
| D | entrenable | EMA del Y-Encoder | — | solo mencionada en §2; sin experimentos |

Lectura de las ablaciones (modelo de 1.6B, 5M muestras vistas, Tabla 7): congelar del todo el
Y-Encoder (multiplicador 0.00) baja clasificación en 7.3 puntos y recuperación en 4.3 respecto de
×0.05; un multiplicador de 1.00 también empeora (−3.6 en clasificación). Con Y-Encoder congelado,
InfoNCE supera a coseno, L1 y L2 en clasificación y recuperación, y coseno gana solo en VQA
(+2.3). El paper concluye que *"only InfoNCE has the anti-collapse regularization and can be applied
with unfrozen Y-Encoder"* (§4.8).

**Protocolo adoptado: opción A**, por ser la del paper: Predictor e Y-Encoder entrenables, Y-Encoder
con LR ×0.05, InfoNCE bidireccional en el espacio compartido de 1 536-D, X-Encoder congelado.
Adaptaciones obligadas, que el paper no cubre y quedan registradas en §4:
- El paper hace ajuste completo; el enunciado exige LoRA sobre base cuantizada. Traducción
  propuesta, paralela a la de los VLM: **LoRA en las capas lineales de las 8 capas del Predictor y
  del Y-Encoder** (este con LR ×0.05), y **proyecciones lineales entrenables en precisión completa**
  (entrada visual del Predictor y cabezas de proyección de Predictor e Y-Encoder), que cumplen el
  rol de "conector" de CLAUDE.md §3 [confirmado por el usuario, D3].
- **Interacción con el Y-Decoder**: en la opción A el espacio objetivo cambia durante el ajuste, de
  modo que un decodificador entrenado sobre los embeddings del Y-Encoder preentrenado deja de ser
  válido y debe reentrenarse sobre el Y-Encoder ajustado. En las opciones B y C el Y-Encoder no
  cambia, aunque en B la proyección sí. Esto no altera la elección de A, pero fija el orden de
  trabajo: primero se ajusta VL-JEPA, después se entrena el Y-Decoder de la etapa 2 (§2.4).

**Presupuesto para VL-JEPA.** El paper da el criterio en su comparación controlada (§4.5): VL-JEPA
y el VLM usan *"the same training iterations with the same effective batch size of 128, same
learning rate scheduler"*. Coincide con CLAUDE.md §3: misma cantidad de pasos y mismo lote efectivo
para los 4 modelos. Salvedad propia de InfoNCE: los negativos son los del lote físico, por lo que
acumular gradientes no reproduce un lote mayor. Como el X-Encoder está congelado, sus salidas no
dependen del entrenamiento; esto permite que el lote **físico** de VL-JEPA sea igual al lote
efectivo común sin acumulación, si la VRAM alcanza. Si no alcanza, la diferencia se registra como
desviación justificada.

**Captions repetidos dentro del lote (medido, D4).** InfoNCE trata como negativos a los demás
textos del lote; si dos filas tienen el mismo caption, uno es un falso negativo. Medición sobre
`data/raw/m3di/train/text/text_raw.txt` [datos]: 125 000 filas, **10 936 captions distintos**, el
más frecuente aparece 100 veces y 622 aparecen una sola vez. La probabilidad de que dos filas al
azar compartan caption es 1.9 × 10⁻⁴; esa es también la fracción de negativos que son falsos, para
cualquier tamaño de lote. Con muestreo uniforme:

| Lote físico | Muestras con al menos un duplicado en su lote | Pares duplicados esperados por lote | Lotes con al menos un par |
|---|---|---|---|
| 16 | 0.3 % | 0.02 | 2 % |
| 32 | 0.6 % | 0.09 | 9 % |
| 64 | 1.2 % | 0.38 | 32 % |
| 128 | 2.4 % | 1.5 | 79 % |
| 256 | 4.7 % | 6.2 | ≈ 100 % |
| 512 | 9.0 % | 25 | ≈ 100 % |

Las tres primeras columnas son esperanzas exactas bajo muestreo con reemplazo; la última usa una
aproximación de Poisson. La medición cuenta cadenas idénticas; dos captions con los mismos
atributos y distinta plantilla son textos distintos y, para el objetivo del paper, negativos
legítimos. **Decisión del usuario (D4)**: no enmascarar si el lote físico es ≤ 128 (afecta a menos del
2.5 % de las muestras y mantiene la pérdida del paper sin modificar), y reportar la tasa. Con lotes
mayores conviene reconsiderarlo; el tratamiento estándar es excluir del denominador los negativos
con el mismo texto [R8].

**Control sin cuantizar en el ajuste** [D6, resuelto con los criterios del proyecto]. El enunciado
pregunta cómo afecta la cuantización *"al comportamiento durante el ajuste"*, y eso solo se
responde entrenando sin cuantizar. Se agrega, para cada uno de los 4 modelos, **una corrida de LoRA
sobre base bf16**, idéntica a la de NF4 en todo lo demás: mismos módulos entrenables, rango,
semilla, lote efectivo, una época y mismo criterio de selección. Es la comparación que hace [8]
(§4: LoRA de 16 bits frente a QLoRA de 4 bits). Cada adaptador se evalúa con la precisión con la
que se entrenó. La comparación principal entre modelos sigue siendo la de NF4; las corridas bf16
son un control y se reportan aparte. Costo: duplica las corridas de entrenamiento (8 en vez de 4),
todas en el cluster.

**Común.**
- Presupuesto: **1 época de train, mismo lote efectivo**, con gradient accumulation según VRAM
  [CLAUDE.md §3]. Ver la salvedad para InfoNCE en §5, C9.
- Selección de checkpoint: mejor exactitud por atributo en **val**. Reporte final en **test**, una
  sola vez. Mismas semillas y mismo protocolo de evaluación que en la etapa 1.
- Validación obligatoria antes de la corrida completa: conteo de parámetros con
  `requires_grad=True` por componente, formas de tensores y métricas sobre ejemplos verificados a
  mano [CLAUDE.md §5].

### 2.4 Y-Decoder de VL-JEPA

**Qué dice el paper.** El Y-Decoder traduce `Ŝ_Y` a texto, es *lightweight*, **no participa del
entrenamiento principal** y se invoca solo en inferencia (§1, §2). No hay más: ni arquitectura, ni
datos, ni pérdida. `open-vljepa` no trae ninguno. Todo lo que sigue es, por tanto, diseño propio
restringido por esas tres propiedades, y debe declararse así en el informe.

**Restricciones derivadas del paper.**
1. Se entrena **aparte y después** de VL-JEPA, con Predictor, X-Encoder e Y-Encoder congelados.
2. Su entrada es un único vector del espacio compartido (1 536-D), el mismo donde vive `Ŝ_Y`.
3. Es de lectura: no ve la imagen ni la consulta. Si las viera, dejaría de medir lo que contiene el
   embedding y pasaría a ser un VLM.

**Protocolo.**
- **Entrenamiento solo con texto**: pares ⟨`S_Y` = Y-Encoder(caption), caption⟩ de **train**, con
  entropía cruzada autorregresiva. Así el decodificador aprende a invertir el Y-Encoder y nunca ve
  predicciones del Predictor; lo que falle al decodificar `Ŝ_Y` es atribuible al Predictor o a la
  brecha entre `Ŝ_Y` y `S_Y`, no a que el decodificador haya memorizado imágenes.
- **Un decodificador por etapa**: uno sobre el Y-Encoder preentrenado (etapa 1) y otro sobre el
  ajustado (etapa 2), por la interacción descrita en §2.3.
- **Zero-shot**: en la etapa 1 el Predictor no ha visto M3DI, pero el Y-Decoder sí se entrena con
  captions de train de M3DI. Es inevitable (no existe un decodificador preentrenado) y es una
  asimetría frente a los VLM, cuyo decodificador de texto es zero-shot de verdad. Se reporta como
  tal; responde en parte a la pregunta del enunciado sobre la evaluación justa de VL-JEPA.
- **Techo del decodificador**: se reporta siempre la decodificación del embedding verdadero,
  `Y-Decoder(S_Y)`, en val y test. Es el máximo alcanzable por VL-JEPA con ese decodificador y
  separa el error del decodificador del error de predicción (§2.5).

**Arquitectura adoptada (D2, decisión del usuario): prefijo sobre Llama-3.2-1B.**
- Una **proyección entrenable** (precisión completa) convierte el embedding de 1 536-D en *k*
  vectores de prefijo en el espacio de embeddings de entrada del LM [R7].
- El LM es **Llama-3.2-1B preentrenado**, cuantizado en **NF4** y adaptado con **LoRA** en sus capas
  lineales, igual que los demás modelos del proyecto (§2.1). Genera el caption de forma
  autorregresiva condicionado al prefijo, con la misma decodificación determinista que los VLM.
- Se usa el modelo preentrenado original, **no** las 8 capas ajustadas del Predictor: el
  decodificador debe ser independiente del Predictor para que su error no se confunda con el de la
  predicción. Comparte con el Predictor solo el tokenizador y el origen de los pesos [5, §3.1].
- Motivo de la elección: no agrega una familia de modelos nueva al proyecto y cumple el enunciado
  (preentrenado, cuantizado, LoRA; nada se entrena desde cero salvo la proyección de prefijo, que
  cumple el rol de conector).

Consecuencias a declarar en el informe:
- **Tamaño**: con 1B de parámetros, el generador de texto de VL-JEPA es aproximadamente el doble de
  grande que los LLM de LLaVA-OV-0.5B e InternVL3-1B (Qwen de 0.5B). La calidad del texto de
  VL-JEPA no es atribuible solo al embedding; el paper además lo describe como *lightweight*, y un
  LM de 1B lo es solo en sentido relativo.
- **Prior de lenguaje**: un LM preentrenado puede completar atributos que el embedding no contiene.
  Se controla con el techo `Y-Decoder(S_Y)` y con las sondas de §2.5: si el texto decodificado
  acierta más que la sonda sobre `Ŝ_Y`, el exceso proviene del decodificador y no del embedding.
- **Licencia**: Llama-3.2-1B requiere aceptar la licencia de Meta; el repo usa una réplica
  (`unsloth/Llama-3.2-1B` [repo, sin verificar]), a revisar en la auditoría.

Alternativas descartadas por el usuario: prefijo sobre Qwen2.5-0.5B, Transformer pequeño entrenado
desde cero (contradice el enunciado) y vecino más cercano en un banco de captions (no genera texto).

Hiperparámetros (número de vectores de prefijo *k*, rango de LoRA, LR, épocas): **PENDIENTE**; se
rigen por las mismas reglas de §4.

### 2.5 Pérdida de información en cada etapa de VL-JEPA

**Objetivo.** Localizar en qué punto de la cadena se pierde cada atributo. VL-JEPA es una cadena de
procesamiento: imagen → X-Encoder → Predictor → proyección → Y-Decoder → texto. Por la desigualdad
de procesamiento de datos, la información sobre un atributo `A` no puede aumentar a lo largo de la
cadena: `I(A; X_V) ≥ I(A; S_V) ≥ I(A; Ŝ_Y) ≥ I(A; Ŷ)`. Medir cuánto queda en cada punto dice qué
componente es responsable de cada error, cosa que la exactitud final no distingue.

**Puntos de medición.**

| Punto | Representación | Qué aísla la diferencia con el punto anterior |
|---|---|---|
| E0 | factores verdaderos de `latents_image.csv` | referencia: `H(A)` |
| E1 | salida del X-Encoder (`S_V`), agregada por promedio | lo que el encoder visual congelado no retiene |
| E2 | estado oculto del Predictor, promediado, antes de la proyección de salida | lo que descartan las capas del Predictor |
| E3 | embedding predicho `Ŝ_Y` (1 536-D) | lo que pierde la proyección al espacio compartido |
| E4 | texto decodificado `Ŷ`, pasado por el extractor de atributos | lo que pierde el Y-Decoder |

Rama de texto, que fija los techos:

| Punto | Representación | Para qué sirve |
|---|---|---|
| T0 | caption original | referencia (los 4 atributos están por construcción) |
| T1 | embedding objetivo `S_Y` = Y-Encoder(caption) | cuánto de cada atributo conserva el objetivo que el Predictor debe alcanzar |
| T2 | `Y-Decoder(S_Y)`, pasado por el extractor | techo del Y-Decoder |

Comparaciones que se reportan: E1 → E3 (pérdida del Predictor); E3 frente a T1 (brecha de
predicción: lo que el objetivo contiene y la predicción no); E3 → E4 frente a T1 → T2 (pérdida del
decodificador sobre predicciones frente a su pérdida intrínseca).

**Estimador.** Se sigue el protocolo de [1] (§5.2), que mide exactamente esto sobre M3DI: predecir
los factores verdaderos desde la representación, con **un MLP de una capa oculta y exactitud para
los factores discretos**, y **regresión kernel ridge con R² para los continuos**. La sonda se ajusta
con representaciones de train y se evalúa en val (test solo en el reporte final). En E4 y T2 no hay
sonda: se usa el extractor de atributos común.

- Factores discretos: forma, x, y, nombre de color (los 4 atributos evaluados).
- Factores continuos, solo en E1–E3: rotación, posición y color del foco, color del fondo. No están
  en el texto, así que se espera que el Predictor los descarte, igual que los encoders contrastivos
  de [1] (§5.2: la información específica de la imagen *"is mostly discarded"*). Verificarlo muestra
  qué abstrae el espacio de embeddings, que es la tesis de [5] (§1).

**Lectura en bits.** La exactitud de la sonda es comparable entre puntos pero no es una cantidad de
información. Para expresarla en bits se usa la cota variacional `I(A; Z) ≥ H(A) − CE` [R2], donde
`CE` es la entropía cruzada de la sonda en datos no vistos y `H(A)` la entropía empírica del
atributo; su uso con sondas sobre representaciones está formalizado en [R3] y [R4]. La cadena de
desigualdades del inicio de esta sección es la desigualdad de procesamiento de datos [R1, §2.8], y
la relación entre exactitud e información condicional es la desigualdad de Fano [R1, §2.10], que
permite además acotar por arriba el error de cualquier lector a partir de la información medida.
Limitaciones a declarar: es una cota inferior (una sonda débil subestima la información) y mide
información *utilizable* por esa familia de sondas [R4]; por eso las estimaciones pueden violar
aparentemente la desigualdad de procesamiento entre puntos de distinta dimensión; y la capacidad de
la sonda debe ser la misma en todos los puntos. Estas referencias no están en `docs/papers/` (§7).

**Cuándo se mide.** En las dos etapas del proyecto, con las mismas sondas y semillas: checkpoint
preentrenado (etapa 1) y ajustado (etapa 2). E1 no cambia entre etapas porque el X-Encoder está
congelado, lo que sirve de control. La diferencia entre etapas en E2–E4 muestra qué componente
ganó información con el ajuste.

**Pendiente**: tamaño de la capa oculta y entrenamiento de la sonda ([1] no los detalla), kernel y
regularización de la regresión, y forma de agregar los tokens en E1 y E2 (se propone el promedio,
por coherencia con el *average pooling* del Predictor en [5] §3.1).

---

## 3. Arquitectura esperada por modelo (lista de verificación para la auditoría)

Lo que los papers dicen que debe tener cada modelo. La auditoría debe confirmar cada fila
inspeccionando el modelo cargado (`named_modules`, `config.json`), no los comentarios del repo.

| Modelo | Encoder visual | Conector | LLM / predictor | Tokens visuales por imagen | Fuente |
|---|---|---|---|---|---|
| LLaVA-OV-0.5B | SigLIP SO400M, 384 px | MLP de 2 capas | Qwen2-0.5B, 24 capas, d = 896 | 729 por vista; AnyRes agrega recortes | 2408.03326 §3, Ap. C.1; 2407.10671 Tabla 1 |
| InternVL3-1B | InternViT-300M-448px-V2.5 | pixel unshuffle + MLP de 2 capas | Qwen2.5-0.5B | 256 por tesela de 448 px | 2504.10479 §2.1, Tabla 1 |
| Qwen3.5-0.8B-Base | el de Qwen3.5 (SigLIP2 según el reporte Omni) | sin fuente en la carpeta (en Qwen2.5-VL: merger 2×2 + MLP de 2 capas) | híbrido con capas Gated DeltaNet | sin fuente en la carpeta | 2604.15804 §2–3; [3] §2.1 |
| VL-JEPA (open) | V-JEPA 2 ViT-L, 256 px, congelado | proyección lineal | últimas 8 capas de Llama-3.2-1B, no causal, mean-pool; Y-Encoder EmbeddingGemma-300M; espacio 1 536-D | imagen duplicada en cuadros; tubelets 2×16×16 | [5] §3.1; 2506.09985 §2 |

Puntos concretos a verificar, derivados de los papers:

1. **LLaVA-OV**: que el proyector sea el MLP de 2 capas y esté fuera de la cuantización; cuántos
   tokens visuales produce realmente una imagen de M3DI con AnyRes (el repo afirma ≈ 2×729 para
   224 px [repo, sin verificar]); que la pérdida enmascare prompt y tokens de imagen.
2. **InternVL3**: que el checkpoint `-hf` aplique pixel unshuffle y entregue 256 tokens por tesela;
   cuántas teselas genera una imagen de M3DI; que los `target_modules` de LoRA apunten al LLM
   Qwen2.5 y no al InternViT.
3. **Qwen3.5**: que el ID exista y sea la variante Base; dónde vive el conector (en Qwen2.5-VL el
   merger está dentro del módulo visual: congelar "el encoder visual" completo congelaría también
   el conector, contra CLAUDE.md §3); que LoRA cubra también las proyecciones lineales de las capas
   Gated DeltaNet, no solo `q/k/v/o_proj` [paper [8], §4: todas las capas lineales]; qué plantilla
   de chat corresponde a un modelo Base sin ajuste por instrucciones.
4. **VL-JEPA**: máscara no causal y promedio sobre tokens no-PAD [paper [5], §3.1]; X-Encoder
   congelado; duplicación de la imagen en cuadros; `max_query_len` y longitud de caption suficientes
   para los prompts y captions de M3DI (10–19 palabras [repo, sin verificar]); si el Y-Encoder usa
   el prefijo de tarea de EmbeddingGemma; coherencia entre la temperatura usada y la del checkpoint.
5. **Todos**: en la etapa 2, que solo LoRA y el conector tengan `requires_grad=True`.

Observaciones preliminares sobre los configs del repo, sin auditar el código:
- `configs/models/qwen35.yaml` declara `vision_modules: [visual]` y un regex de LoRA que incluye
  `in_proj_qkv|in_proj_z|out_proj` (Gated DeltaNet). Lo segundo es coherente con [8]; lo primero
  choca potencialmente con "entrenar el conector" (punto 3).
- `configs/models/vljepa.yaml` define un "banco" de captions de train como decodificador por
  recuperación, y `vljepa_fac.yaml` un banco factorizado. La recuperación sobre candidatos es
  coherente con [5] §2; el banco factorizado **no proviene del paper** y debe justificarse o
  descartarse.
- Existe `qwen35_instruct.yaml`; CLAUDE.md §3 excluye las variantes Instruct de la etapa 2.

---

## 4. Tabla de adaptaciones de escala

Los valores "nuestro" marcados **PENDIENTE** no se fijan: no hay justificación documentada
[CLAUDE.md §2 y §5]. La columna de justificación indica qué fuente podría resolverlos.

| Parámetro | Valor en el paper (modelo) | Valor nuestro | Justificación |
|---|---|---|---|
| Cuantización del modelo base | NF4 + double quantization, cómputo bf16 (LLaMA 7B–65B) [8, Ap. B.2] | igual | Protocolo; no depende de la escala. Exigido por el enunciado. |
| Optimizador paginado | Paged AdamW; crítico solo en 33B/65B [8, §4] | PENDIENTE | Con 0.5–1B en 24 GB el estado del optimizador es pequeño. Decidir si se mantiene por fidelidad a [8] o se usa AdamW estándar. |
| Módulos con LoRA | todas las capas lineales del modelo base [8, §4]; solo `W_q`, `W_v` en [7] | todas las lineales de los bloques del LLM (del predictor en VL-JEPA) | Protocolo de [8]. Los nombres de módulo se verifican por modelo; Qwen3.5 tiene capas Gated DeltaNet. |
| LoRA r | 64 (LLaMA 7B–65B) [8, Ap. B.2] | PENDIENTE | [8] Ap. A.2: r no afecta si LoRA va en todas las capas, pero la evidencia es de 7B. Con d = 896, r = 64 es una fracción mucho mayor de la dimensión. |
| LoRA α | 16 (LLaMA 7B–65B) [8, Ap. B.2] | PENDIENTE | Ligado a r y al LR ([8] Ap. A.2: α es proporcional al LR). |
| LoRA dropout | 0.1 hasta 13B; 0.05 en 33B/65B [8, Ap. B.2] | PENDIENTE | Dependiente de escala según el propio paper. |
| LR de LoRA | 2e-4 (7B/13B), 1e-4 (33B/65B) [8, Tabla 9] | PENDIENTE | Dependiente de escala; barrido corto en val o model card. |
| LR del conector | 1e-3 con solo el proyector, 1e-5 con modelo completo (LLaVA-OV 0.5B) [2408.03326, Tabla 1]; 2e-3 (LLaVA Vicuna) [2, §5] | PENDIENTE | Los papers no cubren "conector + LoRA". Decidir si comparte LR con LoRA o usa grupo propio. |
| Esquema de LR | constante [8, Ap. B.2]; coseno con warmup 3 % [2, Ap.] | PENDIENTE | Dos fuentes válidas y distintas. |
| Warmup | 3 % (LLaVA) [2, Ap.] | PENDIENTE | Ídem. |
| Max grad norm | 0.3 [8, Ap. B.2] | PENDIENTE | — |
| Adam β₂ / weight decay | β₂ = 0.999 [8]; sin weight decay [2, Ap.] | PENDIENTE | — |
| Lote efectivo | 16 (QLoRA 7B) [8]; 32 (LLaVA ft.) [2]; 512 (LLaVA-OV 0.5B) [2408.03326]; 3 072 (VL-JEPA SFT) [5] | PENDIENTE, único para los 4 modelos | CLAUDE.md §3. Ver C9 para VL-JEPA. |
| Épocas | 3 (LLaVA ft.), 1 por etapa (LLaVA-OV), 83k pasos (VL-JEPA SFT) | 1 época de train | CLAUDE.md §3. |
| Módulos entrenados (VLM) | proyector + LLM completo, visión congelada (LLaVA) [2, §4.2]; modelo completo (LLaVA-OV, InternVL3) | LoRA en LLM + conector; visión congelada | CLAUDE.md §3 y enunciado (LoRA sobre modelo cuantizado). Coincide con [2] en congelar la visión. |
| Módulos entrenados (VL-JEPA) | predictor completo (490M) + Y-Encoder con LR ×0.05; X-Encoder congelado [5, §3.1] | opción A del paper con LoRA: LoRA en Predictor e Y-Encoder (×0.05) + proyecciones lineales; X-Encoder congelado | El paper fija qué componentes aprenden; LoRA y NF4 los exige el enunciado. Ver §2.3. |
| Multiplicador de LR del Y-Encoder | ×0.05 (modelo de 1.6B) [5, §3.1, Tabla 7b] | ×0.05 | Es una razón entre LRs, no un valor absoluto; el paper muestra que 0.05–0.10 es el óptimo. |
| Pérdida de VL-JEPA | InfoNCE bidireccional [5, §2] | igual | Protocolo. |
| Y-Decoder: arquitectura | no especificado [5]; solo *lightweight* | prefijo sobre Llama-3.2-1B, NF4 + LoRA | Decisión del usuario (§2.4). Sin fuente en el paper. |
| Y-Decoder: LoRA y optimización | r = 64, α = 16, dropout 0.1, LR 2e-4, lote 16, LR constante, max grad norm 0.3 (LLaMA 7B/13B) [8, Ap. B.2] | los mismos, como punto de partida | Acordado con el usuario: valores de QLoRA como inicio, no justificados para 1B. Revisables tras el barrido. |
| Y-Decoder: vectores de prefijo y épocas | sin fuente | barrido de *k* ∈ {1, 4, 8}; hasta 3 épocas, eligiendo el mejor paso | Se elige en val por el techo `Y-Decoder(S_Y)` (`configs/queues/ydecoder_sweep.txt`). |
| Y-Decoder: entrada | sin fuente | embedding normalizado a norma 1 | InfoNCE no supervisa la norma de `Ŝ_Y`; sin normalizar, una diferencia de norma con `S_Y` sacaría al decodificador de distribución. Diseño propio. |
| Cuantización del encoder visual | no aplica: los papers de los VLM no cuantizan; [8] cuantiza el modelo base completo | NF4, igual en los 4 modelos | Enunciado y comparabilidad. Ver §2.1. |
| Sonda de información | MLP de 1 capa oculta (discretos), kernel ridge (continuos) [1, §5.2]; sin hiperparámetros | igual; grilla elegida en un 20 % retenido: capa oculta ∈ {64, 256, 1024}, weight decay ∈ {1e-4, 1e-2}; núcleo RBF con ancho ∈ {0.25, 1, 4} × heurística de la mediana y α ∈ {1e-3, 1e-1, 10} | Protocolo de [1]. La grilla es propuesta propia (D8), igual en todos los puntos de medición; debe declararse en el informe. |
| Temperatura InfoNCE | no indicada en [5]; 0.07 en `open-vljepa` | PENDIENTE (candidato: la del checkpoint) | Mantener la del checkpoint evita desajustar el espacio preentrenado. |
| Resolución (LLaVA-OV) | 384 px + AnyRes hasta 729×10 tokens [2408.03326, Ap. C.1] | preprocesado por defecto del checkpoint; PENDIENTE si se limita AnyRes | Las imágenes de M3DI son pequeñas (224 px [repo, sin verificar]); el model card manda [CLAUDE.md §2]. |
| Resolución (InternVL3) | teselas de 448 px, 256 tokens por tesela [2504.10479, §2.1] | preprocesado por defecto; PENDIENTE el máximo de teselas | Ídem. |
| Resolución (Qwen3.5) | sin fuente en la carpeta | PENDIENTE (model card) | — |
| Resolución y cuadros (VL-JEPA) | 256² px; 1, 8 o 32 cuadros según fase [5, §3.2] | los del checkpoint (8 cuadros, 256 px [repo, sin verificar]) | Debe coincidir con el entrenamiento del checkpoint. |
| Longitud máxima de consulta (VL-JEPA) | 512 tokens [5, §3.1] | 512 | Valor del paper. El checkpoint usó 64, donde los prompts con ejemplos no caben; el relleno va enmascarado. Decisión del usuario: mismos prompts en los 4 modelos. |
| Longitud máxima de generación | 128 a 1 024 tokens objetivo según dataset [8, Tabla 9] | 256, igual en los tres VLM y en el Y-Decoder | Decisión del usuario (14-11-2026) a partir del piloto en val (200 imágenes): con 128 se cortaba el 38 % de las respuestas de Qwen3.5 con `p1_attributes` y el 64 % con `p0_minimal`, y ese modelo deja la posición para el final. Los captions de referencia tienen como máximo 31 tokens. La fracción de respuestas cortadas se reporta por run. |

---

## 5. Contradicciones y vacíos

Cada punto conserva su planteamiento original. El estado tras las decisiones del usuario es:

| # | Estado | Resolución |
|---|---|---|
| C2 | **resuelto** | Se evalúan 4 atributos: forma, x, y, color. |
| C3 | **resuelto** | Color por nombre exacto, sin contar el prefijo de paleta; se reporta el techo (§2.1). |
| C5 | **resuelto** | VL-JEPA se evalúa con un Y-Decoder que genera el caption (§2.4). Arquitectura: prefijo sobre Llama-3.2-1B (D2). |
| C6 | **no aplica** | Al no evaluarse por selección de candidatos, no se define un conjunto de candidatos. |
| C7 | **resuelto** | Opción A: Predictor + Y-Encoder (×0.05), InfoNCE, X-Encoder congelado; LoRA más proyecciones en precisión completa, confirmado por el usuario (§2.3). |
| C9 | **resuelto** | Mismo número de pasos y mismo lote efectivo, según [5] §4.5. Duplicados medidos: 2.4 % de las muestras con lote 128 (§2.3). No se enmascaran con lote físico ≤ 128. |
| C14 | **resuelto** | Se agrega el control bf16 (§2.1, §2.2). |
| C15 | **resuelto** | Bleu-1, Bleu-4, ROUGE-L y CIDEr con el código de MS COCO, según [9] (§2.1). |
| C8 | **resuelto** | Todo componente congelado en NF4, incluido el encoder visual, en los 4 modelos (§2.1). |
| C1, C4, C10–C13, C16 | declarativos | No requieren decisión; deben constar en el informe. |

**C1. Número y elección de modelos.** El enunciado pide tres modelos: LLaVA, *un* VLM moderno
(Qwen2.5-VL o InternVL 3) y VL-JEPA. CLAUDE.md §2 define cuatro, con InternVL3-1B **y**
Qwen3.5-0.8B-Base. Qwen3.5 no figura en la bibliografía del enunciado y no hay paper propio en
`docs/papers/`. No es una contradicción con un paper, pero debe quedar declarado en el informe.

**C2. Atributos que el texto no contiene.** El enunciado dice que el texto de referencia describe
*"forma, color, posición y ángulo del objeto, color del foco y del fondo"* y pregunta por la
dificultad de *"posición y ángulo"*. Según [1] (Tabla 3, §5.2), rotación, foco y fondo son
**específicos de la imagen** y no aparecen en el texto; los datos lo confirman [datos]. Con el
dataset publicado la exactitud por atributo solo es medible para forma, x, y y color. Medir ángulo,
foco o fondo exigiría captions nuevos con el generador (extensión fuera de alcance). **(decisión:
confirmar que el alcance son 4 atributos)**

**C3. El color no tiene una verdad única. (decisión, etapa 1)** Por diseño de [1] §5.2, el nombre de
color del caption sale de una paleta elegida al azar entre tres, de modo que la imagen no determina
el nombre. La exactitud de nombre exacto tiene un techo inferior a 1 que ningún modelo puede superar
desde la imagen. Ningún paper define cómo puntuar el color. Opciones: (a) nombre exacto, reportando
el techo; (b) categorías de color gruesas derivadas del matiz verdadero de `latents_image.csv`;
(c) ambas. El repo usa 8 categorías básicas [repo, sin verificar]; es una decisión de diseño sin
respaldo en un paper.

**C4. Dependencia x → color.** [1] Ap. B.1 describe la versión con dependencia causal (el intervalo
de matiz depende de x) y Ap. B.2 una versión independiente. El EDA del repo reporta que los datos
en uso son la versión con dependencia (NMI x–color ≈ 0.71 [repo, sin verificar]). Si se confirma,
x se puede adivinar desde el color y viceversa, lo que afecta la interpretación de la exactitud de
posición y de color. No cambia el protocolo; sí obliga a reportar líneas base de azar condicionadas.

**C5. El Y-Decoder de VL-JEPA no está especificado. (decisión, etapas 1 y 2)** El paper [5] no
describe su arquitectura ni su entrenamiento, y `open-vljepa` no lo implementa. El enunciado, en
cambio, pregunta por el *"costo en calidad del texto final"* y por cómo evaluar de forma justa un
modelo *"cuyo texto pasa por un decodificador de lectura adicional"*. Opciones:
(a) **solo recuperación sobre candidatos**, que es lo que el paper sí especifica (§2, §4.1–4.2); el
"texto" de VL-JEPA es el candidato recuperado;
(b) entrenar un decodificador propio embedding → texto, sin respaldo metodológico en [5];
(c) ambas, con (a) como resultado principal.
CLAUDE.md §2 apunta a (a), pero la elección condiciona si BLEU/ROUGE/CIDEr se reportan para VL-JEPA.

**C6. Conjunto de candidatos para VL-JEPA. (decisión, etapa 1)** Derivado de C5(a). El paper usa
como candidatos las etiquetas de clase o las respuestas posibles. Para M3DI hay al menos tres
definiciones: captions completos (¿de qué split, cuántos, con qué plantilla?); un texto por valor de
atributo, con una selección por atributo (análogo a VQA discriminativo, con una consulta por
atributo); o captions canónicos por combinación de atributos. Cada una mide algo distinto y fija un
techo distinto. Los candidatos nunca deben construirse con test. Además, la sensibilidad a la
redacción de los candidatos está documentada en [6] §3.1.4, por lo que la redacción también debe
fijarse en val.

**C7. Qué se entrena en VL-JEPA. (decisión, etapa 2)** El paper entrena el predictor completo y el
Y-Encoder (LR ×0.05), sin LoRA ni cuantización. El enunciado exige modelos cuantizados con LoRA, y
CLAUDE.md fija "LoRA + conector" pensando en los VLM. Falta definir, para VL-JEPA: dónde va LoRA
(¿las 8 capas del predictor?); qué es su "conector" (¿las proyecciones lineales de entrada y salida
del predictor?); y si el Y-Encoder se entrena (con ×0.05, con LoRA) o se congela. Si se congela el
Y-Encoder, el paper indica que el riesgo de colapso desaparece y que otras pérdidas son admisibles
(§2, Tabla 7c), pero InfoNCE sigue siendo la elección del paper.

**C8. Conector entrenable frente a QLoRA.** En [8] todo el modelo base está cuantizado y congelado;
solo se entrenan adaptadores. Entrenar el conector en precisión completa [CLAUDE.md §3] es una
desviación de [8], consistente con [2] §4.2 (el proyector se sigue entrenando). Implica excluir el
conector de la cuantización en los 4 modelos y decidir si el encoder visual, que queda congelado,
se cuantiza o no. Esta última elección afecta la pregunta del enunciado sobre el efecto de la
cuantización y debe ser igual para todos. **(decisión menor)**

**C9. "Mismo lote efectivo" e InfoNCE. (decisión, etapa 2)** En entropía cruzada, acumular gradientes
equivale a un lote mayor. En InfoNCE **no**: los negativos son los del lote físico, por lo que
acumular N pasos de lote B no equivale a un lote N·B. Con 24 GB el lote físico de VL-JEPA queda muy
por debajo de los del paper (3 072) y del checkpoint (512). A esto se suma que M3DI tiene muchos
captions repetidos (10 936 captions únicos en 125 000 filas [repo, sin verificar]), de modo que un
lote contiene falsos negativos: textos idénticos tratados como negativos. El paper no aborda este
caso. Hay que decidir cómo se define la igualdad de presupuesto para VL-JEPA y cómo tratar los
duplicados (p. ej. enmascararlos), y registrarlo como diferencia justificada entre modelos
[CLAUDE.md, comparabilidad].

**C10. El paper de la bibliografía no corresponde al modelo usado.**
- [2] describe CLIP ViT-L/14 + **proyección lineal** + Vicuna, y el enunciado define la línea base
  como *"encoder visual congelado, proyector lineal"*. LLaVA-OV-0.5B usa SigLIP + **MLP de 2 capas**
  + Qwen-2 con AnyRes, y en su entrenamiento original el encoder **no** está congelado. Sigue siendo
  la familia LLaVA, pero no es la "arquitectura antigua" que el enunciado contrasta con el VLM moderno.
- [4] describe InternViT-6B + QLLaMA + Vicuna; InternVL3-1B es ViT-MLP-LLM con InternViT-300M y
  Qwen2.5-0.5B. Los `target_modules` y componentes de [4] no existen en nuestro modelo.
- Consecuencia: LLaVA-OV-0.5B e InternVL3-1B comparten paradigma (ViT + MLP de 2 capas + Qwen de
  0.5B). La pregunta *"¿se traduce la ventaja del VLM moderno sobre LLaVA…?"* compara en la práctica
  encoder visual, compresión de tokens y preentrenamiento, no "proyector lineal vs conector
  expresivo". Debe declararse en el informe.

**C11. Entrenamiento original frente a nuestro esquema.** LLaVA-OV e InternVL3 se entrenaron
actualizando todo el modelo, incluido el encoder visual; aquí se congela la visión y se usa LoRA en
el LLM. Es una adaptación impuesta por el enunciado y el hardware, respaldada por [2] §4.2 y [8].
Queda registrada en §4.

**C12. Checkpoint de VL-JEPA frente al paper.** `open-vljepa` difiere del modelo de [5] en datos
(~1.1M frente a ~3.3B muestras), consulta de preentrenamiento (fija en vez de *query-free*),
longitud de consulta, temperatura y criterio de selección del checkpoint (pérdida de entrenamiento).
Las cifras de [5] (modelo de 1.6B entrenado a gran escala) no son extrapolables. Además el
checkpoint solo vio consultas del tipo `"Describe the image."`: la sensibilidad al prompt de la
etapa 1 se medirá sobre un modelo que casi no fue entrenado con consultas variadas, lo que limita
la lectura de ese resultado para VL-JEPA.

**C13. Qwen3.5-0.8B-Base sin fuente primaria.** No hay paper del modelo en la carpeta (§1.4). El ID
exacto, la arquitectura, la resolución y la plantilla de chat dependen del model card. Al ser un
modelo **Base** (sin ajuste por instrucciones), su zero-shot mezcla percepción con capacidad de
seguir instrucciones; InternVL3 también parte de un LLM base, pero recibe SFT multimodal
[2504.10479, §2.1 y §2.3]. Esta asimetría debe mencionarse al interpretar la etapa 1.

**C14. Efecto de la cuantización.** El enunciado pregunta cómo afecta la cuantización a la calidad
y al ajuste. El protocolo de CLAUDE.md fija NF4 para todo y no contempla una condición de control
(p. ej. bf16 en inferencia). Sin ella la pregunta no se puede responder con datos propios; [8] §4
solo aporta evidencia en modelos de 7B o más. **(decisión: ¿se agrega un control bf16 en la etapa 1?)**

**C15. Métricas de texto.** El paper de CIDEr no está en `docs/papers/`: la variante (CIDEr o
CIDEr-D), la tokenización y el corpus para el idf quedan sin fuente y deben fijarse. Con una sola
referencia por imagen y 5 plantillas, BLEU, ROUGE y CIDEr premian acertar la plantilla (factor
específico del texto en [1]), que no es deducible de la imagen. Conviene reportarlas junto a la
exactitud por atributo y no interpretarlas como fidelidad, en línea con el argumento de [9].

**C16. Vocabulario de posición.** [1] §5.2 habla de *top/center/bottom*; los datos usan
`top/mid/bottom` [datos]. Mandan los datos; el extractor de atributos debe aceptar sinónimos de
forma explícita y testeada.

---

## 6. Decisiones

### 6.1 Tomadas

| # | Decisión | Origen |
|---|---|---|
| C2 | Atributos evaluados: forma, x, y, color. | usuario |
| C3, D1 | Color por nombre exacto; el prefijo de paleta no cuenta (`blue` acierta para `tab:blue`). | usuario |
| C5 | VL-JEPA genera texto con un Y-Decoder y se compara con el caption original. | usuario |
| C14 | Se agrega un control sin cuantizar (bf16) en inferencia. | usuario |
| C15 | Métricas de texto: código de MS COCO, Bleu-1, Bleu-4, ROUGE-L, CIDEr. | usuario; fuente [9] |
| C7, D3 | Fine-tuning de VL-JEPA según el paper (opción A), con X-Encoder congelado; LoRA en Predictor e Y-Encoder (×0.05) y proyecciones lineales en precisión completa. | usuario; fuente [5] |
| D5 (C8) | Todo componente congelado en NF4, incluido el encoder visual, en los 4 modelos. | criterios del proyecto: enunciado, [8] §3, comparabilidad |
| D6 | Control bf16 también en el ajuste: una corrida de LoRA sobre base bf16 por modelo, idéntica en lo demás. | criterios del proyecto: pregunta del enunciado, [8] §4 |
| D2 | Y-Decoder: prefijo sobre Llama-3.2-1B preentrenado, en NF4 con LoRA (§2.4). | usuario |
| D4 | Captions idénticos en el lote de InfoNCE: no se enmascaran con lote físico ≤ 128; se reporta la tasa (§2.3). | usuario |
| D8 | Grilla de selección de las sondas de §2.5 (capa oculta, weight decay, ancho del núcleo y regularización; ver §4). | usuario |
| D7 | Referencias para la cota en bits y las métricas agregadas en §7. | usuario |
| — | Se evalúa la pérdida de información en cada etapa de VL-JEPA (§2.5). | usuario |

### 6.2 Abiertas

| # | Decisión | Propuesta de este documento | Bloquea |
|---|---|---|---|
| §4 | Valores PENDIENTE de la tabla de adaptaciones (r, α, dropout, LR, esquema, lote efectivo, longitudes) y los del Y-Decoder. | Barrido corto en val o model cards, a acordar. | etapa 2 |
| §7 | Agregar a `docs/papers/` las referencias externas y verificarlas antes de citarlas. | — | informe |

---

## 7. Referencias externas a `docs/papers/`

Fuentes que el protocolo necesita y que **no están en la carpeta**. Se citan de memoria y **no
fueron leídas en esta fase**: antes de usarlas en el informe hay que agregarlas a `docs/papers/` y
verificar título, año y la sección citada.

| Ref. | Fuente | Para qué se usa |
|---|---|---|
| R1 | T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2.ª ed., Wiley, 2006. | Desigualdad de procesamiento de datos y desigualdad de Fano (§2.5). |
| R2 | D. Barber, F. Agakov, "The IM Algorithm: A Variational Approach to Information Maximization", NeurIPS, 2003. | Cota inferior variacional de la información mutua (§2.5). |
| R3 | T. Pimentel et al., "Information-Theoretic Probing for Linguistic Structure", ACL, 2020. arXiv:2004.03061. | Sondas como estimadores de información mutua (§2.5). |
| R4 | Y. Xu et al., "A Theory of Usable Information under Computational Constraints", ICLR, 2020. arXiv:2002.10689. | Información utilizable por una familia de sondas; explica las violaciones aparentes de la desigualdad de procesamiento (§2.5). |
| R5 | R. Vedantam, C. L. Zitnick, D. Parikh, "CIDEr: Consensus-based Image Description Evaluation", CVPR, 2015. arXiv:1411.5726. | Definición de CIDEr y CIDEr-D (§2.1). |
| R6 | X. Chen et al., "Microsoft COCO Captions: Data Collection and Evaluation Server", 2015. arXiv:1504.00325. | Código de evaluación de MS COCO: tokenización y variantes de las métricas (§2.1). |
| R7 | R. Mokady, A. Hertz, A. H. Bermano, "ClipCap: CLIP Prefix for Image Captioning", 2021. arXiv:2111.09734. | Decodificación de un embedding mediante prefijo sobre un LM preentrenado (§2.4, D-a). |
| R8 | P. Khosla et al., "Supervised Contrastive Learning", NeurIPS, 2020. arXiv:2004.11362. | Tratamiento de muestras del lote con la misma etiqueta en pérdidas contrastivas (§2.3). |
| R9 | K. Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation", ACL, 2002; C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", 2004. | Definición de BLEU y ROUGE-L (§2.1). |
