# Resultados de la etapa 1 — zero-shot en val

> Registro de lo **medido** en el cluster, separado del protocolo (`docs/metodologia.md`). Todo es
> sobre **val**; test no se ha tocado. Las cifras corresponden a modelos de 0.5–1B y solo son
> comparables entre nuestros modelos bajo el mismo protocolo [CLAUDE.md §2].
>
> Estado (16-11-2026): VLM autorregresivos en NF4 completos; control bf16 de los VLM en curso; VL-JEPA
> pendiente del barrido del Y-Decoder; pérdida de información pendiente.

## 1. Procedencia

| Elemento | Valor |
|---|---|
| Runs | `runs/zeroshot_<modelo>_nf4_<prompt>_val`, 3 VLM × 6 prompts, 10 000 imágenes cada uno |
| Commit de la inferencia | `3feefb0` (jobs `s1_val_g0` y `s1_val_g1`, 14-11-2026, `logs/*/meta.env`) |
| Generación | determinista, `max_new_tokens = 256` (verificado en el `config.yaml` de los 18 runs) |
| Cuantización | NF4 con `bitsandbytes` (versión registrada en el `env.jsonl` de cada run) |
| Puntaje | `scripts/score.py` con el extractor corregido el 16-11-2026 (§6), 48 tests pasan; `pycocoevalcap` con Java 11 |
| Tabla | `runs/_tablas/etapa1_val.csv` (copia en `reports/tables/etapa1_val.csv`); salida con líneas base en `reports/tables/score_etapa1_val.txt` |

## 2. Líneas base sin modelo (val, estimadas con train)

| | forma | x | y | color | attr_mean |
|---|---|---|---|---|---|
| Clase mayoritaria | 0.141 | 0.331 | 0.330 | 0.085 | 0.222 |
| Techo del color (mejor nombre dado el matiz verdadero, 360 bins) | — | — | — | **0.410** | — |

## 3. Tabla completa (NF4, val, n = 10 000)

Exactitud por atributo, fracción de respuestas en que el extractor no encuentra el atributo
(omitir cuenta como error), largo medio y métricas de texto contra el caption original.

| Modelo | Prompt | forma | x | y | color | **attr_mean** | om. forma | om. x | om. y | om. color | palabras | Bleu-1 | Bleu-4 | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-OV-0.5B | p0_minimal | 0.258 | 0.224 | 0.208 | 0.199 | 0.222 | 0.715 | 0.344 | 0.334 | 0.021 | 94.1 | 0.088 | 0.016 | 0.149 | 0.000 |
| LLaVA-OV-0.5B | p1_attributes | 0.271 | 0.445 | 0.298 | 0.210 | 0.306 | 0.710 | 0.027 | 0.068 | 0.004 | 28.2 | 0.267 | 0.008 | 0.329 | 0.024 |
| LLaVA-OV-0.5B | p2_constrained | 0.235 | 0.000 | 0.012 | 0.000 | 0.062 | 0.004 | 0.999 | 0.936 | 1.000 | 1.6 | 0.000 | 0.000 | 0.060 | 0.045 |
| LLaVA-OV-0.5B | **p3_json** | 0.373 | 0.338 | 0.330 | 0.197 | **0.309** | 0.017 | 0.000 | 0.000 | 0.019 | 8.3 | 0.082 | 0.000 | 0.076 | 0.060 |
| LLaVA-OV-0.5B | p4_dataset_format | 0.171 | 0.184 | 0.156 | 0.138 | 0.162 | 0.050 | 0.538 | 0.539 | 0.355 | 8.1 | 0.269 | 0.070 | 0.303 | 0.185 |
| LLaVA-OV-0.5B | p4m_multiexample | 0.212 | 0.057 | 0.063 | 0.179 | 0.128 | 0.168 | 0.769 | 0.769 | 0.053 | 7.2 | 0.233 | 0.043 | 0.278 | 0.190 |
| InternVL3-1B | p0_minimal | 0.148 | 0.190 | 0.188 | 0.224 | 0.187 | 0.817 | 0.542 | 0.479 | 0.005 | 56.9 | 0.133 | 0.021 | 0.195 | 0.000 |
| InternVL3-1B | **p1_attributes** | 0.201 | 0.397 | 0.282 | 0.216 | **0.274** | 0.752 | 0.136 | 0.232 | 0.030 | 55.3 | 0.142 | 0.012 | 0.226 | 0.000 |
| InternVL3-1B | p2_constrained | 0.232 | 0.016 | 0.334 | 0.219 | 0.200 | 0.681 | 0.963 | 0.007 | 0.001 | 10.3 | 0.499 | 0.093 | 0.469 | 0.181 |
| InternVL3-1B | p3_json | 0.303 | 0.032 | 0.314 | 0.210 | 0.215 | 0.498 | 0.927 | 0.048 | 0.000 | 7.3 | 0.073 | 0.000 | 0.073 | 0.056 |
| InternVL3-1B | p4_dataset_format | 0.142 | 0.325 | 0.328 | 0.075 | 0.218 | 0.007 | 0.015 | 0.015 | 0.000 | 9.8 | 0.410 | 0.138 | 0.438 | 0.377 |
| InternVL3-1B | p4m_multiexample | 0.149 | 0.048 | 0.037 | 0.084 | 0.079 | 0.183 | 0.901 | 0.910 | 0.001 | 3.0 | 0.010 | 0.004 | 0.074 | 0.082 |
| Qwen3.5-0.8B-Base | p0_minimal | 0.119 | 0.353 | 0.283 | 0.163 | 0.230 | 0.830 | 0.429 | 0.396 | 0.001 | 116.9 | 0.064 | 0.002 | 0.111 | 0.000 |
| Qwen3.5-0.8B-Base | **p1_attributes** | 0.138 | 0.820 | 0.582 | 0.124 | **0.416** | 0.773 | 0.011 | 0.017 | 0.000 | 84.9 | 0.088 | 0.002 | 0.158 | 0.000 |
| Qwen3.5-0.8B-Base | p2_constrained | 0.339 | 0.099 | 0.346 | 0.031 | 0.204 | 0.036 | 0.865 | 0.141 | 0.921 | 9.1 | 0.479 | 0.089 | 0.460 | 0.111 |
| Qwen3.5-0.8B-Base | p3_json | 0.284 | 0.489 | 0.441 | 0.062 | 0.319 | 0.000 | 0.000 | 0.000 | 0.737 | 9.7 | 0.057 | 0.000 | 0.058 | 0.033 |
| Qwen3.5-0.8B-Base | p4_dataset_format | 0.142 | 0.331 | 0.338 | 0.075 | 0.221 | 0.000 | 0.000 | 0.000 | 0.000 | 11.0 | 0.512 | 0.154 | 0.466 | 0.434 |
| Qwen3.5-0.8B-Base | p4m_multiexample | 0.137 | 0.600 | 0.464 | 0.221 | 0.355 | 0.798 | 0.038 | 0.002 | 0.002 | 12.7 | 0.546 | 0.224 | 0.483 | 0.409 |

CIDEr se calcula con el idf de las referencias de val; no es comparable con valores en test
(`docs/metodologia.md` §2.1).

## 4. Selección de prompt (criterio: `attr_mean` en val, `docs/metodologia.md` §2.2)

| Modelo | Prompt elegido | attr_mean | Segundo | Margen |
|---|---|---|---|---|
| Qwen3.5-0.8B-Base | `p1_attributes` | 0.416 | `p4m_multiexample` (0.355) | 0.061 |
| LLaVA-OV-0.5B | `p3_json` | 0.309 | `p1_attributes` (0.306) | **0.004** |
| InternVL3-1B | `p1_attributes` | 0.274 | `p4_dataset_format` (0.218) | 0.057 |

La selección se reporta pero no decide el prompt de la etapa 2, que es `p1_attributes` en los 4
modelos [CLAUDE.md §3]. Sí fija la condición zero-shot que se controla en bf16 y se evalúa en test.

**LLaVA-OV: selección no robusta.** Se aplicó la regla acordada sin cambiarla tras ver val (decisión
del usuario, 16-11-2026). El margen sobre `p1_attributes` es de 0.004 y la posición de `p3_json` no
depende de la imagen: `y = top` en las 10 000 respuestas y `x = right` en el 90 %, de modo que su x e
y igualan a la clase mayoritaria. Toda su ventaja viene de forma (0.373) y color (0.197). Con
`p1_attributes`, en cambio, LLaVA-OV sí discrimina x (0.445). El informe debe reportar ambos.

## 4b. Control sin cuantizar (bf16, prompt elegido, val completo)

Job `s1_bf16` (16-11-2026), mismos prompt, semilla y protocolo; tabla en `runs/_tablas/etapa1_bf16.csv`.
Δ = NF4 − bf16 (negativo: la cuantización empeora).

| Modelo | Prompt | Condición | forma | x | y | color | attr_mean |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | `p1_attributes` | NF4 | 0.138 | 0.820 | 0.582 | 0.124 | 0.416 |
| | | bf16 | 0.177 | 0.787 | 0.672 | 0.134 | 0.442 |
| | | **Δ** | −0.039 | +0.033 | −0.090 | −0.009 | **−0.027** |
| LLaVA-OV-0.5B | `p3_json` | NF4 | 0.373 | 0.338 | 0.330 | 0.197 | 0.309 |
| | | bf16 | 0.391 | 0.323 | 0.345 | 0.216 | 0.319 |
| | | **Δ** | −0.018 | +0.015 | −0.015 | −0.019 | **−0.009** |
| InternVL3-1B | `p1_attributes` | NF4 | 0.201 | 0.397 | 0.282 | 0.216 | 0.274 |
| | | bf16 | 0.227 | 0.291 | 0.135 | 0.219 | 0.218 |
| | | **Δ** | −0.026 | **+0.106** | **+0.147** | −0.003 | **+0.056** |

- En Qwen3.5 y LLaVA-OV, NF4 cuesta entre 0.01 y 0.03 de `attr_mean`.
- **InternVL3 rinde mejor en NF4 que en bf16, y la causa es de formato, no de percepción**
  (`peek.py`, `logs/peek_internvl_nf4_vs_bf16.txt` en el cluster). En bf16 el modelo deja de declarar
  la posición mucho más a menudo:

  | | omite x | omite y | exactitud x si la declara | exactitud y si la declara |
  |---|---|---|---|---|
  | NF4 | 0.136 | 0.232 | 0.397 / 0.864 = 0.460 | 0.282 / 0.768 = 0.367 |
  | bf16 | 0.303 | **0.610** | 0.291 / 0.697 = 0.418 | 0.135 / 0.390 = 0.346 |

  Cuando declara la posición, acierta de forma parecida en ambas condiciones (diferencia de 0.02–0.04);
  la caída de `attr_mean` viene de las omisiones, que cuentan como error. Además, en bf16 la x
  declarada se concentra en `left` (5 056 de 6 970). La cuantización perturba el estilo de respuesta
  de InternVL3 en este prompt; no se interpreta como que NF4 "mejore" la percepción. Es un ejemplo de
  la sensibilidad al formato que el enunciado pregunta.
- Diferencias sobre las mismas 10 000 imágenes; error estándar por atributo ≈ 0.005, de modo que
  diferencias de 0.02 o más no son ruido de muestreo. El efecto en el prompt elegido no se
  generaliza a los demás prompts, que no se evaluaron en bf16.

## 5. Observaciones para el informe

1. **Métricas de texto y fidelidad divergen.** Los prompts que imitan el formato del caption
   (`p4_dataset_format`, `p4m_multiexample`, `p2_constrained`) obtienen los mejores Bleu, ROUGE-L y
   CIDEr, pero no la mejor exactitud por atributo. En Qwen3.5 e InternVL3, `p4_dataset_format`
   tiene el CIDEr más alto de su modelo (0.434 y 0.377) con exactitudes iguales a las de una
   respuesta constante: forma ≈ 1/7, x e y ≈ 1/3, color 0.075, sin omisiones. **Verificado con
   `scripts/peek.py`:** ambos copian el ejemplo del prompt. Qwen3.5 responde `A "tab:blue" horse is
   at the bottom-right of the image.` en 9 997 de 10 000 imágenes (las 3 restantes, la misma frase sin
   el artículo); InternVL3 responde variantes de esa frase en todas (horse 99.2 %, blue 99.9 %,
   right/bottom 98.5 %). Sus exactitudes son, por tanto, la frecuencia en val de horse, right, bottom
   y blue, y su CIDEr mide solo la coincidencia de plantilla. Apoya el argumento de [9] y C15
   (`docs/metodologia.md` §5).
2. **Fallos de formato, no de percepción.** LLaVA-OV con `p2_constrained` responde 1.6 palabras de
   media y omite x y color en casi el 100 %. Qwen3.5 omite el color en el 92 % con `p2_constrained`
   y en el 74 % con `p3_json` pese a un JSON válido en el 100 %. Varios runs quedan bajo la clase
   mayoritaria por omisiones.
3. **Forma.** Con descripción libre (`p0`, `p1`) los tres modelos omiten la forma en el 71–83 % de las
   respuestas: describen "humanoid figure", "creature" o "toy" en lugar de una de las 7 clases.
4. **Color.** Ningún run supera 0.23, lejos del techo de 0.410. Los modelos usan nombres comunes
   ("teal", "bright green") que no coinciden con el nombre exacto de la paleta.
5. **Qwen3.5-Base y posición.** Con `p1_attributes` Qwen3.5 escribe una sección `Position:` explícita
   ("upper left quadrant", "center-right portion") y alcanza x = 0.820 e y = 0.582, muy por encima del
   resto. Verificado a mano (§6).

## 6. Revisión manual

**Qwen3.5, `p1_attributes`** (20 respuestas completas, `peek.py --n 20 --width 3000`):

- **x:** en las 20, la x extraída corresponde a la sección `Position:` de la respuesta; el modelo
  acierta en 19. El valor x = 0.820 se atribuye al modelo, no al extractor.
- **y:** dos casos discutibles, sin efecto en la selección:
  - `base/val/0002`: "upper-middle portion…, slightly above the horizontal center" → `top`. Con el
    criterio de matices, "slightly above the center" se lee como `mid`; "upper-middle" queda en el
    límite.
  - `base/val/0013`: "center of the image, slightly above the horizontal midline. It is positioned in
    the lower half…" → `bottom`. Por la regla de la primera declaración se esperaría `mid`. El texto
    es contradictorio y la referencia es `top`: es error con cualquiera de las dos lecturas.
    **Causa y corrección (16-11-2026):** "horizontal midline" hacía que la frase se tratara como
    referida solo al eje horizontal, y el `center` no fijaba la y. Ahora "horizontal/vertical" seguido
    de *midline, axis, line, plane, edge* o *direction* no marca la frase como de un solo eje.

**Corrección del color (16-11-2026).** En InternVL3 bf16, `base/val/0004` ("colored in a solid purple
shade. The shadow is cast…") el color quedaba en `None`: las palabras vecinas de "purple" cruzaban el
punto y "shadow" lo clasificaba como luz. Ahora las vecinas se buscan solo dentro de la misma frase.

Ambas correcciones van con sus tests (`tests/test_attributes.py`, 48 pasan en el cluster). Los 21 runs
se volvieron a puntuar el 16-11-2026 y las tablas de §3, §4 y §4b ya tienen los valores corregidos. El
efecto es pequeño: el mayor cambio es +0.004 en la y de Qwen3.5 `p1_attributes` (omisión de y de 0.034 a
0.017); ningún `attr_mean` cambia más de 0.001 y **la selección de prompts no cambia**.

**LLaVA-OV `p3_json` e InternVL3 `p1_attributes`** (8 muestras cada uno): extracción correcta en las
muestras revisadas ("upper right quadrant" → right/top; "positioned vertically in the center" →
y = mid, x sin declarar).

## 7. Inspección de modelos (paso 1.0)

Reportes en `runs/inspect_<modelo>_<quant>/report.json` (commit `4b0d0e9`; entre ese commit y
`3feefb0` solo cambian `max_new_tokens` y el conteo de tokens generados, no la carga ni la
cuantización). Habían sido movidos sin registro a `_para_borrar/runs/` en el cluster y se
devolvieron a `runs/` el 16-11-2026.

### 7.1 Fracción en NF4

| Modelo | Lineales en NF4 | Parámetros en NF4 / total | Fracción | Lineales en bf16 |
|---|---|---|---|---|
| LLaVA-OV-0.5B | 324 (visión 156, LLM 168) | 753.7 M / 893.7 M | **84.3 %** | conector (2), `lm_head` |
| InternVL3-1B | 312 (visión 144, LLM 168) | 659.8 M / 938.2 M | **70.3 %** | conector (2), `lm_head` |
| Qwen3.5-0.8B-Base | 234 (visión 48, LLM 186) | 582.5 M / 853.0 M | **68.3 %** | conector `visual.merger` (2), `lm_head` |
| VL-JEPA | 442 (X-Enc 218, Predictor 56, Y-Enc 168) | 912.0 M / 1 384.5 M | **65.9 %** | `vis_proj`, `out_proj`, `y_encoder.projector` |

- Se cumple D5 (`docs/metodologia.md` §2.1): todo componente congelado, encoder visual incluido, en
  NF4; ninguna lineal fuera del conector, las proyecciones y `lm_head` queda sin cuantizar.
- El resto hasta 100 % son capas que `bitsandbytes` no cuantiza (solo reemplaza `nn.Linear`): tablas
  de embeddings, normalizaciones y convolución de parches. La mayor es `predictor.embed_tokens` de
  VL-JEPA (262.7 M).

### 7.2 Cobertura de LoRA

El patrón de LoRA alcanza todas las lineales del alcance, todas `Linear4bit`, sin lineales del
alcance sin cubrir ni coincidencias fuera de él: 168 en LLaVA-OV e InternVL3 (7 proyecciones × 24
capas), 186 en Qwen3.5 (incluye `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` y `out_proj`
de las 18 capas Gated DeltaNet y q/k/v/o de las 6 de atención completa) y 56 en el Predictor de
VL-JEPA (8 capas).

### 7.3 `lm_head` (P11, `docs/plan_reconstruccion.md` §5)

| Modelo | `lm_head` | ¿Cuantizable con `bitsandbytes`? |
|---|---|---|
| LLaVA-OV-0.5B | atado a `embed_tokens` (`tie_word_embeddings: true`) | No sin desatar los pesos |
| Qwen3.5-0.8B-Base | atado a `embed_tokens` | No sin desatar los pesos |
| InternVL3-1B | independiente (`text_config.tie_word_embeddings: false`), 135.9 M parámetros | Sí |

**Decisión (usuario, 16-11-2026):** excepción uniforme, `lm_head` en bf16 en los tres VLM, por (a) no ser cuantizable sin modificar el modelo en dos de tres, (b) ser el comportamiento por
defecto de `transformers` con `bitsandbytes` (`get_keys_to_not_convert` mantiene la cabeza de
salida en precisión completa; fuente: documentación de la biblioteca, no un paper de
`docs/papers/`) y (c) mantener una regla única. Se declara en el informe; no hay que repetir corridas.

### 7.4 NF4 frente a bf16

- **VLM:** en las 4 imágenes de inspección, las respuestas NF4 y bf16 difieren en detalles pero
  describen el mismo objeto, color y posición aproximada.
- **VL-JEPA:** similitud coseno entre `Ŝ_Y` y `S_Y` (prompt `p1_attributes`):

  | | img 1 | img 2 | img 3 | img 4 | media |
  |---|---|---|---|---|---|
  | bf16 | 0.566 | 0.597 | 0.853 | 0.446 | 0.616 |
  | NF4 | 0.469 | 0.517 | 0.745 | 0.306 | 0.509 |

  NF4 baja la similitud en ≈ 0.1 en las cuatro imágenes. Con n = 4 no es concluyente; puede ser
  efecto de cuantizar los tres componentes o un problema de la cuantización manual de
  `models/vljepa.py`. Se contrasta con el control bf16 de VL-JEPA en val; si la brecha se mantiene,
  revisar la cuantización antes de atribuirla a NF4.

  **Remedición con relleno a la derecha (19-11-2026, commit `598e9bb`).** La tabla anterior se midió con
  la consulta rellenada **a la izquierda** hasta 512 tokens (en el cluster, el tokenizador de Llama-3.2
  se carga con `padding_side = "left"`, y el Predictor numera las posiciones RoPE de corrido sobre
  [visuales; consulta]). Con el relleno corregido (`runs/inspect_vljepa_open_msrvtt_{nf4,bf16}/report.json`):

  | | img 1 | img 2 | img 3 | img 4 | media |
  |---|---|---|---|---|---|
  | bf16 | 0.567 | 0.588 | 0.871 | 0.434 | 0.615 |
  | NF4 | 0.480 | 0.501 | 0.774 | 0.278 | 0.508 |

  Las similitudes cambian poco (≤ 0.03 por imagen) y la brecha NF4 − bf16 se mantiene: −0.107 en la
  media, entre −0.087 y −0.156 por imagen, con el mismo signo en las cuatro. El relleno no la explicaba.
  Las demás mediciones de la etapa 1 no pasan por el Predictor (VLM, Y-Decoder y su techo).

  **Diagnóstico por componente (A2, 20-11-2026; `scripts/diag_quant_vljepa.py`, job `diag_quant`).** Se mide en
  1 024 imágenes de val, con `p1_attributes`, el mismo subconjunto en las cinco condiciones. InfoNCE en lotes fijos
  de 128 (azar ≈ ln 128 = 4.85). Coseno por etapa contra bf16: E1 = X-Encoder, E2/E3 = Predictor, T1 = S_Y.

  | Condición | cos(Ŝ_Y, S_Y) | Δ contra bf16 | R@1 p→t / t→p | InfoNCE | E1 | E2 | E3 | T1 |
  |---|---|---|---|---|---|---|---|---|
  | bf16 (referencia) | 0.682 | — | 0.002 / 0.001 | 5.024 | — | — | — | — |
  | NF4 completo | 0.545 | −0.138 | 0.002 / 0.001 | 5.089 | 0.993 | 0.983 | 0.989 | **0.876** |
  | NF4 solo X-Encoder | 0.681 | −0.001 | 0.002 / 0.007 | 5.022 | 0.993 | 0.986 | 0.990 | 1.000 |
  | NF4 solo Predictor | 0.685 | +0.003 | 0.001 / 0.002 | 5.027 | 1.000 | 0.996 | 0.998 | 1.000 |
  | NF4 solo Y-Encoder | 0.540 | **−0.142** | 0.002 / 0.001 | 5.126 | 1.000 | 1.000 | 1.000 | **0.876** |

  - **La cuantización por componente está aislada.** Las etapas que no dependen del componente cuantizado dan
    1.000 contra bf16. Los efectos son aditivos: la suma de las tres Δ (−0.141) ≈ la Δ de NF4 completo (−0.138).
  - **El Y-Encoder explica toda la brecha** (−0.142 de −0.138). X-Encoder y Predictor en NF4 casi no cambian
    nada (coseno contra bf16 ≥ 0.99). En cambio, S_Y del Y-Encoder (EmbeddingGemma-300m) en NF4 queda con
    coseno 0.876 contra su versión bf16. Cumple el criterio propuesto para revisar la cuantización (≥ 70 % de la
    caída y < 0.95 frente a ≥ 0.99), sea cual sea el umbral exacto.
  - **Sin discriminación ni siquiera en bf16:** R@1 ≈ 1/1 024 y una InfoNCE por encima de ln 128 en las cinco
    condiciones. Para el checkpoint MSR-VTT, Ŝ_Y no distingue entre imágenes de M3DI por recuperación. El
    coseno de 0.68 con su propio S_Y no implica correspondencia si los captions de M3DI (plantillas casi
    iguales) son igual de parecidos entre sí. Pendiente: medir la similitud con los captions de otras imágenes
    para confirmarlo.
  - Pendiente antes de decidir: (a) si la caída del Y-Encoder es un error de la cuantización manual o una
    sensibilidad real del modelo a NF4 (comparar con la carga estándar de `transformers` + `BitsAndBytesConfig`);
    (b) qué hacer con el Y-Encoder en el protocolo (decisión del usuario).

## 7b. Barrido del Y-Decoder con captions de M3DI (paso 1.3) — diseño SUPERADO

> **Superado el 16-11-2026** (decisión del usuario, `docs/metodologia.md` §2.4): el decodificador de la
> etapa 1 ya no se entrena con M3DI, sino con captions de MSR-VTT, para no darle a VL-JEPA el formato del
> dataset en zero-shot. Lo que sigue queda como antecedente: muestra que un lector entrenado desde cero
> con M3DI no se estabiliza en 10 épocas (techo en val 0.739 con 3 épocas, *k* = 4; 0.906 con 10, mejor
> paso 6 800 de 6 840, curva todavía subiendo). Los decodificadores `ydec_vljepa_open_msrvtt_nf4_zs_nf4_k{1,4,8}_r64`
> y `…_k4_r64_e10` no se usan en ninguna condición.

Techo `Y-Decoder(S_Y)`: el decodificador reconstruye el caption desde el embedding del **caption
verdadero** (no desde la predicción de VL-JEPA). 500 captions de val (subconjunto determinista),
VL-JEPA y Llama-3.2-1B en NF4, 3 épocas (2052 pasos, 10 936 captions distintos de train, lote 16).
Job `ydec`, commit `0698e68`.

| *k* | Mejor paso | forma | x | y | color | attr_mean | Caption exacto | Entrenables | Tiempo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2052 (último) | 0.998 | 0.736 | 0.434 | 0.422 | 0.648 | 0.168 | 48.2 M | 10 min |
| **4** | 2052 (último) | 0.994 | **0.840** | **0.550** | **0.572** | **0.739** | **0.292** | 57.7 M | 12 min |
| 8 | 2000 | 1.000 | 0.834 | 0.508 | 0.498 | 0.710 | 0.206 | 70.3 M | 14 min |

- **Elegido: *k* = 4** (regla: mayor `attr_mean` del techo). La ventaja sobre *k* = 8 (0.029) es del
  orden de la oscilación entre evaluaciones consecutivas (±0.03 con n = 500).
- **Ninguna curva se había estabilizado** (*k* = 4: 0.661 → 0.707 → 0.716 → 0.718 → 0.739 en los pasos
  1400–2052). Por eso **no** se puede atribuir todavía al Y-Encoder la pérdida de posición del techo
  (y ≈ 0.5): puede ser subentrenamiento del decodificador. Decisión: reentrenar *k* = 4 con 10 épocas
  (`ydec_vljepa_open_msrvtt_nf4_zs_nf4_k4_r64_e10`).
- El color del techo (0.572) no se compara con el techo de 0.410 de los VLM: aquí se lee el embedding
  del texto, que contiene el nombre de la paleta; aquel es el máximo predecible desde la imagen.
- En las reconstrucciones revisadas los errores son de posición (p. ej. top-left → bottom-left) o de
  tono cercano (`bright blue` → `rich blue`); forma y plantilla salen bien casi siempre.

## 7c. Y-Decoder genérico (MSR-VTT)

Diseño C5′ (`docs/metodologia.md` §2.4): decodificador entrenado solo con captions de MSR-VTT `train_7k`; el paso
se elige por CIDEr en MSR-VTT `test_1k` (500 captions). El techo en M3DI (`m3di_*`) **solo se reporta**. Todas
las cifras son de **reconstrucción**: el decodificador lee el `S_Y` del caption verdadero, no la predicción de
VL-JEPA a partir de la imagen. No son comparables con el CIDEr de captioning de imagen o video de la literatura.

**Barrido de *k*** (1 época, VL-JEPA y Llama-3.2-1B en NF4; job `ydec_msrvtt`, 16-11-2026): CIDEr k = 1 1.321,
k = 4 1.384, **k = 8 1.439**, con k = 8 adelante en las 7 evaluaciones. Techo en M3DI ≈ 0.12 en los tres.

**Ganador k = 8, 3 épocas, NF4** (`runs/ydec_vljepa_open_msrvtt_nf4_zs_nf4_msrvtt_k8_r64_e3`; job `ydec_k8_nf4`,
19-11-2026, `DONE(0)`; 22 143 pasos):

| Paso | 0 | 4 000 | 7 000 | 10 000 | 14 000 | 17 000 | 19 000 | 21 000 | **22 000** | 22 143 (último) |
|---|---|---|---|---|---|---|---|---|---|---|
| CIDEr (MSR-VTT) | 0.001 | 1.389 | 1.439 | 1.521 | 1.652 | 1.674 | 1.692 | 1.717 | **1.827** | 1.805 |
| Bleu-4 | 0.000 | 0.116 | 0.128 | 0.130 | 0.134 | 0.138 | 0.128 | 0.128 | **0.149** | 0.148 |
| ROUGE-L | 0.062 | 0.395 | 0.406 | 0.408 | 0.425 | 0.423 | 0.427 | 0.432 | **0.441** | 0.440 |
| techo M3DI `attr_mean` | 0.000 | 0.110 | 0.120 | 0.121 | 0.113 | 0.116 | 0.124 | 0.127 | 0.120 | 0.119 |

- **Mejor paso: 22 000** (CIDEr 1.827, frente a 1.439 con 1 época). El mejor paso cae casi al final y la
  curva no se ha aplanado del todo: +0.11 entre los pasos 21 000 y 22 000. Parte de esa subida final es
  esperable con el LR que decae al final del schedule. Entre evaluaciones consecutivas el CIDEr oscila
  ±0.05–0.1, con 500 captions. No se entrenan más épocas: 3 era el presupuesto acordado.
- **Techo en M3DI ≈ 0.12, estable desde el paso 4 000**: forma ≈ 0.45, x e y ≈ 0, color ≈ 0.01. El lector
  genérico no escribe posiciones ni nombres de la paleta de M3DI, como se esperaba (lectura acordada,
  `docs/estado_sesion.md` §3). La exactitud zero-shot de VL-JEPA queda acotada por ese techo.
- Pendiente: el `summary.json` del run (parámetros entrenables y tiempo).

## 7d. VL-JEPA zero-shot: piloto de 200 imágenes (A4, 20-11-2026)

NF4, decodificador genérico (§7c), 6 prompts, 200 imágenes de val (`runs/_tablas/vljepa_piloto.csv`, job
`s1_vljepa_piloto`). Alrededor de 37 s por prompt, incluida la carga del modelo.

| Prompt | forma | x | y | color | attr_mean | Bleu-4 | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|---|
| p0_minimal | 0.005 | 0.000 | 0.005 | 0.015 | 0.006 | 0.000 | 0.160 | 0.015 |
| p1_attributes | 0.005 | 0.000 | 0.000 | 0.015 | 0.005 | 0.000 | 0.168 | 0.015 |
| p2_constrained | 0.005 | 0.000 | 0.000 | 0.010 | 0.004 | 0.000 | 0.172 | 0.011 |
| p3_json | 0.010 | 0.000 | 0.000 | 0.010 | 0.005 | 0.000 | 0.168 | 0.011 |
| p4_dataset_format | 0.005 | 0.000 | 0.000 | 0.020 | 0.006 | 0.000 | 0.168 | 0.014 |
| p4m_multiexample | 0.005 | 0.000 | 0.000 | 0.010 | 0.004 | 0.000 | 0.170 | 0.013 |

- **Exactitud ≈ 0, muy por debajo de la clase mayoritaria (0.222) y del techo del lector (≈ 0.12).** Casi todo
  queda omitido: con `p1_attributes`, forma omitida en 199/200 y posición en 200/200. Las diferencias entre
  prompts son de 1 o 2 aciertos en 200 imágenes: ruido.
- **Revisión a mano (`peek`, 12 ejemplos):** el decodificador escribe captions de estilo MSR-VTT, p. ej. "a strange
  creature with long ears is in the woods" o "a pink animated animal is shown in the woods at night". Nunca
  nombra la forma de M3DI ni la posición. Lo que sí parece captar es la *gist*: objetos animados o de dibujo,
  "creature", "animal". Es coherente con §7.4: Ŝ_Y no distingue entre imágenes por recuperación.
- **Falsos positivos del extractor fuera de dominio:** lee como color algunas palabras que son nombres de la
  paleta xkcd usados en otro sentido ("in the **desert**" → `desert`; `dark` × 6). No inflan la exactitud (color
  0.015) y el extractor no se cambia (CLAUDE.md §6: una sola implementación para los 4 modelos). Se declara.
- Detalle menor: `peek` informa `tokens: mediana=0` porque el conteo de tokens generados no se registra para
  VL-JEPA. No afecta la puntuación.

## 8. Cierre de la etapa 1 (CE1, decisión del usuario, 20-11-2026)

La etapa 1 se cierra **sin más entrenamientos** (`docs/metodologia.md` §6.1, CE1). Lo único que se agrega es
**inferencia**: VL-JEPA zero-shot en val con los 6 prompts y el decodificador genérico NF4 ya entrenado.

| Paso | Estado |
|---|---|
| VLM zero-shot, 6 prompts, val completo, NF4 (§3–§4) | hecho |
| Control bf16 de los VLM (prompt elegido, val completo) | hecho (§4b) |
| Correcciones del extractor (§6) | hecho: 21 runs puntuados de nuevo, tablas actualizadas |
| Inspección de modelos (§7) | hecho |
| Barrido del Y-Decoder con M3DI (`ydecoder_sweep`) | hecho, superado (§7b) |
| Y-Decoder genérico: barrido de *k* en MSR-VTT y k = 8 hasta 3 épocas, NF4 | hecho (§7c) |
| Diagnóstico de cuantización de VL-JEPA (A2) | hecho (§7.4); falta la verificación "propio frente a otros" (sin GPU) |
| **VL-JEPA zero-shot en val, 6 prompts** (A4: piloto de 200 → val completo → prompt) | **en curso** (job `s1_vljepa_piloto`, 20-11) |
| Y-Decoder genérico bf16 (A3) y control bf16 de VL-JEPA (A5) | **cancelado** (CE1) |
| Condición intermedia (A6) | **cancelado** (CE1) |
| Pérdida de información de la etapa 1 (A7) | **cancelado** (CE1); el comando queda en `stage1_infoloss.txt` (solo inferencia) |
| P11 (`lm_head`) | resuelto: bf16 en los tres VLM (§7.3) |
| P10 (Y-Decoder del control bf16 de VL-JEPA) | sin efecto en la etapa 1 (control cancelado); queda abierto para la etapa 2 |

**Qué se declara en el informe sobre VL-JEPA en la etapa 1.** Se evalúa solo en NF4, sin control bf16 ni
condición intermedia. Su exactitud zero-shot está acotada por el vocabulario del lector genérico (techo
≈ 0.12, §7c). Además, el lector se entrenó con S_Y del Y-Encoder en NF4, que está desplazado respecto de bf16
(coseno 0.876, §7.4). Por recuperación, el checkpoint no discrimina entre imágenes de M3DI, ni siquiera en bf16
(§7.4).
