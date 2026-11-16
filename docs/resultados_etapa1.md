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
| Puntaje | `scripts/score.py`, extractor de `3feefb0` (sin cambios en `0698e68`), 37 tests pasan; `pycocoevalcap` con Java 11 |
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
| LLaVA-OV-0.5B | p0_minimal | 0.258 | 0.224 | 0.208 | 0.198 | 0.222 | 0.715 | 0.344 | 0.334 | 0.025 | 94.1 | 0.088 | 0.016 | 0.149 | 0.000 |
| LLaVA-OV-0.5B | p1_attributes | 0.271 | 0.445 | 0.298 | 0.210 | 0.306 | 0.710 | 0.027 | 0.069 | 0.004 | 28.2 | 0.267 | 0.008 | 0.329 | 0.024 |
| LLaVA-OV-0.5B | p2_constrained | 0.235 | 0.000 | 0.012 | 0.000 | 0.062 | 0.004 | 0.999 | 0.936 | 1.000 | 1.6 | 0.000 | 0.000 | 0.060 | 0.045 |
| LLaVA-OV-0.5B | **p3_json** | 0.373 | 0.338 | 0.330 | 0.197 | **0.309** | 0.017 | 0.000 | 0.000 | 0.019 | 8.3 | 0.082 | 0.000 | 0.076 | 0.060 |
| LLaVA-OV-0.5B | p4_dataset_format | 0.171 | 0.184 | 0.156 | 0.138 | 0.162 | 0.050 | 0.538 | 0.539 | 0.355 | 8.1 | 0.269 | 0.070 | 0.303 | 0.185 |
| LLaVA-OV-0.5B | p4m_multiexample | 0.212 | 0.057 | 0.063 | 0.179 | 0.128 | 0.168 | 0.769 | 0.769 | 0.053 | 7.2 | 0.233 | 0.043 | 0.278 | 0.190 |
| InternVL3-1B | p0_minimal | 0.148 | 0.190 | 0.188 | 0.224 | 0.187 | 0.817 | 0.542 | 0.479 | 0.005 | 56.9 | 0.133 | 0.021 | 0.195 | 0.000 |
| InternVL3-1B | **p1_attributes** | 0.201 | 0.398 | 0.282 | 0.216 | **0.274** | 0.752 | 0.135 | 0.232 | 0.031 | 55.3 | 0.142 | 0.012 | 0.226 | 0.000 |
| InternVL3-1B | p2_constrained | 0.232 | 0.016 | 0.334 | 0.219 | 0.200 | 0.681 | 0.963 | 0.007 | 0.001 | 10.3 | 0.499 | 0.093 | 0.469 | 0.181 |
| InternVL3-1B | p3_json | 0.303 | 0.032 | 0.314 | 0.210 | 0.215 | 0.498 | 0.927 | 0.048 | 0.000 | 7.3 | 0.073 | 0.000 | 0.073 | 0.056 |
| InternVL3-1B | p4_dataset_format | 0.142 | 0.325 | 0.328 | 0.075 | 0.218 | 0.007 | 0.015 | 0.015 | 0.000 | 9.8 | 0.410 | 0.138 | 0.438 | 0.377 |
| InternVL3-1B | p4m_multiexample | 0.149 | 0.048 | 0.037 | 0.084 | 0.079 | 0.183 | 0.901 | 0.910 | 0.001 | 3.0 | 0.010 | 0.004 | 0.074 | 0.082 |
| Qwen3.5-0.8B-Base | p0_minimal | 0.119 | 0.353 | 0.283 | 0.161 | 0.229 | 0.830 | 0.429 | 0.396 | 0.002 | 116.9 | 0.064 | 0.002 | 0.111 | 0.000 |
| Qwen3.5-0.8B-Base | **p1_attributes** | 0.138 | 0.820 | 0.578 | 0.124 | **0.415** | 0.773 | 0.011 | 0.034 | 0.001 | 84.9 | 0.088 | 0.002 | 0.158 | 0.000 |
| Qwen3.5-0.8B-Base | p2_constrained | 0.339 | 0.099 | 0.346 | 0.031 | 0.204 | 0.036 | 0.865 | 0.141 | 0.921 | 9.1 | 0.479 | 0.089 | 0.460 | 0.111 |
| Qwen3.5-0.8B-Base | p3_json | 0.284 | 0.489 | 0.441 | 0.062 | 0.319 | 0.000 | 0.000 | 0.000 | 0.737 | 9.7 | 0.057 | 0.000 | 0.058 | 0.033 |
| Qwen3.5-0.8B-Base | p4_dataset_format | 0.142 | 0.331 | 0.338 | 0.075 | 0.221 | 0.000 | 0.000 | 0.000 | 0.000 | 11.0 | 0.512 | 0.154 | 0.466 | 0.434 |
| Qwen3.5-0.8B-Base | p4m_multiexample | 0.137 | 0.600 | 0.464 | 0.221 | 0.355 | 0.798 | 0.038 | 0.002 | 0.002 | 12.7 | 0.546 | 0.224 | 0.483 | 0.409 |

CIDEr se calcula con el idf de las referencias de val; no es comparable con valores en test
(`docs/metodologia.md` §2.1).

## 4. Selección de prompt (criterio: `attr_mean` en val, `docs/metodologia.md` §2.2)

| Modelo | Prompt elegido | attr_mean | Segundo | Margen |
|---|---|---|---|---|
| Qwen3.5-0.8B-Base | `p1_attributes` | 0.415 | `p4m_multiexample` (0.355) | 0.060 |
| LLaVA-OV-0.5B | `p3_json` | 0.309 | `p1_attributes` (0.306) | **0.004** |
| InternVL3-1B | `p1_attributes` | 0.274 | `p4_dataset_format` (0.218) | 0.057 |

La selección se reporta pero no decide el prompt de la etapa 2, que es `p1_attributes` en los 4
modelos [CLAUDE.md §3]. Sí fija la condición zero-shot que se controla en bf16 y se evalúa en test.

**LLaVA-OV: selección no robusta.** Se aplicó la regla acordada sin cambiarla tras ver val (decisión
del usuario, 16-11-2026). El margen sobre `p1_attributes` es de 0.004 y la posición de `p3_json` no
depende de la imagen: `y = top` en las 10 000 respuestas y `x = right` en el 90 %, de modo que su x e
y igualan a la clase mayoritaria. Toda su ventaja viene de forma (0.373) y color (0.197). Con
`p1_attributes`, en cambio, LLaVA-OV sí discrimina x (0.445). El informe debe reportar ambos.

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
   ("upper left quadrant", "center-right portion") y alcanza x = 0.820 e y = 0.578, muy por encima del
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
    **Pendiente revisar en el extractor** por qué no se aplicó la primera declaración.

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

## 8. Pendiente para cerrar la etapa 1

| Paso | Estado |
|---|---|
| Control bf16 de los VLM (prompt elegido, val completo) | en curso (`s1_bf16`) |
| Barrido del Y-Decoder (`ydecoder_sweep`, *k* ∈ {8, 4, 1}) | en curso (`ydec`) |
| VL-JEPA zero-shot en val (piloto de 200 y completo) | espera el Y-Decoder |
| Control bf16 de VL-JEPA | espera el *k* elegido: entrenar el Y-Decoder bf16 y luego inferir |
| Pérdida de información (`stage1_infoloss`) | espera el Y-Decoder |
| P11 (`lm_head`) | resuelto: bf16 en los tres VLM (§7.3) |
| P10 (Y-Decoder del control bf16 de VL-JEPA) | resuelto: decodificador propio, todo en bf16 |
| Caso `base/val/0013` del extractor (§6) | revisar |
