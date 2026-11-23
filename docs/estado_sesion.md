# Estado de la sesión — 20-11-2026 (cierre de la etapa 1)

> Documento de traspaso: dónde quedó el trabajo, qué está corriendo y qué sigue. Para retomar, leer
> este archivo primero. Los resultados medidos están en `docs/resultados_etapa1.md`; el protocolo, en
> `docs/metodologia.md`; los comandos del cluster, paso a paso, en `docs/pasos_cluster.md`.

## 0. Al retomar (22-11-2026, tarde)

**Hecho:** grilla de los 4 (`resultados_etapa2.md` §2–3) y ajuste completo de los 3 VLM, NF4 y bf16 (§4: attr_mean
0.82–0.83 en las 500 de val). Ganador de VL-JEPA lr 5e-4 · r64 · ×1, en el borde del eje de LR: aceptado y declarado
(**G5**). `hparams_confirmed: true`. Adaptadores anotados en `final_test.yaml`, `stage2_val.txt`, `stage2_infoloss.txt`
y `ydecoder_msrvtt.txt` §4 (exp_id esperados; comprobar con `ls -d runs/train_*_e1/adapter_best`).

**Siguiente en el cluster** (`pasos_cluster.md` B5–B7): GPU 0 → ajuste de VL-JEPA (NF4 y luego bf16, job `s2_train_vj`);
GPU 1 → val completo de los 6 VLM ajustados (job `s2_val_vlm`). Después: B6 (Y-Decoder de la etapa 2, NF4, 20 épocas)
→ B7 de VL-JEPA → B8.

**Pendientes del usuario:** P10 (decodificador bf16 de VL-JEPA; bloquea solo su control bf16 en B6/B7) y P4
(CLAUDE.md). Sin GPU: notebooks 02 (§11c-bis) y 03.

## 1. Dónde estamos

**Etapa 1 cerrada (CE1, 20-11-2026)**, incluida la inferencia de VL-JEPA en val. **Etapa 2:** humo hecho (los 4);
sigue la grilla.

- **Cierre de la etapa 1 (CE1, `metodologia.md` §6.1):** se cierra con los 3 VLM, el Y-Decoder genérico NF4 (k = 8,
  3 épocas, CIDEr 1.827, techo M3DI ≈ 0.12) y el diagnóstico de cuantización. Se agrega solo inferencia: VL-JEPA
  zero-shot NF4 con los 6 prompts. **Cancelados:** Y-Decoder bf16 y control bf16 de VL-JEPA (A3, A5), condición
  intermedia (A6) y pérdida de información de la etapa 1 (A7). `final_test.yaml` queda con 15 condiciones.
- **VL-JEPA zero-shot en val (A4, hecho 20-11):** 6 prompts × 10 000, ≈ 580 s cada uno. Gana `p4m_multiexample`
  (attr_mean 0.0039; todos entre 0.0026 y 0.0039, es decir, ruido); anotado en `final_test.yaml` junto con los
  prompts zero-shot de los VLM. **Etapa 1 cerrada.** La GPU 0 sigue ocupada por un proceso ajeno.
- **Diagnóstico de cuantización (A2, hecho):** toda la brecha NF4 − bf16 de VL-JEPA viene del **Y-Encoder** (S_Y
  contra bf16: coseno 0.876; X-Encoder y Predictor ≥ 0.99). Además, el checkpoint **no discrimina entre imágenes de
  M3DI** ni siquiera en bf16 (R@1 ≈ azar). Detalle en `resultados_etapa1.md` §7.4. Falta la verificación "propio
  frente a otros" (notebook 02 §11c-bis, sin GPU).
- **Etapa 2:** lista para el humo (B2) y la grilla (B4). Nada se ha ejecutado, ni sus tests. Antes del ajuste
  completo hay que decidir qué hacer con la cuantización del Y-Encoder (ver §4).
- **Notebooks:** `02_resultados_etapa1.ipynb` (con la nota de cierre y §11c-bis) y `03_resultados_etapa2.ipynb`.
  Ninguno se ha ejecutado.
- **Sin commitear** (cierre CE1): notebooks 02 y 03, docs (`estado_sesion`, `metodologia`, `pasos_cluster`,
  `resultados_etapa1`), `configs/final_test.yaml`, colas `ydecoder_msrvtt`, `stage1_vljepa_val`, `stage1_infoloss`
  y `tests/test_final_test.py`. **No hacer `git pull`
  mientras corra una cola.**

## 2. Lo cerrado en la etapa 1

| Qué | Resultado | Dónde |
|---|---|---|
| VLM zero-shot, 3 modelos × 6 prompts, val completo, NF4 | 18 runs puntuados con el extractor corregido y revisados a mano | `resultados_etapa1.md` §3 |
| Prompt elegido (`attr_mean` en val) | Qwen3.5 `p1_attributes` 0.416 · LLaVA-OV `p3_json` 0.309 (empate técnico con `p1`, 0.306; su posición es constante) · InternVL3 `p1_attributes` 0.274 | §4 |
| Líneas base | clase mayoritaria `attr_mean` 0.222; techo del color 0.410 | §2 |
| Control bf16 de los VLM | Δ(NF4 − bf16): Qwen −0.027, LLaVA −0.009, InternVL3 +0.056 (en bf16 omite la posición: formato, no percepción) | §4b |
| Inspección de modelos | fracción NF4: LLaVA 84 %, InternVL3 70 %, Qwen 68 %, VL-JEPA 66 %; LoRA cubre todas las lineales | §7 |
| Extractor | dos correcciones (vecinas del color dentro de la frase; "horizontal midline") y `json_mode_for` (VL-JEPA nunca se lee como JSON); 48+ tests | §6 |

## 3. Decisiones tomadas hoy (todas registradas en `metodologia.md` §6.1)

- **Selección de prompts:** se mantuvo la regla `attr_mean` también en LLaVA (`p3_json`).
- **P11:** `lm_head` en bf16 en los tres VLM (excepción uniforme; atado a `embed_tokens` en LLaVA y Qwen).
- **bf16 frente al enunciado:** las condiciones bf16 son *controles*; toda selección y comparación
  principal es en NF4 (`metodologia.md` §2.1).
- **P10:** el control bf16 de VL-JEPA lleva su propio Y-Decoder, todo en bf16. La carga del
  decodificador verifica adaptador **y** cuantización (`check_decoder_matches`).
- **C5′ — Y-Decoder genérico (el cambio grande del día):** el decodificador de la etapa 1 se entrena
  solo con captions de **MSR-VTT `train_7k`** (140 200; 118 094 distintos), y el mejor paso se elige por
  **CIDEr en MSR-VTT `test_1k`**. Así VL-JEPA llega a la etapa 1 sin conocer el formato de M3DI, igual
  que los VLM. Los decodificadores ajustados a M3DI (**condición intermedia** y **etapa 2**) continúan
  desde el genérico (`decoder.text_source=m3di decoder.init_from=…`). Se agregó la condición intermedia
  al plan de test (`configs/final_test.yaml`, entrada `vljepa_zs_nf4_decm3di`).
- *k* del decodificador genérico se barre en MSR-VTT; el ganador se entrena hasta 3 épocas.
- **Superado:** el barrido de *k* con captions de M3DI (`ydecoder_sweep.txt`, runs `…_k{1,4,8}_r64` y
  `…_k4_r64_e10`). No se usan; quedan como antecedente (`resultados_etapa1.md` §7b).

**Lectura acordada de VL-JEPA en la etapa 1:** con el lector genérico, su techo en M3DI será ≈ 0.1 (no
escribe posiciones ni nombres de paleta). Su exactitud zero-shot será baja por vocabulario, no
necesariamente por percepción. Lo que más informa sobre VL-JEPA en la etapa 1 es la **condición
intermedia** y las **sondas** de §2.5.

## 3b. Decisiones de la etapa 2 (16-11-2026, `metodologia.md` §4 y §6.1)

- **P7:** VL-JEPA se selecciona por la InfoNCE en val (mínima). Contradice la letra de CLAUDE.md §3 para VL-JEPA → P4.
- **P9:** `y_encoder.projector` con el LR del conector (completo). open-vljepa la entrena a ×0.05: se declara.
- **P5:** código + propuesta citada; LR por barrido corto en val ({1e-4, 2e-4, 5e-4}, 150 pasos × 128).
- **G1 (20-11-2026): grilla completa de hiperparámetros**, reemplaza al barrido de solo LR: LR {1e-4, 2e-4, 5e-4} ×
  LoRA r {16, 64} (α = 16 fijo) × multiplicador del conector {1, 5}; 48 corridas de 150 pasos, solo NF4, **ganador por
  modelo** (diferencia de protocolo que se declara). `configs/train/grid.yaml` → `scripts/make_grid.py` →
  `configs/queues/stage2_grid{,_g0,_g1}.txt`; resultado con `scripts/summarize_train.py --winners`. Sin ejecutar.
- **G2 (20-11-2026): sin parada temprana**, ni en la grilla (150 pasos completos, gana el último paso evaluado) ni
  en el ajuste completo (época entera; solo selección de checkpoint `adapter_best`). Se descartó *successive halving*.
  Limitación declarada: la grilla compara tras un schedule corto completo, no a mitad de una época.
- Pendiente del usuario: confirmar la propuesta (filas "Propuesta" de `metodologia.md` §4), P8 tras el humo,
  épocas del Y-Decoder ajustado a M3DI (intermedio y etapa 2), y autorizar la actualización de CLAUDE.md (P4).
- **Verificado en el cluster (19-11-2026): el tokenizador de Llama-3.2 se carga con `padding_side = "left"`**
  (EmbeddingGemma, `right`). La inferencia rellenaba la consulta hasta 512 por la izquierda y corría las
  posiciones RoPE. Corregido: `models/vljepa.py` fija `right` en ambos al cargar. Única medición afectada: la
  similitud coseno de la inspección (`resultados_etapa1.md` §7.4), **ya remedida**: cambia ≤ 0.03 y la
  brecha NF4 − bf16 se mantiene (−0.107). El Y-Decoder y su techo no pasan por el Predictor: no se repiten.
- **Abierto:** la brecha NF4 − bf16 de VL-JEPA (≈ −0.1 en coseno, n = 4) persiste. Según §7.4, revisar la
  cuantización manual antes de atribuirla a NF4. Propuesta pendiente de aprobación: diagnóstico cuantizando
  un componente a la vez (X-Encoder, Predictor, Y-Encoder). **Aprobado y escrito (19-11):**
  `scripts/diag_quant_vljepa.py`, cola `configs/queues/diag_quant_vljepa.txt` (bf16, NF4 completo y NF4 por
  componente; 1 024 imágenes de val; coseno, recall@k, InfoNCE en lotes de 128 y coseno por etapa contra bf16).
  Tests nuevos: `test_embedding_metrics.py`, `test_vljepa_quant_components.py`. Sin ejecutar.

## 4. Pasos que siguen

> **Guía ejecutable paso a paso (19-11-2026): `docs/pasos_cluster.md`** — comandos, verificación y qué mandar a
> Claude para cerrar la etapa 1 (A1–A8), la etapa 2 (B0–B8) y test (C). Análisis de la etapa 1:
> `notebooks/02_resultados_etapa1.ipynb`. Nuevos para la etapa 2: `scripts/summarize_train.py` (humo, P8,
> barrido con `--winners`) y `configs/queues/stage2_val.txt` (adapter_best en val completo, P1).

| # | Paso | Dónde | Depende de |
|---|---|---|---|
| 1 | ~~VL-JEPA zero-shot (A4)~~ hecho: `p4m_multiexample` | — | — |
| 2 | ~~Prompts zero-shot en `final_test.yaml`~~ hecho (los 4 modelos) | — | — |
| 3 | Commit + push; `git pull` en el cluster **cuando no corra ninguna cola** | usuario | — |
| 4 | Verificación "propio frente a otros" (notebook 02 §11c-bis o el comando del 20-11, sin GPU) | cluster | — |
| 5 | Correr el notebook 02 completo y actualizar `resultados_etapa1.md` con VL-JEPA (A8) | cluster + laptop | 2, 3 |
| 6 | Tests de entrenamiento y humo de la etapa 2 (B1–B2) | cluster | 3 |
| 7 | ~~P5~~ confirmado (P5′); revisar `max_grad_norm` con el humo; P8 tras el humo; grilla de los **VLM** (B4a, 36 corridas) → ganadores | usuario + Claude → cluster | 6 |
| 8 | ~~YQ~~ **decidido: Y-Encoder en NF4** (20-11). Pendiente solo la corrida completa de `diag_yencoder_quant.py` (cifras del informe) | cluster | — |
| 9 | `stage2_train.txt` → Y-Decoder de la etapa 2 (`ydecoder_msrvtt.txt` §4) → `stage2_val.txt` → `stage2_infoloss.txt` (B5–B8; notebook 03) | cluster | 7, 8 |
| 10 | Llenar `configs/final_test.yaml` y correr `scripts/final_test.py` una sola vez (Parte C) | cluster | 9 |

**Decisiones del usuario pendientes:** de dónde parte el Y-Decoder del control bf16 de la
etapa 2, ya que no hay genérico bf16 (P10, por CE1); P4 (actualizar CLAUDE.md §3: prompt de la etapa 2, controles bf16,
Y-Decoder, P7, P9 y el cierre CE1). **Tomadas el 20-11:** P5 (P5′), grilla en dos tandas (G3), Y-Encoder en NF4 (YQ), lote 64 (P8, por OOM de
VL-JEPA con 128), Y-Decoder M3DI con 20 épocas (YD).

## 5. Comandos para retomar

Los comandos completos de cada paso están en `docs/pasos_cluster.md`. Los más inmediatos:

**Estado del cluster al retomar:**
```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM
scripts/jobs.sh status
scripts/jobs.sh gpu
grep -E "\[ydec\]\[val\]|\[ydec\] listo" logs/ydec_k8_nf4/job.log | tail -5
```

**Y-Decoder genérico bf16 (A3), cuando haya una GPU libre:**
```bash
grep '^model' configs/queues/ydecoder_msrvtt.txt | tail -1 > /tmp/ydec_k8_bf16.txt
SCRIPT=scripts/train_ydecoder.py GPU=0 scripts/launch.sh ydec_k8_bf16 scripts/run_queue.sh /tmp/ydec_k8_bf16.txt
```

**Tests y humo de la etapa 2 (B1–B2; requiere una GPU libre y `git pull` hecho sin colas corriendo):**
```bash
uv run --no-sync pytest tests/test_train_units.py tests/test_train_e2e.py tests/test_make_grid.py tests/test_summarize_train.py -q
SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_smoke scripts/run_queue.sh configs/queues/stage2_smoke.txt
uv run --no-sync python scripts/summarize_train.py runs/train_*_m6
```

**Notebooks (cluster, VS Code Remote-SSH, kernel `vlmfid`):** `notebooks/02_resultados_etapa1.ipynb` y
`notebooks/03_resultados_etapa2.ipynb`. Solo leen `runs/`; correrlos de nuevo a medida que terminan las colas.

**Traer resultados al laptop:** ver el anexo de `docs/pasos_cluster.md`. El destino del `scp` es la carpeta de
trabajo de la sesión de Claude, que cambia en cada sesión: pedirle la ruta antes de copiar. `/tmp` del cluster
también puede limpiarse.

## 6. Cómo se trabaja (recordatorios)

- **Flujo:** editar en el laptop → commit y push (los hace el usuario) → `git pull` en el cluster → ejecutar.
  Nada se ejecuta en el laptop, ni tests.
- **No hacer `git pull` en el cluster con un job de cola corriendo:** cada línea de la cola arranca
  con el código que haya en ese momento.
- **Al pegar en la terminal del cluster:** copiar solo los bloques de código, sin texto alrededor ni
  el prompt `t7-vLLM@deepmind:…$`.
- **GPU compartidas:** `scripts/jobs.sh gpu` antes de lanzar; si hay un proceso ajeno pequeño, `FORCE=1`.
  `launch.sh` toma la GPU de `GPU=…`; un comando suelto, de `CUDA_VISIBLE_DEVICES=…`.
- **Entorno del cluster:** `.env` define `HF_HOME` y `UV_CACHE_DIR` (`/mnt/home2/t7-vLLM/.cache/…`).
  Si falta un paquete, `uv sync --extra quant --extra eval --extra track --extra dev` (un `uv sync`
  sin extras los desinstala).
- **Lo descartado no se borra:** se mueve a `_para_borrar/` con su motivo; el usuario decide.

## 7. Archivos clave

| Archivo | Para qué |
|---|---|
| `docs/resultados_etapa1.md` | todo lo medido en la etapa 1, con procedencia |
| `docs/resultados_etapa2.md` | lo medido en la etapa 2 (humo; grilla y ajuste cuando existan) |
| `docs/metodologia.md` §2.4, §4, §6 | Y-Decoder, tabla de adaptaciones, decisiones tomadas y abiertas |
| `docs/plan_reconstruccion.md` §5–§6 | puntos abiertos P1–P11 y estado por paso |
| `configs/ydecoder.yaml` | decodificador: `text_source`, bloque `msrvtt`, *k* = 8, épocas |
| `docs/pasos_cluster.md` | guía ejecutable: A1–A8 (cierre de la etapa 1), B0–B8 (etapa 2), C (test) |
| `configs/queues/ydecoder_msrvtt.txt` | barrido de *k* (cerrado), ganador k = 8 NF4/bf16 y, comentados, intermedio y etapa 2 |
| `configs/queues/stage1_vljepa_val.txt` | VL-JEPA en val, ya apunta al genérico k = 8, 3 épocas |
| `configs/queues/stage1_infoloss.txt` | pérdida de información de la etapa 1 (genérico k = 8) |
| `configs/queues/diag_quant_vljepa.txt` | diagnóstico de cuantización de VL-JEPA por componente |
| `configs/train/budget.yaml` | etapa 2: presupuesto común y propuesta de hiperparámetros (candado `hparams_confirmed`) |
| `configs/train/models/*.yaml` | por modelo: micro-lote, conector; aquí va el ganador de la grilla |
| `configs/train/grid.yaml`, `scripts/make_grid.py` | grilla LR × r × conector → `configs/queues/stage2_grid{,_g0,_g1}.txt` |
| `configs/queues/stage2_{smoke,train,val,infoloss}.txt` | colas de la etapa 2 (val e infoloss con marcadores `<…>` a llenar) |
| `scripts/summarize_train.py` | resumen de humo, grilla (`--winners`) y ajuste completo |
| `notebooks/02_resultados_etapa1.ipynb`, `notebooks/03_resultados_etapa2.ipynb` | análisis de la etapa 1 y de la etapa 2 (solo leen `runs/`) |
| `configs/final_test.yaml` | plan de test (17 condiciones); se llena con lo elegido en val |
