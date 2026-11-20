# Estado de la sesión — 16-11-2026 (actualizado al cierre)

> Documento de traspaso: dónde quedó el trabajo, qué está corriendo y qué sigue. Para retomar, leer
> este archivo primero. Los resultados medidos están en `docs/resultados_etapa1.md`; el protocolo, en
> `docs/metodologia.md`.

## 1. Dónde estamos

**Fase 4 (etapa 1), casi terminada; la etapa 2 ya está escrita (sin ejecutar).**

- Barrido del Y-Decoder genérico **terminado** (`ydec_msrvtt`, `DONE(0)`): **k = 8** (CIDEr en MSR-VTT
  `test_1k`: k=1 1.321, k=4 1.384, k=8 1.439; adelante en las 7 evaluaciones). Techo en M3DI ≈ 0.12.
- Siguiente en el cluster: k = 8 hasta 3 épocas, NF4 (GPU 0) y bf16 (GPU 1): `configs/queues/ydecoder_msrvtt.txt` §2.
- Etapa 2 escrita con la propuesta de hiperparámetros (`configs/train/budget.yaml`), candado
  `train.hparams_confirmed: false` y colas `stage2_{smoke,sweep,train,infoloss}.txt`. Detalle en
  `docs/plan_reconstruccion.md` §6.0b. Nada de la etapa 2 se ha ejecutado, ni sus tests.
- Cambios sin commitear: todo lo anterior (el usuario hace commit y push).

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
- Pendiente del usuario: confirmar la propuesta (filas "Propuesta" de `metodologia.md` §4), P8 tras el humo,
  épocas del Y-Decoder ajustado a M3DI (intermedio y etapa 2), y autorizar la actualización de CLAUDE.md (P4).
- **Verificado en el cluster (19-11-2026): el tokenizador de Llama-3.2 se carga con `padding_side = "left"`**
  (EmbeddingGemma, `right`). La inferencia rellenaba la consulta hasta 512 por la izquierda y corría las
  posiciones RoPE. Corregido: `models/vljepa.py` fija `right` en ambos al cargar. Única medición afectada: la
  similitud coseno de la inspección (`resultados_etapa1.md` §7.4, marcada inválida) → repetir
  `inspect_vljepa_{nf4,bf16}`. El Y-Decoder y su techo no pasan por el Predictor: no se repiten.

## 4. Pasos que siguen

| # | Paso | Dónde | Depende de |
|---|---|---|---|
| 1 | Commit + push; `git pull` en el cluster | usuario | — |
| 2 | ~~Verificar el lado de relleno~~ hecho (19-11): era `left`, corregido. Repetir la inspección de VL-JEPA (NF4 y bf16) | cluster | 1 |
| 3 | Genérico k = 8 hasta 3 épocas, NF4 y bf16 (`ydecoder_msrvtt.txt` §2) | cluster | 1 |
| 4 | `pytest tests/test_train_units.py tests/test_train_e2e.py` y luego `stage2_smoke.txt` (puede ir en paralelo con 3 si hay GPU) | cluster | 1 |
| 5 | VL-JEPA en val con el genérico (piloto 200 × 6 prompts → val completo → prompt) | cluster | 2, 3 |
| 6 | Decidir épocas del decodificador intermedio y entrenarlo; VL-JEPA con él; control bf16 | usuario → cluster | 3 |
| 7 | Pérdida de información de la etapa 1 (`stage1_infoloss`, humo primero) | cluster | 3 |
| 8 | Revisar humo de la etapa 2 → P8 → `stage2_sweep.txt` → confirmar hiperparámetros | Claude + usuario | 4 |
| 9 | Actualizar `resultados_etapa1.md` y **aprobación del cierre de la etapa 1** | laptop | 5–7 |
| 10 | `stage2_train.txt` → Y-Decoder de la etapa 2 (`ydecoder_msrvtt.txt` §4) → `stage2_infoloss.txt` | cluster | 8, 9 |

## 5. Comandos para retomar

**Lado de relleno de los tokenizadores de VL-JEPA (cluster, un segundo):**
```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM
uv run --no-sync python -c "from transformers import AutoTokenizer as A; [print(n, type(t).__name__, t.padding_side) for n in ('unsloth/Llama-3.2-1B','google/embeddinggemma-300m') for t in [A.from_pretrained(n)]]"
```

**Y-Decoder genérico k = 8, 3 épocas (cluster):**
```bash
grep '^model' configs/queues/ydecoder_msrvtt.txt | head -1 > /tmp/ydec_k8_nf4.txt
grep '^model' configs/queues/ydecoder_msrvtt.txt | tail -1 > /tmp/ydec_k8_bf16.txt
SCRIPT=scripts/train_ydecoder.py GPU=0 scripts/launch.sh ydec_k8_nf4 scripts/run_queue.sh /tmp/ydec_k8_nf4.txt
SCRIPT=scripts/train_ydecoder.py GPU=1 scripts/launch.sh ydec_k8_bf16 scripts/run_queue.sh /tmp/ydec_k8_bf16.txt
```

**Tests y humo de la etapa 2 (cluster; requiere una GPU libre):**
```bash
uv run --no-sync pytest tests/test_train_units.py tests/test_train_e2e.py -q
SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_smoke scripts/run_queue.sh configs/queues/stage2_smoke.txt
```

**Ver cómo va el barrido del Y-Decoder (cerrado; referencia):**
```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM
scripts/jobs.sh status | grep ydec_msrvtt              # RUNNING / DONE(0)
cat logs/ydec_msrvtt/queue_status.tsv                  # una línea por k terminado (faltan si < 3)
grep -E "\[ydec\]\[val\]|\[ydec\] listo|fuente=" logs/ydec_msrvtt/job.log | cut -c1-170
```

**Cuando diga `DONE(0)`, juntar los resultados (cluster):**
```bash
mkdir -p /tmp/para_claude && rm -f /tmp/para_claude/*
for d in runs/ydec_vljepa*_msrvtt_k*_e1; do n=$(basename $d); cp $d/summary.json /tmp/para_claude/${n}_summary.json; cp $d/val_log.jsonl /tmp/para_claude/${n}_val_log.jsonl; head -n 30 $d/val_ceiling.jsonl > /tmp/para_claude/${n}_val_ceiling.jsonl; head -n 30 $d/m3di_ceiling.jsonl > /tmp/para_claude/${n}_m3di_ceiling.jsonl; done
cp logs/ydec_msrvtt/queue_status.tsv /tmp/para_claude/
ls -la /tmp/para_claude                                 # 13 archivos
```

**Copiarlos al laptop (terminal UCRT64):**
```bash
scp -r t7-vLLM@172.17.28.130:/tmp/para_claude /c/Users/bruno/AppData/Local/Temp/claude/bef8a9dbb391d17f/1b849f48-11bd-40d2-9502-128a7d217a5f/scratchpad/
```
Esa carpeta es temporal y de esta sesión: si al retomar la sesión es otra, pedir a Claude la ruta de su
carpeta de trabajo y usarla en el `scp`. `/tmp` del cluster también puede limpiarse; si falta algo,
repetir el bloque anterior.

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
| `docs/metodologia.md` §2.4, §4, §6 | Y-Decoder, tabla de adaptaciones, decisiones tomadas y abiertas |
| `docs/plan_reconstruccion.md` §5–§6 | puntos abiertos P1–P11 y estado por paso |
| `configs/ydecoder.yaml` | decodificador: `text_source`, bloque `msrvtt`, *k* = 8, épocas |
| `configs/queues/ydecoder_msrvtt.txt` | barrido de *k* (en curso) y, comentados, ganador, bf16 e intermedio |
| `configs/queues/stage1_vljepa_val.txt` | VL-JEPA en val, ya apunta al genérico k = 8, 3 épocas |
| `configs/queues/stage1_infoloss.txt` | pérdida de información de la etapa 1 (genérico k = 8) |
| `configs/train/budget.yaml`, `configs/queues/stage2_*.txt` | etapa 2: propuesta de hiperparámetros y colas |
| `configs/final_test.yaml` | plan de test (17 condiciones); se llena con lo elegido en val |
