# Pasos en el cluster: cierre de la etapa 1 y etapa 2 completa

> Guía de ejecución, en orden. Cada paso dice **qué lanzar**, **cómo saber que salió bien** y **qué mandarle a
> Claude**. Los puntos marcados **DECISIÓN** los resuelve el usuario antes de seguir: ningún hiperparámetro se fija
> sin fuente (CLAUDE.md §5). Estado al 19-11-2026: commit `89c0558` + este documento; `ydec_k8_nf4` corriendo en
> la GPU 1. El estado vivo está en `docs/estado_sesion.md`; el protocolo, en `docs/metodologia.md`.

## 0. Reglas que valen en todos los pasos

- **Copiar solo los bloques de código** a la terminal, sin el prompt `t7-vLLM@deepmind:…$` ni el texto alrededor.
- **Siempre partir en la carpeta del repo:**
  ```bash
  cd /mnt/home2/t7-vLLM/Multimodal_LLM
  ```
- **`git pull` solo sin colas corriendo.** Cada línea de una cola arranca con el código que haya en ese momento.
  Antes de cada `pull`: `scripts/jobs.sh status` y que nada diga `RUNNING`.
- **GPU compartidas.** Antes de lanzar: `scripts/jobs.sh gpu`. `launch.sh` usa la GPU de `GPU=…` y se niega si
  está ocupada (> 1 GB); `FORCE=1` lo fuerza (solo si el proceso ajeno es chico). Un comando suelto usa
  `CUDA_VISIBLE_DEVICES=…`.
- **Seguir un job:** `scripts/jobs.sh status`, `scripts/jobs.sh logs <job> -f` (Ctrl-C deja de seguir, no mata el
  job), `scripts/jobs.sh runs` (progreso por run). Detener limpio: `scripts/jobs.sh stop <job>`.
- **Relanzar es seguro:** `infer.py`, `train.py` y `train_ydecoder.py` reanudan desde lo guardado y saltan lo
  completo. Si una cola falla a medias, se corrige y se relanza la misma.
- **Si falta un paquete:** `uv sync --extra quant --extra eval --extra track --extra dev` (un `uv sync` sin
  extras los desinstala).
- **Disco (~80 %):** solo se guardan adaptadores LoRA y `*_best`; nunca modelos completos. Revisar con
  `du -sh runs/ ; df -h /mnt/home2`.
- **Test no se toca** hasta la Parte C. Solo `scripts/final_test.py` lo lee, una vez.

---

## Parte A — Cerrar la etapa 1

### A1. Esperar el Y-Decoder genérico NF4 (k = 8, 3 épocas)

Lanzado el 19-11 en la GPU 1 (`ydec_k8_nf4`, ≈ 1 h 45 min).

```bash
scripts/jobs.sh status | grep ydec_k8
grep -E "\[ydec\]\[val\]|\[ydec\] listo" logs/ydec_k8_nf4/job.log | cut -c1-170 | tail -25
```

**Bien si:** `DONE(0)` y existe `runs/ydec_vljepa_open_msrvtt_nf4_zs_nf4_msrvtt_k8_r64_e3/decoder_best`.

```bash
ls runs/ydec_vljepa_open_msrvtt_nf4_zs_nf4_msrvtt_k8_r64_e3/
python3 -m json.tool runs/ydec_vljepa_open_msrvtt_nf4_zs_nf4_msrvtt_k8_r64_e3/summary.json | head -40
```

**Mandar a Claude:** las líneas `[ydec][val]` y el `summary.json`.

### A2. Actualizar el código y correr el diagnóstico de cuantización

Con **nada corriendo**:

```bash
scripts/jobs.sh status
git pull
git log --oneline -1
```

Tests que no necesitan GPU (métricas, validaciones, resumen de entrenamiento):

```bash
uv run --no-sync pytest tests/test_embedding_metrics.py tests/test_vljepa_quant_components.py tests/test_vljepa_decoder_check.py tests/test_summarize_train.py -q
```

Humo del diagnóstico (64 imágenes, ~1 min por línea):

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/diag_quant_vljepa.py model=vljepa model.quant=bf16 diag.n=64 diag.batch_size=32 diag.tag=humo
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/diag_quant_vljepa.py model=vljepa model.quant_components=[predictor] diag.n=64 diag.batch_size=32 diag.tag=humo
```

**Bien si:** la segunda imprime `NF4 aplicado a capas lineales: {'predictor': …}` y un bloque `vs_bf16` con números.
Entonces, la cola completa (5 condiciones × 1 024 imágenes):

```bash
SCRIPT=scripts/diag_quant_vljepa.py GPU=1 scripts/launch.sh diag_quant scripts/run_queue.sh configs/queues/diag_quant_vljepa.txt
```

Al terminar (`DONE(0)`):

```bash
for d in runs/diagquant_vljepa_open_msrvtt_*; do case $d in *humo*) continue;; esac; echo "== $d"; python3 -c "import json,sys; m=json.load(open('$d/metrics.json')); a=m['alignment']['cos_pred_target']; d=m['discrimination']; print('cos', round(a['mean'],4), '| R@1', round(d['recall@1_p2t'],3), round(d['recall@1_t2p'],3), '| InfoNCE', round(d['infonce'],3)); v=m.get('vs_bf16'); print('vs_bf16', {k: round(x['mean'],4) for k,x in v.items() if isinstance(x,dict)} if isinstance(v,dict) else v)"; done
```

**Mandar a Claude:** esa salida. **DECISIÓN** según el resultado: si un componente explica casi toda la caída, se
revisa su cuantización antes de seguir con VL-JEPA en NF4; si la caída se reparte y la recuperación se mantiene,
se sigue.

### A3. Y-Decoder genérico bf16 (control P10)

Necesita una GPU libre (≈ 1 h 45 min). Si la GPU 0 sigue ocupada por el proceso ajeno, va en la GPU 1 después de A2.

```bash
scripts/jobs.sh gpu
grep '^model' configs/queues/ydecoder_msrvtt.txt | sed -n 2p > /tmp/ydec_k8_bf16.txt
cat /tmp/ydec_k8_bf16.txt
SCRIPT=scripts/train_ydecoder.py GPU=1 scripts/launch.sh ydec_k8_bf16 scripts/run_queue.sh /tmp/ydec_k8_bf16.txt
```

El `cat` debe mostrar `model=vljepa model.quant=bf16 decoder.quant=bf16 decoder.prefix_len=8`.
**Bien si:** existe `runs/ydec_vljepa_open_msrvtt_bf16_zs_bf16_msrvtt_k8_r64_e3/decoder_best`.

### A4. VL-JEPA zero-shot en val (piloto → val completo → prompt)

Puede ir en paralelo con A3 si hay dos GPU libres. Piloto: 200 imágenes × 6 prompts.

```bash
GPU=1 scripts/launch.sh s1_vljepa_piloto scripts/run_queue.sh configs/queues/stage1_vljepa_val.txt
```

Al terminar:

```bash
uv run --no-sync python scripts/score.py runs/zeroshot_vljepa_open_msrvtt_nf4_*_val_n200 --table vljepa_piloto
uv run --no-sync python scripts/peek.py runs/zeroshot_vljepa_open_msrvtt_nf4_p1_attributes_val_n200 --n 12
```

**Mandar a Claude:** la tabla y el `peek` (revisión manual del extractor sobre los captions del decodificador).
Si está bien, val completo (las mismas 6 líneas sin `data.limit`):

```bash
grep '^model' configs/queues/stage1_vljepa_val.txt | sed 's/ data.limit=200//' > /tmp/vljepa_val.txt
cat /tmp/vljepa_val.txt
GPU=1 scripts/launch.sh s1_vljepa_val scripts/run_queue.sh /tmp/vljepa_val.txt
```

Al terminar, puntuar (con métricas de texto; necesita Java) y elegir el prompt por `attr_mean`:

```bash
uv run --no-sync python scripts/score.py runs/zeroshot_vljepa_open_msrvtt_nf4_*_val --table vljepa_val --baselines
```

**Mandar a Claude:** la tabla. Claude anota el prompt elegido en `configs/final_test.yaml` (`vljepa_zs_*`).

### A5. Control bf16 de VL-JEPA

Requiere A3 y el prompt de A4. Reemplazar `<PROMPT>` por el elegido:

```bash
PROMPT=<PROMPT>
GPU=1 scripts/launch.sh s1_vljepa_bf16 uv run --no-sync python scripts/infer.py model=vljepa prompt=$PROMPT model.quant=bf16 infer.save_embeddings=true model.decoder_path=runs/ydec_vljepa_open_msrvtt_bf16_zs_bf16_msrvtt_k8_r64_e3/decoder_best
```

```bash
uv run --no-sync python scripts/score.py runs/zeroshot_vljepa_open_msrvtt_*_${PROMPT}_val --table vljepa_bf16
```

### A6. Condición intermedia (decodificador genérico ajustado a M3DI)

**DECISIÓN (pendiente):** épocas del decodificador ajustado a M3DI (`decoder.train.epochs`; hoy 10 en
`configs/ydecoder.yaml`, sin fuente). Las mismas épocas se usan después en la etapa 2 (B6). Con la decisión,
Claude descomenta la línea de `configs/queues/ydecoder_msrvtt.txt` §3. Luego:

```bash
grep '^model=vljepa decoder.text_source=m3di' configs/queues/ydecoder_msrvtt.txt | grep -v adapter_path > /tmp/ydec_inter.txt
cat /tmp/ydec_inter.txt
SCRIPT=scripts/train_ydecoder.py GPU=1 scripts/launch.sh ydec_inter scripts/run_queue.sh /tmp/ydec_inter.txt
```

Y VL-JEPA con ese decodificador, mismo prompt que A4, con `exp_id` propio para no pisar el zero-shot
(`<E>` = épocas elegidas):

```bash
PROMPT=<PROMPT>; E=<E>
GPU=1 scripts/launch.sh s1_vljepa_inter uv run --no-sync python scripts/infer.py model=vljepa prompt=$PROMPT infer.save_embeddings=true model.decoder_path=runs/ydec_vljepa_open_msrvtt_nf4_zs_nf4_k8_r64_e${E}_cont/decoder_best exp_id=zeroshot_vljepa_open_msrvtt_nf4_${PROMPT}_val_decm3di
uv run --no-sync python scripts/score.py runs/zeroshot_vljepa_open_msrvtt_nf4_${PROMPT}_val_decm3di --table vljepa_inter
```

### A7. Pérdida de información de la etapa 1

Humo primero (2–5 min), luego la cola (10 000 de train para ajustar las sondas, 2 000 de val para medir):

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/info_loss.py model=vljepa prompt=p1_attributes info.fit_limit=256 info.eval_limit=128 info.tag=humo
SCRIPT=scripts/info_loss.py GPU=1 scripts/launch.sh infoloss scripts/run_queue.sh configs/queues/stage1_infoloss.txt
```

**Bien si:** existe `runs/infoloss_vljepa_open_msrvtt_nf4_p1_attributes/tabla_discretos.csv`.
**Mandar a Claude:** ese CSV.

### A8. Análisis y cierre de la etapa 1

Re-puntuar todo con la versión actual del extractor y abrir el notebook `notebooks/02_resultados_etapa1.ipynb`
(VS Code Remote-SSH, kernel `vlmfid`; ver `notebooks/README.md`):

```bash
uv run --no-sync python scripts/score.py runs/zeroshot_*_val --table etapa1_val --baselines
```

Ejecutar el notebook completo. Las secciones de VL-JEPA se llenan solas con lo que ya exista. **Mandar a Claude:**
las tablas de §3, §7 y §12 del notebook y `runs/_tablas/etapa1_val.csv`. Claude actualiza
`docs/resultados_etapa1.md` y se pide la **aprobación del cierre de la etapa 1** (CLAUDE.md §5).

---

## Parte B — Etapa 2 (fine-tuning)

### B0. Decisiones pendientes del usuario (bloquean B5 en adelante)

| # | Qué | Dónde | Cuándo |
|---|---|---|---|
| P8 | Lote efectivo común: 128 (propuesta, [5] §4.5) o 64 si VL-JEPA no cabe | `configs/train/budget.yaml` | tras B2 (memoria medida) |
| P5 | Confirmar la propuesta: optimizador, esquema, warmup, `max_grad_norm`, LoRA α/dropout, weight decay (filas "Propuesta" de `docs/metodologia.md` §4) | `configs/train/budget.yaml` → `train.hparams_confirmed: true` | tras B4 |
| Grilla | Combinación ganadora por modelo (LR, r, conector; B4) | `configs/train/models/<modelo>.yaml` (`train.lr`, `lora.r`, `train.connector_lr_mult`) | tras B4 |
| Épocas | Épocas del Y-Decoder ajustado a M3DI (mismo valor que A6) | `configs/ydecoder.yaml` | antes de A6 |
| P4 | Autorizar la actualización de CLAUDE.md §3 (prompt de la etapa 2, controles bf16, Y-Decoder, P7) | `CLAUDE.md` | cuando quieras |

Mientras `train.hparams_confirmed` sea `false`, `scripts/train.py` solo acepta corridas cortas (`train.max_steps`):
el humo (B2) y la grilla (B4) se pueden correr ya; el ajuste completo (B5), no.

### B1. Tests del entrenamiento

Usan modelos diminutos (`tests/_tiny_vlm.py`); con GPU libre tardan pocos minutos.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync pytest tests/test_train_units.py tests/test_train_e2e.py -q
```

**Mandar a Claude:** la salida completa si algo falla (no seguir a B2 con fallos).

### B2. Prueba de humo (6 pasos × lote 128, los 4 modelos, NF4 y bf16)

Repartida en las dos GPU si están libres (NF4 en una, bf16 en otra); si no, toda la cola en una:

```bash
grep '^model' configs/queues/stage2_smoke.txt | grep -v 'quant=bf16' > /tmp/s2_smoke_nf4.txt
grep '^model' configs/queues/stage2_smoke.txt | grep 'quant=bf16' > /tmp/s2_smoke_bf16.txt
wc -l /tmp/s2_smoke_nf4.txt /tmp/s2_smoke_bf16.txt
SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_smoke_nf4 scripts/run_queue.sh /tmp/s2_smoke_nf4.txt
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2_smoke_bf16 scripts/run_queue.sh /tmp/s2_smoke_bf16.txt
```

(Una sola GPU: `SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2_smoke scripts/run_queue.sh configs/queues/stage2_smoke.txt`.)

Al terminar:

```bash
cat logs/s2_smoke_nf4/queue_status.tsv logs/s2_smoke_bf16/queue_status.tsv
uv run --no-sync python scripts/summarize_train.py runs/train_*_m6 --table s2_smoke
grep -hE "\[train\] modelo listo|\[train\] grupos de LR|query_tokens" logs/s2_smoke_*/job.log | cut -c1-400
```

**Comprobar** (los valores esperados están en la cabecera de `configs/queues/stage2_smoke.txt`):

| Qué | Esperado |
|---|---|
| `lora_layers_llm` | 168 (LLaVA-OV, InternVL3), 186 (Qwen3.5) |
| VL-JEPA `lora_layers_predictor` / `lora_layers_y_encoder` | 56 / 168 |
| Grupos de LR de VL-JEPA | `lora_y_encoder` = 0.05 × `lora` |
| `peak_mem_gb` de VL-JEPA (NF4 y bf16) | < 24 GB con micro-lote 128 → P8 = 128 |
| `loss_last` < `loss_first` | en los 8 runs |
| `val_step0` (adaptador en cero) | ≈ zero-shot de `p1_attributes` en las mismas 500 de val |

**Mandar a Claude:** la tabla de `summarize_train` y el `grep`. **DECISIÓN P8** con el pico de memoria de VL-JEPA.

### B3. Ajustar el lote si P8 = 64

Solo si VL-JEPA no cabe con 128: Claude cambia `train.effective_batch_size` a 64 en `configs/train/budget.yaml`
(para los cuatro, por comparabilidad) y regenera la grilla (`scripts/make_grid.py`). Push → `git pull` → repetir solo la línea de humo de
VL-JEPA para confirmar.

### B4. Grilla de hiperparámetros (LR × r × conector, 150 pasos × lote, en val)

Grilla completa, decidida el 20-11-2026 (`configs/train/grid.yaml`; `docs/metodologia.md` §6.1, G1):
LR {1e-4, 2e-4, 5e-4} × LoRA r {16, 64} (α = 16 fijo) × multiplicador del LR del conector {1, 5} = 12 combinaciones
por modelo, **48 corridas**, solo NF4. Ganador **por modelo**. Primero, verificar que la cola versionada coincide con
la grilla (falla si alguien editó una de las dos):

```bash
uv run --no-sync python scripts/make_grid.py --check
```

Repartida por modelo entre las dos GPU (`stage2_grid_g0.txt`: VL-JEPA e InternVL3; `stage2_grid_g1.txt`: Qwen3.5 y
LLaVA-OV; 24 corridas cada una):

```bash
scripts/jobs.sh gpu
SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_grid_g0 scripts/run_queue.sh configs/queues/stage2_grid_g0.txt
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2_grid_g1 scripts/run_queue.sh configs/queues/stage2_grid_g1.txt
```

(Una sola GPU: `SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2_grid scripts/run_queue.sh configs/queues/stage2_grid.txt`.)
Duración: con los `img/s` del humo (B2), cada corrida procesa 150 × 128 = 19 200 imágenes más 4 evaluaciones en val.

Seguimiento y resultado:

```bash
cat logs/s2_grid_g0/queue_status.tsv logs/s2_grid_g1/queue_status.tsv | awk '{print $2}' | sort | uniq -c
uv run --no-sync python scripts/summarize_train.py runs/train_*_m150 --table s2_grid --winners
```

La primera línea cuenta los códigos de salida (todos deben ser `0`). **Regla** (`configs/train/grid.yaml`): por
modelo, gana la combinación con mejor métrica en el **último** paso (mayor `attr_mean` en los VLM, menor InfoNCE en
VL-JEPA), no el mejor paso; un run que no llegó al paso 150 no compite. **Mandar a Claude:** la salida y
`runs/_tablas/s2_grid.csv`. Claude escribe `train.lr`, `lora.r` y `train.connector_lr_mult` de cada modelo en
`configs/train/models/<modelo>.yaml`, y con tu **confirmación de P5** pone `train.hparams_confirmed: true`.
Push → `git pull` en el cluster.

### B5. Ajuste completo (1 época, NF4 y control bf16)

977 pasos con lote 128 por run; la duración por modelo se estima con los `img/s` del humo (columna
`samples_per_s` de `train_log.jsonl`). NF4 en la GPU 0 y bf16 en la GPU 1:

```bash
git log --oneline -1
grep -n "hparams_confirmed" configs/train/budget.yaml
grep '^model' configs/queues/stage2_train.txt | grep -v 'quant=bf16' > /tmp/s2_train_nf4.txt
grep '^model' configs/queues/stage2_train.txt | grep 'quant=bf16' > /tmp/s2_train_bf16.txt
SCRIPT=scripts/train.py GPU=0 scripts/launch.sh s2_train_nf4 scripts/run_queue.sh /tmp/s2_train_nf4.txt
SCRIPT=scripts/train.py GPU=1 scripts/launch.sh s2_train_bf16 scripts/run_queue.sh /tmp/s2_train_bf16.txt
```

El `grep` debe mostrar `hparams_confirmed: true`. Seguimiento:

```bash
scripts/jobs.sh status | grep s2_train
grep -E "\[train\] [0-9]+/|\[val\]|\*mejor\*" logs/s2_train_nf4/job.log | tail -20
```

Al terminar:

```bash
uv run --no-sync python scripts/summarize_train.py runs/train_*_e1 --table s2_train
ls -d runs/train_*_e1/adapter_best
```

**Mandar a Claude:** la tabla. Claude anota cada `adapter_best` en `configs/final_test.yaml` (`*_ft_*`) y llena los
marcadores de `configs/queues/stage2_val.txt` y de las colas de VL-JEPA.

### B6. Y-Decoder de la etapa 2 (sobre el Y-Encoder ajustado)

Requiere el adaptador de VL-JEPA (B5) y las épocas decididas (A6). Claude descomenta `configs/queues/ydecoder_msrvtt.txt`
§4 con el `exp_id` real; luego:

```bash
grep '^model=vljepa.*adapter_path' configs/queues/ydecoder_msrvtt.txt | grep -v 'quant=bf16' > /tmp/ydec_ft_nf4.txt
grep '^model=vljepa.*adapter_path' configs/queues/ydecoder_msrvtt.txt | grep 'quant=bf16' > /tmp/ydec_ft_bf16.txt
cat /tmp/ydec_ft_nf4.txt /tmp/ydec_ft_bf16.txt
SCRIPT=scripts/train_ydecoder.py GPU=0 scripts/launch.sh ydec_ft_nf4 scripts/run_queue.sh /tmp/ydec_ft_nf4.txt
SCRIPT=scripts/train_ydecoder.py GPU=1 scripts/launch.sh ydec_ft_bf16 scripts/run_queue.sh /tmp/ydec_ft_bf16.txt
```

### B7. Modelos ajustados en val completo (P1)

El ajuste eligió el paso con 500 imágenes de val; aquí se evalúa `adapter_best` en las 10 000, con el mismo
`infer.py` y `score.py` que la etapa 1. Comprobar que no quedan marcadores y lanzar:

```bash
grep -v '^#' configs/queues/stage2_val.txt | grep '<' || echo "sin marcadores"
GPU=0 scripts/launch.sh s2_val scripts/run_queue.sh configs/queues/stage2_val.txt
```

Al terminar:

```bash
uv run --no-sync python scripts/score.py runs/finetuned_*_val --table etapa2_val
uv run --no-sync python scripts/peek.py runs/finetuned_*_nf4_p1_attributes_val --n 8
```

**Mandar a Claude:** la tabla y el `peek`.

### B8. Pérdida de información de la etapa 2

Mismas sondas e imágenes que A7, sobre VL-JEPA ajustado con su decodificador de B6 (Claude llena los marcadores de
`configs/queues/stage2_infoloss.txt`):

```bash
grep -v '^#' configs/queues/stage2_infoloss.txt | grep '<' || echo "sin marcadores"
SCRIPT=scripts/info_loss.py GPU=1 scripts/launch.sh infoloss_ft scripts/run_queue.sh configs/queues/stage2_infoloss.txt
```

**Mandar a Claude:** `runs/infoloss_vljepa_open_msrvtt_nf4_p1_attributes_ft/tabla_discretos.csv`.

---

## Parte C — Evaluación final en test (una sola vez)

Solo cuando las etapas 1 y 2 estén aprobadas y `configs/final_test.yaml` no tenga ningún `null`.

```bash
uv run --no-sync python scripts/final_test.py --dry-run
```

**Bien si:** `plan válido: 17 runs. No se leyó test`. Revisar la lista con Claude **antes** de confirmar: al primer
lanzamiento el plan se congela (`runs/final_test/plan.lock.yaml`) y no puede cambiar.

```bash
GPU=0 scripts/launch.sh final_test uv run --no-sync python scripts/final_test.py --confirm
```

Resultado: `runs/final_test/tabla_final.csv`.

---

## Anexo — Traer resultados al laptop (solo archivos chicos)

En el cluster, juntar lo pedido en una carpeta temporal (ejemplo: tablas y resúmenes):

```bash
mkdir -p /tmp/para_claude && rm -f /tmp/para_claude/*
cp runs/_tablas/*.csv /tmp/para_claude/ 2>/dev/null
for d in runs/train_*_e1 runs/ydec_*_e3* runs/diagquant_*; do [ -f $d/summary.json ] && cp $d/summary.json /tmp/para_claude/$(basename $d)_summary.json; [ -f $d/metrics.json ] && cp $d/metrics.json /tmp/para_claude/$(basename $d)_metrics.json; done
ls -la /tmp/para_claude
```

En el laptop (terminal UCRT64), copiarla a la carpeta de trabajo de la sesión de Claude (pedirle la ruta; cambia
por sesión):

```bash
scp -r t7-vLLM@172.17.28.130:/tmp/para_claude <RUTA_QUE_DA_CLAUDE>
```
