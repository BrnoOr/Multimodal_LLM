# Plan de reconstrucción — estructura y pasos (fase 3)

> Propuesta. Nada de esto está implementado. Se apoya en `docs/metodologia.md` (**M**) y en
> `docs/auditoria.md` (**A**). Toda ejecución se deja preparada para el cluster; en el laptop solo
> se escribe código.

## 1. Principios

1. **Se conserva la infraestructura que la auditoría clasificó como reutilizable** (A §3): paquete
   `vlmfid`, composición de configs, manifiestos, registro por run, bucle reanudable, colas y
   scripts de lanzamiento. Reescribir eso no aporta y arriesga errores nuevos.
2. **Lo descartado se mueve, no se borra**, a `_para_borrar/` con un `README.md` que explica el
   motivo de cada archivo. Cada movimiento va en un commit propio. El usuario decide cuándo se
   elimina de verdad.
3. **Test queda bloqueado por defecto.** El split por defecto pasa a `val`; evaluar en test exige
   una bandera explícita y un único script final. Es la corrección estructural de A H1.
4. **Una sola implementación** de la extracción de atributos y de las métricas, usada por los 4
   modelos, el bucle de entrenamiento y los notebooks [CLAUDE.md §6].
5. **Ningún hiperparámetro se fija sin fuente.** Los valores PENDIENTE de M §4 se resuelven en el
   paso 2.0, antes de entrenar.

## 2. Estructura propuesta

Los archivos sin marca se conservan tal cual. `~` = se modifica; `+` = nuevo; `→` = se mueve a
`_para_borrar/`.

```
configs/
  defaults.yaml                 ~  data.split: val por defecto
  prompts.yaml                  ~  revisión de los prompts frente a la métrica de color
  models/
    llava_ov.yaml  internvl3.yaml  qwen35.yaml   ~  cuantización de todo lo congelado
    vljepa.yaml                 ~  sin sección bank
    ydecoder.yaml               +  Y-Decoder: prefijo sobre Llama-3.2-1B
    qwen35_instruct.yaml  vljepa_fac.yaml        →
  train/
    budget.yaml                 ~  caption original, selección nueva, sin test automático
    models/{llava_ov,internvl3,qwen35,vljepa}.yaml   ~  prompt p1; conectores
    models/ydecoder.yaml        +
    models/vljepa_fac.yaml      →
  queues/                       →  todas las actuales; se escriben colas nuevas (§4)
  experiments/                  →

src/vlmfid/
  paths.py  config.py  tracking.py
  data/
    m3di.py  manifest.py
    targets.py                  →
  prompts/
  models/
    base.py  registry.py  qwen_vl.py
    quant.py                    ~  NF4 para todo componente congelado; informe de fracción en NF4
    hf_generative.py            ~  verificación de cuantización invertida
    vljepa.py                   ~  sin banco; NF4 en X/Y-Encoder; expone S_V, estado del Predictor, Ŝ_Y, S_Y
    ydecoder.py                 +  proyección de prefijo + Llama-3.2-1B (NF4 + LoRA)
    bank.py                     →
  eval/
    attributes.py               ~  color por nombre exacto sin prefijo; agregado = media de 4 atributos
    text_metrics.py             ~  envoltorio fino de pycocoevalcap (Bleu-1, Bleu-4, ROUGE-L, CIDEr)
    probes.py                   +  MLP de 1 capa oculta y kernel ridge [1]; cota en bits
    infoloss.py                 ~  cadena E0–E4 y T0–T2 de M §2.5, sin banco
    probe.py  copying.py        →
  train/
    data.py
    collate.py  objectives.py  lora.py  loop.py  evaluate.py  config.py   ~
    jepa.py                     ~  solo InfoNCE bidireccional del lote; Y-Encoder con LoRA ×0.05
    ydecoder.py                 +  entrenamiento del Y-Decoder con pares ⟨S_Y, caption⟩

scripts/
  launch.sh  run_queue.sh  jobs.sh  check_env.py  build_manifest.py
  infer.py  train.py  smoke.py  ~
  inspect_models.py             +  vuelca named_modules, tokens visuales y fracción en NF4 (A §6)
  train_ydecoder.py             +
  info_loss.py                  ~
  final_test.py                 +  único punto de entrada que lee test
  probe.py  probe_p4.py  copy_report.py          →

tests/                          ~  se reescriben los de atributos; se agregan métricas, sondas y decodificador
notebooks/01, 06                ~  EDA de datos
notebooks/02–05                 →
reports/                        →  resultados anteriores (se conserva hallazgos.md del EDA)
docs/
  metodologia.md  auditoria.md  plan_reconstruccion.md  enunciado_T7.pdf  papers/
  GUIA_LAPTOP_CLUSTER.md        ~
  ETAPA2_FINETUNING.md  ETAPA2B_REFORMULACION.md  VLJEPA_AJUSTE_V3.pdf  ESTRUCTURA_REPO.md  →
_para_borrar/README.md          +  motivo de cada archivo movido
```

## 3. Dependencias

- `pycocoevalcap` ya está en el extra `eval` de `pyproject.toml`. Su tokenizador requiere Java:
  `scripts/check_env.py` debe comprobarlo. Si el cluster no tiene Java, se vuelve a consultar.
- Kernel ridge: `scikit-learn`, nueva dependencia sin extensiones CUDA (sin riesgo de wheels).
- No se agregan dependencias con extensiones CUDA. `bitsandbytes` ya está.

## 4. Pasos de implementación

Cada paso termina con una corrida pequeña en el cluster antes de la completa [CLAUDE.md §5].

### Paso 0 — Limpieza (sin cambios de comportamiento)
1. Mover lo marcado `→` a `_para_borrar/`, un commit por grupo (VL-JEPA banco; reformulación
   canónica y holdout; colas; notebooks y reportes; documentación antigua; diagnósticos fuera de
   alcance), con el motivo en `_para_borrar/README.md`.
2. Quitar del código conservado las referencias a lo movido.

### Paso 1.0 — Inspección de modelos (cluster)
`scripts/inspect_models.py` por modelo, sin entrenar: ID y `config.json`, `named_modules`, módulos
alcanzados por el regex de LoRA y por el conector, tokens visuales por imagen de M3DI, fracción de
parámetros en NF4, y comparación de salidas NF4 frente a bf16 en unas pocas imágenes. Responde A §6
y la lista de M §3. **Sin este paso no se da por buena ninguna configuración de modelo.**

### Paso 1.1 — Evaluación común
1. `eval/attributes.py`: color por nombre exacto sin prefijo; agregado. Tests con casos verificados
   a mano: las 5 plantillas, respuestas sin comillas, sinónimos de posición, omisiones.
2. `eval/text_metrics.py` sobre `pycocoevalcap`. Test con un ejemplo calculado a mano.
3. Líneas base sin modelo: azar, clase mayoritaria y techo del color dado el matiz (M §2.1).

### Paso 1.2 — VLMs en zero-shot
Cuantización de todo lo congelado; cola de **val** con los prompts del catálogo, en NF4, más el
control bf16 con el prompt seleccionado (M §2.2). Corrida pequeña (pocas imágenes por modelo y
prompt) con inspección manual de las respuestas y de la extracción.

### Paso 1.3 — VL-JEPA en zero-shot
1. `models/vljepa.py` sin banco y con NF4 en los tres componentes congelados.
2. `models/ydecoder.py` y `scripts/train_ydecoder.py`: decodificador de la etapa 1, entrenado con
   texto de train sobre el Y-Encoder preentrenado (M §2.4). Validación: techo `Y-Decoder(S_Y)` en
   val; si no reconstruye el caption desde su propio embedding, el decodificador está mal antes de
   mirar ninguna predicción.
3. Inferencia en val con los mismos prompts como consulta, y control bf16.

### Paso 1.4 — Pérdida de información, etapa 1
Extracción de E1–E3 y T1, sondas de M §2.5, tabla por atributo. Corrida pequeña primero.

### Paso 2.0 — Hiperparámetros
Resolver los PENDIENTE de M §4 (r, α, dropout, LR, esquema, lote efectivo, longitudes, y los del
Y-Decoder). Propuesta a acordar: tomar de los model cards lo que exista y hacer un barrido corto en
un subconjunto de train, evaluado en val, igual para los 4 modelos. **Requiere acuerdo del usuario
antes de lanzar nada.**

### Paso 2.1 — Ajuste de los VLM
Caption original, prompt `p1_attributes`, LoRA en todas las lineales del LLM más el conector,
selección en val. Antes de la corrida completa: conteo de entrenables por componente, formas de
tensores, alineación de etiquetas y una corrida de pocos pasos que confirme que la pérdida baja.
Después: la corrida de control sobre base bf16 (M §2.3, D6).

### Paso 2.2 — Ajuste de VL-JEPA
Opción A del paper con LoRA: Predictor e Y-Encoder (×0.05), proyecciones en precisión completa,
InfoNCE bidireccional del lote sin máscara, X-Encoder congelado. Después se entrena el Y-Decoder de
la etapa 2 sobre el Y-Encoder ajustado. Control bf16.

### Paso 2.3 — Pérdida de información, etapa 2
Mismas sondas y semillas que en 1.4, sobre el modelo ajustado.

### Paso 3 — Evaluación final
`scripts/final_test.py`, una sola vez: para cada modelo, zero-shot NF4, zero-shot bf16, ajustado NF4
y ajustado bf16, en test. Tablas y figuras del informe en notebooks nuevos que importan el paquete.

## 5. Puntos abiertos antes de empezar

| # | Punto | Estado |
|---|---|---|
| P1 | Tamaño del subconjunto de val para seleccionar checkpoint (A3). | Propuesta: 500 durante el entrenamiento, val completo para el checkpoint elegido. |
| P2 | Agregado para seleccionar (A4): media de forma, x, y y color. | Por defecto, revisable. |
| P3 | Alcance de la decisión "prompt 1": se interpretó como `p1_attributes` en los 4 modelos para la etapa 2, manteniendo varios prompts en la etapa 1. Si significaba usar solo `p1` también en la etapa 1, el plan se simplifica, pero se pierde la pregunta del enunciado sobre sensibilidad al prompt. | A confirmar. |
| P4 | Actualizar CLAUDE.md §3 con las decisiones que lo contradicen (prompt de la etapa 2; controles bf16; Y-Decoder). | Requiere autorización: es el archivo de instrucciones del proyecto. |
| P5 | Hiperparámetros de M §4 (paso 2.0). | Se acuerda al llegar a la etapa 2. |
| P6 | Java en el cluster para `pycocoevalcap`. | Se comprueba en el paso 1.0. |

## 6. Estado (actualizado al escribir `scripts/final_test.py`)

Código escrito y **sin ejecutar**; cada punto se valida en el cluster antes de darlo por bueno.

| Paso | Estado |
|---|---|
| 0 Limpieza | hecho (`_para_borrar/`), salvo lo que depende de reescribir el entrenamiento |
| 1.0 Inspección de modelos | escrito: `scripts/inspect_models.py`, cola `stage1_inspect` |
| 1.1 Evaluación común | escrito: `eval/attributes.py`, `eval/text_metrics.py`, `scripts/score.py` |
| 1.2 VLMs zero-shot | escrito: colas `stage1_pilot_val`, `stage1_val` |
| 1.3 VL-JEPA zero-shot | escrito: `models/ydecoder.py`, `train/ydecoder.py`, colas `ydecoder_sweep`, `stage1_vljepa_val` |
| 1.4 Pérdida de información | escrito: `eval/probes.py`, `scripts/info_loss.py`, cola `stage1_infoloss` |
| 3 Evaluación final | escrito: `scripts/final_test.py`, plan `configs/final_test.yaml` (pendiente de completar) |
| 2.0–2.3 Etapa 2 | pendiente: `src/vlmfid/train/` sigue con el protocolo anterior |
