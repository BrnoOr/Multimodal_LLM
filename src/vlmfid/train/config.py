"""Composición y validación de la configuración de entrenamiento (etapa 2).

    configuración = configs/train/budget.yaml                (común a los 4 modelos)
                  ⊕ configs/train/models/<model>.yaml        (micro-lote, conector, objetivo, selección, LR)
                  ⊕ model:  configs/models/<model>.yaml      (la MISMA config de inferencia de la etapa 1)
                  ⊕ prompt: configs/prompts.yaml[<prompt>]   (p1_attributes en los 4, budget.yaml)
                  ⊕ overrides por línea de comandos (dotlist)

La config del modelo es la de inferencia, sin copias: el modelo que se entrena se carga, cuantiza y
preprocesa igual que en la etapa 1 y que en la evaluación final.

El archivo por modelo solo puede tocar `ALLOWED_MODEL_KEYS`. Los tres ejes de la grilla de hiperparámetros
(`train.lr`, `lora.r`, `train.connector_lr_mult`) están permitidos porque se eligen POR MODELO con la misma
grilla completa en val (decisiones del usuario 16-11-2026 y 20-11-2026; configs/train/grid.yaml). El resto
del presupuesto (épocas, lote efectivo, esquema, α, dropout...) es común. Desde la línea de comandos se
puede cambiar todo, y el cambio queda en config.yaml del run (y en el exp_id lo que lo distingue).

Candado de hiperparámetros: mientras `train.hparams_confirmed` sea false (los valores de budget.yaml son
una propuesta, docs/metodologia.md §4), solo se aceptan corridas cortas con `train.max_steps`.

exp_id por defecto:
    train_<model.name>_<quant>_<prompt>_r<r>_lr<lr>[_c<connector_lr_mult>]_e<epochs>[_s<seed>][_m<max_steps>]
    (`_c` solo si el multiplicador del conector no es 1: los nombres anteriores no cambian)
"""

from __future__ import annotations

import json

from omegaconf import DictConfig, OmegaConf

from ..config import _load_model, _load_prompt, as_dict
from ..eval.attributes import METRICS
from ..paths import CONFIGS

TRAIN_CONFIGS = CONFIGS / "train"
# Ejes de la grilla (configs/train/grid.yaml): se eligen por modelo, así que el archivo del modelo puede fijarlos.
GRID_KEYS = ("train.lr", "lora.r", "train.connector_lr_mult")
ALLOWED_MODEL_KEYS = {"train.micro_batch_size", "train.selection_metric", "lora.connector_modules",
                      "objective.kind", *GRID_KEYS}
SELECTION_METRICS = tuple(METRICS) + ("val_loss",)
FAMILY_OBJECTIVE = {"hf": "lm", "qwen_vl": "lm", "vljepa": "infonce"}
OPTIMIZERS = ("adamw", "paged_adamw")
SCHEDULES = ("cosine", "constant")
# D4 (docs/metodologia.md §2.3): sin máscara de captions duplicados solo con lote físico ≤ 128
MAX_BATCH_WITHOUT_MASK = 128
# Claves que pueden cambiar al reanudar sin alterar el resultado. El micro-lote de los VLM solo reparte el
# lote efectivo en la GPU (la pérdida se normaliza por los tokens del lote efectivo); en VL-JEPA el
# micro-lote es el lote efectivo, que no está aquí. max_steps no: un run corto escribe summary.json.
RESUME_SAFE = {"data.num_workers", "train.micro_batch_size", "train.log_every", "train.save_every",
               "train.eval_every", "train.x_encoder_chunk", "train.hparams_confirmed"}
REQUIRED = ("train.epochs", "train.effective_batch_size", "train.lr", "train.connector_lr_mult",
            "train.weight_decay", "train.schedule", "train.warmup_ratio", "train.max_grad_norm",
            "train.optimizer", "lora.r", "lora.alpha", "lora.dropout")


def _flat_keys(d: dict, prefix: str = "") -> set[str]:
    out = set()
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key not in ALLOWED_MODEL_KEYS:
            out |= _flat_keys(v, key + ".")
        else:
            out.add(key)
    return out


def load_model_train(name: str) -> DictConfig:
    path = TRAIN_CONFIGS / "models" / f"{name}.yaml"
    if not path.exists():
        avail = sorted(p.stem for p in (TRAIN_CONFIGS / "models").glob("*.yaml"))
        raise FileNotFoundError(f"No existe {path}. Modelos con config de entrenamiento: {avail}")
    cfg = OmegaConf.load(path)
    extra = _flat_keys(as_dict(cfg)) - ALLOWED_MODEL_KEYS
    if extra:
        raise ValueError(f"{path}: claves no permitidas {sorted(extra)}. El presupuesto común vive en "
                         f"configs/train/budget.yaml; por modelo solo {sorted(ALLOWED_MODEL_KEYS)}.")
    return cfg


def _fmt_lr(lr: float) -> str:
    return f"{float(lr):g}"


def default_exp_id(cfg: DictConfig) -> str:
    t = cfg.train
    mult = float(t.connector_lr_mult)
    return (f"train_{cfg.model.name}_{cfg.model.quant}_{cfg.prompt.id}_r{cfg.lora.r}_lr{_fmt_lr(t.lr)}"
            + (f"_c{mult:g}" if mult != 1.0 else "") + f"_e{t.epochs}"
            + (f"_s{cfg.seed}" if int(cfg.seed) != 0 else "")
            + (f"_m{t.max_steps}" if t.get("max_steps") else ""))


def compose_train(args: list[str] | None = None) -> DictConfig:
    kv = {}
    for a in args or []:
        if "=" not in a:
            raise ValueError(f"Argumento inválido '{a}': se espera clave=valor")
        k, v = a.split("=", 1)
        kv[k] = v
    if "model" not in kv:
        raise ValueError("Falta model=<nombre> (llava_ov | internvl3 | qwen35 | vljepa)")
    model_name = kv.pop("model")

    cfg = OmegaConf.load(TRAIN_CONFIGS / "budget.yaml")
    cfg = OmegaConf.merge(cfg, load_model_train(model_name))
    prompt_name = kv.pop("prompt", cfg.prompt)
    cfg.model = _load_model(model_name)
    cfg.model_key = model_name
    cfg.prompt = _load_prompt(str(prompt_name))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in kv.items()]))
    validate(cfg)
    if cfg.get("exp_id") in (None, "", "null"):
        cfg.exp_id = default_exp_id(cfg)
    return cfg


def validate(cfg: DictConfig) -> None:
    """Falla ante cualquier configuración que rompa el protocolo de la etapa 2."""
    t, o = cfg.train, cfg.objective
    missing = [k for k in REQUIRED if OmegaConf.select(cfg, k) in (None, "", "null")]
    if missing:
        raise ValueError(f"Hiperparámetros sin valor: {missing}. Cada uno necesita fuente (docs/metodologia.md §4).")
    if int(cfg.lora.r) <= 0 or float(t.connector_lr_mult) <= 0 or float(t.lr) <= 0:
        raise ValueError(f"lora.r={cfg.lora.r}, train.lr={t.lr} y train.connector_lr_mult={t.connector_lr_mult} "
                         "deben ser positivos")
    if t.get("max_steps") in ("", "null"):
        t.max_steps = None
    if not bool(t.hparams_confirmed) and not t.get("max_steps"):
        raise ValueError(
            "Los hiperparámetros de configs/train/budget.yaml son una propuesta sin confirmar "
            "(docs/metodologia.md §4). Hasta fijarlos (train.hparams_confirmed=true) solo se aceptan corridas "
            "cortas con train.max_steps: prueba de humo y barrido (configs/queues/stage2_*.txt).")

    family = str(cfg.model.family)
    expected = FAMILY_OBJECTIVE.get(family)
    if expected is None:
        raise ValueError(f"Familia de modelo desconocida: {family}")
    if o.kind != expected:
        raise ValueError(f"objective.kind='{o.kind}' no corresponde a la familia {family} (se espera '{expected}')")
    if t.micro_batch_size is None:
        t.micro_batch_size = t.effective_batch_size
    if int(t.effective_batch_size) % int(t.micro_batch_size):
        raise ValueError(f"effective_batch_size={t.effective_batch_size} no es múltiplo de "
                         f"micro_batch_size={t.micro_batch_size}")
    if o.kind == "infonce":
        if int(t.micro_batch_size) != int(t.effective_batch_size):
            raise ValueError("VL-JEPA: InfoNCE toma los negativos del lote, así que el lote físico debe ser el "
                             "efectivo (micro_batch_size == effective_batch_size; docs/metodologia.md §2.3).")
        if int(t.effective_batch_size) > MAX_BATCH_WITHOUT_MASK and not bool(o.mask_duplicate_captions):
            raise ValueError(f"Lote de {t.effective_batch_size} > {MAX_BATCH_WITHOUT_MASK} sin máscara de captions "
                             "duplicados: D4 pide reconsiderarlo (docs/metodologia.md §2.3).")
        if t.selection_metric != "val_loss":
            raise ValueError("VL-JEPA se selecciona por la InfoNCE en val (P7): train.selection_metric=val_loss. "
                             "Sin Y-Decoder no hay texto que puntuar durante el ajuste.")
    elif t.selection_metric not in METRICS:
        raise ValueError(f"VLM: train.selection_metric debe ser una exactitud por atributo {METRICS} (CLAUDE.md §3)")
    if t.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"train.selection_metric='{t.selection_metric}' inválido {SELECTION_METRICS}")
    if t.optimizer not in OPTIMIZERS:
        raise ValueError(f"train.optimizer='{t.optimizer}' inválido {OPTIMIZERS}")
    if t.schedule not in SCHEDULES:
        raise ValueError(f"train.schedule='{t.schedule}' inválido {SCHEDULES}")
    if not cfg.lora.get("connector_modules"):
        raise ValueError("lora.connector_modules vacío: el conector se entrena (CLAUDE.md §3)")
    if cfg.model.get("adapter_path"):
        raise ValueError("model.adapter_path debe ser null al entrenar (el adaptador se crea desde cero)")
    if family == "vljepa" and cfg.model.get("decoder_path"):
        raise ValueError("model.decoder_path debe ser null al entrenar VL-JEPA: el Y-Decoder de la etapa 2 se "
                         "entrena después, sobre el Y-Encoder ajustado (docs/metodologia.md §2.3)")
    if cfg.data.get("train_limit") not in (None, "", "null"):
        raise ValueError("data.train_limit no existe: el entrenamiento usa train completo. Para una corrida "
                         "corta usa train.max_steps (agrega _m<N> al exp_id).")
    for split in (cfg.data.train_split, cfg.data.val_split):
        if str(split) == "test":
            raise ValueError("El entrenamiento no lee test: solo scripts/final_test.py (CLAUDE.md §5)")


def strip_resume_safe(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg))
    for key in RESUME_SAFE:
        node, parts = c, key.split(".")
        for p in parts[:-1]:
            node = node.get(p, {})
        node.pop(parts[-1], None)
    return c
