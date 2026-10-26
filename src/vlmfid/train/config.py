"""Composición de la configuración de entrenamiento (etapa 2).

    configuración = configs/train/budget.yaml                (común a los 4 modelos)
                  ⊕ configs/train/models/<model>.yaml        (prompt, micro-lote, conector, objetivo)
                  ⊕ model:  configs/models/<model>.yaml      (la MISMA config de inferencia)
                  ⊕ prompt: configs/prompts.yaml[<prompt>]
                  ⊕ overrides por línea de comandos (dotlist)

La config del modelo es la de inferencia, sin copias: así el modelo que se entrena se carga, cuantiza
y preprocesa exactamente igual que en la etapa 1 y que en la evaluación final con `scripts/infer.py`.

El archivo por modelo solo puede tocar las claves de `ALLOWED_MODEL_KEYS`. Cualquier otra (lr,
épocas, rango LoRA...) rompería el "mismo presupuesto" y se rechaza. Desde la línea de comandos sí se
puede cambiar todo (pruebas de humo, ablaciones), pero el cambio queda en el exp_id o en config.yaml.

exp_id por defecto (reformulación, docs/ETAPA2B_REFORMULACION.md):
    train_<model>_<quant>_<prompt>_t<can|orig>_r<r>_e<epochs>[_hc][_s<seed>][_m<max_steps>]
    (_hc = split composicional; _s<seed> solo si seed != 0; _m<max_steps> solo en pruebas de humo)

Train se usa SIEMPRE completo (las 125 000 imágenes, o todas las no retiradas con el split
composicional): no hay `data.train_limit` y pasarlo es un error.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig, OmegaConf

from ..config import _load_model, _load_prompt, as_dict
from ..data.targets import HOLDOUTS, TARGET_FORMATS
from ..paths import CONFIGS

TRAIN_CONFIGS = CONFIGS / "train"
ALLOWED_MODEL_KEYS = {"prompt", "train.micro_batch_size", "lora.connector_modules", "objective.kind",
                      "objective.temperature", "objective.mask_duplicate_captions", "objective.negatives"}
SELECTION_METRICS = ("macro", "shape_pos", "macro_cat", "shape", "pos", "color", "color_adj", "name", "val_loss")
TARGET_TAG = {"canonical": "tcan", "original": "torig"}
# Claves que pueden cambiar al reanudar un entrenamiento sin alterar su resultado.
# (max_steps no: un run cortado con max_steps escribe summary.json y adapter_best, así que va en el exp_id)
RESUME_SAFE = {"data.num_workers", "train.micro_batch_size", "train.log_every", "train.save_every",
               "train.eval_every", "test"}
TEST_DEFAULT_LIMIT = 10000      # el subconjunto de test del zero-shot: sin sufijo _n en el exp_id


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
    per_model = load_model_train(model_name)
    prompt_name = kv.pop("prompt", per_model.get("prompt"))
    if not prompt_name:
        raise ValueError(f"configs/train/models/{model_name}.yaml no define `prompt`")
    cfg = OmegaConf.merge(cfg, {k: v for k, v in per_model.items() if k != "prompt"})
    cfg.model = _load_model(model_name)
    cfg.model_key = model_name
    cfg.prompt = _load_prompt(str(prompt_name))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in kv.items()]))

    t = cfg.train
    if t.micro_batch_size is None:
        t.micro_batch_size = t.effective_batch_size
    if t.effective_batch_size % t.micro_batch_size:
        raise ValueError(f"effective_batch_size={t.effective_batch_size} no es múltiplo de "
                         f"micro_batch_size={t.micro_batch_size}")
    if cfg.objective.kind not in ("lm", "infonce"):
        raise ValueError(f"objective.kind='{cfg.objective.kind}' inválido (lm | infonce)")
    negatives = cfg.objective.get("negatives", "batch")
    if negatives not in ("batch", "bank"):
        raise ValueError(f"objective.negatives='{negatives}' inválido (batch | bank)")
    if (cfg.objective.kind == "infonce" and negatives == "batch"
            and t.micro_batch_size != t.effective_batch_size):
        raise ValueError("objective=infonce con negatives=batch: los negativos salen del micro-lote, que debe ser "
                         "igual al lote efectivo (acumular gradientes cambia la pérdida). Ajusta "
                         "train.effective_batch_size o usa objective.negatives=bank.")
    if cfg.model.get("adapter_path"):
        raise ValueError("model.adapter_path debe ser null al entrenar (el adaptador se crea desde cero)")
    if cfg.target.format not in TARGET_FORMATS:
        raise ValueError(f"target.format='{cfg.target.format}' inválido ({' | '.join(TARGET_FORMATS)})")
    if cfg.data.get("train_limit") not in (None, "", "null"):
        raise ValueError("data.train_limit no existe: el entrenamiento usa train completo. Para una prueba "
                         "corta usa train.max_steps (agrega _m<N> al exp_id).")
    if cfg.data.get("holdout") in ("", "null"):
        cfg.data.holdout = None
    if cfg.data.get("holdout") not in HOLDOUTS:
        raise ValueError(f"data.holdout='{cfg.data.holdout}' inválido (null | compositional)")
    if t.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"train.selection_metric='{t.selection_metric}' inválido {SELECTION_METRICS}")
    if cfg.model.family == "vljepa":
        # el banco de recuperación (val durante el ajuste y test después) debe estar en el formato aprendido
        cfg.model.bank.target_format = cfg.target.format

    if cfg.get("exp_id") in (None, "", "null"):
        cfg.exp_id = (f"train_{cfg.model.name}_{cfg.model.quant}_{cfg.prompt.id}_{TARGET_TAG[cfg.target.format]}"
                      f"_r{cfg.lora.r}_e{t.epochs}"
                      + ("_hc" if cfg.data.holdout == "compositional" else "")
                      + (f"_s{cfg.seed}" if int(cfg.seed) != 0 else "")
                      + (f"_m{t.max_steps}" if t.get("max_steps") else ""))
    return cfg


def compose_test(cfg: DictConfig) -> DictConfig:
    """Config de `scripts/infer.py` para evaluar en test el adaptador elegido en val.

    Parte de configs/defaults.yaml (como el zero-shot) y toma de la config de entrenamiento el modelo
    tal como se entrenó (cuantización, banco de VL-JEPA en el formato aprendido), el prompt y la semilla.
    El exp_id coincide con el de configs/queues/stage2b_test.txt, así que ambos caminos comparten run.
    """
    from ..config import compose
    from ..paths import ROOT, RUNS

    t = cfg.test
    if t.split == cfg.data.val_split or t.split == cfg.data.train_split:
        raise ValueError(f"test.split='{t.split}' no puede ser el split de train o val")
    limit = None if t.get("limit") in (None, "", "null") else int(t.limit)
    exp_id = cfg.exp_id.removeprefix("train_")
    exp_id = f"finetuned_{exp_id}_test" + (f"_n{limit}" if limit != TEST_DEFAULT_LIMIT else "")
    tcfg = compose([f"model={cfg.model_key}", f"prompt={cfg.prompt.id}", "stage=finetuned", f"exp_id={exp_id}",
                    f"seed={cfg.seed}", f"data.split={t.split}", f"data.limit={'null' if limit is None else limit}",
                    f"data.variant={cfg.data.variant}", f"data.subset_seed={cfg.data.subset_seed}"])
    adapter = RUNS / cfg.exp_id / "adapter_best"
    if adapter.is_relative_to(ROOT):        # relativa, como en stage2b_test.txt (config.yaml comparable)
        adapter = adapter.relative_to(ROOT)
    tcfg.model = OmegaConf.merge(cfg.model, {"adapter_path": adapter.as_posix()})
    if cfg.data.get("root") is not None:
        tcfg.data.root = cfg.data.root
    tcfg.infer.batch_size = cfg.model.get("batch_size", 8)
    tcfg.infer.save_embeddings = cfg.model.family == "vljepa"
    return tcfg


def strip_resume_safe(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg))
    for key in RESUME_SAFE:
        node, parts = c, key.split(".")
        for p in parts[:-1]:
            node = node.get(p, {})
        node.pop(parts[-1], None)
    return c
