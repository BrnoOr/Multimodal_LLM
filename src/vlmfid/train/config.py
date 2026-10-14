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
"""

from __future__ import annotations

import json

from omegaconf import DictConfig, OmegaConf

from ..config import _load_model, _load_prompt, as_dict
from ..paths import CONFIGS

TRAIN_CONFIGS = CONFIGS / "train"
ALLOWED_MODEL_KEYS = {"prompt", "train.micro_batch_size", "lora.connector_modules", "objective.kind",
                      "objective.temperature", "objective.mask_duplicate_captions"}
# Claves que pueden cambiar al reanudar un entrenamiento sin alterar su resultado.
RESUME_SAFE = {"data.num_workers", "train.micro_batch_size", "train.log_every", "train.save_every",
               "train.max_steps", "train.eval_every"}


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
    if cfg.objective.kind == "infonce" and t.micro_batch_size != t.effective_batch_size:
        raise ValueError("objective=infonce: los negativos salen del micro-lote, que debe ser igual al lote "
                         "efectivo (acumular gradientes cambia la pérdida). Ajusta train.effective_batch_size.")
    if cfg.objective.kind not in ("lm", "infonce"):
        raise ValueError(f"objective.kind='{cfg.objective.kind}' inválido (lm | infonce)")
    if cfg.model.get("adapter_path"):
        raise ValueError("model.adapter_path debe ser null al entrenar (el adaptador se crea desde cero)")

    if cfg.get("exp_id") in (None, "", "null"):
        lim = cfg.data.get("train_limit")
        cfg.exp_id = (f"train_{cfg.model.name}_{cfg.model.quant}_{cfg.prompt.id}_r{cfg.lora.r}_e{t.epochs}"
                      + (f"_n{lim}" if lim else ""))
    return cfg


def strip_resume_safe(cfg: dict) -> dict:
    c = json.loads(json.dumps(cfg))
    for key in RESUME_SAFE:
        node, parts = c, key.split(".")
        for p in parts[:-1]:
            node = node.get(p, {})
        node.pop(parts[-1], None)
    return c
