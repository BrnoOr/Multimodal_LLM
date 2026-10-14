"""Composición declarativa de configuraciones (OmegaConf, sin Hydra).

Una configuración resuelta = defaults.yaml
    ⊕ configs/models/<model>.yaml       (en la clave `model`)
    ⊕ configs/prompts.yaml[<prompt>]    (catálogo único; la entrada va en la clave `prompt`)
    ⊕ configs/experiments/<exp>.yaml (opcional; puede fijar model, prompt y overrides)
    ⊕ overrides por línea de comandos (sintaxis dotlist: data.limit=100)

exp_id por defecto: <stage>_<model.name>_<quant>_<prompt.id>_<split>[_<variant>][_n<limit>]
(la variante solo aparece si no es "base")

Ejemplos:
    compose(["model=llava_ov", "prompt=p0_minimal", "data.limit=50"])
    compose(["experiment=e01_zeroshot_llava_p1", "infer.batch_size=4"])
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .paths import CONFIGS

GROUPS = ("model", "prompt")


def as_dict(node) -> dict:
    """Nodo de configuración opcional -> dict de Python.

    Acepta DictConfig, dict común o None (clave ausente o `null` en el YAML). OmegaConf.to_container
    solo acepta objetos de OmegaConf, así que un `cfg.get(clave) or {}` directo falla cuando la
    clave no existe.
    """
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=True) or {}
    return dict(node)
PROMPT_CATALOG = CONFIGS / "prompts.yaml"


def _load_model(name: str) -> DictConfig:
    path = CONFIGS / "models" / f"{name}.yaml"
    if not path.exists():
        avail = sorted(p.stem for p in (CONFIGS / "models").glob("*.yaml"))
        raise FileNotFoundError(f"No existe {path}. Modelos disponibles: {avail}")
    return OmegaConf.load(path)


def load_prompt_catalog() -> DictConfig:
    return OmegaConf.load(PROMPT_CATALOG)


def _load_prompt(name: str) -> DictConfig:
    catalog = load_prompt_catalog()
    if name not in catalog:
        raise FileNotFoundError(f"'{name}' no está en {PROMPT_CATALOG}. Prompts: {list(catalog)}")
    entry = catalog[name]
    if entry.get("id") != name:
        raise ValueError(f"{PROMPT_CATALOG}: la entrada '{name}' tiene id '{entry.get('id')}'")
    return entry


_LOADERS = {"model": _load_model, "prompt": _load_prompt}


def _load_experiment(name_or_path: str) -> DictConfig:
    p = Path(name_or_path)
    if p.suffix != ".yaml":
        p = CONFIGS / "experiments" / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No existe el experimento {p}")
    return OmegaConf.load(p)


def compose(args: list[str] | None = None) -> DictConfig:
    args = list(args or [])
    kv = {}
    for a in args:
        if "=" not in a:
            raise ValueError(f"Argumento inválido '{a}': se espera clave=valor")
        k, v = a.split("=", 1)
        kv[k] = v

    cfg = OmegaConf.load(CONFIGS / "defaults.yaml")
    exp_overrides: DictConfig = OmegaConf.create()

    if "experiment" in kv:
        exp = _load_experiment(kv.pop("experiment"))
        for key in ("exp_id", "stage", "model", "prompt"):
            if key in exp:
                cfg[key] = exp[key]
        exp_overrides = exp.get("overrides", OmegaConf.create())

    # selección de grupos (CLI > experimento > defaults)
    for group in GROUPS:
        if group in kv:
            cfg[group] = kv.pop(group)
    model_name, prompt_name = str(cfg.model), str(cfg.prompt)
    cfg.model = _LOADERS["model"](model_name)
    cfg.prompt = _LOADERS["prompt"](prompt_name)

    # overrides del experimento y luego de la CLI (dotlist)
    for dotted, value in as_dict(exp_overrides).items():
        OmegaConf.update(cfg, dotted, value, merge=True)
    cli = OmegaConf.from_dotlist([f"{k}={v}" for k, v in kv.items()])
    cfg = OmegaConf.merge(cfg, cli)

    if cfg.get("exp_id") in (None, "", "null"):
        limit = cfg.data.get("limit")
        variant = cfg.data.get("variant", "base")
        suffix = (f"_{variant}" if variant != "base" else "") + (f"_n{limit}" if limit else "")
        cfg.exp_id = (f"{cfg.stage}_{cfg.model.name}_{cfg.model.quant}_{cfg.prompt.id}_"
                      f"{cfg.data.split}{suffix}")
    if cfg.infer.get("batch_size") is None:
        cfg.infer.batch_size = cfg.model.get("batch_size", 8)
    return cfg


def to_yaml(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)
