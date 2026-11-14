"""Entrena el Y-Decoder de VL-JEPA (docs/metodologia.md §2.4; vlmfid.train.ydecoder).

    # etapa 1: decodificador sobre el Y-Encoder preentrenado
    uv run --no-sync python scripts/train_ydecoder.py model=vljepa
    # barrido del largo del prefijo
    uv run --no-sync python scripts/train_ydecoder.py model=vljepa decoder.prefix_len=4
    # etapa 2: decodificador sobre el Y-Encoder ajustado
    uv run --no-sync python scripts/train_ydecoder.py model=vljepa model.adapter_path=runs/<train_vljepa...>/adapter_best
    # prueba de humo (pocos pasos, val chico)
    uv run --no-sync python scripts/train_ydecoder.py model=vljepa decoder.train.epochs=1 decoder.train.eval_every=20 \
        decoder.train.val_limit=32 smoke_steps=40 exp_id=ydec_humo

En el cluster:
    SCRIPT=scripts/train_ydecoder.py GPU=0 scripts/launch.sh ydec scripts/run_queue.sh configs/queues/ydecoder_sweep.txt

Configuración = defaults.yaml ⊕ configs/models/<model>.yaml ⊕ configs/ydecoder.yaml (clave `decoder`)
⊕ overrides. Entrena solo con texto de train y elige el mejor paso en val; test no se usa.
Para usar el decodificador en inferencia: `model.decoder_path=runs/<exp_id>/decoder_best`.
"""

from __future__ import annotations

import sys

from omegaconf import OmegaConf

from vlmfid.config import compose
from vlmfid.paths import CONFIGS
from vlmfid.train.ydecoder import default_exp_id, train_ydecoder


def compose_ydecoder(argv: list[str]):
    dec_args = [a for a in argv if a.startswith("decoder.") or a.startswith("smoke_steps=")]
    rest = [a for a in argv if a not in dec_args]
    explicit_id = any(a.startswith("exp_id=") for a in rest)
    cfg = compose(rest)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIGS / "ydecoder.yaml"),
                          OmegaConf.from_dotlist(dec_args))
    smoke = cfg.get("smoke_steps")
    if smoke is not None:
        # prueba de humo: recorta las épocas a un número de pasos; exige exp_id propio para no pisar un run real
        if not explicit_id:
            raise ValueError("smoke_steps requiere un exp_id explícito (p. ej. exp_id=ydec_humo)")
    cfg.stage = "ydecoder"
    if not explicit_id:
        cfg.exp_id = default_exp_id(cfg)
    return cfg


def main(argv=None) -> int:
    cfg = compose_ydecoder(sys.argv[1:] if argv is None else argv)
    return train_ydecoder(cfg)


if __name__ == "__main__":
    sys.exit(main())
