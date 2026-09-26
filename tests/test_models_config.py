"""Configs de modelos: registro, nombres únicos y patrones de exclusión de NF4.

No carga pesos (no requiere GPU ni torch): verifica que cada YAML sea coherente con el adaptador
y con la semántica real de transformers >= 5.17 para `llm_int8_skip_modules`.
"""

import re

import pytest
from omegaconf import OmegaConf

from vlmfid.models.registry import MODELS
from vlmfid.paths import CONFIGS

MODEL_FILES = sorted((CONFIGS / "models").glob("*.yaml"))
HF_FILES = [p for p in MODEL_FILES if OmegaConf.load(p).family != "vljepa"]

# nombres representativos de módulos por familia (tomados de transformers 5.17)
SAMPLE_MODULES = {
    "vision": {
        "vision_tower": ["model.vision_tower.encoder.layers.0.self_attn.q_proj",
                         "model.vision_tower.vision_model.encoder.layers.3.mlp.fc1"],
        "multi_modal_projector": ["model.multi_modal_projector.linear_1"],
        "visual": ["model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc1"],
    },
    "language": ["model.language_model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.0.mlp.gate_proj",
                 "model.language_model.layers.1.linear_attn.in_proj_qkv"],
}


def should_convert(full_name: str, patterns: list[str]) -> bool:
    """Réplica de transformers.quantizers.quantizers_utils.should_convert_module (5.17)."""
    return not any(
        re.match(f"{key}\\.", full_name) or re.match(f"{key}", full_name) or full_name.endswith(key)
        for key in patterns
    )


def test_expected_models_present():
    stems = {p.stem for p in MODEL_FILES}
    assert {"llava_ov", "internvl3", "qwen35", "vljepa"} <= stems


@pytest.mark.parametrize("path", MODEL_FILES, ids=lambda p: p.stem)
def test_family_registered_and_fields(path):
    cfg = OmegaConf.load(path)
    assert cfg.family in MODELS
    assert cfg.name and cfg.hf_id and cfg.quant in ("nf4", "int8", "bf16")
    assert int(cfg.batch_size) > 0


def test_names_unique():
    names = [OmegaConf.load(p).name for p in MODEL_FILES]
    assert len(names) == len(set(names))            # el nombre forma parte del exp_id


@pytest.mark.parametrize("path", HF_FILES, ids=lambda p: p.stem)
def test_quant_skip_patterns_exclude_vision_only(path):
    cfg = OmegaConf.load(path)
    patterns = list(cfg.quant_skip_modules)
    assert "lm_head" in patterns
    assert cfg.vision_modules, "vision_modules es necesario para la verificación post-carga"
    for key in cfg.vision_modules:
        for name in SAMPLE_MODULES["vision"][key]:
            assert not should_convert(name, patterns), f"{name} quedaría cuantizado"
    for name in SAMPLE_MODULES["language"]:
        assert should_convert(name, patterns), f"{name} no se cuantizaría"
    assert not should_convert("lm_head", patterns)


def test_unanchored_pattern_would_fail():
    # documenta el error que evitan los patrones con ".*\\.": el nombre simple no excluye nada
    assert should_convert("model.vision_tower.encoder.layers.0.self_attn.q_proj", ["vision_tower"])


def test_qwen35_lora_regex_targets_language_model_only():
    rx = re.compile(OmegaConf.load(CONFIGS / "models" / "qwen35.yaml").lora_target_modules)
    assert rx.match("model.language_model.layers.1.linear_attn.in_proj_qkv")
    assert rx.match("model.language_model.layers.3.self_attn.q_proj")
    assert not rx.match("model.visual.blocks.0.attn.qkv")
    assert not rx.match("model.visual.merger.linear_fc1")
