import pytest

from vlmfid.config import compose
from vlmfid.models.registry import MODELS


@pytest.mark.parametrize("model", ["llava_ov", "qwen35", "internvl3", "vljepa", "qwen35_instruct"])
def test_every_model_config_resolves(model):
    cfg = compose([f"model={model}", "prompt=p2_constrained", "data.limit=50"])
    assert cfg.model.family in MODELS
    assert cfg.model.quant == "nf4"                   # esquema común
    assert cfg.infer.batch_size == cfg.model.batch_size
    assert cfg.exp_id == f"zeroshot_{cfg.model.name}_nf4_p2_constrained_test_n50"


def test_experiment_and_cli_precedence():
    cfg = compose(["experiment=e01_zeroshot_llava_p0", "infer.batch_size=3", "model.quant=bf16"])
    assert cfg.exp_id == "e01_zeroshot_llava_p0"
    assert cfg.model.name == "llava_ov_05b" and cfg.model.quant == "bf16"
    assert cfg.infer.batch_size == 3


def test_variant_in_exp_id():
    cfg = compose(["model=llava_ov", "prompt=p0_minimal", "data.variant=ood_colors"])
    assert cfg.exp_id == "zeroshot_llava_ov_05b_nf4_p0_minimal_test_ood_colors"
    assert compose(["model=llava_ov"]).exp_id.endswith("_test")      # base no aparece


def test_bad_group():
    with pytest.raises(FileNotFoundError):
        compose(["model=no_existe"])
    with pytest.raises(FileNotFoundError):
        compose(["prompt=no_existe"])


def test_as_dict_accepts_missing_plain_and_omegaconf():
    from omegaconf import OmegaConf

    from vlmfid.config import as_dict

    cfg = compose(["model=llava_ov"])                       # sin chat_template_kwargs
    assert as_dict(cfg.model.get("chat_template_kwargs")) == {}
    assert as_dict(None) == {} and as_dict({}) == {}
    assert as_dict(OmegaConf.create({"a": 1})) == {"a": 1}
    q = compose(["model=qwen35"])
    assert as_dict(q.model.get("chat_template_kwargs")) == {"enable_thinking": False}
    assert as_dict(cfg.model.get("generation"))["max_new_tokens"] == 128
