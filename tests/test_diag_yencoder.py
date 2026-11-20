"""scripts/diag_yencoder_quant.py: partes que no cargan pesos (promedio con máscara y validación de la config)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diag_yencoder_quant as dy  # noqa: E402


def test_masked_mean_ignora_relleno():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])     # 3.er token = relleno
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(dy.masked_mean(h, mask), torch.tensor([[2.0, 3.0]]))


def test_masked_mean_igual_a_open_vljepa():
    # misma fórmula que openvljepa.models.y_encoder.YEncoder.forward
    torch.manual_seed(0)
    h, mask = torch.randn(4, 7, 5), (torch.rand(4, 7) > 0.3).long()
    mask[:, 0] = 1
    m = mask.unsqueeze(-1).to(h.dtype)
    ref = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    assert torch.allclose(dy.masked_mean(h, mask), ref)


def test_config_por_defecto_bf16_y_exp_id():
    cfg = dy.compose_diag(["model=vljepa", "diag.n=64", "diag.tag=humo"])
    assert cfg.model.quant == "bf16"
    assert cfg.exp_id == "diagyq_vljepa_open_msrvtt_humo"
    assert int(cfg.diag.n) == 64


@pytest.mark.parametrize("argv", [["model=vljepa", "model.quant=nf4"],
                                  ["model=vljepa", "model.adapter_path=runs/x/adapter_best"],
                                  ["model=vljepa", "data.split=test"]])
def test_config_invalida(argv):
    with pytest.raises(ValueError):
        dy.compose_diag(argv)
