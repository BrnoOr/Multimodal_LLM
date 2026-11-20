"""`quant_components` de VL-JEPA: por defecto los tres; un subconjunto solo para el diagnóstico.

No carga pesos: prueba `VLJEPADescriber.quant_components` con configuraciones mínimas.
"""

import pytest
from omegaconf import OmegaConf

from vlmfid.models.vljepa import QUANT_COMPONENTS, VLJEPADescriber


def _d(**kw):
    base = {"name": "vljepa_open_msrvtt", "family": "vljepa", "quant": "nf4", "quant_components": None}
    return VLJEPADescriber(OmegaConf.create(base | kw))


def test_por_defecto_los_tres():
    assert _d().quant_components() == QUANT_COMPONENTS


def test_lista_completa_en_otro_orden_equivale_al_defecto():
    assert _d(quant_components=["y_encoder", "x_encoder", "predictor"]).quant_components() == QUANT_COMPONENTS


def test_subconjunto_en_orden_canonico():
    assert _d(quant_components=["y_encoder", "predictor"]).quant_components() == ("predictor", "y_encoder")


@pytest.mark.parametrize("comps", [[], ["vision"], ["predictor", "predictor"]])
def test_invalidos(comps):
    with pytest.raises(ValueError):
        _d(quant_components=comps).quant_components()


def test_parcial_con_bf16_falla():
    with pytest.raises(ValueError):
        _d(quant="bf16", quant_components=["predictor"]).quant_components()


@pytest.mark.parametrize("extra", [{"decoder_path": "runs/x/decoder_best"}, {"adapter_path": "runs/x/adapter_best"}])
def test_parcial_no_admite_decodificador_ni_adaptador(extra):
    with pytest.raises(ValueError):
        _d(quant_components=["predictor"], **extra).quant_components()


def test_completo_si_admite_decodificador():
    assert _d(decoder_path="runs/x/decoder_best").quant_components() == QUANT_COMPONENTS
