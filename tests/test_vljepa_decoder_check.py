"""Verificación de que el Y-Decoder corresponde al Y-Encoder cargado (adaptador y cuantización).

No carga pesos: prueba la función pura `check_decoder_matches` con metadatos como los que escribe
`vlmfid.train.ydecoder` (docs/metodologia.md §2.4; P10).
"""

import pytest

from vlmfid.models.vljepa import check_decoder_matches


def _meta(quant="nf4", adapter=None):
    return {"vljepa_name": "vljepa_open_msrvtt", "vljepa_quant": quant,
            "vljepa_adapter_path": adapter, "exp_id": "ydec_prueba"}


def test_coincide_etapa1_nf4():
    check_decoder_matches(_meta("nf4"), quant="nf4", adapter_path=None, path="x")


def test_coincide_etapa1_bf16():
    check_decoder_matches(_meta("bf16"), quant="bf16", adapter_path=None, path="x")


def test_coincide_etapa2():
    check_decoder_matches(_meta("nf4", "runs/a/adapter_best"), quant="nf4",
                          adapter_path="runs/a/adapter_best", path="x")


def test_adaptador_vacio_equivale_a_none():
    check_decoder_matches(_meta("nf4", ""), quant="nf4", adapter_path=None, path="x")


@pytest.mark.parametrize("trained, current", [("nf4", "bf16"), ("bf16", "nf4")])
def test_cuantizacion_distinta_falla(trained, current):
    with pytest.raises(RuntimeError, match="cuantización"):
        check_decoder_matches(_meta(trained), quant=current, adapter_path=None, path="x")


def test_sin_cuantizacion_registrada_falla():
    meta = _meta()
    del meta["vljepa_quant"]
    with pytest.raises(RuntimeError, match="no registra"):
        check_decoder_matches(meta, quant="nf4", adapter_path=None, path="x")


@pytest.mark.parametrize("trained, current", [(None, "runs/a/adapter_best"), ("runs/a/adapter_best", None)])
def test_etapa_distinta_falla(trained, current):
    with pytest.raises(RuntimeError, match="adapter_path"):
        check_decoder_matches(_meta("nf4", trained), quant="nf4", adapter_path=current, path="x")
