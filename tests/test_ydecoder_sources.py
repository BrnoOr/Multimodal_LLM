"""Y-Decoder: lectura de captions de MSR-VTT, nombres de los runs y continuación desde otro decodificador.

No descarga nada ni carga pesos (docs/metodologia.md §2.4; decisión del 16-11-2026: decodificador
genérico de MSR-VTT en la etapa 1, ajustado con M3DI desde él en la intermedia y en la etapa 2).
"""

import pytest
from omegaconf import OmegaConf

from vlmfid.data.msrvtt import parse_captions
from vlmfid.paths import CONFIGS
from vlmfid.train.ydecoder import (
    SELECTION,
    check_init_compatible,
    default_exp_id,
    epochs_for,
    eval_every_for,
    text_source,
)


# ── formatos de anotación de MSR-VTT ──────────────────────────────────────────────────────────────
def test_un_caption_por_registro():
    obj = [{"video_id": "video7020", "caption": "a man  is talking"}, {"video_id": "video7021", "caption": "x"}]
    assert parse_captions(obj) == [("video7020", "a man is talking"), ("video7021", "x")]


def test_lista_de_captions_por_registro():
    obj = [{"video_id": "video0", "caption": ["a cat", "a cat plays"]}]
    assert parse_captions(obj) == [("video0", "a cat"), ("video0", "a cat plays")]


def test_diccionario_por_video():
    assert parse_captions({"video0": ["a", "b"]}) == [("video0", "a"), ("video0", "b")]
    assert parse_captions({"video0": {"captions": ["a"], "split": "train"}}) == [("video0", "a")]


def test_captions_vacios_se_descartan_y_formato_desconocido_falla():
    assert parse_captions([{"video_id": "v", "caption": ["  ", "ok"]}]) == [("v", "ok")]
    with pytest.raises(ValueError):
        parse_captions([{"video_id": "v", "title": "sin caption"}])
    with pytest.raises(ValueError):
        parse_captions("texto")


# ── configuración ─────────────────────────────────────────────────────────────────────────────────
def _cfg(**dec):
    base = OmegaConf.load(CONFIGS / "ydecoder.yaml")
    cfg = OmegaConf.merge(OmegaConf.create({"model": {"name": "vljepa_open_msrvtt", "quant": "nf4"}, "seed": 0}),
                          base, OmegaConf.create({"decoder": dec}))
    return cfg


def test_por_defecto_el_genérico_de_msrvtt_elige_por_cider():
    cfg = _cfg()
    assert text_source(cfg) == "msrvtt" and SELECTION["msrvtt"] == "cider"
    assert epochs_for(cfg) == int(cfg.decoder.msrvtt.epochs)
    assert eval_every_for(cfg) == int(cfg.decoder.msrvtt.eval_every)


def test_nombres_de_los_runs():
    k = OmegaConf.load(CONFIGS / "ydecoder.yaml").decoder.prefix_len
    assert default_exp_id(_cfg(msrvtt={"epochs": 1})) == f"ydec_vljepa_open_msrvtt_nf4_zs_nf4_msrvtt_k{k}_r64_e1"
    # M3DI sin marca de fuente: los runs anteriores conservan su nombre
    assert default_exp_id(_cfg(text_source="m3di", train={"epochs": 10})) == f"ydec_vljepa_open_msrvtt_nf4_zs_nf4_k{k}_r64_e10"
    cont = _cfg(text_source="m3di", init_from="runs/x/decoder_best", train={"epochs": 10})
    assert default_exp_id(cont).endswith("_e10_cont")
    with pytest.raises(ValueError):
        text_source(_cfg(text_source="coco"))


def test_continuar_exige_la_misma_arquitectura(tmp_path):
    d = _cfg().decoder
    (tmp_path / "decoder.yaml").write_text(OmegaConf.to_yaml(d), encoding="utf-8")
    check_init_compatible(d, tmp_path)                                    # misma arquitectura: pasa
    otro = OmegaConf.merge(d, {"prefix_len": int(d.prefix_len) + 4})
    with pytest.raises(ValueError, match="prefix_len"):
        check_init_compatible(otro, tmp_path)
    with pytest.raises(FileNotFoundError):
        check_init_compatible(d, tmp_path / "no_existe")
