"""Objetivo canónico y split composicional de la etapa 2 reformulada (vlmfid.data.targets)."""

import itertools

import pytest

pytest.importorskip("matplotlib")

from vlmfid.data.targets import (
    HELD_OUT_COMBOS,
    apply_holdout,
    canonical_caption,
    combo,
    is_held_out,
    target_fn,
    target_text,
)
from vlmfid.eval.attributes import score, extract, truth_from_record


def _rec(shape, x, y, hue, name="tab:blue"):
    return {"id": f"{shape}{x}{y}{hue}", "caption": f'A "{name}" thing.',
            "latents_image": {"object_shape": shape, "object_xpos": x, "object_ypos": y, "object_zpos": 0,
                              "object_color": hue},
            "latents_text": {"object_color_name": name}}


ALL = [_rec(s, x, y, h) for s, x, y, h in itertools.product(range(7), range(3), range(3), (0.0, 0.1, 0.17, 0.3,
                                                                                          0.5, 0.62, 0.75, 0.9))]


def test_canonico_es_exacto_para_el_extractor():
    """El extractor de la etapa 1 recupera todos los atributos del caption canónico (techo = 100 %)."""
    for r in ALL:
        s = score(extract(canonical_caption(r)), truth_from_record(r))
        assert s["shape"] and s["pos"] and s["color"] and s["color_adj"], canonical_caption(r)


def test_formato_canonico():
    r = _rec(2, 0, 2, 0.62)
    assert canonical_caption(r) == "A blue dragon is at the bottom-left of the image."
    assert target_text(r, "original") == r["caption"]
    assert target_fn("canonical")(r) == canonical_caption(r)
    with pytest.raises(ValueError):
        target_fn("json")
    assert len({canonical_caption(r) for r in ALL}) == 7 * 9 * 8


def test_split_composicional():
    assert len(HELD_OUT_COMBOS) == 14
    assert all(sum(1 for s, _ in HELD_OUT_COMBOS if s == k) == 2 for k in range(7))   # 2 posiciones por forma
    assert {p for _, p in HELD_OUT_COMBOS} == set(range(9))                          # toda posición se retira alguna vez
    kept = apply_holdout(ALL, "compositional")
    assert not any(is_held_out(r) for r in kept)
    assert len(kept) == len(ALL) * 49 // 63
    assert {combo(r)[0] for r in kept} == set(range(7)) and {combo(r)[1] for r in kept} == set(range(9))
    assert apply_holdout(ALL, None) is ALL
