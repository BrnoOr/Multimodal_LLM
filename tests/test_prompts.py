"""Catálogo de prompts y parseo de respuestas estructuradas."""

import pytest

from vlmfid.config import compose, load_prompt_catalog
from vlmfid.prompts import parse_json_response, render

CATALOG = load_prompt_catalog()


def test_catalog_ids_match_keys_and_have_rationale():
    assert list(CATALOG) == ["p0_minimal", "p1_attributes", "p2_constrained", "p3_json", "p4_dataset_format"]
    for key, entry in CATALOG.items():
        assert entry.id == key
        assert entry.text.strip() and entry.rationale.strip()


@pytest.mark.parametrize("name", list(CATALOG))
def test_every_prompt_renders(name):
    # p2 y p3 contienen llaves literales: no deben pasar por str.format
    cfg = compose([f"prompt={name}"])
    text = render(cfg.prompt, {"id": "test/000001"})
    assert text and "\n" not in text and "  " not in text
    if name == "p2_constrained":
        assert "{top, mid, bottom}" in text
    if name == "p3_json":
        assert '{"shape":' in text


@pytest.mark.parametrize("raw, expected", [
    ('{"shape": "teapot", "color": "red", "vertical": "top", "horizontal": "left"}',
     {"shape": "teapot", "color": "red", "vertical": "top", "horizontal": "left"}),
    ('```json\n{"Shape": "cow ", "color": "tab:blue"}\n```', {"shape": "cow", "color": "tab:blue"}),
    ('Here is the answer: {"shape": "hare", "note": "a {weird} value"} Hope it helps.',
     {"shape": "hare", "note": "a {weird} value"}),
    ("{'shape': 'horse', 'vertical': 'mid'}", {"shape": "horse", "vertical": "mid"}),
])
def test_parse_json_response_tolerant(raw, expected):
    assert parse_json_response(raw) == expected


@pytest.mark.parametrize("raw", ["", "A red teapot is at the top-left of the image.", '{"shape": teapot}', "[1, 2]"])
def test_parse_json_response_returns_none(raw):
    assert parse_json_response(raw) is None
