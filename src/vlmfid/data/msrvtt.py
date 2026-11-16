"""Captions de MSR-VTT para el Y-Decoder genérico de la etapa 1 (docs/metodologia.md §2.4).

Solo texto: no se descargan videos. El Y-Decoder de la etapa 1 se entrena con captions que NO son de
M3DI, para que VL-JEPA llegue a la etapa 1 sin haber visto el formato del dataset, igual que los VLM
(decisión del usuario, 16-11-2026). Se usa MSR-VTT porque es el corpus del checkpoint `open-vljepa`
(Stage B, "MSRVTT 140K" en su README).

Fuente: `friedrichor/MSR-VTT` (Hugging Face, protocolo 1K-A de JSFusion):
  * `msrvtt_train_7k.json`  7 010 videos × 20 captions = 140 200: el entrenamiento del checkpoint
    (deducido de los conteos del README de open-vljepa; no está dicho explícitamente). Entrena el
    decodificador.
  * `msrvtt_test_1k.json`   1 000 videos, 1 caption cada uno: VL-JEPA nunca los vio. Elige el mejor paso
    del decodificador. Es el test de MSR-VTT, no el de M3DI: el test de M3DI no se toca.

El formato exacto de los JSON no está documentado en la ficha del dataset; `parse_captions` acepta las
formas habituales y `load_msrvtt` exige los conteos esperados, para fallar en vez de entrenar con un
corpus mal leído.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "friedrichor/MSR-VTT"
FILES = {"train_7k": "msrvtt_train_7k.json", "test_1k": "msrvtt_test_1k.json"}
EXPECTED = {"train_7k": 140_200, "test_1k": 1_000}          # captions (README de friedrichor/MSR-VTT)
CAPTION_KEYS = ("caption", "captions", "sentence", "sentences", "text")
VIDEO_KEYS = ("video_id", "video", "vid", "id")


def _captions_of(item: dict) -> list[str]:
    for k in CAPTION_KEYS:
        if k in item:
            v = item[k]
            return [str(c) for c in v] if isinstance(v, list) else [str(v)]
    raise ValueError(f"Registro de MSR-VTT sin campo de caption ({CAPTION_KEYS}): claves {sorted(item)[:8]}")


def _video_of(item: dict, fallback: str) -> str:
    for k in VIDEO_KEYS:
        if k in item and item[k] is not None:
            return str(item[k])
    return fallback


def parse_captions(obj) -> list[tuple[str, str]]:
    """(video_id, caption) en el orden del archivo, para las formas habituales de anotación:

    * lista de registros, un caption por registro     [{"video_id": ..., "caption": "..."}, ...]
    * lista de registros, varios captions por registro [{"video_id": ..., "caption": ["...", ...]}, ...]
    * diccionario video -> captions                   {"video0": ["...", ...], ...}
    * diccionario video -> registro                   {"video0": {"captions": [...]}, ...}
    """
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for vid, v in obj.items():
            caps = v if isinstance(v, list) else _captions_of(v) if isinstance(v, dict) else [v]
            out += [(str(vid), str(c)) for c in caps]
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                raise ValueError(f"Registro {i} de MSR-VTT no es un objeto: {type(item).__name__}")
            vid = _video_of(item, f"row{i}")
            out += [(vid, c) for c in _captions_of(item)]
    else:
        raise ValueError(f"Formato de anotaciones de MSR-VTT no reconocido: {type(obj).__name__}")
    out = [(v, " ".join(c.split())) for v, c in out]
    return [(v, c) for v, c in out if c]


def _read_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_msrvtt(split: str, repo: str = REPO, filename: str | None = None,
                check_count: bool = True) -> list[tuple[str, str]]:
    """(video_id, caption) de un subconjunto de MSR-VTT. Descarga solo el JSON de anotaciones (caché de
    Hugging Face, `HF_HOME`)."""
    if filename is None:
        if split not in FILES:
            raise ValueError(f"Subconjunto de MSR-VTT desconocido: {split!r} (opciones: {sorted(FILES)})")
        filename = FILES[split]
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))
    pairs = parse_captions(_read_json_or_jsonl(path))
    if check_count and split in EXPECTED and len(pairs) != EXPECTED[split]:
        raise RuntimeError(
            f"MSR-VTT {split}: se leyeron {len(pairs)} captions y se esperaban {EXPECTED[split]} ({path}). "
            "Revisa el formato del archivo antes de entrenar.")
    return pairs
