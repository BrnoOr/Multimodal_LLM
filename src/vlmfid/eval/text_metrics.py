"""Métricas de texto: Bleu-1, Bleu-4, ROUGE-L y CIDEr con el código de evaluación de MS COCO.

Es un envoltorio fino de `pycocoevalcap` (docs/metodologia.md §2.1): no reimplementa ninguna métrica.
SPICE (Anderson et al., 2016) reporta estas cuatro con ese mismo código, y VL-JEPA (Chen et al., 2025)
usa CIDEr para captioning.

Lo que se hereda de `pycocoevalcap`, y hay que declarar en el informe:
  * Tokenización: PTBTokenizer de Stanford CoreNLP (minúsculas, sin puntuación). **Requiere Java**
    en el PATH; `scripts/check_env.py` lo comprueba.
  * `Cider` es la variante CIDEr-D (TF recortado y penalización de largo, σ = 6, escala 0–10).
  * La IDF de CIDEr se calcula sobre las referencias que se le pasan: los valores de val y de test
    no son comparables entre sí, y un subconjunto pequeño da una IDF ruidosa.
  * Bleu es de corpus; ROUGE-L y CIDEr son la media de los puntajes por imagen.

Hay UNA referencia por imagen: el caption original de M3DI. Como el caption usa una de cinco
plantillas elegida al azar, estas métricas premian acertar la plantilla, que la imagen no determina.
Se reportan junto a la exactitud por atributo y no deben leerse como fidelidad.
"""

from __future__ import annotations

from ..prompts.parsing import strip_think_tags

TEXT_METRICS = ("bleu1", "bleu4", "rouge_l", "cider")


def _clean(text: str | None) -> str:
    # el tokenizador PTB procesa una frase por línea: un salto de línea desalinearía las muestras
    return " ".join(strip_think_tags(text or "").split())


def text_metrics(hypotheses: list[str], references: list[str], ids: list | None = None,
                 per_sample: bool = False) -> dict:
    """Métricas de `hypotheses` contra `references` (alineadas, una referencia por hipótesis).

    Devuelve {"bleu1", "bleu4", "rouge_l", "cider", "n"}; con `per_sample=True` agrega
    "per_sample": {"ids", "rouge_l", "cider"} con los puntajes por imagen en el orden de entrada.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hipótesis para {len(references)} referencias")
    if not hypotheses:
        return {m: float("nan") for m in TEXT_METRICS} | {"n": 0}
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError("vlmfid.eval.text_metrics necesita pycocoevalcap: `uv sync --extra eval`") from e

    keys = list(range(len(hypotheses)))                 # claves propias: los ids pueden repetirse o no ser str
    gts = {k: [{"caption": _clean(r)}] for k, r in zip(keys, references)}
    res = {k: [{"caption": _clean(h)}] for k, h in zip(keys, hypotheses)}
    tok = PTBTokenizer()
    gts, res = tok.tokenize(gts), tok.tokenize(res)
    if set(gts) != set(keys) or set(res) != set(keys):
        raise RuntimeError("El tokenizador PTB devolvió un conjunto distinto de muestras (¿Java disponible?)")

    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    rouge, rouge_each = Rouge().compute_score(gts, res)
    cider, cider_each = Cider().compute_score(gts, res)
    out = {"bleu1": float(bleu[0]), "bleu4": float(bleu[3]), "rouge_l": float(rouge), "cider": float(cider),
           "n": len(keys)}
    if per_sample:
        # Rouge y Cider recorren gts.keys() en orden de inserción, que es el de entrada
        out["per_sample"] = {"ids": list(ids) if ids is not None else keys,
                             "rouge_l": [float(x) for x in rouge_each],
                             "cider": [float(x) for x in cider_each]}
    return out
