"""Métricas estándar de generación: BLEU-4, ROUGE-L y CIDEr-D contra una referencia por imagen.

Es el `TextScorer` del notebook 03, sin cambios de lógica (más la tasa de texto idéntico y el BLEU
pareado), para que los notebooks 03 y 05 midan lo mismo. Reproduce las definiciones de `pycocoevalcap` (verificado numéricamente en el notebook 03)
sin Java:

  * BLEU-4 (Papineni et al., 2002): BP · exp(¼ Σ log p_n), precisión de n-gramas recortada y
    BP = min(1, e^{1 - r/c}), a nivel de corpus.
  * ROUGE-L (Lin, 2004): F_β de la subsecuencia común más larga, β = 1.2.
  * CIDEr-D (Vedantam et al., 2015): coseno TF-IDF de 1-4-gramas con TF recortado y penalización
    de largo e^{-δ²/2σ²}, σ = 6. La IDF se estima sobre las referencias del scorer, así que los
    n-gramas presentes en todas (la plantilla) pesan 0. Escala 0-10.

Tokenización común a las tres: minúsculas, sin <think>, secuencias alfanuméricas.

La referencia es un argumento: el mismo texto generado se puede medir contra el caption original
de M3DI y contra el caption canónico (`vlmfid.data.targets.canonical_caption`).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..prompts.parsing import parse_json_response, strip_think_tags

_TOK = re.compile(r"[a-z0-9]+")
TEXT_METRICS = ("bleu", "rouge_l", "cider_d")


def tokenize(t) -> tuple[str, ...]:
    return tuple(_TOK.findall(strip_think_tags(t or "").lower()))


def ngrams(toks, n):
    return Counter(toks[i:i + n] for i in range(len(toks) - n + 1))


def json_to_template(t):
    """Respuesta JSON (p3) reescrita en la plantilla de M3DI con los valores del modelo; el resto queda igual."""
    o = parse_json_response(strip_think_tags(t or ""))
    if not o:
        return t
    return (f'A "{str(o.get("color", "")).strip()}" {str(o.get("shape", "")).strip()} is at the '
            f'{str(o.get("vertical", "")).strip()}-{str(o.get("horizontal", "")).strip()} of the image.')


def derangement(n: int, seed: int = 0) -> np.ndarray:
    """Índices src con src[i] != i: a cada imagen le toca la referencia de otra (línea base permutada)."""
    order = np.random.default_rng(seed).permutation(n)
    src = np.empty_like(order)
    src[order] = np.roll(order, 1)
    return src


class TextScorer:
    """BLEU-4 de corpus, ROUGE-L y CIDEr-D por imagen contra una referencia fija por id (definiciones de pycocoevalcap)."""

    def __init__(self, refs: dict, n=4, beta=1.2, sigma=6.0):
        self.n, self.beta, self.sigma = n, beta, sigma
        self.refs = {i: tokenize(t) for i, t in refs.items()}
        self.df_ = Counter()
        for r in self.refs.values():
            self.df_.update({g for k in range(1, n + 1) for g in ngrams(r, k)})
        self.logN = math.log(len(self.refs))
        self._ng, self._vec = {}, {}

    def ng(self, t):
        if t not in self._ng:
            self._ng[t] = [ngrams(t, k) for k in range(1, self.n + 1)]
        return self._ng[t]

    def vec(self, t):
        if t not in self._vec:
            v = [{g: tf * (self.logN - math.log(max(1.0, self.df_[g]))) for g, tf in c.items()} for c in self.ng(t)]
            self._vec[t] = (v, [math.sqrt(sum(x * x for x in d.values())) for d in v], max(0, len(t) - 1))
        return self._vec[t]

    def bleu_stats(self, h, r):
        out = np.zeros(2 + 2 * self.n)
        out[0], out[1] = len(h), len(r)
        for k, (hc, rc) in enumerate(zip(self.ng(h), self.ng(r)), 1):
            out[1 + k] = sum(min(c, rc[g]) for g, c in hc.items())
            out[1 + self.n + k] = max(0, len(h) - k + 1)
        return out

    def corpus_bleu(self, s):
        s = s.sum(0) if s.ndim == 2 else s
        c, r, m, g = s[0], s[1], s[2:2 + self.n], s[2 + self.n:]
        if c == 0 or (m == 0).any():
            return 0.0
        return (1.0 if c > r else math.exp(1 - r / c)) * math.exp(np.mean(np.log(m / g)))

    def rouge(self, h, r):
        if not h or not r:
            return 0.0
        prev = [0] * (len(r) + 1)
        for x in h:
            cur = [0]
            for j, y in enumerate(r, 1):
                cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
            prev = cur
        lcs = prev[-1]
        if lcs == 0:
            return 0.0
        p, rc = lcs / len(h), lcs / len(r)
        return (1 + self.beta ** 2) * p * rc / (rc + self.beta ** 2 * p)

    def cider(self, h, r):
        vh, nh, lh = self.vec(h)
        vr, nr, lr = self.vec(r)
        pen = math.exp(-((lh - lr) ** 2) / (2 * self.sigma ** 2))
        return 10.0 / self.n * sum(
            sum(min(w, vr[k].get(g, 0.0)) * vr[k].get(g, 0.0) for g, w in vh[k].items()) / (nh[k] * nr[k]) * pen
            for k in range(self.n) if nh[k] and nr[k])

    def per_sample(self, hyps, ids):
        H = [tokenize(t) for t in hyps]
        R = [self.refs[i] for i in ids]
        return {"bleu_stats": np.stack([self.bleu_stats(h, r) for h, r in zip(H, R)]),
                "rouge_l": np.array([self.rouge(h, r) for h, r in zip(H, R)]),
                "cider_d": np.array([self.cider(h, r) for h, r in zip(H, R)]),
                "exact": np.array([h == r for h, r in zip(H, R)], dtype=float)}

    def summary(self, ps, B=0, seed=0, alpha=0.05):
        """Promedios de corpus; con B > 0, IC (1 − alpha) por bootstrap de imágenes."""
        out = {"bleu": self.corpus_bleu(ps["bleu_stats"]), "rouge_l": ps["rouge_l"].mean(), "cider_d": ps["cider_d"].mean()}
        if B:
            n, rng_ = len(ps["rouge_l"]), np.random.default_rng(seed)
            draws = np.empty((B, 3))
            for b in range(B):
                w = np.bincount(rng_.integers(0, n, n), minlength=n).astype(float)
                draws[b] = self.corpus_bleu(w @ ps["bleu_stats"]), w @ ps["rouge_l"] / n, w @ ps["cider_d"] / n
            lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
            for j, k in enumerate(TEXT_METRICS):
                out[f"{k}_lo"], out[f"{k}_hi"] = lo[j], hi[j]
        return out

    def paired_bleu(self, ps_a, ps_b, B=1000, seed=0, alpha=0.05):
        """BLEU-4 de corpus B − A con IC bootstrap pareado (mismas imágenes remuestreadas en ambos)."""
        n, rng_ = len(ps_a["bleu_stats"]), np.random.default_rng(seed)
        d = np.empty(B)
        for b in range(B):
            w = np.bincount(rng_.integers(0, n, n), minlength=n).astype(float)
            d[b] = self.corpus_bleu(w @ ps_b["bleu_stats"]) - self.corpus_bleu(w @ ps_a["bleu_stats"])
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        return {"Δ": self.corpus_bleu(ps_b["bleu_stats"]) - self.corpus_bleu(ps_a["bleu_stats"]), "lo": lo, "hi": hi,
                "p": max(p, 1 / B)}
