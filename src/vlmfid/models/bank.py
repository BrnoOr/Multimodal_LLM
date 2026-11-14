"""Banco de referencias factorizado de VL-JEPA (diseño: docs/VLJEPA_AJUSTE_V3.pdf).

El banco anterior (`bank.kind: captions`) contiene frases completas: 504 captions canónicos o ~11 000
originales. Dos frases que solo difieren en una palabra ("top" / "mid") quedan casi paralelas en el
espacio del Y-Encoder, así que la lectura de ese atributo depende de una fracción mínima del vector, y
una recuperación exitosa no dice qué atributo decidió el resultado.

El banco factorizado (`bank.kind: factorized`) tiene UNA frase de referencia por valor de atributo:

    forma (7)   "The shape of the object is teapot."            ... head
    color (8)   "The color of the object is red."               ... pink
    y     (3)   "The vertical position of the object is top."   ... bottom
    x     (3)   "The horizontal position of the object is left." ... right

21 frases legibles en vez de 504. Al modelo se le hace una consulta por atributo (el uso discriminativo
de VL-JEPA: VQA con candidatos, Chen et al. 2025, §3.2 y §4) y cada ŝ se compara solo con las frases de su
atributo. Es explicable por construcción:

  * cada decisión es un softmax sobre 3, 7 u 8 frases que se pueden leer;
  * la respuesta trae su probabilidad y su margen en nats frente al segundo candidato;
  * el caption final se compone con la misma plantilla canónica de la etapa 2
    (`vlmfid.data.targets.CANONICAL_TEMPLATE`), así que pasa por el mismo extractor que los VLM;
  * el techo de la cota InfoNCE de cada atributo es log K = la entropía máxima del atributo, de modo
    que la información que entrega el decodificador se puede leer en nats contra H(atributo).

Este módulo no carga modelos: recibe los embeddings ya calculados. Así se puede probar en CPU.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..data.targets import CANONICAL_TEMPLATE
from ..eval.attributes import COLOR_CATS, SHAPE_NAMES, XPOS_NAMES, YPOS_NAMES, truth_from_record

BANK_KINDS = ("captions", "factorized")
# Orden fijo: define el orden de los embeddings guardados (N, 4, D) y de la rotación de consultas.
ATTRS = ("shape", "color", "y", "x")
VALUES = {
    "shape": [SHAPE_NAMES[k] for k in sorted(SHAPE_NAMES)],
    "color": list(COLOR_CATS),
    "y": [YPOS_NAMES[k] for k in sorted(YPOS_NAMES)],
    "x": [XPOS_NAMES[k] for k in sorted(XPOS_NAMES)],
}
QUERIES = {
    "shape": "What is the shape of the object in the image?",
    "color": "What is the color of the object in the image?",
    "y": "What is the vertical position of the object in the image?",
    "x": "What is the horizontal position of the object in the image?",
}
PHRASES = {
    "shape": "The shape of the object is {v}.",
    "color": "The color of the object is {v}.",
    "y": "The vertical position of the object is {v}.",
    "x": "The horizontal position of the object is {v}.",
}
_TRUTH_KEY = {"shape": "shape_true", "color": "cat_true", "y": "y_true", "x": "x_true"}


def truth_values(rec: dict) -> dict[str, str]:
    """Valor verdadero de cada atributo (misma verdad que `vlmfid.eval.attributes.score`)."""
    t = truth_from_record(rec)
    return {a: t[k] for a, k in _TRUTH_KEY.items()}


def query_rotation(rec_id, seed: int, epoch: int) -> str:
    """Atributo que se le pregunta a una imagen en una época de entrenamiento.

    Rotación determinista: desplazamiento fijo por (semilla, id) + época, módulo 4. En cada época los
    cuatro atributos reciben ~1/4 de las imágenes y, tras 4 épocas, cada imagen fue consultada
    exactamente una vez por cada atributo. No depende del RNG global, así que reanudar no la altera.
    """
    h = int(hashlib.sha1(f"{int(seed)}:{rec_id}".encode()).hexdigest()[:8], 16)
    return ATTRS[(h + int(epoch)) % len(ATTRS)]


@dataclass
class FactorizedBank:
    """Índice del banco factorizado. `texts[i]` es la frase i; `slices[attr]` su tramo contiguo."""

    texts: list[str] = field(default_factory=list)
    entries: list[tuple[str, str]] = field(default_factory=list)       # (atributo, valor) por frase
    slices: dict[str, slice] = field(default_factory=dict)

    @classmethod
    def build(cls) -> "FactorizedBank":
        b = cls()
        for a in ATTRS:
            start = len(b.texts)
            for v in VALUES[a]:
                b.texts.append(PHRASES[a].format(v=v))
                b.entries.append((a, v))
            b.slices[a] = slice(start, len(b.texts))
        return b

    def __len__(self) -> int:
        return len(self.texts)

    def phrase(self, attr: str, value: str) -> str:
        return PHRASES[attr].format(v=value)

    def index(self, attr: str, value: str) -> int:
        return self.slices[attr].start + VALUES[attr].index(value)

    def target_phrases(self, rec: dict) -> dict[str, str]:
        """Frase de referencia correcta de cada atributo para un registro."""
        return {a: self.phrase(a, v) for a, v in truth_values(rec).items()}

    def group_mask(self, device=None) -> torch.Tensor:
        """M[i, j] = True si las frases i y j son del mismo atributo (candidatas entre sí)."""
        g = torch.tensor([ATTRS.index(a) for a, _ in self.entries], device=device)
        return g[:, None] == g[None, :]

    @staticmethod
    def compose(values: dict[str, str]) -> str:
        return CANONICAL_TEMPLATE.format(color=values["color"], shape=values["shape"], y=values["y"],
                                         x=values["x"])

    def decode(self, pred: dict[str, torch.Tensor], bank_emb: torch.Tensor, temperature: float) -> list[dict]:
        """Lectura explicable. `pred[attr]`: (B, D) normalizado, la respuesta a la consulta de `attr`.

        Devuelve, por imagen: valores elegidos, probabilidad de cada candidato (softmax a `temperature`
        sobre las frases del atributo), margen en nats frente al segundo y el caption compuesto.
        """
        bank = F.normalize(bank_emb.float(), dim=-1)
        B = next(iter(pred.values())).shape[0]
        out = [{"values": {}, "prob": {}, "margin_nats": {}} for _ in range(B)]
        for a in ATTRS:
            logits = F.normalize(pred[a].float(), dim=-1) @ bank[self.slices[a]].T / temperature
            prob = logits.softmax(-1)
            top = logits.topk(2, dim=-1)
            for j in range(B):
                out[j]["values"][a] = VALUES[a][top.indices[j, 0].item()]
                out[j]["prob"][a] = {v: round(p, 4) for v, p in zip(VALUES[a], prob[j].tolist())}
                out[j]["margin_nats"][a] = round((top.values[j, 0] - top.values[j, 1]).item(), 4)
        for o in out:
            o["text"] = self.compose(o["values"])
        return out


# ───────────────────────────────────────────────────────── geometría del banco
def pairwise_cosine(emb: torch.Tensor) -> torch.Tensor:
    e = F.normalize(emb.float(), dim=-1)
    return e @ e.T


def margin_bound(cos: float, temperature: float) -> float:
    """Máxima diferencia de logits (nats) que un ŝ unitario puede poner entre dos referencias unitarias
    con similitud coseno `cos`: sup_ŝ ⟨ŝ, a − b⟩ / τ = ‖a − b‖ / τ = sqrt(2 (1 − cos)) / τ."""
    return math.sqrt(max(0.0, 2.0 * (1.0 - float(cos)))) / float(temperature)


def group_capacity(emb: torch.Tensor, temperature: float, steps: int = 300, lr: float = 0.05) -> dict:
    """Capacidad de un grupo de K referencias: la mayor información (nats) que un predictor ideal puede
    transmitir a través del softmax a temperatura τ sobre esas K frases, con clases equiprobables.

        C = log K − (1/K) Σ_k min_{‖ŝ‖=1} −log softmax_k(⟨ŝ, t_·⟩ / τ)

    Es una propiedad del banco, no del predictor: si C << log K, ningún entrenamiento del predictor
    puede pasar más información por ese decodificador. Se resuelve por descenso de gradiente sobre la
    esfera (un ŝ por clase), partiendo de la propia referencia de la clase.
    """
    t = F.normalize(emb.float(), dim=-1)
    K = t.shape[0]
    s = t.clone().requires_grad_(True)
    opt = torch.optim.Adam([s], lr=lr)
    labels = torch.arange(K, device=t.device)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = F.cross_entropy(F.normalize(s, dim=-1) @ t.T / temperature, labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ce = F.cross_entropy(F.normalize(s, dim=-1) @ t.T / temperature, labels, reduction="none")
    return {"capacity_nats": float(math.log(K) - ce.mean()), "max_nats": math.log(K),
            "worst_class_ce": float(ce.max())}


def geometry_report(emb: torch.Tensor, groups: dict[str, list[int]], temperature: float) -> dict:
    """Por grupo de candidatos: cosenos entre referencias, cota del margen y capacidad en nats.

    `groups[attr]` = índices de las referencias que compiten entre sí para ese atributo.
    """
    out = {}
    for name, idx in groups.items():
        e = emb[torch.as_tensor(idx)]
        c = pairwise_cosine(e)
        off = c[~torch.eye(len(idx), dtype=torch.bool)]
        cap = group_capacity(e, temperature)
        out[name] = {"k": len(idx), "cos_mean": float(off.mean()), "cos_max": float(off.max()),
                     "cos_min": float(off.min()),
                     "margin_bound_nats_hardest_pair": margin_bound(float(off.max()), temperature), **cap}
    return out


def caption_groups(texts: list[str], attr_of: dict[str, dict[str, str]], fixed: bool = True) -> dict:
    """Grupos de candidatos para un banco de captions canónicos: para cada atributo, conjuntos de
    captions que solo difieren en ese atributo (los pares difíciles). `attr_of[texto]` = valores.

    Devuelve {attr: [[índices], ...]}: una lista de grupos por atributo (un grupo por combinación de los
    otros tres atributos).
    """
    out: dict[str, list[list[int]]] = {a: [] for a in ATTRS}
    for a in ATTRS:
        by_rest: dict[tuple, list[tuple[int, int]]] = {}
        for i, t in enumerate(texts):
            v = attr_of[t]
            rest = tuple(v[b] for b in ATTRS if b != a)
            by_rest.setdefault(rest, []).append((VALUES[a].index(v[a]), i))
        for members in by_rest.values():
            if len(members) == len(VALUES[a]) or not fixed:
                out[a].append([i for _, i in sorted(members)])
    return out
