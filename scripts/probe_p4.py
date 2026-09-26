"""Prueba de prompts alternativos a p4 (8 imágenes x 4 modelos x 2 variantes) y resumen.

Uso en el cluster, desde la raíz del repo:
    CUDA_VISIBLE_DEVICES=1 uv run --no-sync python probe_p4.py
No modifica configs/: los runs quedan en runs/probe_<variante>_<modelo>_n8/.
"""

import glob
import json
import re
import runpy

import torch

from vlmfid.config import compose

VARIANTES = {
    # plantilla con marcadores: no hay ningún valor real que copiar
    "p4t_template": (
        'Fill in this template with what you see in the image: '
        'A "<color>" <object> is at the <vertical>-<horizontal> of the image. '
        '<color> is the color name of the object, <object> is what the object is, '
        '<vertical> is top, mid or bottom, and <horizontal> is left, center or right. '
        'Keep the color name in double quotes. Answer with the completed sentence only.'
    ),
    # tres ejemplos distintos entre sí: si copia, se nota y el sesgo no cae en un solo valor
    "p4m_multiexample": (
        'Describe the object in the image with one sentence in the same format as these examples: '
        'A "tab:red" dragon is at the top-left of the image. '
        'A "gold" teapot is at the mid-center of the image. '
        'A "xkcd:light blue" cow is at the bottom-right of the image. '
        'Put the color name in double quotes. Describe only the object in this image.'
    ),
}
EJEMPLOS = {
    "p4_dataset_format": ['"tab:blue" horse'],
    "p4t_template": ["<color>", "<object>", "<vertical>", "<horizontal>"],
    "p4m_multiexample": ['"tab:red" dragon', '"gold" teapot', '"xkcd:light blue" cow'],
}
MODELOS = ["qwen35", "qwen35_instruct", "internvl3", "llava_ov"]

infer = runpy.run_path("scripts/infer.py")
for variante, texto in VARIANTES.items():
    for m in MODELOS:
        cfg = compose([f"model={m}", "prompt=p4_dataset_format", "data.limit=8"])
        cfg.prompt.id = variante
        cfg.prompt.text = texto
        cfg.exp_id = f"probe_{variante}_{cfg.model.name}_n8"
        print(f"\n######## {cfg.exp_id}", flush=True)
        infer["run"](cfg)
        torch.cuda.empty_cache()

# ------------------------------------------------------------------ resumen
FORMAS = ["armadillo", "cow", "dragon", "hare", "head", "horse", "teapot"]
POS = re.compile(r"\b(top|mid|middle|bottom)[- ](left|center|centre|right)\b")


def forma(t):
    t = t.lower()
    return next((f for f in FORMAS if re.search(rf"\b{f}s?\b", t)), None)


def pos(t):
    m = POS.search(t.lower())
    if not m:
        return None
    v = "mid" if m.group(1) == "middle" else m.group(1)
    return f"{v}-{m.group(2).replace('centre', 'center')}"


archivos = sorted(glob.glob("runs/probe_p4*_n8/predictions.jsonl"))
archivos += sorted(glob.glob("runs/zeroshot_*p4_dataset_format_test_n*/predictions.jsonl"))
print(f"\n{'run':58s} {'copia':>6s} {'distintas':>9s} {'forma':>6s} {'pos':>6s}")
print("azar aprox.: forma 1/7 = 14 %, posición 1/9 = 11 %")
for f in archivos:
    run = f.split("/")[1]
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        continue
    clave = next((k for k in EJEMPLOS if k in run), "p4_dataset_format")
    preds = [r["prediction"] for r in rows]
    copia = sum(any(e in p for e in EJEMPLOS[clave]) for p in preds) / len(preds)
    f_ok = sum(forma(p) is not None and forma(p) == forma(r["reference"]) for p, r in zip(preds, rows))
    p_ok = sum(pos(p) is not None and pos(p) == pos(r["reference"]) for p, r in zip(preds, rows))
    n = len(rows)
    print(f"{run:58s} {copia:6.0%} {len(set(preds)):>5d}/{n:<4d} {f_ok / n:6.0%} {p_ok / n:6.0%}")

print("\nEjemplos (primeras 2 predicciones de cada prueba):")
for f in sorted(glob.glob("runs/probe_p4*_n8/predictions.jsonl")):
    rows = [json.loads(l) for l in open(f)][:2]
    print(f"\n{f.split('/')[1]}")
    for r in rows:
        print("  PRED:", r["prediction"][:150].replace("\n", " "))
        print("  REF :", r["reference"])
