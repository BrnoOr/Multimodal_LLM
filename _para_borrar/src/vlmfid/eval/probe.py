"""Sondas lineales: ¿qué atributos son linealmente decodificables de la representación del modelo?

Es el diagnóstico previo al ajuste fino (docs/ETAPA2B_REFORMULACION.md, §2). Si una sonda lineal sobre el
modelo CONGELADO ya separa forma y posición, el error zero-shot es de "formato" (el LM no lo verbaliza) y el
ajuste fino debería cerrarlo. Si la sonda también falla, la información no está en la representación y
LoRA sobre el LM difícilmente la cree. Repetida sobre el modelo ajustado (`model.adapter_path`), mide
cuánto cambió la representación.

Representaciones (una por imagen, con el mismo prompt y plantilla de chat que la inferencia):

  VLM autorregresivo (una pasada hacia adelante, sin generar; `output_hidden_states`)
      img_l0       media de los estados de entrada en las posiciones de tokens de imagen
                   = salida del conector (sin contexto del LM)
      img_l<k>     media de los estados de la capa k del LM en las posiciones de imagen
      last_l<k>    estado de la capa k en la última posición del prompt (de donde sale la 1.ª palabra)
                   con k ∈ round(f·L), f ∈ probe.layers (por defecto 0,5 · 0,75 · 1)
  VL-JEPA
      vis_mean     media de los tokens del X-Encoder (V-JEPA 2)
      pred         ŝ_Y, la salida del predictor (lo que se compara contra el banco)

La media de los tokens de imagen pierde buena parte de la disposición espacial: que `img_*` falle en
posición y `last_*` acierte indica que el LM sí integra la posición. Es parte de lo que se reporta.

Clasificador: regresión logística multinomial (L-BFGS, lote completo) sobre rasgos estandarizados; la
regularización L2 se elige en un 20 % retenido de las imágenes de sonda-train y luego se reajusta con
todas. Se evalúa en imágenes de val (test no se toca). Atributos: forma (7), x (3), y (3), posición
conjunta (9) y color básico (8, categoría del tono latente, la misma verdad que la evaluación).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .attributes import COLOR_CATS, hue_to_cat

PROBE_DEFAULTS = {
    "train_split": "train",
    "train_limit": 10_000,
    "test_split": "val",
    "test_limit": 2_000,
    "layers": [0.5, 0.75, 1.0],
    "batch_size": None,              # null -> model.batch_size // 2 (output_hidden_states usa más memoria)
    "weight_decays": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "max_iter": 300,
    "tag": None,                     # sufijo del exp_id (p. ej. ft_tcan para un adaptador)
}
ATTRS = {"shape": 7, "x": 3, "y": 3, "pos": 9, "color": 8}


def probe_labels(records: list[dict]) -> dict[str, np.ndarray]:
    out = {a: [] for a in ATTRS}
    for r in records:
        li = r["latents_image"]
        x, y = int(li["object_xpos"]), int(li["object_ypos"])
        out["shape"].append(int(li["object_shape"]))
        out["x"].append(x)
        out["y"].append(y)
        out["pos"].append(3 * y + x)
        out["color"].append(COLOR_CATS.index(hue_to_cat(li["object_color"])))
    return {a: np.asarray(v, dtype=np.int64) for a, v in out.items()}


# ─────────────────────────────────────────────────────────────── rasgos
def _image_token_id(model) -> int | None:
    cfg = getattr(model, "config", None)
    for k in ("image_token_id", "image_token_index"):
        v = getattr(cfg, k, None)
        if v is not None:
            return int(v)
    return None


@torch.inference_mode()
def hf_features(describer, images, prompt: str, layer_fracs) -> dict[str, torch.Tensor]:
    tkw = describer.chat_template_kwargs()
    text = describer.processor.apply_chat_template(describer.build_messages(prompt), add_generation_prompt=True,
                                                   tokenize=False, **tkw)
    inputs = describer.processor(text=[text] * len(images), images=images, padding=True, return_tensors="pt")
    model = describer.model
    dev = model.device
    inputs = {k: (v.to(dev, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(dev))
              for k, v in inputs.items()}
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    hs = out.hidden_states
    n_layers = len(hs) - 1
    layers = sorted({max(1, min(n_layers, round(float(f) * n_layers))) for f in layer_fracs})
    feats = {}
    img_id = _image_token_id(model)
    mask = (inputs["input_ids"] == img_id) if img_id is not None else None
    if mask is not None and bool(mask.any(dim=1).all()):
        w = mask.unsqueeze(-1).float()
        for k in [0] + layers:
            feats[f"img_l{k}"] = (hs[k].float() * w).sum(1) / w.sum(1)
    for k in layers:
        feats[f"last_l{k}"] = hs[k][:, -1].float()              # relleno a la izquierda: -1 es el prompt
    return feats


@torch.inference_mode()
def jepa_features(describer, images, prompt: str) -> dict[str, torch.Tensor]:
    model = describer.model
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    frames = torch.stack([describer.transform(im) for im in images])
    pv = frames.unsqueeze(1).repeat(1, describer.num_frames, 1, 1, 1).to(describer.device, dtype=torch.bfloat16)
    q = [describer.cfg.get("query_override") or prompt] * len(images)
    ids, mask = describer._tok(describer.query_tok, q, describer.max_query_len)
    with torch.autocast(describer.device.type, dtype=torch.bfloat16, enabled=describer.device.type == "cuda"):
        vis = inner.x_encoder(pv)
        vis_t = vis[0] if isinstance(vis, (tuple, list)) else vis
        pred = inner.predictor(vis, ids, mask)
    return {"vis_mean": vis_t.float().mean(1), "pred": pred.float()}


def extract_features(describer, loader, prompt: str, layer_fracs, log=print) -> tuple[list[str], dict[str, np.ndarray]]:
    ids, chunks, n = [], {}, 0
    for recs, imgs in loader:
        f = (jepa_features(describer, imgs, prompt) if describer.family == "vljepa"
             else hf_features(describer, imgs, prompt, layer_fracs))
        for k, v in f.items():
            chunks.setdefault(k, []).append(v.cpu().numpy().astype(np.float32))
        ids += [r["id"] for r in recs]
        n += len(recs)
        if n % (20 * len(recs)) < len(recs):
            log(f"[probe] rasgos: {n} imágenes")
    return ids, {k: np.concatenate(v) for k, v in chunks.items()}


# ───────────────────────────────────────────────────────── clasificador
def _fit(X: torch.Tensor, y: torch.Tensor, n_classes: int, wd: float, max_iter: int):
    W = torch.zeros(X.shape[1], n_classes, device=X.device, requires_grad=True)
    b = torch.zeros(n_classes, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + wd * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def _acc(X, y, W, b) -> float:
    return float(((X @ W + b).argmax(1) == y).float().mean())


def linear_probe(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, n_classes: int,
                 weight_decays, max_iter: int = 300, seed: int = 0, device: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Xtr = torch.tensor((X_tr - mu) / sd, device=dev)
    Xte = torch.tensor((X_te - mu) / sd, device=dev)
    ytr, yte = torch.tensor(y_tr, device=dev), torch.tensor(y_te, device=dev)
    perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(seed)).to(dev)
    cut = int(0.8 * len(Xtr))
    fit_i, hold_i = perm[:cut], perm[cut:]
    sel = []
    for wd in weight_decays:
        W, b = _fit(Xtr[fit_i], ytr[fit_i], n_classes, float(wd), max_iter)
        sel.append((_acc(Xtr[hold_i], ytr[hold_i], W, b), float(wd)))
    hold_acc, best_wd = max(sel, key=lambda t: (t[0], t[1]))      # empate -> más regularización
    W, b = _fit(Xtr, ytr, n_classes, best_wd, max_iter)
    majority = int(np.bincount(y_tr, minlength=n_classes).argmax())
    return {"acc": _acc(Xte, yte, W, b), "acc_train": _acc(Xtr, ytr, W, b), "acc_holdout": hold_acc,
            "weight_decay": best_wd, "majority_acc": float((y_te == majority).mean()),
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "dim": int(X_tr.shape[1])}
