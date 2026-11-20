"""Etapa 2 de punta a punta en CPU: Qwen3.5 y VL-JEPA diminutos (arquitectura real, pesos aleatorios).

Cubre lo que no se puede ver sin GPU en el cluster: que LoRA + conector se adjunten donde corresponde
(y solo ahí), que la pérdida baje, que la selección en val guarde `adapter_best` con el criterio de cada
familia, que reanudar tras SIGTERM dé el mismo resultado que no interrumpir, y que el adaptador guardado
se cargue por el camino de `model.adapter_path` (el de scripts/infer.py y scripts/final_test.py).
"""

import json
import signal

import pytest
import torch
from PIL import Image

pytest.importorskip("peft")

from _tiny_vlm import tiny_describer, tiny_vljepa_describer, train_cfg, write_m3di

from vlmfid.data.manifest import to_records
from vlmfid.prompts import render
from vlmfid.train import loop


@pytest.fixture(autouse=True)
def solo_cpu(monkeypatch):
    """Los modelos diminutos viven en CPU. Trainer elige el dispositivo con torch.cuda.is_available(),
    así que en el cluster (con GPU) mandaría los tensores a cuda:0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "m3di"
    recs = {"train": to_records(write_m3di(root, 64, "train")), "val": to_records(write_m3di(root, 8, "val"))}

    def fake_load(split, dataset="m3di", limit=None, subset_seed=0, variant="base", **_):
        return recs[split][:limit] if limit else recs[split]

    monkeypatch.setattr(loop, "load_manifest", fake_load)
    monkeypatch.setattr(loop, "data_root", lambda *a, **k: root)
    return recs


def _run(cfg, describer, runs, stop_after=None):
    tr = loop.Trainer(cfg, describer=describer, run_root=runs)
    if stop_after is not None:
        orig, calls = tr._step, {"n": 0}

        def step(*a, **k):
            out = orig(*a, **k)
            calls["n"] += 1
            if calls["n"] == stop_after:
                loop._STOP.update(flag=True, signum=signal.SIGTERM)
            return out

        tr._step = step
    try:
        return tr.run(), tr
    finally:
        loop._STOP.update(flag=False, signum=None)


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


# ───────────────────────────────────────────────────────────────────────────── VLM (Qwen3.5)
def test_qwen35_entrena_selecciona_y_recarga(data, tmp_path):
    cfg = train_cfg()
    d = tiny_describer()
    rc, tr = _run(cfg, d, tmp_path / "runs")
    assert rc == 0
    rd = tmp_path / "runs" / cfg.exp_id
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["steps"] == 8 and summary["samples_seen"] == 64            # 64 / 8: todo train
    assert summary["trainable_connector"] > 0 and summary["lora_layers_llm"] > 0
    names = [n for n, p in d.model.named_parameters() if p.requires_grad]
    assert all("language_model" in n or "visual.merger.modules_to_save" in n for n in names)
    assert not any("visual.blocks" in n for n in names)                        # encoder visual congelado
    assert {g["name"] for g in tr.optimizer.param_groups} == {"lora", "connector"}
    lrs = {g["name"]: g["initial_lr"] for g in tr.optimizer.param_groups}
    assert lrs["connector"] == pytest.approx(0.1 * lrs["lora"])               # connector_lr_mult

    val = _lines(rd / "val_log.jsonl")
    assert [r["step"] for r in val] == [0, 2, 4, 6, 8]
    assert all(r["selection_metric"] == "attr_mean" and "val_loss" in r for r in val)
    best = json.loads((rd / "best.json").read_text())
    assert best["value"] == pytest.approx(max(r["attr_mean"] for r in val))
    preds0 = _lines(rd / "val_predictions" / "step_00000.jsonl")
    assert {p["reference"] for p in preds0} == {r["caption"] for r in data["val"]}        # caption original
    train = _lines(rd / "train_log.jsonl")
    assert train[-1]["loss"] < train[0]["loss"]
    assert (rd / "adapter_best" / "adapter_config.json").exists()

    # recarga como en inferencia (HFGenerativeDescriber con model.adapter_path)
    from peft import PeftModel

    fresh = tiny_describer()
    fresh.model = PeftModel.from_pretrained(fresh.model, str(rd / "adapter_last")).eval()
    pairs = [(r, Image.open(tmp_path / "m3di" / r["image_path"]).convert("RGB")) for r in data["val"][:2]]
    batch = tr.featurizer.pad(tr.featurizer(pairs))
    d.model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        a = d.model(**batch["inputs"]).logits.float()
        b = fresh.model(**batch["inputs"]).logits.float()
    assert torch.allclose(a, b, atol=0.15, rtol=0.05), (a - b).abs().max()     # conector fp32 vs bf16

    # idempotente: relanzar no vuelve a entrenar
    assert loop.Trainer(cfg, describer=tiny_describer(), run_root=tmp_path / "runs").run() == 0


def test_qwen35_reanudar_es_identico(data, tmp_path):
    cfg = train_cfg()
    rc, ta = _run(cfg, tiny_describer(), tmp_path / "a")
    assert rc == 0
    rc, _ = _run(cfg, tiny_describer(), tmp_path / "b", stop_after=3)
    assert rc == 128 + signal.SIGTERM
    st = json.loads((tmp_path / "b" / cfg.exp_id / "checkpoint" / "state.json").read_text())
    assert st["global_step"] == 3
    rc, tb = _run(cfg, tiny_describer(), tmp_path / "b")
    assert rc == 0
    pa = {n: p for n, p in ta.describer.model.named_parameters() if p.requires_grad}
    pb = {n: p for n, p in tb.describer.model.named_parameters() if p.requires_grad}
    assert pa.keys() == pb.keys()
    for n, a in pa.items():
        assert torch.allclose(a, pb[n], atol=1e-6), n


def test_usa_todo_train_con_ultimo_lote_incompleto(data, tmp_path):
    """64 imágenes con lote efectivo 24 -> 3 pasos (24, 24, 16): ninguna imagen queda fuera."""
    cfg = train_cfg(**{"train.effective_batch_size": 24, "train.micro_batch_size": 8})
    rc, _tr = _run(cfg, tiny_describer(), tmp_path / "runs")
    assert rc == 0
    summary = json.loads((tmp_path / "runs" / cfg.exp_id / "summary.json").read_text())
    assert summary["steps"] == 3 and summary["samples_seen"] == 64 == summary["n_train"]


def test_corrida_corta_completa_su_schedule(data, tmp_path):
    """train.max_steps corta el run y el coseno se completa dentro de la corrida (barrido de LR)."""
    cfg = train_cfg(**{"train.max_steps": 4, "train.hparams_confirmed": "false"})
    assert cfg.exp_id.endswith("_m4")
    rc, _tr = _run(cfg, tiny_describer(), tmp_path / "runs")
    assert rc == 0
    rd = tmp_path / "runs" / cfg.exp_id
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["steps"] == 4 == summary["steps_schedule"] and summary["short_run"]
    train = _lines(rd / "train_log.jsonl")
    assert train[-1]["lr_lora"] < 1e-9 < train[0]["lr_lora"]                  # coseno terminado
    assert [r["step"] for r in _lines(rd / "val_log.jsonl")] == [0, 2, 4]


def test_reanudar_con_otra_config_falla(data, tmp_path):
    cfg = train_cfg()
    _run(cfg, tiny_describer(), tmp_path / "r", stop_after=2)
    with pytest.raises(RuntimeError, match="difiere"):
        loop.Trainer(train_cfg(**{"train.max_grad_norm": 0.5}), describer=tiny_describer(),
                     run_root=tmp_path / "r").run()
    # cambiar solo el micro-lote es seguro (no altera la matemática)
    assert _run(train_cfg(**{"train.micro_batch_size": 2}), tiny_describer(), tmp_path / "r")[0] == 0


def test_micro_lote_no_cambia_el_gradiente(data, tmp_path):
    """Mismo lote efectivo partido en micro-lotes de 8, 4 y 2 -> mismo gradiente acumulado (fp32).
    Es lo que garantiza la normalización por tokens del lote efectivo en `Trainer._step`."""
    import contextlib

    from vlmfid.train.collate import GenerativeFeaturizer

    d = tiny_describer()
    d.model = d.model.float()
    feat = GenerativeFeaturizer(d, "Describe the object.", 128)
    pairs = [(r, Image.open(tmp_path / "m3di" / r["image_path"]).convert("RGB")) for r in data["train"][:8]]
    samples = feat(pairs)
    grads = []
    for mb in (8, 4, 2):
        tr = loop.Trainer(train_cfg(), describer=d, run_root=tmp_path)
        tr.featurizer, tr.micro = feat, mb
        tr.amp = contextlib.nullcontext
        d.model.zero_grad(set_to_none=True)
        for p in d.model.parameters():
            p.requires_grad_(True)
        stats = tr._step(samples, d.model)
        grads.append(({n: p.grad.clone() for n, p in d.model.named_parameters() if p.grad is not None}, stats["loss"]))
    g0, l0 = grads[0]
    for g, loss in grads[1:]:
        assert abs(loss - l0) < 1e-5
        for n in g0:
            assert torch.allclose(g0[n], g[n], atol=1e-6, rtol=1e-4), n


# ───────────────────────────────────────────────────────────────────────────── VL-JEPA (opción A)
def _jepa_cfg(**kw):
    return train_cfg(**{"model": "vljepa", "train.micro_batch_size": 8, "model.batch_size": 4,
                        "train.x_encoder_chunk": 3, **kw})


def _pred(d, img, prompt, query_len):
    """Ŝ_Y por el camino de inferencia (consulta rellena hasta `query_len`), con autocast en CPU."""
    inner = d.model.get_base_model() if hasattr(d.model, "get_base_model") else d.model
    ids, mask = d._tok(d.query_tok, [prompt], query_len)
    pv = d._pixel_values([img])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        return inner.predictor(inner.x_encoder(pv), ids, mask).float()


def test_vljepa_opcion_a(data, tmp_path):
    cfg = _jepa_cfg()
    assert cfg.train.selection_metric == "val_loss"
    d = tiny_vljepa_describer(tmp_path)
    base_before = {n: p.detach().clone() for n, p in d.model.named_parameters()}
    rc, tr = _run(cfg, d, tmp_path / "runs")
    assert rc == 0
    rd = tmp_path / "runs" / cfg.exp_id

    names = [n for n, p in d.model.named_parameters() if p.requires_grad]
    assert any("predictor.layers" in n and "lora_" in n for n in names)          # LoRA en el Predictor
    assert any("y_encoder.model.layers" in n and "lora_" in n for n in names)    # y en el Y-Encoder
    for mod in ("predictor.vis_proj", "predictor.out_proj", "y_encoder.projector"):
        assert any(f"{mod}.modules_to_save" in n for n in names), mod
    assert not any("x_encoder" in n for n in names)
    inner = d.model.get_base_model()
    for n, p in inner.named_parameters():                     # la base congelada no cambió (solo LoRA y copias)
        if n in base_before and "lora_" not in n and "modules_to_save" not in n:
            assert torch.equal(p, base_before[n]), n

    lrs = {g["name"]: g["initial_lr"] for g in tr.optimizer.param_groups}
    assert set(lrs) == {"lora", "lora_y_encoder", "connector"}
    assert lrs["lora_y_encoder"] == pytest.approx(0.05 * lrs["lora"])            # [5] §3.1
    assert lrs["connector"] == pytest.approx(0.1 * lrs["lora"])                  # incluye y_encoder.projector (P9)

    info = json.loads((rd / "status.json").read_text())["model_info"]
    assert info["lora_layers_predictor"] == 2 * 7 and info["lora_layers_y_encoder"] == 1 * 7
    val = _lines(rd / "val_log.jsonl")
    assert [r["step"] for r in val] == [0, 2, 4, 6, 8]
    assert all(r["selection_metric"] == "val_loss" and r["n"] == 8 for r in val)
    best = json.loads((rd / "best.json").read_text())
    assert best["value"] == pytest.approx(min(r["val_loss"] for r in val))      # P7: mínima InfoNCE
    assert min(r["val_loss"] for r in val[1:]) < val[0]["val_loss"]
    assert not (rd / "val_predictions").exists()                                 # sin Y-Decoder no hay texto
    train = _lines(rd / "train_log.jsonl")
    assert all({"loss_p2t", "loss_t2p", "acc_p2t", "dup_pairs"} <= r.keys() for r in train)

    # la consulta sin relleno del entrenamiento da el mismo Ŝ_Y que la de inferencia (relleno hasta 512)
    d.model.eval()
    rec = data["val"][0]
    img = Image.open(tmp_path / "m3di" / rec["image_path"]).convert("RGB")
    batch = tr.featurizer.pad(tr.featurizer([(rec, img)]))
    assert batch["query_ids"].shape[1] == tr.featurizer.query_tokens < 512
    with torch.no_grad():
        pred_train, _ = tr.objective.embeddings(batch)
    pred_infer = _pred(d, img, render(cfg.prompt), d.max_query_len)
    assert torch.nn.functional.cosine_similarity(pred_train.float(), pred_infer).item() > 0.999

    # recarga por el camino de model.adapter_path: LoRA del Y-Encoder y proyecciones viajan con el adaptador
    from peft import PeftModel

    (tmp_path / "f").mkdir()
    fresh = tiny_vljepa_describer(tmp_path / "f")
    fresh.model = PeftModel.from_pretrained(fresh.model, str(rd / "adapter_last")).eval()
    pa, pb = _pred(d, img, "q", 16), _pred(fresh, img, "q", 16)
    assert torch.nn.functional.cosine_similarity(pa, pb).item() > 0.99
    ids, mask = d._tok(d.target_tok, [rec["caption"]], d.max_caption_len)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ya = d.model.get_base_model().y_encoder(ids, mask).float()
        yb = fresh.model.get_base_model().y_encoder(ids, mask).float()
    assert torch.nn.functional.cosine_similarity(ya, yb).item() > 0.99


def test_vljepa_reanudar_es_identico(data, tmp_path):
    cfg = _jepa_cfg()
    for sub in ("a", "b", "c"):
        (tmp_path / sub).mkdir()
    rc, _ = _run(cfg, tiny_vljepa_describer(tmp_path / "a"), tmp_path / "ra")
    assert rc == 0
    rc, _ = _run(cfg, tiny_vljepa_describer(tmp_path / "b"), tmp_path / "rb", stop_after=3)
    assert rc == 128 + signal.SIGTERM
    rc, _ = _run(cfg, tiny_vljepa_describer(tmp_path / "c"), tmp_path / "rb")
    assert rc == 0
    la = _lines(tmp_path / "ra" / cfg.exp_id / "train_log.jsonl")
    lb = _lines(tmp_path / "rb" / cfg.exp_id / "train_log.jsonl")
    assert [r["loss"] for r in la] == pytest.approx([r["loss"] for r in lb], rel=1e-3)


def test_vljepa_x_encoder_por_trozos_no_cambia_la_perdida(data, tmp_path):
    """El X-Encoder congelado en trozos (memoria) da la misma pérdida que de una vez."""
    cfg = _jepa_cfg()
    d = tiny_vljepa_describer(tmp_path)
    tr = loop.Trainer(cfg, describer=d, run_root=tmp_path / "runs")
    tr.root = tmp_path / "m3di"
    tr._setup(data["train"], data["val"])
    pairs = [(r, Image.open(tmp_path / "m3di" / r["image_path"]).convert("RGB")) for r in data["train"][:8]]
    batch = tr.featurizer.pad(tr.featurizer(pairs))
    d.model.eval()
    out = []
    for chunk in (0, 3):
        tr.objective.chunk = chunk
        with torch.no_grad():
            out.append(float(tr.objective.loss(batch)[0]))
    assert out[0] == pytest.approx(out[1], rel=1e-3)
