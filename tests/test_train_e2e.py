"""Etapa 2 de punta a punta en CPU: Qwen3.5 y VL-JEPA diminutos (arquitectura real, pesos aleatorios).

Cubre lo que no se puede ver sin GPU en el cluster: que LoRA + conector se adjunten donde corresponde,
que la pérdida baje, que la selección en val guarde `adapter_best`, que reanudar tras SIGTERM dé
exactamente el mismo resultado que no interrumpir, y que el adaptador guardado se cargue por el
camino de `model.adapter_path` (el que usa scripts/infer.py en test).
"""

import json
import signal

import pytest
import torch
from PIL import Image

pytest.importorskip("peft")
pytest.importorskip("matplotlib")

from _tiny_vlm import tiny_describer, tiny_vljepa_describer, train_cfg, write_m3di

from vlmfid.data.manifest import to_records
from vlmfid.train import loop


@pytest.fixture(autouse=True)
def solo_cpu(monkeypatch):
    """Los modelos diminutos viven en CPU. Trainer y VLJEPADescriber eligen el dispositivo con
    torch.cuda.is_available(), así que en el cluster (con GPU) mandarían los tensores a cuda:0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "m3di"
    recs = {"train": to_records(write_m3di(root, 64, "train")), "val": to_records(write_m3di(root, 8, "val"))}

    def fake_load(split, dataset="m3di", limit=None, subset_seed=0, variant="base", **_):
        return recs[split][:limit] if limit else recs[split]

    monkeypatch.setattr(loop, "load_manifest", fake_load)
    monkeypatch.setattr(loop, "data_root", lambda *a, **k: root)
    monkeypatch.setattr(loop, "CACHE", tmp_path / "cache")
    return recs


def _run(cfg, describer, runs, stop_after=None, monkeypatch=None):
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


def test_qwen35_entrena_selecciona_y_recarga(data, tmp_path):
    cfg = train_cfg()
    d = tiny_describer()
    rc, tr = _run(cfg, d, tmp_path / "runs")
    assert rc == 0
    rd = tmp_path / "runs" / cfg.exp_id
    summary = json.loads((rd / "summary.json").read_text())
    assert summary["steps"] == 8 and summary["samples_seen"] == 64            # 64 // 8, drop_last
    assert summary["trainable_connector"] > 0 and summary["lora_layers"] > 0
    names = [n for n, p in d.model.named_parameters() if p.requires_grad]
    assert all("language_model" in n or "merger.modules_to_save" in n for n in names)
    assert not any("visual.blocks" in n for n in names)                        # encoder visual congelado

    val = _lines(rd / "val_log.jsonl")
    assert [r["step"] for r in val] == [0, 2, 4, 6, 8]
    best = json.loads((rd / "best.json").read_text())
    assert best["value"] == max(r["macro"] for r in val)
    train = _lines(rd / "train_log.jsonl")
    assert train[-1]["loss"] < train[0]["loss"]
    assert (rd / "adapter_best" / "adapter_config.json").exists()
    assert (rd / "val_predictions" / "step_00000.jsonl").exists()

    # recarga como en inferencia (HFGenerativeDescriber con model.adapter_path)
    from peft import PeftModel

    fresh = tiny_describer()
    fresh.model = PeftModel.from_pretrained(fresh.model, str(rd / "adapter_last")).eval()
    pairs = [(r, Image.open(tmp_path / "m3di" / r["image_path"]).convert("RGB")) for r in data["val"][:2]]
    batch = tr.featurizer.pad(tr.featurizer(pairs))
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


def test_reanudar_con_otra_config_falla(data, tmp_path):
    cfg = train_cfg()
    _run(cfg, tiny_describer(), tmp_path / "r", stop_after=2)
    with pytest.raises(RuntimeError, match="difiere"):
        loop.Trainer(train_cfg(**{"train.lr": 1e-3}), describer=tiny_describer(), run_root=tmp_path / "r").run()
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


def test_vljepa_entrena_con_lora_en_el_predictor(data, tmp_path):
    cfg = train_cfg(**{"model": "vljepa", "train.micro_batch_size": 8, "model.batch_size": 4})
    d = tiny_vljepa_describer(tmp_path, data)
    bank_before = d.bank_emb.clone()
    rc, _tr = _run(cfg, d, tmp_path / "runs")
    assert rc == 0
    rd = tmp_path / "runs" / cfg.exp_id
    names = [n for n, p in d.model.named_parameters() if p.requires_grad]
    assert any("predictor.layers" in n and "lora_" in n for n in names)
    assert any("vis_proj.modules_to_save" in n for n in names)
    assert not any("y_encoder" in n or "x_encoder" in n or "out_proj" in n for n in names)
    assert torch.equal(bank_before, d.bank_emb)                 # Y-Encoder congelado: el banco sigue válido
    val = _lines(rd / "val_log.jsonl")
    assert [r["step"] for r in val] == [0, 2, 4, 6, 8]
    train = _lines(rd / "train_log.jsonl")
    assert train[-1]["loss"] < train[0]["loss"]
    preds = _lines(rd / "val_predictions" / "step_00008.jsonl")
    assert all(p["prediction"] in set(d.bank_texts) for p in preds)       # texto por recuperación

    # recarga por el camino de VLJEPADescriber.load (bandera is_loaded_in_4bit incluida si fuera NF4)
    from peft import PeftModel

    (tmp_path / "f").mkdir()
    fresh = tiny_vljepa_describer(tmp_path / "f", data)
    fresh.model = PeftModel.from_pretrained(fresh.model, str(rd / "adapter_last")).eval()
    img = [Image.new("RGB", (64, 64), (200, 30, 30))]
    d.model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ea = d.predict_embeddings(img, ["q"])
        eb = fresh.predict_embeddings(img, ["q"])
    assert torch.nn.functional.cosine_similarity(ea, eb).item() > 0.99
