"""Bucle de entrenamiento de la etapa 2 (QLoRA + conector), reanudable y con selección en val.

Diseñado para el cluster compartido, con las mismas garantías que `scripts/infer.py`:
  * Reanudable: cada `train.save_every` pasos se guarda un checkpoint atómico (parámetros entrenables,
    estado del optimizador y del schedule, RNG y paso). Relanzar el mismo comando continúa desde ahí, con
    el mismo orden de datos (vlmfid.train.data).
  * Idempotente: si el run ya terminó, sale de inmediato (útil en colas).
  * Apagado limpio: SIGTERM/SIGINT terminan el paso en curso, guardan y salen con 143/130.
  * OOM en los VLM: se descarta el paso, se parte el micro-lote a la mitad y se repite con las mismas
    muestras; la normalización por tokens del lote efectivo deja el gradiente igual. VL-JEPA no se parte
    (InfoNCE depende del lote completo).

Selección del checkpoint en val (docs/metodologia.md §2.3):
    VLM      `attr_mean` (o el atributo de `train.selection_metric`) del texto generado, máximo
    VL-JEPA  InfoNCE bidireccional en val, mínima (P7)
Test no se lee aquí: la evaluación final es `scripts/final_test.py`, una sola vez.

Corridas cortas (`train.max_steps`, humo y barrido de LR): cortan el entrenamiento Y el schedule, que se
calcula sobre `max_steps` (el calentamiento y el coseno se completan dentro de la corrida corta).

runs/<exp_id>/
    config.yaml, env.jsonl, status.json     como en inferencia (vlmfid.tracking.RunTracker)
    train_log.jsonl                         una línea cada `log_every` pasos
    val_log.jsonl                           una línea por evaluación (paso 0, cada eval_every, último)
    val_predictions/step_XXXXX.jsonl        respuestas en val de cada evaluación (solo VLM)
    adapter_best/                           adaptador PEFT del mejor paso en val  -> model.adapter_path
    adapter_last/                           adaptador del último paso
    best.json                               mejor paso y sus métricas
    checkpoint/                             estado para reanudar (se puede borrar al terminar)
    summary.json                            resumen final: mejor paso, entrenables, tiempos
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..config import to_yaml
from ..data.manifest import ImageRecordDataset, data_root, load_manifest
from ..eval.attributes import json_mode_for
from ..models import build_describer
from ..paths import RUNS
from ..prompts import render
from ..tracking import RunTracker
from .collate import GenerativeFeaturizer
from .config import strip_resume_safe
from .data import EpochSampler, steps_per_epoch
from .evaluate import generate_and_score
from .jepa import JEPAObjective
from .lora import attach_lora, load_trainable_state, param_groups, set_train_mode, trainable_state
from .objectives import lm_loss

_STOP = {"flag": False, "signum": None}


def _handle(signum, _frame):
    _STOP.update(flag=True, signum=signum)
    print(f"\n[train] señal {signal.Signals(signum).name}: se termina el paso en curso y se guarda.", flush=True)


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:d}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def lr_lambda_factory(total_steps: int, warmup_steps: int, schedule: str, min_ratio: float):
    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def build_optimizer(groups: list[dict], t) -> torch.optim.Optimizer:
    kw = {"betas": tuple(t.betas), "eps": float(t.eps), "weight_decay": float(t.weight_decay)}
    if t.optimizer == "paged_adamw":
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW(groups, **kw)
    return torch.optim.AdamW(groups, **kw)


def _atomic_dir(final: Path, writer) -> None:
    tmp, old = final.with_name(final.name + ".tmp"), final.with_name(final.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    writer(tmp)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Trainer:
    def __init__(self, cfg: DictConfig, describer=None, run_root: Path | None = None):
        self.cfg = cfg
        self.describer = describer                     # inyectable en tests (modelo pequeño en CPU)
        self.dir = Path(run_root or RUNS) / cfg.exp_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jepa = cfg.objective.kind == "infonce"
        self.objective: JEPAObjective | None = None    # solo VL-JEPA; se crea en _setup
        self.prompt_text = render(cfg.prompt)
        self.micro = int(cfg.train.micro_batch_size)
        self.json_mode = json_mode_for(cfg)
        self.minimize = cfg.train.selection_metric == "val_loss"

    # ------------------------------------------------------------------ utilidades
    def amp(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def _frozen_modules(self) -> list[str]:
        if self.jepa:
            return ["x_encoder"]
        conn = set(self.cfg.lora.get("connector_modules") or [])
        return [m for m in (self.cfg.model.get("vision_modules") or []) if m not in conn]

    def _seed(self) -> None:
        s = int(self.cfg.seed)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    def _check_resume_compatible(self) -> None:
        old = self.dir / "config.yaml"
        if not old.exists() or not (self.dir / "checkpoint").exists():
            return
        prev = OmegaConf.to_container(OmegaConf.load(old), resolve=True)
        curr = OmegaConf.to_container(self.cfg, resolve=True)
        if strip_resume_safe(prev) != strip_resume_safe(curr):
            raise RuntimeError(f"La configuración difiere de la guardada en {old} y hay un checkpoint a medias. "
                               "Usa otro exp_id o borra runs/<exp_id> para empezar de cero.")

    def _records(self):
        """train: split completo, sin límite. val: las mismas `val_limit` imágenes para los 4 modelos."""
        d = self.cfg.data
        train = load_manifest(d.train_split, d.dataset, None, d.subset_seed, variant=d.variant)
        val = load_manifest(d.val_split, d.dataset, d.get("val_limit"), d.subset_seed, variant=d.variant)
        return train, val

    # ------------------------------------------------------------------- forward
    def _to_device(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.device)
            else:
                out[k] = v
        return out

    def _step(self, samples: list, model) -> dict:
        """Un paso de optimización sobre `samples` (= un lote efectivo): solo forward y backward."""
        if self.jepa:
            loss, stats = self.objective.loss(self.featurizer.pad(samples))
            loss.backward()
            return {"loss": float(loss.detach()), "tokens": len(samples), **stats}
        n_eff = sum(len(s["target"]) for s in samples)             # tokens objetivo del lote efectivo
        while True:
            try:
                total = 0.0
                for i in range(0, len(samples), self.micro):
                    batch = self.featurizer.pad(samples[i:i + self.micro])
                    batch["inputs"] = self._to_device(batch["inputs"])
                    with self.amp():
                        loss, loss_sum, _ = lm_loss(model, batch, n_eff)
                    loss.backward()
                    total += loss_sum
                return {"loss": total / n_eff, "tokens": n_eff}
            except torch.cuda.OutOfMemoryError:
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if self.micro == 1:
                    raise
                self.micro = max(1, self.micro // 2)
                print(f"[train] OOM: micro-lote -> {self.micro}; se repite el paso con las mismas muestras", flush=True)

    # ------------------------------------------------------------- evaluación
    @torch.no_grad()
    def _val_loss_lm(self, model, val_records) -> float:
        """Entropía cruzada por token en val (teacher forcing). Solo se registra; no elige el paso."""
        loader = DataLoader(ImageRecordDataset(val_records, self.root), batch_size=self.micro, num_workers=2,
                            collate_fn=self.featurizer, shuffle=False, generator=torch.Generator())
        tot, n = 0.0, 0
        for samples in loader:
            batch = self.featurizer.pad(samples)
            batch["inputs"] = self._to_device(batch["inputs"])
            with self.amp():
                _, s, k = lm_loss(model, batch, 1)
            tot, n = tot + s, n + k
        return tot / max(n, 1)

    @torch.no_grad()
    def _val_jepa(self, val_records) -> dict:
        """InfoNCE bidireccional en val: mismos lotes (orden fijo, tamaño del lote efectivo) en cada evaluación."""
        bs = int(self.cfg.train.effective_batch_size)
        loader = DataLoader(ImageRecordDataset(val_records, self.root), batch_size=bs, num_workers=2,
                            collate_fn=self.featurizer, shuffle=False, generator=torch.Generator())
        agg: dict[str, float] = {}
        n = 0
        for samples in loader:
            loss, stats = self.objective.loss(self.featurizer.pad(samples))
            b = len(samples)
            for k, v in ({"val_loss": float(loss)} | stats).items():
                agg[k] = agg.get(k, 0.0) + float(v) * (b if not k.startswith("dup_") else 1)
            n += b
        return {k: (v / n if not k.startswith("dup_") else v) for k, v in agg.items()} | {"n": n}

    def evaluate(self, model, val_records, step: int, state: dict) -> dict:
        t0 = time.time()
        model.eval()
        if self.jepa:
            metrics, preds = self._val_jepa(val_records), None
        else:
            val_loss = self._val_loss_lm(model, val_records)
            bs = int(self.cfg.model.get("batch_size", 16))
            metrics, preds = generate_and_score(self.describer, val_records, self.root, self.prompt_text, bs,
                                                self.amp, self.json_mode)
            metrics = {"val_loss": val_loss} | metrics
        set_train_mode(model, self._frozen_modules())
        if preds is not None:
            pred_dir = self.dir / "val_predictions"
            pred_dir.mkdir(exist_ok=True)
            with (pred_dir / f"step_{step:05d}.jsonl").open("w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        metric = self.cfg.train.selection_metric
        value = float(metrics[metric])
        best = state.get("best")
        better = best is None or (value < best["value"] if self.minimize else value > best["value"])
        row = {"step": step, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()},
               "selection_metric": metric, "is_best": better, "eval_seconds": round(time.time() - t0, 1)}
        if better:
            state["best"] = {"step": step, "value": value, "metrics": row}
            _atomic_dir(self.dir / "adapter_best", lambda d: model.save_pretrained(d))
            (self.dir / "best.json").write_text(json.dumps(state["best"], indent=2, ensure_ascii=False))
        state["last_eval_step"] = step
        _append_jsonl(self.dir / "val_log.jsonl", row)
        if self.jepa:
            detail = (f"infonce={metrics['val_loss']:.4f} (p→t {metrics['loss_p2t']:.4f}, t→p {metrics['loss_t2p']:.4f}) "
                      f"top1 p→t={metrics['acc_p2t']:.3f} t→p={metrics['acc_t2p']:.3f}")
        else:
            detail = (f"attr_mean={metrics['attr_mean']:.4f} forma={metrics['shape']:.3f} x={metrics['x']:.3f} "
                      f"y={metrics['y']:.3f} color={metrics['color']:.3f} val_loss={metrics['val_loss']:.4f} "
                      f"cortadas={metrics['truncated']:.3f}")
        sample = f" | {preds[0]['prediction'][:80]!r}" if preds else ""
        print(f"[val] paso {step}: {detail}{'  *mejor*' if better else ''}  ({row['eval_seconds']} s){sample}",
              flush=True)
        return row

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, model, state: dict) -> None:
        def write(d: Path):
            torch.save(trainable_state(model), d / "trainable.pt")
            torch.save(self.optimizer.state_dict(), d / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), d / "scheduler.pt")
            rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
            if torch.cuda.is_available():
                rng["cuda"] = torch.cuda.get_rng_state_all()
            torch.save(rng, d / "rng.pt")
            (d / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
        _atomic_dir(self.dir / "checkpoint", write)

    def load_checkpoint(self, model) -> dict:
        d = self.dir / "checkpoint"
        load_trainable_state(model, torch.load(d / "trainable.pt", map_location="cpu"))
        self.optimizer.load_state_dict(torch.load(d / "optimizer.pt", map_location="cpu", weights_only=False))
        self.scheduler.load_state_dict(torch.load(d / "scheduler.pt", map_location="cpu", weights_only=False))
        rng = torch.load(d / "rng.pt", map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        return json.loads((d / "state.json").read_text())

    # ---------------------------------------------------------------------- run
    def run(self) -> int:
        cfg, t = self.cfg, self.cfg.train
        summary_path = self.dir / "summary.json"
        if summary_path.exists() and (self.dir / "adapter_best").exists():
            print(f"[train] {cfg.exp_id}: entrenamiento completo, nada pendiente ({summary_path}).")
            return 0
        self._check_resume_compatible()
        self._seed()
        train_records, val_records = self._records()
        self.root = data_root(cfg.data.variant, cfg.data.get("root"))
        n_steps_epoch = steps_per_epoch(len(train_records), int(t.effective_batch_size), bool(t.drop_last))
        total_steps = n_steps_epoch * int(t.epochs)
        max_steps = t.get("max_steps")
        schedule_steps = min(total_steps, int(max_steps)) if max_steps else total_steps
        warmup = round(float(t.warmup_ratio) * schedule_steps)
        print(to_yaml(cfg))
        print(f"[train] exp_id={cfg.exp_id}  train={len(train_records)}  val={len(val_records)}  "
              f"pasos/época={n_steps_epoch}  pasos={schedule_steps}"
              + (f" (corrida corta de {total_steps})" if schedule_steps < total_steps else "")
              + f"  calentamiento={warmup}", flush=True)

        tracker = RunTracker(self.dir, cfg)
        tracker.start()
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)

        try:
            model, info, lora_info = self._setup(train_records, val_records)
        except Exception as e:      # sin esto el run queda en "running" para siempre en status.json
            tracker.finish("failed", error=f"{type(e).__name__}: {e}", step=0)
            raise
        tracker.update(model_info=info, total_steps=schedule_steps)
        print(f"[train] modelo listo: {json.dumps(info, default=str)}", flush=True)

        groups = param_groups(model, cfg)
        print("[train] grupos de LR: " + ", ".join(f"{g['name']}={g['lr']:.2e} "
                                                   f"({sum(p.numel() for p in g['params']):,} parámetros)"
                                                   for g in groups), flush=True)
        self.optimizer = build_optimizer(groups, t)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda_factory(schedule_steps, warmup, str(t.schedule), float(t.min_lr_ratio)))

        state = {"global_step": 0, "best": None, "last_eval_step": None, "train_seconds": 0.0,
                 "eval_seconds": 0.0, "micro_batch_size": self.micro, "samples_seen": 0}
        if (self.dir / "checkpoint").exists():
            state = self.load_checkpoint(model)
            if not self.jepa:
                self.micro = min(self.micro, int(state.get("micro_batch_size") or self.micro))
            print(f"[train] reanudando desde el paso {state['global_step']} "
                  f"(mejor: {state['best'] and state['best']['step']})", flush=True)
        set_train_mode(model, self._frozen_modules())
        return self._train_loop(model, state, train_records, val_records, n_steps_epoch, schedule_steps,
                                lora_info, tracker, summary_path)

    def _setup(self, train_records, val_records):
        """Carga el modelo, adjunta LoRA y arma el featurizer. Devuelve (modelo, info, lora_info)."""
        cfg, t = self.cfg, self.cfg.train
        t_load = time.time()
        if self.describer is None:
            self.describer = build_describer(cfg.model)
            self.describer.load()
        lora_info = attach_lora(self.describer, cfg)
        model = self.describer.model
        info = self.describer.info() | lora_info | {"load_seconds": round(time.time() - t_load, 1),
                                                    "n_train": len(train_records), "n_val": len(val_records)}
        if self.jepa:
            self.objective = JEPAObjective(cfg, self.describer, self.device, self.amp)
            self.featurizer = self.objective.featurizer(self.prompt_text)
            # falla temprano si algún caption excede max_caption_len (todos los distintos de train y val)
            self.featurizer.caption_ids(sorted({r["caption"] for r in train_records + val_records}))
            info |= {"query_tokens": self.featurizer.query_tokens, **self.objective.info()}
        else:
            self.featurizer = GenerativeFeaturizer(self.describer, self.prompt_text, int(t.max_target_tokens))
            info |= {"turn_end": self.featurizer.turn_end}
            # falla temprano si algún objetivo excede max_target_tokens (todos los captions distintos)
            for txt in {r["caption"] for r in train_records + val_records}:
                self.featurizer.target_ids(txt)
        return model, info, lora_info

    def _train_loop(self, model, state, train_records, val_records, n_steps_epoch, total_steps, lora_info,
                    tracker, summary_path) -> int:
        cfg, t = self.cfg, self.cfg.train
        if t.eval_at_start and state["global_step"] == 0 and state["last_eval_step"] is None:
            t0 = time.time()
            self.evaluate(model, val_records, 0, state)
            state["eval_seconds"] += time.time() - t0

        accum = int(t.effective_batch_size) // int(t.micro_batch_size)
        status, t_run = "completed", time.time()
        win: dict[str, float] = {}
        log_tok, log_n, t_log = 0, 0, time.time()
        trainable = [p for p in model.parameters() if p.requires_grad]
        try:
            while state["global_step"] < total_steps:
                epoch, step_in_epoch = divmod(state["global_step"], n_steps_epoch)
                sampler = EpochSampler(len(train_records), int(t.effective_batch_size), int(cfg.seed), epoch,
                                       step_in_epoch, bool(t.drop_last))
                loader = DataLoader(ImageRecordDataset(train_records, self.root), batch_size=int(t.micro_batch_size),
                                    sampler=sampler, num_workers=int(cfg.data.num_workers),
                                    collate_fn=self.featurizer, persistent_workers=False,
                                    pin_memory=self.device.type == "cuda",
                                    # generador propio: crear un DataLoader consume el RNG global, y al
                                    # reanudar desfasaría las máscaras de dropout de LoRA
                                    generator=torch.Generator().manual_seed(int(cfg.seed) + epoch))
                it = iter(loader)
                for _ in range(step_in_epoch, n_steps_epoch):
                    # el último paso de la época puede traer menos de `accum` micro-lotes (drop_last=False)
                    samples = [s for mb in itertools.islice(it, accum) for s in mb]
                    t_step = time.time()
                    stats = self._step(samples, model)
                    gnorm = torch.nn.utils.clip_grad_norm_(trainable, float(t.max_grad_norm))
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    state["global_step"] += 1
                    state["samples_seen"] = state.get("samples_seen", 0) + len(samples)
                    state["train_seconds"] += time.time() - t_step
                    step = state["global_step"]
                    w = stats.pop("tokens")
                    for k, v in stats.items():                    # promedio ponderado en la ventana de log
                        win[k] = win.get(k, 0.0) + float(v) * (1 if k.startswith("dup_") else w)
                    log_tok += w
                    log_n += len(samples)

                    if step % int(t.log_every) == 0 or step == total_steps:
                        el = time.time() - t_log
                        rate = log_n / max(el, 1e-6)
                        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                        lrs = {g["name"]: g["lr"] for g in self.optimizer.param_groups}
                        means = {k: round(v if k.startswith("dup_") else v / max(log_tok, 1), 5) for k, v in win.items()}
                        row = {"step": step, "epoch": epoch, **means, "grad_norm": round(float(gnorm), 4),
                               **{f"lr_{k}": v for k, v in lrs.items()}, "samples_per_s": round(rate, 2),
                               "micro_batch": self.micro, "peak_mem_gb": round(mem, 2)}
                        _append_jsonl(self.dir / "train_log.jsonl", row)
                        eta = (total_steps - step) * int(t.effective_batch_size) / max(rate, 1e-6)
                        print(f"[train] {step}/{total_steps}  loss={row['loss']:.4f}  |g|={row['grad_norm']:.3f}  "
                              f"lr={lrs['lora']:.2e}  {rate:.1f} img/s  ETA {_fmt(eta)}  pico {mem:.1f} GB", flush=True)
                        tracker.update(step=step, total_steps=total_steps, loss=row["loss"], img_per_s=row["samples_per_s"])
                        win, log_tok, log_n, t_log = {}, 0, 0, time.time()

                    is_last = step == total_steps
                    if step % int(t.eval_every) == 0 or is_last:
                        t0 = time.time()
                        self.evaluate(model, val_records, step, state)
                        state["eval_seconds"] += time.time() - t0
                        t_log += time.time() - t0          # img/s mide solo entrenamiento
                    state["micro_batch_size"] = self.micro
                    if step % int(t.save_every) == 0 or is_last or _STOP["flag"]:
                        self.save_checkpoint(model, state)
                    if _STOP["flag"]:
                        status = "interrupted"
                        break
                    if is_last:
                        break
                del it, loader
                if status == "interrupted":
                    break
        except Exception as e:
            tracker.finish("failed", error=f"{type(e).__name__}: {e}", step=state["global_step"])
            raise

        if status == "interrupted":
            tracker.finish(status, step=state["global_step"])
            print(f"[train] interrumpido en el paso {state['global_step']}; relanzar el mismo comando reanuda.")
            return 128 + (_STOP["signum"] or signal.SIGTERM)

        _atomic_dir(self.dir / "adapter_last", lambda d: model.save_pretrained(d))
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        summary = {
            "exp_id": cfg.exp_id, "model": cfg.model.name, "prompt": cfg.prompt.id, "quant": cfg.model.quant,
            "seed": int(cfg.seed), "steps": state["global_step"], "steps_schedule": total_steps,
            "short_run": bool(t.get("max_steps")), "epochs": int(t.epochs), "n_train": len(train_records),
            "samples_seen": state.get("samples_seen", 0), "effective_batch_size": int(t.effective_batch_size),
            "lr": float(t.lr), "selection_metric": cfg.train.selection_metric, "best": state["best"],
            "adapter_best": str(self.dir / "adapter_best"), "hparams_confirmed": bool(t.hparams_confirmed),
            "train_hours": round(state["train_seconds"] / 3600, 3), "eval_hours": round(state["eval_seconds"] / 3600, 3),
            "gpu": gpu, "micro_batch_final": self.micro, **lora_info,
            "wall_hours_this_attempt": round((time.time() - t_run) / 3600, 3),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        tracker.finish("completed", step=state["global_step"], best_step=state["best"]["step"],
                       best_value=state["best"]["value"])
        print(f"[train] completo. Mejor paso {state['best']['step']} "
              f"({cfg.train.selection_metric}={state['best']['value']:.4f}) -> {self.dir / 'adapter_best'}")
        return 0


def train(cfg: DictConfig) -> int:
    return Trainer(cfg).run()
