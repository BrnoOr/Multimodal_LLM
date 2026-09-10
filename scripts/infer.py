"""Etapa 1: inferencia sin ajuste. Genera descripciones y las persiste como datos.

Las predicciones se guardan en JSONL, no se puntuan aqui: asi las metricas se
recalculan sin re-inferir cuando el extractor de atributos mejore.

Uso:
    # un modelo, un prompt
    uv run python scripts/infer.py --model llava --prompt p0_minimal -n 1000

    # barrido completo modelo x prompt
    uv run python scripts/infer.py --model llava qwen --prompt all -n 1000

    # subconjunto reproducible: mismas escenas para todos los modelos
    uv run python scripts/infer.py --model llava --prompt all --manifest data/manifests/m3di_val.parquet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlmfid.models.base import GenConfig, ModelSpec, build  # noqa: E402

MODELS = {
    "llava": ModelSpec(name="llava-1.5-7b", hf_id="llava-hf/llava-1.5-7b-hf"),
    "qwen": ModelSpec(name="qwen2.5-vl-7b", hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
                      extra={"min_visual_tokens": 64, "max_visual_tokens": 256}),
}


def stratified_sample(df: pl.DataFrame, n: int, seed: int = 0) -> pl.DataFrame:
    """Subconjunto estratificado por forma x posicion: evita que el azar sesgue la
    comparacion entre modelos. El mismo seed da el mismo subconjunto siempre."""
    if n >= df.height:
        return df
    per = max(1, n // (df["object_shape"].n_unique() * 9))
    out = (
        df.with_columns(
            (pl.col("object_shape").cast(str) + "_" +
             pl.col("object_xpos").cast(str) + "_" +
             pl.col("object_ypos").cast(str)).alias("_stratum")
        )
        .filter(pl.int_range(pl.len()).shuffle(seed=seed).over("_stratum") < per)
        .drop("_stratum")
    )
    return out.head(n) if out.height > n else out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "n/a"


def run_one(model_key: str, prompt_id: str, prompt_text: str, df: pl.DataFrame,
            batch_size: int, cfg: GenConfig, out_root: Path, quant: str) -> Path:
    exp_id = f"e1_{model_key}_{prompt_id}" + ("" if quant == "nf4" else f"_{quant}")
    out_dir = out_root / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = MODELS[model_key]
    spec.quantization = quant
    print(f"\n=== {exp_id} ===")
    t0 = time.time()
    model = build(spec)
    load_s = time.time() - t0
    print(f"cargado en {load_s:.0f}s | {model.memory_footprint_gib():.2f} GiB")

    rows = df.to_dicts()
    pred_path = out_dir / "predictions.jsonl"
    t0 = time.time()
    with pred_path.open("w", encoding="utf-8") as fh:
        for i in tqdm(range(0, len(rows), batch_size), desc=exp_id, unit="batch"):
            chunk = rows[i:i + batch_size]
            imgs = [Image.open(r["image_path"]).convert("RGB") for r in chunk]
            try:
                texts = model.describe(imgs, prompt_text, cfg)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"\nOOM con batch={batch_size}; reintentando de a uno")
                texts = [model.describe([im], prompt_text, cfg)[0] for im in imgs]
            for r, t in zip(chunk, texts):
                fh.write(json.dumps({
                    "image_id": r["image_id"],
                    "row_idx": r["row_idx"],
                    "prediction": t,
                    "caption_ref": r["caption_ref"],
                    "object_shape": r["object_shape"],
                    "object_xpos": r["object_xpos"],
                    "object_ypos": r["object_ypos"],
                    "object_color": r["object_color"],
                    "text_object_color_name": r["text_object_color_name"],
                    "text_phrasing": r["text_phrasing"],
                }, ensure_ascii=False) + "\n")
    infer_s = time.time() - t0

    meta = {
        "exp_id": exp_id, "stage": 1, "model": spec.name, "hf_id": spec.hf_id,
        "quantization": quant, "prompt_id": prompt_id, "prompt_text": prompt_text,
        "n_examples": len(rows), "batch_size": batch_size,
        "generation": vars(cfg),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "memory_gib": round(model.memory_footprint_gib(), 3),
        "load_seconds": round(load_s, 1), "infer_seconds": round(infer_s, 1),
        "seconds_per_example": round(infer_s / max(len(rows), 1), 4),
        "commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"-> {pred_path}  ({infer_s:.0f}s, {meta['seconds_per_example']:.3f}s/ej)")

    del model
    torch.cuda.empty_cache()
    return pred_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["llava"], choices=list(MODELS) + ["all"])
    ap.add_argument("--prompt", nargs="+", default=["p0_minimal"])
    ap.add_argument("--manifest", type=Path, default=Path("data/manifests/m3di_val.parquet"))
    ap.add_argument("--prompts-file", type=Path,
                    default=Path("configs/prompt/prompts.yaml"))
    ap.add_argument("-n", type=int, default=1000, help="tamano del subconjunto estratificado")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "none"])
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    catalogue = yaml.safe_load(args.prompts_file.read_text(encoding="utf-8"))
    prompt_ids = list(catalogue) if "all" in args.prompt else args.prompt
    unknown = [p for p in prompt_ids if p not in catalogue]
    if unknown:
        sys.exit(f"prompts desconocidos: {unknown}\ndisponibles: {list(catalogue)}")
    model_keys = list(MODELS) if "all" in args.model else args.model

    df = pl.read_parquet(args.manifest)
    sub = stratified_sample(df, args.n, args.seed)
    print(f"manifiesto {args.manifest.name}: {df.height} -> subconjunto {sub.height}")
    print(f"modelos: {model_keys} | prompts: {prompt_ids}")

    cfg = GenConfig(max_new_tokens=args.max_new_tokens, do_sample=False, seed=args.seed)
    torch.manual_seed(args.seed)

    for mk in model_keys:
        for pid in prompt_ids:
            run_one(mk, pid, " ".join(catalogue[pid]["text"].split()),
                    sub, args.batch_size, cfg, args.out, args.quant)


if __name__ == "__main__":
    main()
