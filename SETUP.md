# Preparación del entorno — T7 (VLMs autorregresivos vs VL-JEPA)

Dos máquinas, un solo `pyproject.toml` + `uv.lock`:

| | Laptop | Cluster `deepmind` |
|---|---|---|
| SO | Windows | Linux, home en `/mnt/home2/t7-vLLM` |
| GPU | RTX 5090 laptop, Blackwell **sm_120**, ~24 GB | 2× RTX 4090, Ada **sm_89**, 24 GB c/u |
| Driver / build PyTorch | ≥ 570 → **cu128** | 550.120 → **cu126** |
| Compilador CUDA | — | sin `nvcc` → solo wheels precompiladas |
| GPU a usar | 0 | **1** (la 0 la ocupa otro usuario) |
| Sesiones largas | — | **`scripts/launch.sh`** (setsid + nohup), logs en `logs/`; no hay tmux |

## 1. Cluster

```bash
# prerrequisitos
nvidia-smi                                   # anota la GPU libre
df -h /mnt/home2 /                           # modelos + dataset ≈ 60 GB: la caché va en /mnt/home2
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# repo + submodule fijado
cd /mnt/home2/t7-vLLM/Multimodal_LLM
git submodule update --init --recursive      # external/open-vljepa @ f92107f

# variables (antes de cualquier uv sync, para que la caché no llene $HOME)
cp .env.example .env                         # completar HF_TOKEN
mkdir -p /mnt/home2/t7-vLLM/.cache/{hf,uv}
set -a; . ./.env; set +a

# entorno
uv sync --extra quant --extra eval --extra track --extra dev
GPU=1 scripts/launch.sh check_env uv run --no-sync python scripts/check_env.py
scripts/jobs.sh logs check_env               # esperado: cc (8, 9), build 12.6, NF4 OK
```

`open-vljepa` no es un paquete instalable (no trae `setup.py`/`pyproject`): el adaptador lo importa desde `external/open-vljepa` añadiéndolo a `sys.path`. No hace falta `pip install -e`.

## 2. Laptop (Windows)

```powershell
uv sync --extra quant --extra eval --extra dev
uv run python scripts/check_env.py           # esperado: cc (12, 0), build 12.8, NF4 OK
uv run python -m ipykernel install --user --name vlmfid --display-name "vlmfid (cu128)"
```

En Windows no hay `setsid`: los scripts `.sh` son para el cluster; en el laptop se corre `uv run python scripts/infer.py ...` directamente. Con WSL2 el marcador de plataforma elige cu126 y la 5090 no funciona: forzar `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --reinstall`.

## 3. Modelos y licencias

VL-JEPA depende de dos modelos con licencia que hay que **aceptar en la web de Hugging Face** con la cuenta del `HF_TOKEN`: `meta-llama/Llama-3.2-1B` y `google/embeddinggemma-300m`.

```bash
uv run hf download llava-hf/llava-onevision-qwen2-0.5b-ov-hf   # ~2 GB
uv run hf download OpenGVLab/InternVL3-1B-hf                    # ~2 GB
uv run hf download Qwen/Qwen3.5-0.8B-Base                       # ~2 GB
uv run hf download Qwen/Qwen3.5-0.8B                            # opcional: control Instruct
uv run hf download cun-bjy/open-vljepa best.pt          # ~2 GB (predictor + Y-Encoder)
uv run hf download facebook/vjepa2-vitl-fpc64-256       # X-Encoder congelado
uv run hf download meta-llama/Llama-3.2-1B              # arquitectura del predictor
uv run hf download google/embeddinggemma-300m           # Y-Encoder
```

Descargas largas también con `launch.sh` (p. ej. `scripts/launch.sh dl_vjepa uv run hf download facebook/vjepa2-vitl-fpc64-256`).

Qwen3.5 requiere transformers ≥ 5.17. Sus capas Gated DeltaNet usan kernels rápidos si están disponibles y, si no, una implementación en PyTorch (más lenta, mismo resultado). Opcional: `uv sync ... --extra linattn` instala `flash-linear-attention==0.4.2` (Triton, no requiere `nvcc`); **no usar la 0.5.0**, que corrompe las salidas multimodales.

## 4. Hitos

1. `check_env.py` verde en ambas máquinas.
2. Manifiestos: `uv run python scripts/build_manifest.py --report-disagreement` (ver `data/README.md`). Revisar en la salida qué latentes de texto discretos detecta.
3. Smoke test: `scripts/launch.sh smoke uv run --no-sync python scripts/smoke.py --set model.bank.max_captions=2000`.
4. Extractor de atributos con tests (≈100 % sobre captions de referencia).
5. Etapa 1: `configs/queues/stage1_pilot.txt` → `stage1_full.txt` (+ una cola con `model.quant=bf16`).
6. Etapa 2: QLoRA con `configs/train/budget.yaml` común.

## 5. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `launch.sh` aborta con "GPU ocupada" | otro usuario en esa GPU: `scripts/jobs.sh gpu`; probar `GPU=0` o esperar |
| el log se ve vacío durante minutos | carga de pesos; `launch.sh` ya exporta `PYTHONUNBUFFERED=1` |
| job en estado `DIED` | murió sin cerrar el wrapper (reinicio, SIGKILL, OOM del host): relanzar el mismo comando reanuda |
| `RuntimeError: La configuración difiere` | se reanudó un run con otra configuración: usar otro `exp_id` o borrar el run |
| `401/403` al cargar VL-JEPA | falta aceptar la licencia de Llama-3.2-1B o EmbeddingGemma, o `HF_TOKEN` no está en `.env` |
| `check_env` reporta build 12.8 en el cluster | torch no vino de cu126: borrar `.venv` y repetir `uv sync` |
| `bitsandbytes` no importa | `uv pip install -U bitsandbytes` y repetir el check |
| `uv sync` llena `$HOME` | `UV_CACHE_DIR` no exportado: cargar `.env` antes |
| OOM | `infer` parte el lote solo; si persiste, bajar `infer.batch_size` |
