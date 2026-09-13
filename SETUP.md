# Preparación del entorno — proyecto T7 (VLMs autorregresivos vs VL-JEPA)

Dos máquinas, un solo `pyproject.toml` + `uv.lock`:

| | Laptop | Cluster `deepmind` |
|---|---|---|
| SO | Windows | Linux, home en `/mnt/home2/t7-vLLM` |
| GPU | RTX 5090 laptop, Blackwell **sm_120**, ~24 GB | 2× RTX 4090, Ada **sm_89**, 24 GB c/u |
| Driver / CUDA | ≥ 570 (CUDA 12.8) | 550.120 (CUDA 12.4) |
| Build PyTorch | **cu128** | **cu126** (cu128 no es compatible con driver 550) |
| Compilador CUDA | — | **no hay `nvcc`** → solo wheels precompiladas |
| GPU a usar | 0 | **1** (la 0 la ocupa otro usuario) |
| Disco | — | `/` con ~87 GB libres; verificar `/mnt/home2` |

Cuantización común para los tres modelos: **NF4 (bitsandbytes) + LoRA, cómputo en bf16**. AWQ solo si su wheel resuelve sin compilar; si no, se descarta.

---

## 0. Archivos de este bundle

```
pyproject.toml        dependencias, extras y selección cu128/cu126 por plataforma
.python-version       3.12
.env.example          plantilla de variables (copiar a .env)
.gitignore
scripts/check_env.py  verificación del entorno (ejecutar en ambas máquinas)
scripts/run.sh        lanzador para el cluster: elige GPU libre y carga .env
```

Cópialos a la raíz del repo `vlm-fidelidad/` y crea `src/vlmfid/__init__.py` (vacío) para que el paquete instale.

---

## 1. Laptop (Windows)

```powershell
# 1.1 Prerrequisitos
nvidia-smi                      # debe mostrar "CUDA Version: 12.8" o superior
winget install Git.Git
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 1.2 Repo
git clone <url-repo> vlm-fidelidad
cd vlm-fidelidad
git submodule add <url-vl-jepa> external/vl-jepa     # solo la primera vez
git submodule update --init --recursive

# 1.3 Variables
copy .env.example .env          # editar si quieres mover HF_HOME a otro disco

# 1.4 Entorno (AWQ y wandb no aplican en Windows)
uv sync --extra quant --extra qwen --extra eval --extra dev
uv pip install -e external/vl-jepa    # revisar antes su setup.py / pyproject; puede requerir deps extra

# 1.5 Verificación
uv run python scripts/check_env.py    # esperado: cc (12, 0), build cu128, NF4 OK

# 1.6 Jupyter
uv run python -m ipykernel install --user --name vlmfid --display-name "vlmfid (cu128)"
```

Commitea `uv.lock` después del primer `uv sync` exitoso.

Restricciones en Windows: `autoawq` y `flash-attn` no compilan → cargar modelos con `attn_implementation="sdpa"`. Si trabajas desde **WSL2**, el marcador de plataforma te dará cu126 y la 5090 no funcionará; en ese caso, tras `uv sync`, fuerza el build:
`uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --reinstall`.

---

## 2. Cluster (`deepmind`, Linux)

```bash
# 2.1 Prerrequisitos
nvidia-smi                      # 2× RTX 4090; anota cuál está libre (memoria ~2 MiB)
df -h /mnt/home2 /              # decide dónde va la caché de modelos (~50 GB)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2.2 Repo
cd /mnt/home2/t7-vLLM
git clone <url-repo> vlm-fidelidad
cd vlm-fidelidad
git submodule update --init --recursive

# 2.3 Variables: caché fuera de / si /mnt/home2 tiene más espacio
cp .env.example .env
mkdir -p /mnt/home2/t7-vLLM/.cache/{hf,uv}
# .env ya trae HF_HOME y UV_CACHE_DIR apuntando ahí; ajusta si df dice otra cosa
set -a; . ./.env; set +a

# 2.4 Entorno
uv sync --extra quant --extra qwen --extra eval --extra track --extra dev
uv pip install -e external/vl-jepa
# AWQ (opcional): solo si instala en <1 min sin compilar; si falla, ignorar
timeout 120 uv pip install autoawq || echo "AWQ descartado: se usa NF4"

# 2.5 Verificación en la GPU libre
chmod +x scripts/run.sh
GPU=1 ./scripts/run.sh python scripts/check_env.py   # esperado: cc (8, 9), build cu126, NF4 OK

# 2.6 Jupyter (si usas VS Code remoto)
uv run python -m ipykernel install --user --name vlmfid --display-name "vlmfid (cu126)"
```

Trabajo en el cluster, siempre:

```bash
tmux new -s t7                      # sesiones largas sobreviven a la desconexión SSH
./scripts/run.sh python scripts/infer.py experiment=e01_zeroshot_llava_p1
```

`run.sh` aborta si la GPU elegida tiene > 1 GB en uso, exporta `CUDA_VISIBLE_DEVICES` y registra GPU + commit + hora en la salida. Cambia de GPU con `GPU=0 ./scripts/run.sh ...`.

---

## 3. Descarga de modelos y datos (una vez por máquina)

Con `HF_HOME` ya apuntando al disco correcto:

```bash
uv run hf download llava-hf/llava-1.5-7b-hf
uv run hf download Qwen/Qwen2.5-VL-7B-Instruct

#vl-JEPA
git clone https://github.com/dion-jy/open-vljepa
cd open-vljepa
uv pip install webdataset decord pyyaml huggingface-hub

# Download ckpt
uv run hf download cun-bjy/open-vljepa best.pt --local-dir checkpoints_msrvtt

# VL-JEPA: seguir las instrucciones del submodule para sus checkpoints
# Multimodal3DIdent: script de descarga del repo original (data/raw/, ignorado por git)
```

Tamaños aproximados: LLaVA ~14 GB, Qwen2.5-VL-7B ~16 GB, VL-JEPA 2–10 GB, dataset unos GB. Los adaptadores LoRA son de decenas de MB.

---

## 4. Orden de hitos tras el setup

1. `check_env.py` verde en ambas máquinas.
2. Smoke test: una imagen de M3DI descrita por LLaVA, Qwen2.5-VL y VL-JEPA en NF4 (VL-JEPA primero, es el de mayor riesgo).
3. Manifiesto único del dataset + extractor de atributos con tests (≈100 % sobre los captions de referencia).
4. Etapa 1 (zero-shot, modelo × prompt), también en bf16 completo para cuantificar el costo de NF4.
5. Etapa 2 (QLoRA, `budget.yaml` común).

---

## 5. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `check_env` reporta build cu128 en el cluster | torch no vino del índice cu126: `uv pip show torch`; borra `.venv` y repite `uv sync` |
| `CUDA error: no kernel image` en el laptop | build ≠ cu128 o driver < 570: actualiza driver; en WSL2 ver §1 |
| `bitsandbytes` no importa | versión antigua para la GPU: `uv pip install -U bitsandbytes` y repite el check |
| `autoawq` intenta compilar | no hay `nvcc`; descartar AWQ |
| `uv sync` llena `$HOME` en el cluster | `UV_CACHE_DIR` no exportado antes de `uv sync`; cargar `.env` primero |
| OOM al entrenar en la 4090 | bajar micro-batch a la mitad y subir acumulación; gradient checkpointing activado |
| `run.sh` aborta con "GPU ocupada" | otro usuario en esa GPU; revisar `nvidia-smi` y probar `GPU=0` o esperar |
