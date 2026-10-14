# Guía: laptop → GitHub → cluster

Flujo de trabajo del repo `Multimodal_LLM` (rama `modelos`) entre dos máquinas:

```
 laptop (Windows)            GitHub                     cluster deepmind (Linux)
 edita, prueba, commit  ──►  BrnoOr/Multimodal_LLM  ──►  git pull --ff-only, ejecuta
                             rama modelos               runs/ y logs/ quedan solo aquí
```

**Regla:** el código se edita en el laptop. En el cluster solo se hace `git pull` y se ejecuta. Si alguna vez hay que corregir algo directamente en el cluster, se hace commit y push de inmediato desde el cluster, y se hace `git pull` en el laptop antes de volver a editar ahí. Así las dos historias no vuelven a separarse, que fue lo que pasó el 26-09-2026.

---

## Parte A — Puesta al día del 26-09-2026 (una sola vez)

### Qué pasó

El laptop subió la reorganización (`1cccb6b`). El cluster aplicó la misma reorganización por su cuenta y quedó con 3 commits locales que nunca se subieron, entre ellos `reports/` y los manifiestos `m3di_base_*`. Un `git pull` en el cluster producía unos 100 conflictos.

La versión nueva parte de `origin/modelos` e incluye estos arreglos:

| Arreglo | Dónde |
|---|---|
| Plantilla de chat de Qwen3.5-0.8B-Base (el processor no trae; se toma la del Instruct) | `hf_generative.py`, `configs/models/qwen35.yaml` |
| `as_dict` para claves opcionales de la config | `src/vlmfid/config.py` |
| Prompts con llaves literales (p2, p3) sin `str.format` | `src/vlmfid/prompts/` |
| Umbral de cardinalidad 256 (color nominal de M3DI, ~106 valores) | `src/vlmfid/data/m3di.py` |
| Predictor de VL-JEPA desde `unsloth/Llama-3.2-1B` (réplica sin aprobación de Meta) | `vljepa.py`, `configs/models/vljepa.yaml` |
| Un run que falló sin guardar predicciones se reanuda con la config corregida | `src/vlmfid/tracking.py` (+ `tests/test_tracking.py`) |
| Scripts `.sh` y `.py` ejecutables, LF forzado para `.sh`/`.py`/`.yaml` | modos en git + `.gitattributes` |
| `external/open-vljepa` registrado como submodule fijado a `f92107f` | `.gitmodules` |
| Carpeta `OTHER/` ordenada: notebooks → `notebooks/legacy/`, figuras/tablas → `reports/`, contexto → `docs/` | — |

Se eliminó lo que ya estaba reemplazado: código y configs antiguos, logs viejos, `Multimodal_LLM_5.zip` y `MIGRACION.md`. Todo sigue disponible en el historial de git, en el commit `1cccb6b`.

### A1. Laptop: descomprimir y subir

El .zip trae el repo completo, con su historial de git. No se copia sobre tu clon actual: se usa en su lugar.

```powershell
# 1. aparta el clon anterior (por si tiene algo que quieras recuperar)
cd $HOME\Documents                                   # o donde tengas el repo
Rename-Item Multimodal_LLM Multimodal_LLM_anterior   # si existe

# 2. descomprime (tar de Windows conserva .git y nombres con tildes)
tar -xf $HOME\Downloads\Multimodal_LLM.zip
cd Multimodal_LLM

# 3. ajustes de Windows para este clon
git config core.filemode false        # NTFS no guarda el bit de ejecución: evita falsos "modified"
git config core.autocrlf true         # CRLF en tu editor, LF en el repo (los .sh quedan LF igual)

# 4. comprobar
git status                            # esperado: "nothing to commit, working tree clean"
git log --oneline -3                  # arriba: el commit nuevo, debajo 1cccb6b
git submodule status                  # f92107f external/open-vljepa

# 5. subir (es fast-forward sobre 1cccb6b: no reescribe nada)
git push origin modelos
```

Si `git status` muestra archivos modificados que no tocaste, casi siempre son saltos de línea. Revisa con `git diff --stat` y descártalos con `git checkout -- .`.

Si el push es rechazado porque alguien subió algo en el intermedio, haz `git pull --rebase origin modelos` y repite el push.

Los tests son opcionales en el laptop porque no necesitan GPU ni datos:

```powershell
uv sync --extra dev
uv run pytest -q                      # esperado: todos pasan (63 al 26-09-2026)
```

Antes de borrar `Multimodal_LLM_anterior`, revisa si tiene cambios sin subir con `git -C ..\Multimodal_LLM_anterior status` y `git -C ..\Multimodal_LLM_anterior log origin/modelos..HEAD`.

### A2. Cluster: respaldar y alinear con GitHub

Se trabaja en `/mnt/home2/t7-vLLM/Multimodal_LLM`. `runs/`, `data/raw/`, `.env` y `.venv/` no están versionados y el reset no los toca. Lo versionado solo en el cluster (reports, manifiestos, notebooks) se respalda primero.

```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM

# 0. que no haya nada corriendo (un reset con un job activo le cambia el código bajo los pies)
pgrep -af "scripts/infer.py|run_queue.sh" || echo "nada corriendo"

# 1. respaldo en una rama local (no se sube) + copia comprimida fuera del repo
git merge --abort 2>/dev/null; true                 # por si quedó un merge a medias
git switch -c respaldo-cluster-20260926
git add -u && git commit -qm "respaldo del cluster antes de alinear con origin/modelos" || true
mkdir -p ~/respaldo_t7
tar czf ~/respaldo_t7/cluster_20260926.tgz reports notebooks data/manifests configs src scripts logs 2>/dev/null
ls -lh ~/respaldo_t7/

# 2. alinear la rama modelos con GitHub
git switch modelos
git fetch origin
git reset --hard origin/modelos
git log --oneline -3                                # el commit nuevo arriba
```

`git stash list` puede mostrar `cluster-local-antes-pull`: es un respaldo más y no molesta. Bórralo cuando ya no lo necesites, con `git stash drop`.

```bash
# 3. submodule de VL-JEPA
git submodule sync
git submodule update --init --recursive
# si dice "already exists and is not an empty directory":
#   mv external/open-vljepa ~/respaldo_t7/open-vljepa_anterior && git submodule update --init --recursive
git submodule status                                # f92107f external/open-vljepa

# 4. manifiestos y figuras del cluster (vuelven al árbol de trabajo, sin commit)
git restore --source=respaldo-cluster-20260926 --worktree -- data/manifests
git restore --source=respaldo-cluster-20260926 --worktree -- reports
ls data/manifests                                   # m3di_base_{train,val,test}.parquet

# 5. comparar notebooks del cluster con los del repo y traer los que quieras conservar
git diff --stat HEAD respaldo-cluster-20260926 -- notebooks
# p. ej.: git restore --source=respaldo-cluster-20260926 --worktree -- "notebooks/03_eda_predicciones.ipynb"
```

Si algún `restore` dice `pathspec ... did not match`, esa carpeta no estaba en el respaldo. Los manifiestos se regeneran con `uv run --no-sync python scripts/build_manifest.py --report-disagreement`.

`reports/` queda exactamente como estaba en el cluster, que es la versión más nueva. Las figuras antiguas que el cluster ya no tenía aparecen como borradas en `git status`. Es lo esperado.

### A3. Cluster: verificar el entorno y relanzar

```bash
set -a; . ./.env; set +a                            # HF_TOKEN y cachés en /mnt/home2
ls -l scripts/*.sh                                  # deben tener x (rwxr-xr-x)

uv run --no-sync python -c "import vlmfid, vlmfid.config as c; print(vlmfid.__file__, hasattr(c, 'as_dict'))"
# esperado: .../Multimodal_LLM/src/vlmfid/__init__.py True
uv run --no-sync pytest -q                          # esperado: todos pasan (63 al 26-09-2026)
```

Si falla la importación de alguna librería, sincroniza el entorno: `uv sync --extra quant --extra eval --extra dev`. Después comprueba que torch siga siendo cu126 con `uv run --no-sync python scripts/check_env.py`.

Haz una prueba corta con Qwen, que era el modelo que fallaba:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/infer.py model=qwen35 prompt=p4_dataset_format data.limit=8
```

En la salida deberías ver:

- `[hf] Qwen/Qwen3.5-0.8B-Base sin plantilla de chat: se usa la de Qwen/Qwen3.5-0.8B`;
- `"chat_template_origin": "Qwen/Qwen3.5-0.8B"` en la línea `modelo cargado`;
- `8/8` predicciones escritas.

Luego relanza la cola del piloto:

```bash
GPU=1 scripts/launch.sh stage1_pilot scripts/run_queue.sh configs/queues/stage1_pilot.txt
scripts/jobs.sh status
scripts/jobs.sh logs stage1_pilot -f                # Ctrl-C solo deja de mirar
```

Qué esperar de la cola:

- **Runs completos:** se saltan con `nada pendiente: run completo.`.
- **Runs de Qwen que fallaron sin predicciones:** se reanudan con la configuración corregida.
- **Runs completos con `La configuración difiere`:** el run ya está terminado, pero su `runs/<exp_id>/config.yaml` no coincide con la config actual. Compáralos con `diff <(uv run --no-sync python -c "from vlmfid.config import compose; from omegaconf import OmegaConf; print(OmegaConf.to_yaml(compose(['model=llava_ov','prompt=p0_minimal','data.limit=500']), resolve=True))") runs/<exp_id>/config.yaml`. Si la diferencia no afecta el resultado, puedes ignorar el aviso: las predicciones siguen ahí. Si sí lo afecta, borra el run y relánzalo.

### A4. Opcional: subir las figuras y notebooks del cluster

Las figuras de `reports/` y los notebooks que restauraste quedan como cambios sin commit en el cluster. Para versionarlos, súbelos desde ahí y luego actualiza el laptop:

```bash
# cluster
git add reports notebooks
git commit -m "reports: figuras y tablas de evaluación generadas en el cluster"
git push origin modelos
```

Después, en el laptop: `git pull origin modelos`.

Si el push desde el cluster pide credenciales, el túnel de VS Code se cortó. Abre una terminal nueva de VS Code Remote-SSH, o configura una clave SSH del cluster en GitHub y cambia el remoto con `git remote set-url origin git@github.com:BrnoOr/Multimodal_LLM.git`.

---

## Parte B — Ciclo de todos los días

**Laptop** (PowerShell):

```powershell
git pull --ff-only origin modelos      # siempre antes de editar
# ... editar ...
uv run pytest -q
git add -A
git commit -m "fix: ..."
git push origin modelos
```

**Cluster** (bash):

```bash
cd /mnt/home2/t7-vLLM/Multimodal_LLM
scripts/jobs.sh status                 # no actualizar el código con un job corriendo
git pull --ff-only origin modelos
git submodule update --init --recursive
GPU=1 scripts/launch.sh <job> scripts/run_queue.sh configs/queues/<cola>.txt
```

Si `git pull --ff-only` falla con `Not possible to fast-forward`, alguien hizo commit en el cluster sin subirlo. No hagas merge. Mira qué commits son con `git log --oneline origin/modelos..HEAD`, respáldalos con `git branch respaldo-$(date +%Y%m%d)`, súbelos si deben quedar, y vuelve a alinear con `git reset --hard origin/modelos`.

### Qué vive dónde

| Qué | Dónde | En git |
|---|---|---|
| código, configs, colas, tests, docs | laptop → GitHub → cluster | sí |
| manifiestos `data/manifests/*.parquet` | se generan en el cluster | sí (si se suben desde el cluster) |
| figuras y tablas del informe `reports/` | se generan en el cluster | sí (si se suben desde el cluster) |
| predicciones `runs/`, logs `logs/` | solo el cluster | no |
| dataset `data/raw/`, pesos, cachés | solo el cluster | no |
| secretos `.env` | cada máquina | no (plantilla: `.env.example`) |
