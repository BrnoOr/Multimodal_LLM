"""launch.sh / jobs.sh: el job sobrevive desacoplado, deja su log en logs/ y se detiene limpio."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = "pytest_job_tmp"
pytestmark = pytest.mark.skipif(shutil.which("setsid") is None, reason="requiere util-linux")


def sh(*args, **kw):
    env = {**os.environ, "FORCE": "1"}
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, **kw)


def test_launch_status_stop():
    jobdir = ROOT / "logs" / JOB
    shutil.rmtree(jobdir, ignore_errors=True)
    try:
        r = sh("scripts/launch.sh", JOB, "bash", "-c", "echo hola; sleep 60")
        assert r.returncode == 0, r.stderr
        time.sleep(0.5)
        assert "RUNNING" in sh("scripts/jobs.sh", "status").stdout
        assert "hola" in sh("scripts/jobs.sh", "logs", JOB).stdout
        assert sh("scripts/launch.sh", JOB, "true").returncode == 1        # no duplica
        assert sh("scripts/jobs.sh", "stop", JOB).returncode == 0
        time.sleep(0.5)
        assert "RUNNING" not in sh("scripts/jobs.sh", "status").stdout
    finally:
        shutil.rmtree(jobdir, ignore_errors=True)


def test_exit_code_recorded():
    jobdir = ROOT / "logs" / JOB
    shutil.rmtree(jobdir, ignore_errors=True)
    try:
        assert sh("scripts/launch.sh", JOB, "bash", "-c", "exit 3").returncode == 0
        for _ in range(50):
            if (jobdir / "exit_code").exists():
                break
            time.sleep(0.1)
        assert (jobdir / "exit_code").read_text().strip() == "3"
        assert "DONE(3)" in sh("scripts/jobs.sh", "status").stdout
    finally:
        shutil.rmtree(jobdir, ignore_errors=True)


def test_queue_missing_or_empty_fails(tmp_path):
    missing = sh("scripts/run_queue.sh", "configs/queues/no_existe.txt")
    assert missing.returncode == 2 and "no existe el archivo de cola" in missing.stderr
    assert "stage1_pilot.txt" in missing.stderr                     # sugiere las colas disponibles
    empty = tmp_path / "vacia.txt"
    empty.write_text("# solo comentarios\n\n")
    r = sh("scripts/run_queue.sh", str(empty))
    assert r.returncode == 3 and "no tiene runs" in r.stderr


def test_queue_preflight_aborts_when_package_missing():
    env_py = {"PYTHON": "false"}          # simula un intérprete que no puede importar vlmfid
    r = subprocess.run(["scripts/run_queue.sh", "configs/queues/stage1_pilot.txt"], cwd=ROOT,
                       env={**os.environ, **env_py}, capture_output=True, text=True)
    assert r.returncode == 4 and "no puede importar el paquete vlmfid" in r.stderr
    assert "=================== [1]" not in r.stdout          # no se lanzó ningún run
