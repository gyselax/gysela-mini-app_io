"""IO test"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GYS_IO = REPO_ROOT / "build" / "apps" / "io" / "gys_io"
IO_DIR = REPO_ROOT / "apps" / "io"
GYS_COMPRESS = REPO_ROOT / "build" / "apps" / "compression" / "gys_compress"
COMPRESSION_DIR = REPO_ROOT / "apps" / "compression"
REPO_PYTHON = REPO_ROOT / "src" / "python"


def mpirun_env():
    env = os.environ.copy()
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{REPO_PYTHON}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(REPO_PYTHON)

    cmd = ["mpirun", "-n", "1"]
    for var in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH"):
        if var in env:
            cmd.extend(["-x", var])
    return cmd, env


def test_gys_io_runs():
    assert GYS_IO.is_file(), f"Build gys_io first (missing {GYS_IO})"
    assert (IO_DIR / "gys_io.yaml").is_file()
    assert (IO_DIR / "pdi_default.yaml").is_file()

    cmd, env = mpirun_env()
    cmd.extend(
        [
            str(GYS_IO),
            str(IO_DIR / "gys_io.yaml"),
            str(IO_DIR / "pdi_default.yaml"),
        ]
    )
    result = subprocess.run(cmd, cwd=IO_DIR, env=env, check=False)
    assert result.returncode == 0


def test_gys_compress_runs():
    assert GYS_COMPRESS.is_file(), f"Build gys_compress first (missing {GYS_COMPRESS})"
    assert (COMPRESSION_DIR / "params_two_stream.yaml").is_file()
    assert (COMPRESSION_DIR / "pdi_out.yaml").is_file()

    cmd, env = mpirun_env()
    cmd.extend(
        [
            str(GYS_COMPRESS),
            str(COMPRESSION_DIR / "params_two_stream.yaml"),
            str(COMPRESSION_DIR / "pdi_out.yaml"),
        ]
    )
    result = subprocess.run(cmd, cwd=COMPRESSION_DIR, env=env, check=False)
    assert result.returncode == 0