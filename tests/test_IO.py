"""IO test"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GYS_IO = REPO_ROOT / "build" / "apps" / "io" / "gys_io"
IO_DIR = REPO_ROOT / "apps" / "io"
REPO_PYTHON = REPO_ROOT


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
    assert (IO_DIR / "params" / "gys_io.yaml").is_file()
    assert (IO_DIR / "params" / "seq_pdi.yaml").is_file()

    cmd, env = mpirun_env()
    cmd.extend(
        [
            str(GYS_IO),
            str(IO_DIR / "params" / "gys_io.yaml"),
            str(IO_DIR / "params" / "seq_pdi.yaml"),
        ]
    )
    result = subprocess.run(cmd, cwd=IO_DIR, env=env, check=False)
    assert result.returncode == 0