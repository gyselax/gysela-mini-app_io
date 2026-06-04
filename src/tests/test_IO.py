"""IO test"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GYS_IO = REPO_ROOT / "build" / "apps" / "io" / "gys_io"
IO_DIR = REPO_ROOT / "apps" / "io"
REPO_PYTHON = REPO_ROOT / "src" / "python"


def test_gys_io_runs():
    assert GYS_IO.is_file(), f"Build gys_io first (missing {GYS_IO})"
    assert (IO_DIR / "gys_io.yaml").is_file()
    assert (IO_DIR / "pdi_default.yaml").is_file()

    env = os.environ.copy()
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{REPO_PYTHON}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(REPO_PYTHON)

    cmd = ["mpirun", "-n", "1"]
    for var in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH"):
        if var in env:
            cmd.extend(["-x", var])
    cmd.extend(
        [
            str(GYS_IO),
            str(IO_DIR / "gys_io.yaml"),
            str(IO_DIR / "pdi_default.yaml"),
        ]
    )
    result = subprocess.run(cmd, cwd=IO_DIR, env=env, check=False)
    assert result.returncode == 0