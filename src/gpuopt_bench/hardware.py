from __future__ import annotations

import json
import shutil
import subprocess

from .cuopt_adapter import cuopt_available


def environment_report() -> dict[str, object]:
    report: dict[str, object] = {"cuopt_importable": cuopt_available(), "nvidia_smi": False}
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return report
    proc = subprocess.run(
        [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    report["nvidia_smi"] = proc.returncode == 0
    report["gpu_lines"] = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return report


def environment_json() -> str:
    return json.dumps(environment_report(), indent=2, sort_keys=True)
