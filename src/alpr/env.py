"""Runtime environment probing.

Colab sessions are ephemeral and silently vary: a runtime that was a T4
yesterday can be a CPU box today, and the failure shows up hours later as
training that never finishes. Every notebook driver calls `require_gpu()`
in its first cell so that mistake surfaces in seconds instead.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    """A single visible CUDA device."""

    name: str
    memory_mb: int

    def __str__(self) -> str:
        return f"{self.name} ({self.memory_mb / 1024:.1f} GB)"


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def detect_gpus() -> list[GpuInfo]:
    """Return visible CUDA devices, or an empty list if there are none.

    Shells out to `nvidia-smi` rather than importing torch: this is called
    before the heavy dependencies are installed, and importing torch just to
    read a device name costs seconds on every notebook start.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        # A driver that is present but broken is a no-GPU environment for our
        # purposes; the caller's error message is more useful than a traceback.
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        name, _, memory = line.partition(",")
        if not memory.strip():
            continue
        gpus.append(GpuInfo(name=name.strip(), memory_mb=int(memory.strip())))
    return gpus


def require_gpu() -> GpuInfo:
    """Return the first CUDA device, raising if the runtime has none.

    Raises:
        RuntimeError: when no CUDA device is visible.
    """
    gpus = detect_gpus()
    if not gpus:
        hint = (
            "Runtime > Change runtime type > T4 GPU, then rerun this cell."
            if in_colab()
            else "This step needs a CUDA GPU; run it on Colab."
        )
        raise RuntimeError(f"No CUDA GPU visible. {hint}")
    return gpus[0]
