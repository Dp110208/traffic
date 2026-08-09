"""Tests for runtime probing.

`detect_gpus` shells out, so these tests stub the boundary rather than the
function under test — the parsing is the part that has bugs.
"""

from __future__ import annotations

import subprocess

import pytest

from alpr import env


def _fake_smi(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(env.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=0, stdout=stdout),
    )


def test_no_driver_means_no_gpus(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda _: None)
    assert env.detect_gpus() == []


def test_parses_a_t4(monkeypatch):
    _fake_smi(monkeypatch, "Tesla T4, 15360\n")
    (gpu,) = env.detect_gpus()
    assert gpu.name == "Tesla T4"
    assert gpu.memory_mb == 15360


def test_parses_multiple_gpus(monkeypatch):
    _fake_smi(monkeypatch, "Tesla T4, 15360\nTesla T4, 15360\n")
    assert len(env.detect_gpus()) == 2


def test_ignores_blank_trailing_lines(monkeypatch):
    # nvidia-smi emits a trailing newline; a naive split yields a phantom device.
    _fake_smi(monkeypatch, "Tesla T4, 15360\n\n")
    assert len(env.detect_gpus()) == 1


def test_broken_driver_is_treated_as_no_gpu(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(returncode=1, cmd="nvidia-smi")

    monkeypatch.setattr(env.subprocess, "run", _boom)
    assert env.detect_gpus() == []


def test_require_gpu_raises_without_one(monkeypatch):
    monkeypatch.setattr(env, "detect_gpus", lambda: [])
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        env.require_gpu()


def test_require_gpu_returns_the_first(monkeypatch):
    monkeypatch.setattr(env, "detect_gpus", lambda: [env.GpuInfo("Tesla T4", 15360)])
    assert env.require_gpu().name == "Tesla T4"
