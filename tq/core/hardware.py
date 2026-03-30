from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str
    gpu_vram_mb: int
    gpu_compute_cap: str
    system_ram_mb: int
    cpu_name: str
    cpu_cores: int
    os: str
    arch: str
    backend: str
    available_vram_mb: int


def _run(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_via_llmfit() -> dict | None:
    output = _run(["llmfit", "system", "--json"])
    if output:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
    return None


def _detect_via_nvidia_smi() -> dict | None:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    lines = output.strip().split("\n")
    if not lines:
        return None
    first_gpu = lines[0].strip()
    parts = [p.strip() for p in first_gpu.split(",")]
    if len(parts) < 4:
        return None
    return {
        "gpu_name": parts[0],
        "vram_total_mb": int(float(parts[1])),
        "vram_free_mb": int(float(parts[2])),
        "compute_cap": parts[3],
    }


def _detect_via_torch_cuda() -> dict | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        vram_total_mb = int(total_mem / (1024 * 1024))
        vram_free_mb = int((total_mem - torch.cuda.memory_allocated(0)) / (1024 * 1024))
        return {
            "gpu_name": props.name,
            "vram_total_mb": vram_total_mb,
            "vram_free_mb": vram_free_mb,
            "compute_cap": f"{props.major}.{props.minor}",
        }
    except (ImportError, RuntimeError):
        return None


def _detect_via_torch_mps() -> dict | None:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "gpu_name": "Apple Metal Performance Shaders",
                "vram_total_mb": 0,
                "vram_free_mb": 0,
                "compute_cap": "",
            }
    except (ImportError, AttributeError):
        pass
    return None


def detect_hardware() -> HardwareProfile:
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores = os.cpu_count() or 1
    system_ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    os_name = platform.system().lower()
    arch = platform.machine()

    gpu_name = "No GPU"
    gpu_vram_mb = 0
    gpu_compute_cap = ""
    available_vram_mb = 0
    backend = "cpu"

    info = _detect_via_llmfit()
    if info:
        gpu_name = info.get("gpu_name", gpu_name)
        gpu_vram_mb = info.get("vram_total_mb", gpu_vram_mb)
        available_vram_mb = info.get("vram_free_mb", info.get("available_vram_mb", gpu_vram_mb))
        gpu_compute_cap = info.get("compute_cap", gpu_compute_cap)
        if gpu_vram_mb > 0:
            backend = "cuda"
    else:
        nvidia = _detect_via_nvidia_smi()
        if nvidia:
            gpu_name = nvidia["gpu_name"]
            gpu_vram_mb = nvidia["vram_total_mb"]
            available_vram_mb = nvidia["vram_free_mb"]
            gpu_compute_cap = nvidia["compute_cap"]
            backend = "cuda"
        else:
            torch_cuda = _detect_via_torch_cuda()
            if torch_cuda:
                gpu_name = torch_cuda["gpu_name"]
                gpu_vram_mb = torch_cuda["vram_total_mb"]
                available_vram_mb = torch_cuda["vram_free_mb"]
                gpu_compute_cap = torch_cuda["compute_cap"]
                backend = "cuda"
            else:
                mps = _detect_via_torch_mps()
                if mps:
                    gpu_name = mps["gpu_name"]
                    backend = "metal"

    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_compute_cap=gpu_compute_cap,
        system_ram_mb=system_ram_mb,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        os=os_name,
        arch=arch,
        backend=backend,
        available_vram_mb=available_vram_mb,
    )
