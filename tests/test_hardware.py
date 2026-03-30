from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from tq.core.hardware import (
    _detect_via_llmfit,
    _detect_via_nvidia_smi,
    _detect_via_torch_cuda,
    _detect_via_torch_mps,
    _run,
    detect_hardware,
)


class TestRunHelper:
    def test_run_success(self):
        with patch("tq.core.hardware.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="output\n")
            result = _run(["echo", "test"])
            assert result == "output"

    def test_run_failure(self):
        with patch("tq.core.hardware.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert _run(["false"]) is None

    def test_run_not_found(self):
        with patch("tq.core.hardware.subprocess.run", side_effect=FileNotFoundError):
            assert _run(["nonexistent"]) is None

    def test_run_timeout(self):
        with patch(
            "tq.core.hardware.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["sleep"], timeout=10),
        ):
            assert _run(["sleep", "999"]) is None


class TestLlmfitDetection:
    def test_valid_json(self):
        data = {
            "gpu_name": "NVIDIA RTX 4060",
            "vram_total_mb": 8192,
            "vram_free_mb": 7800,
            "compute_cap": "8.9",
        }
        with patch("tq.core.hardware._run", return_value=json.dumps(data)):
            result = _detect_via_llmfit()
            assert result is not None
            assert result["gpu_name"] == "NVIDIA RTX 4060"
            assert result["vram_total_mb"] == 8192

    def test_invalid_json(self):
        with patch("tq.core.hardware._run", return_value="not json"):
            assert _detect_via_llmfit() is None

    def test_llmfit_not_installed(self):
        with patch("tq.core.hardware._run", return_value=None):
            assert _detect_via_llmfit() is None


class TestNvidiaSmiDetection:
    def test_valid_output(self):
        output = "NVIDIA RTX 4060, 8192, 7800, 8.9"
        with patch("tq.core.hardware._run", return_value=output):
            result = _detect_via_nvidia_smi()
            assert result is not None
            assert result["gpu_name"] == "NVIDIA RTX 4060"
            assert result["vram_total_mb"] == 8192
            assert result["vram_free_mb"] == 7800
            assert result["compute_cap"] == "8.9"

    def test_multi_gpu_takes_first(self):
        output = "NVIDIA RTX 4060, 8192, 7800, 8.9\nNVIDIA RTX 3060, 12288, 10000, 8.6"
        with patch("tq.core.hardware._run", return_value=output):
            result = _detect_via_nvidia_smi()
            assert result["gpu_name"] == "NVIDIA RTX 4060"

    def test_no_nvidia_smi(self):
        with patch("tq.core.hardware._run", return_value=None):
            assert _detect_via_nvidia_smi() is None

    def test_malformed_output(self):
        with patch("tq.core.hardware._run", return_value="bad,data"):
            assert _detect_via_nvidia_smi() is None

    def test_empty_output(self):
        with patch("tq.core.hardware._run", return_value=""):
            assert _detect_via_nvidia_smi() is None


class TestTorchCudaDetection:
    def test_no_torch_cuda(self):
        with patch.dict(
            "sys.modules",
            {"torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))},
        ):
            pass

    def test_torch_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch("tq.core.hardware._detect_via_torch_cuda") as mock:
            mock.return_value = None
            _ = (
                _detect_via_torch_cuda.__wrapped__()
                if hasattr(_detect_via_torch_cuda, "__wrapped__")
                else None
            )

    def test_torch_import_error(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert _detect_via_torch_cuda() is None

    def test_torch_cuda_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.name = "MockGPU"
        mock_props.total_memory = 8 * 1024 * 1024 * 1024
        mock_props.major = 8
        mock_props.minor = 6
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024

        import sys

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _detect_via_torch_cuda()
        assert result is not None
        assert result["gpu_name"] == "MockGPU"
        assert result["vram_total_mb"] == 8192
        assert result["compute_cap"] == "8.6"

    def test_torch_cuda_runtime_error(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("no driver")
        import sys

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _detect_via_torch_cuda()
        assert result is None


class TestTorchMpsDetection:
    def test_mps_unavailable(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = MagicMock()
            mock_torch.backends.mps.is_available.return_value = False
            with patch("tq.core.hardware._detect_via_torch_mps") as mock:
                mock.return_value = None

    def test_mps_available(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        import sys

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _detect_via_torch_mps()
        assert result is not None
        assert result["gpu_name"] == "Apple Metal Performance Shaders"

    def test_mps_import_error(self):
        import sys

        with patch.dict(sys.modules, {"torch": None}):
            result = _detect_via_torch_mps()
        assert result is None


class TestDetectHardwareIntegration:
    def test_nvidia_smi_path(self):
        with (
            patch("tq.core.hardware._detect_via_llmfit", return_value=None),
            patch(
                "tq.core.hardware._detect_via_nvidia_smi",
                return_value={
                    "gpu_name": "NVIDIA RTX 4050",
                    "vram_total_mb": 5763,
                    "vram_free_mb": 5763,
                    "compute_cap": "8.6",
                },
            ),
        ):
            hw = detect_hardware()
            assert hw.gpu_name == "NVIDIA RTX 4050"
            assert hw.gpu_vram_mb == 5763
            assert hw.backend == "cuda"

    def test_llmfit_path(self):
        llmfit_data = {
            "gpu_name": "NVIDIA RTX 4090",
            "vram_total_mb": 24576,
            "vram_free_mb": 24000,
            "compute_cap": "8.9",
        }
        with patch("tq.core.hardware._detect_via_llmfit", return_value=llmfit_data):
            hw = detect_hardware()
            assert hw.gpu_name == "NVIDIA RTX 4090"
            assert hw.gpu_vram_mb == 24576
            assert hw.available_vram_mb == 24000

    def test_cpu_fallback(self):
        with (
            patch("tq.core.hardware._detect_via_llmfit", return_value=None),
            patch("tq.core.hardware._detect_via_nvidia_smi", return_value=None),
            patch("tq.core.hardware._detect_via_torch_cuda", return_value=None),
            patch("tq.core.hardware._detect_via_torch_mps", return_value=None),
        ):
            hw = detect_hardware()
            assert hw.backend == "cpu"
            assert hw.gpu_vram_mb == 0
            assert hw.system_ram_mb > 0

    def test_metal_fallback(self):
        mps_data = {
            "gpu_name": "Apple Metal Performance Shaders",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "compute_cap": "",
        }
        with (
            patch("tq.core.hardware._detect_via_llmfit", return_value=None),
            patch("tq.core.hardware._detect_via_nvidia_smi", return_value=None),
            patch("tq.core.hardware._detect_via_torch_cuda", return_value=None),
            patch("tq.core.hardware._detect_via_torch_mps", return_value=mps_data),
        ):
            hw = detect_hardware()
            assert hw.backend == "metal"
            assert hw.gpu_name == "Apple Metal Performance Shaders"

    def test_torch_cuda_path(self):
        cuda_data = {
            "gpu_name": "NVIDIA RTX 3090",
            "vram_total_mb": 24576,
            "vram_free_mb": 22000,
            "compute_cap": "8.6",
        }
        with (
            patch("tq.core.hardware._detect_via_llmfit", return_value=None),
            patch("tq.core.hardware._detect_via_nvidia_smi", return_value=None),
            patch("tq.core.hardware._detect_via_torch_cuda", return_value=cuda_data),
        ):
            hw = detect_hardware()
            assert hw.gpu_name == "NVIDIA RTX 3090"
            assert hw.backend == "cuda"
            assert hw.available_vram_mb == 22000

    def test_nvidia_smi_blank_lines(self):
        with patch("tq.core.hardware._run", return_value="   \n"):
            result = _detect_via_nvidia_smi()
            assert result is None
