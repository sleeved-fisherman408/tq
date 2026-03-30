from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tq.cli.start import start
from tq.core.turbo import TQConfig


def _fake_recommendation():
    tq = TQConfig(
        key_bits=4,
        value_bits=4,
        key_method="mse",
        value_method="mse",
        outlier_channels=32,
        outlier_bits=8,
        compression_ratio=4.0,
        estimated_quality="lossless",
    )
    rec = MagicMock()
    rec.model_id = "Qwen/Qwen3-8B"
    rec.tq_config = tq
    rec.max_context_tq = 32768
    return rec


@pytest.fixture
def runner():
    return CliRunner()


class TestStartCommand:
    @patch("tq.cli.start.uvicorn")
    @patch("tq.cli.start.set_engine")
    @patch("tq.cli.start.InferenceEngine")
    @patch("tq.cli.start.pull_model")
    @patch("tq.cli.start.recommend")
    @patch("tq.cli.start.detect_hardware")
    @patch("tq.cli.start.load_global_config")
    @patch("tq.cli.start.ensure_dirs")
    def test_start_coding(
        self,
        mock_dirs,
        mock_cfg,
        mock_hw,
        mock_rec,
        mock_pull,
        mock_engine_cls,
        mock_set,
        mock_uvicorn,
        runner,
    ):
        from tq.core.config import TQGlobalConfig
        from tq.core.hardware import HardwareProfile

        mock_cfg.return_value = TQGlobalConfig()
        mock_hw.return_value = HardwareProfile(
            gpu_name="Test GPU",
            gpu_vram_mb=8000,
            gpu_compute_cap="8.9",
            system_ram_mb=16000,
            cpu_name="Test CPU",
            cpu_cores=8,
            os="linux",
            arch="x86_64",
            backend="cuda",
            available_vram_mb=7000,
        )
        mock_rec.return_value = _fake_recommendation()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(start, ["--coding"])
        assert result.exit_code == 0
        assert "Step 1/5" in result.output
        assert "Step 5/5" in result.output
        mock_engine.load.assert_called_once()
        mock_set.assert_called_once_with(mock_engine)
        mock_uvicorn.run.assert_called_once()

    @patch("tq.cli.start.uvicorn")
    @patch("tq.cli.start.set_engine")
    @patch("tq.cli.start.InferenceEngine")
    @patch("tq.cli.start.pull_model")
    @patch("tq.cli.start.recommend")
    @patch("tq.cli.start.detect_hardware")
    @patch("tq.cli.start.load_global_config")
    @patch("tq.cli.start.ensure_dirs")
    def test_start_cpu_flag(
        self,
        mock_dirs,
        mock_cfg,
        mock_hw,
        mock_rec,
        mock_pull,
        mock_engine_cls,
        mock_set,
        mock_uvicorn,
        runner,
    ):
        from tq.core.config import TQGlobalConfig
        from tq.core.hardware import HardwareProfile

        mock_cfg.return_value = TQGlobalConfig()
        mock_hw.return_value = HardwareProfile(
            gpu_name="NVIDIA RTX 4050",
            gpu_vram_mb=6000,
            gpu_compute_cap="8.9",
            system_ram_mb=16000,
            cpu_name="Test CPU",
            cpu_cores=8,
            os="linux",
            arch="x86_64",
            backend="cuda",
            available_vram_mb=5000,
        )
        mock_rec.return_value = _fake_recommendation()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(start, ["--cpu"])
        assert result.exit_code == 0
        assert "No GPU (forced CPU)" in result.output
        call_kwargs = mock_engine_cls.call_args
        assert call_kwargs.kwargs.get("device") == "cpu" or call_kwargs[1].get("device") == "cpu"

    @patch("tq.cli.start.uvicorn")
    @patch("tq.cli.start.set_engine")
    @patch("tq.cli.start.InferenceEngine")
    @patch("tq.cli.start.pull_model")
    @patch("tq.cli.start.recommend")
    @patch("tq.cli.start.detect_hardware")
    @patch("tq.cli.start.load_global_config")
    @patch("tq.cli.start.ensure_dirs")
    def test_start_verbose(
        self,
        mock_dirs,
        mock_cfg,
        mock_hw,
        mock_rec,
        mock_pull,
        mock_engine_cls,
        mock_set,
        mock_uvicorn,
        runner,
    ):
        from tq.core.config import TQGlobalConfig
        from tq.core.hardware import HardwareProfile

        mock_cfg.return_value = TQGlobalConfig()
        mock_hw.return_value = HardwareProfile(
            gpu_name="Test",
            gpu_vram_mb=8000,
            gpu_compute_cap="8.9",
            system_ram_mb=16000,
            cpu_name="CPU",
            cpu_cores=8,
            os="linux",
            arch="x86_64",
            backend="cuda",
            available_vram_mb=7000,
        )
        mock_rec.return_value = _fake_recommendation()
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(start, ["--verbose"])
        assert result.exit_code == 0
        assert "TQ:" in result.output
        assert "4-bit" in result.output

    @patch("tq.cli.start.uvicorn")
    @patch("tq.cli.start.set_engine")
    @patch("tq.cli.start.InferenceEngine")
    @patch("tq.cli.start.pull_model")
    @patch("tq.cli.start.recommend")
    @patch("tq.cli.start.detect_hardware")
    @patch("tq.cli.start.load_global_config")
    @patch("tq.cli.start.ensure_dirs")
    def test_start_custom_bits(
        self,
        mock_dirs,
        mock_cfg,
        mock_hw,
        mock_rec,
        mock_pull,
        mock_engine_cls,
        mock_set,
        mock_uvicorn,
        runner,
    ):
        from tq.core.config import TQGlobalConfig
        from tq.core.hardware import HardwareProfile

        mock_cfg.return_value = TQGlobalConfig()
        mock_hw.return_value = HardwareProfile(
            gpu_name="Test",
            gpu_vram_mb=8000,
            gpu_compute_cap="8.9",
            system_ram_mb=16000,
            cpu_name="CPU",
            cpu_cores=8,
            os="linux",
            arch="x86_64",
            backend="cuda",
            available_vram_mb=7000,
        )
        mock_rec.return_value = _fake_recommendation()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(start, ["--bits", "3"])
        assert result.exit_code == 0
        engine_call = mock_engine_cls.call_args
        tq_cfg = engine_call.kwargs.get("tq_config") or engine_call[1].get("tq_config")
        assert tq_cfg.key_bits == 3
        assert tq_cfg.value_bits == 3

    @patch("tq.cli.start.uvicorn")
    @patch("tq.cli.start.set_engine")
    @patch("tq.cli.start.InferenceEngine")
    @patch("tq.cli.start.pull_model")
    @patch("tq.cli.start.recommend")
    @patch("tq.cli.start.detect_hardware")
    @patch("tq.cli.start.load_global_config")
    @patch("tq.cli.start.ensure_dirs")
    def test_start_custom_port_host(
        self,
        mock_dirs,
        mock_cfg,
        mock_hw,
        mock_rec,
        mock_pull,
        mock_engine_cls,
        mock_set,
        mock_uvicorn,
        runner,
    ):
        from tq.core.config import TQGlobalConfig
        from tq.core.hardware import HardwareProfile

        mock_cfg.return_value = TQGlobalConfig()
        mock_hw.return_value = HardwareProfile(
            gpu_name="Test",
            gpu_vram_mb=8000,
            gpu_compute_cap="8.9",
            system_ram_mb=16000,
            cpu_name="CPU",
            cpu_cores=8,
            os="linux",
            arch="x86_64",
            backend="cuda",
            available_vram_mb=7000,
        )
        mock_rec.return_value = _fake_recommendation()
        mock_engine_cls.return_value = MagicMock()

        result = runner.invoke(start, ["--port", "9000", "--host", "0.0.0.0"])
        assert result.exit_code == 0
        uv_call = mock_uvicorn.run.call_args
        assert uv_call.kwargs.get("port") == 9000 or uv_call[1].get("port") == 9000
        assert uv_call.kwargs.get("host") == "0.0.0.0" or uv_call[1].get("host") == "0.0.0.0"

    @patch("tq.cli.start.uvicorn")
    @patch("tq.cli.start.set_engine")
    @patch("tq.cli.start.InferenceEngine")
    @patch("tq.cli.start.pull_model")
    @patch("tq.cli.start.recommend")
    @patch("tq.cli.start.detect_hardware")
    @patch("tq.cli.start.load_global_config")
    @patch("tq.cli.start.ensure_dirs")
    def test_start_keyboard_interrupt(
        self,
        mock_dirs,
        mock_cfg,
        mock_hw,
        mock_rec,
        mock_pull,
        mock_engine_cls,
        mock_set,
        mock_uvicorn,
        runner,
    ):
        from tq.core.config import TQGlobalConfig
        from tq.core.hardware import HardwareProfile

        mock_cfg.return_value = TQGlobalConfig()
        mock_hw.return_value = HardwareProfile(
            gpu_name="Test",
            gpu_vram_mb=8000,
            gpu_compute_cap="8.9",
            system_ram_mb=16000,
            cpu_name="CPU",
            cpu_cores=8,
            os="linux",
            arch="x86_64",
            backend="cuda",
            available_vram_mb=7000,
        )
        mock_rec.return_value = _fake_recommendation()
        mock_engine_cls.return_value = MagicMock()
        mock_uvicorn.run.side_effect = KeyboardInterrupt()

        result = runner.invoke(start, [])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()
