from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from tq.cli.main import cli
from tq.core.config import InstalledModel


@pytest.fixture
def runner():
    return CliRunner()


class TestMainCLI:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Run local LLMs" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestSystemCommand:
    def test_system(self, runner):
        result = runner.invoke(cli, ["system"])
        assert result.exit_code == 0
        assert "Hardware Profile" in result.output

    def test_system_shows_vram(self, runner):
        result = runner.invoke(cli, ["system"])
        assert result.exit_code == 0
        assert "VRAM" in result.output


class TestRecommendCommand:
    def test_recommend(self, runner):
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "Recommended Model" in result.output

    def test_recommend_coding(self, runner):
        result = runner.invoke(cli, ["recommend", "--coding"])
        assert result.exit_code == 0
        assert "coding" in result.output

    def test_recommend_chat(self, runner):
        result = runner.invoke(cli, ["recommend", "--chat"])
        assert result.exit_code == 0
        assert "chat" in result.output


class TestListCommand:
    def test_list_empty(self, runner):
        with patch("tq.cli.models.list_models", return_value=[]):
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "No models installed" in result.output

    def test_list_with_models(self, runner):
        fake = [
            InstalledModel(
                model_id="Qwen/Qwen3-8B-Instruct",
                shortname="qwen3-8b",
                gguf_path="/tmp/test.gguf",
                quant="Q4_K_M",
                size_bytes=4200000000,
                downloaded_at="2026-03-30",
            )
        ]
        with patch("tq.cli.models.list_models", return_value=fake):
            result = runner.invoke(cli, ["list"])
            assert result.exit_code == 0
            assert "Qwen/Qwen3-8B-Instruct" in result.output
            assert "3.9 GB" in result.output


class TestPullCommand:
    def test_pull(self, runner):
        with patch("tq.cli.models.pull_model") as mock_pull:
            result = runner.invoke(cli, ["pull", "qwen3-8b"])
            assert result.exit_code == 0
            mock_pull.assert_called_once()

    def test_pull_custom_quant(self, runner):
        with patch("tq.cli.models.pull_model") as mock_pull:
            result = runner.invoke(cli, ["pull", "qwen3-8b", "--quant", "Q8_0"])
            assert result.exit_code == 0
            mock_pull.assert_called_once_with("qwen3-8b", "Q8_0")


class TestRemoveCommand:
    def test_remove_found(self, runner):
        with (
            patch("tq.core.download.resolve_model") as mock_resolve,
            patch("tq.core.download.unregister_installed_model", return_value=True),
            patch("tq.core.download.find_installed_model", return_value=None),
        ):
            mock_resolve.return_value = MagicMock(model_id="TestModel")
            result = runner.invoke(cli, ["remove", "test-model", "--yes"])
            assert result.exit_code == 0
            assert "Removed" in result.output

    def test_remove_not_found(self, runner):
        with (
            patch("tq.core.download.resolve_model") as mock_resolve,
            patch("tq.core.download.unregister_installed_model", return_value=False),
        ):
            mock_resolve.return_value = MagicMock(model_id="TestModel")
            result = runner.invoke(cli, ["remove", "nonexistent", "--yes"])
            assert result.exit_code == 0
            assert "not found" in result.output


class TestStopStatus:
    def test_status_with_server(self, runner):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "qwen3-8b",
            "turboquant": {
                "enabled": True,
                "key_bits": 4,
                "value_bits": 4,
                "compression_ratio": 4.0,
            },
            "performance": {"tokens_generated": 100, "uptime_seconds": 60},
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("tq.cli.stop_status.httpx.get", return_value=mock_resp):
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "qwen3-8b" in result.output
            assert "TurboQuant" in result.output

    def test_status_no_server(self, runner):
        with patch("tq.cli.stop_status.httpx.get", side_effect=httpx.ConnectError("no")):
            result = runner.invoke(cli, ["status"])
            assert result.exit_code == 0
            assert "Cannot connect" in result.output

    def test_stop_no_server(self, runner):
        with patch("tq.cli.stop_status.httpx.post", side_effect=httpx.ConnectError("no")):
            result = runner.invoke(cli, ["stop"])
            assert result.exit_code == 0
            assert "No server running" in result.output
