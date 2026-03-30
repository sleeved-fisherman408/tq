from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tq.core.config import (
    InstalledModel,
    load_global_config,
    load_installed_models,
)


class TestLoadGlobalConfigWithFile:
    def test_load_from_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '[defaults]\nport = 9999\nhost = "0.0.0.0"\n'
            'preferred_quant = "Q8_0"\ndefault_bits = 3\n'
            '[storage]\nmodels_dir = "/custom/models"\n'
        )
        with patch("tq.core.config.GLOBAL_CONFIG_PATH", config_path):
            cfg = load_global_config()
        assert cfg.port == 9999
        assert cfg.host == "0.0.0.0"
        assert cfg.preferred_quant == "Q8_0"
        assert cfg.default_bits == 3
        assert cfg.models_dir == "/custom/models"

    def test_load_partial_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("[defaults]\nport = 7777\n")
        with patch("tq.core.config.GLOBAL_CONFIG_PATH", config_path):
            cfg = load_global_config()
        assert cfg.port == 7777
        assert cfg.host == "127.0.0.1"

    def test_load_corrupt_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("{{invalid toml:::")
        with patch("tq.core.config.GLOBAL_CONFIG_PATH", config_path):
            cfg = load_global_config()
        assert cfg.port == 8000


class TestInstalledModelsEdgeCases:
    def test_load_corrupt_json(self, tmp_path):
        index_path = tmp_path / "models.json"
        index_path.write_text("not valid json{{}}")
        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            result = load_installed_models()
        assert result == []

    def test_load_json_with_bad_schema(self, tmp_path):
        index_path = tmp_path / "models.json"
        index_path.write_text(json.dumps([{"bad": "data"}]))
        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            result = load_installed_models()
        assert result == []

    def test_find_by_model_id(self, tmp_path):
        index_path = tmp_path / "models.json"
        model = InstalledModel(
            model_id="org/model-name",
            shortname="model",
            gguf_path="/tmp/m.gguf",
            quant="Q4_K_M",
            size_bytes=100,
            downloaded_at="2026-01-01",
        )
        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            from tq.core.config import find_installed_model, register_installed_model

            register_installed_model(model)
            found = find_installed_model("org/model-name")
        assert found is not None
        assert found.model_id == "org/model-name"


class TestStopStatusCLI:
    def test_stop_success(self):
        from click.testing import CliRunner

        from tq.cli.stop_status import stop

        mock_resp = MagicMock()
        with patch("tq.cli.stop_status.httpx.post", return_value=mock_resp):
            result = CliRunner().invoke(stop, [])
        assert result.exit_code == 0
        assert "shutting down" in result.output.lower()

    def test_status_tq_disabled(self):
        from click.testing import CliRunner

        from tq.cli.stop_status import status

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "test",
            "turboquant": {
                "enabled": False,
                "key_bits": None,
                "value_bits": None,
                "compression_ratio": None,
            },
            "performance": {"tokens_generated": 0, "uptime_seconds": 5},
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("tq.cli.stop_status.httpx.get", return_value=mock_resp):
            result = CliRunner().invoke(status, [])
        assert result.exit_code == 0
        assert "disabled" in result.output

    def test_status_generic_error(self):
        from click.testing import CliRunner

        from tq.cli.stop_status import status

        with patch("tq.cli.stop_status.httpx.get", side_effect=ValueError("boom")):
            result = CliRunner().invoke(status, [])
        assert result.exit_code == 0
        assert "Error" in result.output


class TestRecommendCLIAlternatives:
    def test_with_alternatives(self):
        from click.testing import CliRunner

        from tq.cli.main import cli
        from tq.core.turbo import TQConfig

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
        main_rec = MagicMock()
        main_rec.model_id = "Main/Model"
        main_rec.quant = "Q4_K_M"
        main_rec.model_size_mb = 4000
        main_rec.max_context_vanilla = 8192
        main_rec.max_context_tq = 32768
        main_rec.tq_config = tq
        main_rec.estimated_tok_s = 30.0
        main_rec.fit_score = 0.95
        main_rec.alternatives = [
            MagicMock(
                model_id="Alt/Model",
                max_context_tq=16384,
                tq_config=tq,
                fit_score=0.80,
            )
        ]

        with (
            patch("tq.cli.recommend.detect_hardware"),
            patch("tq.cli.recommend.recommend", return_value=main_rec),
        ):
            result = CliRunner().invoke(cli, ["recommend"])
        assert result.exit_code == 0
        assert "Alt/Model" in result.output
        assert "Alternatives" in result.output
