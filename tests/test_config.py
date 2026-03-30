from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tq.core.config import (
    InstalledModel,
    TQGlobalConfig,
    find_installed_model,
    load_global_config,
    load_installed_models,
    register_installed_model,
    save_installed_models,
    unregister_installed_model,
)


class TestGlobalConfig:
    def test_default_config(self):
        cfg = TQGlobalConfig()
        assert cfg.port == 8000
        assert cfg.host == "127.0.0.1"
        assert cfg.preferred_quant == "Q4_K_M"
        assert cfg.default_bits == 4

    def test_load_config_missing_file(self):
        with patch("tq.core.config.GLOBAL_CONFIG_PATH", Path("/nonexistent/config.toml")):
            cfg = load_global_config()
            assert cfg.port == 8000


class TestInstalledModels:
    def test_round_trip(self, tmp_path):
        index_path = tmp_path / "models.json"
        models = [
            InstalledModel(
                model_id="Qwen/Qwen3-8B-Instruct",
                shortname="qwen3-8b",
                gguf_path="/tmp/qwen3.gguf",
                quant="Q4_K_M",
                size_bytes=4800000000,
                downloaded_at="2026-03-30T00:00:00",
            )
        ]

        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            save_installed_models(models)
            loaded = load_installed_models()

        assert len(loaded) == 1
        assert loaded[0].model_id == "Qwen/Qwen3-8B-Instruct"
        assert loaded[0].size_bytes == 4800000000

    def test_register_and_find(self, tmp_path):
        index_path = tmp_path / "models.json"
        model = InstalledModel(
            model_id="Qwen/Qwen3-8B-Instruct",
            shortname="qwen3-8b",
            gguf_path="/tmp/qwen3.gguf",
            quant="Q4_K_M",
            size_bytes=1000,
            downloaded_at="2026-03-30",
        )

        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            register_installed_model(model)
            found = find_installed_model("qwen3-8b")
            assert found is not None
            assert found.model_id == "Qwen/Qwen3-8B-Instruct"

    def test_unregister(self, tmp_path):
        index_path = tmp_path / "models.json"
        model = InstalledModel(
            model_id="test/model",
            shortname="test",
            gguf_path="/tmp/test.gguf",
            quant="Q4_K_M",
            size_bytes=100,
            downloaded_at="2026-03-30",
        )

        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            register_installed_model(model)
            assert unregister_installed_model("test/model") is True
            assert find_installed_model("test") is None

    def test_unregister_nonexistent(self, tmp_path):
        index_path = tmp_path / "models.json"
        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            assert unregister_installed_model("nonexistent") is False

    def test_empty_index(self, tmp_path):
        index_path = tmp_path / "models.json"
        with patch("tq.core.config.MODELS_INDEX_PATH", index_path):
            assert load_installed_models() == []
