from __future__ import annotations

import pytest

from tq.core.hardware import HardwareProfile, detect_hardware
from tq.core.models import (
    list_available_models,
    load_model_configs,
    resolve_model,
)


class TestHardwareDetection:
    def test_detect_returns_hardware_profile(self):
        hw = detect_hardware()
        assert isinstance(hw, HardwareProfile)
        assert hw.backend in ("cuda", "metal", "cpu")
        assert hw.system_ram_mb > 0
        assert hw.cpu_cores >= 1
        assert hw.os in ("linux", "darwin", "windows")
        assert hw.arch

    def test_hardware_profile_is_frozen(self):
        hw = detect_hardware()
        with pytest.raises(AttributeError):
            hw.gpu_name = "modified"


class TestModelConfigs:
    def test_load_all_models(self):
        configs = load_model_configs()
        assert len(configs) == 4

    def test_qwen3_8b_config(self):
        cfg = resolve_model("qwen3-8b")
        assert cfg.model_id == "Qwen/Qwen3-8B-Instruct"
        assert cfg.params == "8B"
        assert cfg.num_layers == 32
        assert cfg.num_kv_heads == 8
        assert cfg.head_dim == 128

    def test_coder_7b_alias(self):
        cfg = resolve_model("coder-7b")
        assert cfg.model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_llama_alias(self):
        cfg = resolve_model("llama3.3-8b")
        assert cfg.model_id == "meta-llama/Llama-3.3-8B-Instruct"

    def test_qwen35_4b_alias(self):
        cfg = resolve_model("qwen3.5-4b")
        assert cfg.model_id == "Qwen/Qwen3.5-4B-Instruct"

    def test_full_model_id(self):
        cfg = resolve_model("Qwen/Qwen3-8B-Instruct")
        assert cfg.model_id == "Qwen/Qwen3-8B-Instruct"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_model("nonexistent-model")

    def test_list_available_models(self):
        models = list_available_models()
        assert len(models) == 4
        assert all(m.model_id for m in models)

    def test_model_config_fields(self):
        cfg = resolve_model("qwen3-8b")
        assert "coding" in cfg.use_cases
        assert cfg.gguf_q4km_size_mb > 0
        assert cfg.min_vram_mb > 0
        assert cfg.quality_tier in ("high", "medium", "low")

    def test_fuzzy_match(self):
        cfg = resolve_model("qwen3-8b-instruct")
        assert cfg.model_id == "Qwen/Qwen3-8B-Instruct"
