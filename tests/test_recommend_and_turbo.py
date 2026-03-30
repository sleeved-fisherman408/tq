from __future__ import annotations

import pytest

from tq.core.hardware import HardwareProfile
from tq.core.recommend import (
    InsufficientVRAM,
    ModelRecommendation,
    calc_compression_ratio,
    calc_kv_per_token_bytes,
    calc_max_context,
    recommend,
)
from tq.core.turbo import auto_configure_tq


def _rtx_4060() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA RTX 4060",
        gpu_vram_mb=8192,
        gpu_compute_cap="8.9",
        system_ram_mb=24576,
        cpu_name="AMD Ryzen 7 7800X",
        cpu_cores=16,
        os="linux",
        arch="x86_64",
        backend="cuda",
        available_vram_mb=7800,
    )


def _m2_macbook() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Apple M2",
        gpu_vram_mb=16384,
        gpu_compute_cap="",
        system_ram_mb=16384,
        cpu_name="Apple M2",
        cpu_cores=8,
        os="darwin",
        arch="arm64",
        backend="metal",
        available_vram_mb=16384,
    )


def _gtx_1650() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA GTX 1650",
        gpu_vram_mb=4096,
        gpu_compute_cap="7.5",
        system_ram_mb=16384,
        cpu_name="Intel i5",
        cpu_cores=6,
        os="linux",
        arch="x86_64",
        backend="cuda",
        available_vram_mb=3800,
    )


def _cpu_only() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="No GPU",
        gpu_vram_mb=0,
        gpu_compute_cap="",
        system_ram_mb=16384,
        cpu_name="Intel i7",
        cpu_cores=8,
        os="linux",
        arch="x86_64",
        backend="cpu",
        available_vram_mb=0,
    )


class TestVRAMMath:
    def test_kv_per_token_calculation(self):
        kv = calc_kv_per_token_bytes(num_layers=32, num_kv_heads=8, head_dim=128)
        assert kv == 2 * 32 * 8 * 128 * 2
        assert kv == 131072

    def test_max_context_with_turbo4(self):
        kv = calc_kv_per_token_bytes(32, 8, 128)
        available = (7800 - 4800 - 500) * 1024 * 1024
        ctx = calc_max_context(available, kv, 4.0)
        assert ctx > 60000

    def test_max_context_vanilla(self):
        kv = calc_kv_per_token_bytes(32, 8, 128)
        available = (7800 - 4800 - 500) * 1024 * 1024
        ctx = calc_max_context(available, kv, 1.0)
        assert ctx > 0
        assert ctx < 25000

    def test_zero_kv_returns_zero(self):
        assert calc_max_context(1000000, 0, 4.0) == 0

    def test_compression_ratio(self):
        assert calc_compression_ratio(4) == 4.0
        assert calc_compression_ratio(3) == pytest.approx(5.333, rel=0.01)


class TestTurboConfig:
    def test_turbo4_when_enough_vram(self):
        cfg = auto_configure_tq(
            model_id="Qwen/Qwen3-8B-Instruct",
            available_vram_mb=7800,
            model_size_mb=4800,
            target_context=65536,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            param_count="8B",
        )
        assert cfg.key_bits == 4
        assert cfg.value_bits == 4
        assert cfg.compression_ratio == 4.0
        assert cfg.estimated_quality == "lossless"

    def test_turbo3_for_large_model(self):
        cfg = auto_configure_tq(
            model_id="Qwen/Qwen3-8B-Instruct",
            available_vram_mb=6000,
            model_size_mb=4800,
            target_context=65536,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            param_count="8B",
        )
        assert cfg.key_bits == 3
        assert cfg.value_bits == 3
        assert cfg.estimated_quality == "near-lossless"

    def test_asymmetric_for_small_model(self):
        cfg = auto_configure_tq(
            model_id="Qwen/Qwen3.5-4B-Instruct",
            available_vram_mb=5000,
            model_size_mb=2500,
            target_context=200000,
            num_layers=24,
            num_kv_heads=4,
            head_dim=128,
            param_count="4B",
        )
        assert cfg.key_bits == 4
        assert cfg.value_bits == 3
        assert cfg.estimated_quality == "near-lossless"

    def test_insufficient_vram_raises(self):
        with pytest.raises(InsufficientVRAM):
            auto_configure_tq(
                model_id="Qwen/Qwen3-8B-Instruct",
                available_vram_mb=3000,
                model_size_mb=4800,
                target_context=16384,
            )

    def test_tq_config_is_frozen(self):
        cfg = auto_configure_tq(
            model_id="Qwen/Qwen3-8B-Instruct",
            available_vram_mb=7800,
            model_size_mb=4800,
        )
        with pytest.raises(AttributeError):
            cfg.key_bits = 3

    def test_best_effort_turbo3_when_target_unreachable(self):
        cfg = auto_configure_tq(
            model_id="Qwen/Qwen3-8B-Instruct",
            available_vram_mb=6000,
            model_size_mb=4800,
            target_context=9999999,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            param_count="8B",
        )
        assert cfg.key_bits == 3
        assert cfg.value_bits == 3
        assert cfg.estimated_quality == "near-lossless"

    def test_default_params(self):
        cfg = auto_configure_tq(
            model_id="Qwen/Qwen3-8B-Instruct",
            available_vram_mb=7800,
            model_size_mb=4800,
        )
        assert cfg.key_bits in (3, 4)
        assert cfg.key_method == "mse"
        assert cfg.value_method == "mse"
        assert cfg.outlier_channels == 32
        assert cfg.outlier_bits == 8


class TestRecommendation:
    def test_recommend_coding_on_rtx_4060(self):
        rec = recommend(_rtx_4060(), use_case="coding")
        assert isinstance(rec, ModelRecommendation)
        assert rec.model_id
        assert rec.max_context_tq > rec.max_context_vanilla
        assert rec.tq_config.key_bits in (3, 4)
        assert rec.fit_score > 0

    def test_recommend_chat(self):
        rec = recommend(_rtx_4060(), use_case="chat")
        assert rec.model_id

    def test_recommend_on_low_vram(self):
        rec = recommend(_gtx_1650(), use_case="general")
        assert rec.model_id
        assert rec.model_size_mb < 4096

    def test_recommend_has_alternatives(self):
        rec = recommend(_rtx_4060(), use_case="general")
        assert isinstance(rec.alternatives, list)

    def test_recommend_cpu_mode(self):
        rec = recommend(_cpu_only(), use_case="general")
        assert rec.model_id
        assert rec.estimated_tok_s == 5.0

    def test_recommend_on_m2(self):
        rec = recommend(_m2_macbook(), use_case="coding")
        assert rec.model_id

    def test_recommend_fit_score_between_0_and_1(self):
        for hw in [_rtx_4060(), _gtx_1650()]:
            rec = recommend(hw, use_case="general")
            assert 0 < rec.fit_score <= 1.0

    def test_kv_cache_per_1k_positive(self):
        rec = recommend(_rtx_4060(), use_case="coding")
        assert rec.kv_cache_per_1k_fp16_mb > 0
        assert rec.kv_cache_per_1k_tq_mb > 0
        assert rec.kv_cache_per_1k_tq_mb < rec.kv_cache_per_1k_fp16_mb

    def test_recommend_general_includes_all_models(self):
        rec = recommend(_rtx_4060(), use_case="general")
        assert rec.model_id

    def test_recommend_coding_filters_non_coding(self):
        rec = recommend(_rtx_4060(), use_case="coding")
        all_ids = [rec.model_id] + [a.model_id for a in rec.alternatives]
        from tq.core.models import load_model_configs

        for mid in all_ids:
            cfg = load_model_configs()[mid]
            assert "coding" in cfg.use_cases

    def test_recommend_model_recommendation_is_frozen(self):
        rec = recommend(_rtx_4060(), use_case="general")
        with pytest.raises(AttributeError):
            rec.model_id = "modified"

    def test_recommend_high_min_context_on_small_gpu(self):
        hw = _gtx_1650()
        rec = recommend(hw, use_case="general", min_context=8192)
        assert rec.model_id
        assert rec.fit_score > 0

    def test_recommend_gguf_file_has_extension(self):
        rec = recommend(_rtx_4060(), use_case="general")
        assert rec.gguf_file.endswith(".gguf")
