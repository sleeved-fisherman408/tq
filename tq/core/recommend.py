from __future__ import annotations

from dataclasses import dataclass

from tq.core.hardware import HardwareProfile
from tq.core.models import ModelConfig
from tq.core.turbo import InsufficientVRAM, TQConfig, auto_configure_tq

OVERHEAD_MB = 500


def calc_kv_per_token_bytes(num_layers: int, num_kv_heads: int, head_dim: int) -> int:
    return 2 * num_layers * num_kv_heads * head_dim * 2


def calc_max_context(
    available_vram_bytes: int,
    kv_per_token_bytes: int,
    compression_ratio: float,
) -> int:
    if kv_per_token_bytes <= 0 or compression_ratio <= 0:
        return 0
    return int(available_vram_bytes / (kv_per_token_bytes / compression_ratio))


def calc_compression_ratio(bits: int) -> float:
    return 16.0 / bits


@dataclass(frozen=True)
class ModelRecommendation:
    model_id: str
    gguf_file: str
    quant: str
    param_count: str
    model_size_mb: int
    max_context_vanilla: int
    max_context_tq: int
    kv_cache_per_1k_fp16_mb: float
    kv_cache_per_1k_tq_mb: float
    tq_config: TQConfig
    estimated_tok_s: float
    fit_score: float
    alternatives: list[ModelRecommendation]


def _gguf_filename(model_id: str, quant: str = "Q4_K_M") -> str:
    name = model_id.split("/")[-1]
    return f"{name}-{quant}.gguf"


def _estimate_speed(hw: HardwareProfile, model_cfg: ModelConfig) -> float:
    if hw.backend == "cpu":
        return 5.0
    param_b = float(model_cfg.params.replace("B", ""))
    if param_b <= 4:
        return 65.0
    if param_b <= 8:
        return 45.0
    return 25.0


def _compute_fit_score(
    model_cfg: ModelConfig,
    hw: HardwareProfile,
    max_ctx_tq: int,
    target_ctx: int,
) -> float:
    if max_ctx_tq <= 0:
        return 0.0
    ctx_score = min(max_ctx_tq / target_ctx, 1.0) if target_ctx > 0 else 1.0
    quality_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
    quality_score = quality_map.get(model_cfg.quality_tier, 0.5)
    vram_util = model_cfg.gguf_q4km_size_mb / max(hw.gpu_vram_mb, 1)
    vram_score = 1.0 - abs(vram_util - 0.7)
    return round(0.4 * ctx_score + 0.35 * quality_score + 0.25 * vram_score, 3)


def recommend(
    hardware: HardwareProfile,
    use_case: str = "general",
    min_context: int = 16384,
    prefer_quality: bool = True,
) -> ModelRecommendation:
    from tq.core.models import list_available_models

    candidates: list[tuple[ModelConfig, TQConfig, int, int]] = []

    for model_cfg in list_available_models():
        if use_case not in ("general",) and use_case not in model_cfg.use_cases:
            continue

        vram_for_model = (
            hardware.available_vram_mb if hardware.backend != "cpu" else hardware.system_ram_mb // 2
        )
        if model_cfg.gguf_q4km_size_mb > vram_for_model:
            continue

        try:
            tq_config = auto_configure_tq(
                model_id=model_cfg.model_id,
                available_vram_mb=vram_for_model,
                model_size_mb=model_cfg.gguf_q4km_size_mb,
                target_context=min_context,
                num_layers=model_cfg.num_layers,
                num_kv_heads=model_cfg.num_kv_heads,
                head_dim=model_cfg.head_dim,
                param_count=model_cfg.params,
            )
        except InsufficientVRAM:
            continue

        kv_bytes = calc_kv_per_token_bytes(
            model_cfg.num_layers, model_cfg.num_kv_heads, model_cfg.head_dim
        )
        remaining_bytes = (vram_for_model - model_cfg.gguf_q4km_size_mb - OVERHEAD_MB) * 1024 * 1024
        max_ctx_tq = calc_max_context(remaining_bytes, kv_bytes, tq_config.compression_ratio)
        max_ctx_vanilla = calc_max_context(remaining_bytes, kv_bytes, 1.0)

        candidates.append((model_cfg, tq_config, max_ctx_vanilla, max_ctx_tq))

    if not candidates:
        raise InsufficientVRAM(
            "No model fits your hardware with the requested constraints. "
            "Try a smaller model or reduce --context."
        )

    scored: list[tuple[float, tuple[ModelConfig, TQConfig, int, int]]] = []
    for entry in candidates:
        model_cfg, tq_config, max_ctx_v, max_ctx_t = entry
        score = _compute_fit_score(model_cfg, hardware, max_ctx_t, min_context)
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, (best_cfg, best_tq, best_vanilla, best_tq_ctx) = scored[0]
    alts: list[ModelRecommendation] = []
    for score, (mc, tc, mv, mt) in scored[1:]:
        kv = calc_kv_per_token_bytes(mc.num_layers, mc.num_kv_heads, mc.head_dim)
        alts.append(
            ModelRecommendation(
                model_id=mc.model_id,
                gguf_file=_gguf_filename(mc.model_id),
                quant="Q4_K_M",
                param_count=mc.params,
                model_size_mb=mc.gguf_q4km_size_mb,
                max_context_vanilla=mv,
                max_context_tq=mt,
                kv_cache_per_1k_fp16_mb=kv * 1000 / (1024 * 1024),
                kv_cache_per_1k_tq_mb=kv * 1000 / (tc.compression_ratio * 1024 * 1024),
                tq_config=tc,
                estimated_tok_s=_estimate_speed(hardware, mc),
                fit_score=score,
                alternatives=[],
            )
        )

    kv = calc_kv_per_token_bytes(best_cfg.num_layers, best_cfg.num_kv_heads, best_cfg.head_dim)
    return ModelRecommendation(
        model_id=best_cfg.model_id,
        gguf_file=_gguf_filename(best_cfg.model_id),
        quant="Q4_K_M",
        param_count=best_cfg.params,
        model_size_mb=best_cfg.gguf_q4km_size_mb,
        max_context_vanilla=best_vanilla,
        max_context_tq=best_tq_ctx,
        kv_cache_per_1k_fp16_mb=kv * 1000 / (1024 * 1024),
        kv_cache_per_1k_tq_mb=kv * 1000 / (best_tq.compression_ratio * 1024 * 1024),
        tq_config=best_tq,
        estimated_tok_s=_estimate_speed(hardware, best_cfg),
        fit_score=best_score,
        alternatives=alts,
    )
