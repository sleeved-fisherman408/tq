from __future__ import annotations

from dataclasses import dataclass


class InsufficientVRAM(Exception):
    pass


@dataclass(frozen=True)
class TQConfig:
    key_bits: int
    value_bits: int
    key_method: str
    value_method: str
    outlier_channels: int
    outlier_bits: int
    compression_ratio: float
    estimated_quality: str


def auto_configure_tq(
    model_id: str,
    available_vram_mb: int,
    model_size_mb: int,
    target_context: int = 65536,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    param_count: str = "8B",
) -> TQConfig:
    overhead_mb = 500
    remaining_mb = available_vram_mb - model_size_mb - overhead_mb

    if remaining_mb <= 0:
        raise InsufficientVRAM(
            f"Not enough VRAM for {model_id}. "
            f"Need at least {model_size_mb + overhead_mb} MB, have {available_vram_mb} MB."
        )

    remaining_bytes = remaining_mb * 1024 * 1024
    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2

    param_b = float(param_count.replace("B", ""))

    # Try turbo4 (4-bit symmetric)
    compression_4 = 16.0 / 4
    kv_per_token_tq4 = kv_per_token / compression_4
    max_ctx_4 = int(remaining_bytes / kv_per_token_tq4)

    if max_ctx_4 >= target_context:
        return TQConfig(
            key_bits=4,
            value_bits=4,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=round(compression_4, 1),
            estimated_quality="lossless",
        )

    # Try turbo3 (3-bit)
    compression_3 = 16.0 / 3
    kv_per_token_tq3 = kv_per_token / compression_3
    max_ctx_3 = int(remaining_bytes / kv_per_token_tq3)

    if max_ctx_3 >= target_context:
        if param_b >= 8:
            return TQConfig(
                key_bits=3,
                value_bits=3,
                key_method="mse",
                value_method="mse",
                outlier_channels=32,
                outlier_bits=8,
                compression_ratio=round(compression_3, 1),
                estimated_quality="near-lossless",
            )
        else:
            return TQConfig(
                key_bits=4,
                value_bits=3,
                key_method="mse",
                value_method="mse",
                outlier_channels=32,
                outlier_bits=8,
                compression_ratio=round(16.0 / 3.5, 1),
                estimated_quality="near-lossless",
            )

    # Nothing fits — return best effort turbo3
    if max_ctx_3 > 0:
        return TQConfig(
            key_bits=3,
            value_bits=3,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=round(compression_3, 1),
            estimated_quality="near-lossless",
        )

    raise InsufficientVRAM(
        f"Cannot fit {model_id} with any TurboQuant config in {available_vram_mb} MB VRAM. "
        f"Suggest using a smaller model."
    )
