from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_CONFIGS_PATH = DATA_DIR / "model_configs.toml"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    params: str
    use_cases: list[str]
    gguf_q4km_size_mb: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    min_vram_mb: int
    recommended_tq: str
    quality_tier: str
    notes: str = ""


ALIASES: dict[str, str] = {
    "qwen3-8b": "Qwen/Qwen3-8B-Instruct",
    "qwen3-8b-instruct": "Qwen/Qwen3-8B-Instruct",
    "coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder-7b-instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "llama3.3-8b": "meta-llama/Llama-3.3-8B-Instruct",
    "llama3.3-8b-instruct": "meta-llama/Llama-3.3-8B-Instruct",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B-Instruct",
    "qwen3.5-4b-instruct": "Qwen/Qwen3.5-4B-Instruct",
}

_MODEL_CACHE: dict[str, ModelConfig] | None = None


def _parse_model_id(raw: str) -> str:
    lower = raw.lower().strip()
    if lower in ALIASES:
        return ALIASES[lower]
    return raw


def load_model_configs() -> dict[str, ModelConfig]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    with open(MODEL_CONFIGS_PATH, "rb") as f:
        data = tomllib.load(f)

    models: dict[str, ModelConfig] = {}
    for model_id, entry in data.get("models", {}).items():
        models[model_id] = ModelConfig(
            model_id=model_id,
            params=entry["params"],
            use_cases=entry["use_cases"],
            gguf_q4km_size_mb=entry["gguf_q4km_size_mb"],
            num_layers=entry["num_layers"],
            num_kv_heads=entry["num_kv_heads"],
            head_dim=entry["head_dim"],
            min_vram_mb=entry["min_vram_mb"],
            recommended_tq=entry["recommended_tq"],
            quality_tier=entry["quality_tier"],
            notes=entry.get("notes", ""),
        )

    _MODEL_CACHE = models
    return models


def resolve_model(name: str) -> ModelConfig:
    model_id = _parse_model_id(name)
    configs = load_model_configs()
    if model_id in configs:
        return configs[model_id]

    lower = model_id.lower()
    for mid, cfg in configs.items():
        if lower in mid.lower():
            return cfg

    available = ", ".join(sorted(ALIASES.keys()))
    raise ValueError(f"Model '{name}' not found. Available: {available}")


def list_available_models() -> list[ModelConfig]:
    return list(load_model_configs().values())
