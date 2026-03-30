from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


TQ_HOME = Path(os.environ.get("TQ_HOME", Path.home() / ".tq"))
MODELS_DIR = TQ_HOME / "models"
CONFIGS_DIR = TQ_HOME / "configs"
CODEBOOKS_DIR = TQ_HOME / "codebooks"
GLOBAL_CONFIG_PATH = TQ_HOME / "config.toml"
MODELS_INDEX_PATH = TQ_HOME / "models.json"


@dataclass(frozen=True)
class TQGlobalConfig:
    port: int = 8000
    host: str = "127.0.0.1"
    preferred_quant: str = "Q4_K_M"
    default_bits: int = 4
    models_dir: str = str(MODELS_DIR)
    configs_dir: str = str(CONFIGS_DIR)


DEFAULT_CONFIG = TQGlobalConfig()


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    CODEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def load_global_config() -> TQGlobalConfig:
    if not GLOBAL_CONFIG_PATH.exists():
        return DEFAULT_CONFIG
    try:
        with open(GLOBAL_CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        defaults = data.get("defaults", {})
        storage = data.get("storage", {})
        return TQGlobalConfig(
            port=defaults.get("port", DEFAULT_CONFIG.port),
            host=defaults.get("host", DEFAULT_CONFIG.host),
            preferred_quant=defaults.get("preferred_quant", DEFAULT_CONFIG.preferred_quant),
            default_bits=defaults.get("default_bits", DEFAULT_CONFIG.default_bits),
            models_dir=storage.get("models_dir", DEFAULT_CONFIG.models_dir),
            configs_dir=storage.get("configs_dir", DEFAULT_CONFIG.configs_dir),
        )
    except Exception:
        return DEFAULT_CONFIG


@dataclass
class InstalledModel:
    model_id: str
    shortname: str
    gguf_path: str
    quant: str
    size_bytes: int
    downloaded_at: str


def load_installed_models() -> list[InstalledModel]:
    if not MODELS_INDEX_PATH.exists():
        return []
    try:
        with open(MODELS_INDEX_PATH) as f:
            data = json.load(f)
        return [InstalledModel(**m) for m in data]
    except (json.JSONDecodeError, TypeError):
        return []


def save_installed_models(models: list[InstalledModel]) -> None:
    ensure_dirs()
    with open(MODELS_INDEX_PATH, "w") as f:
        json.dump([asdict(m) for m in models], f, indent=2)


def register_installed_model(model: InstalledModel) -> None:
    models = load_installed_models()
    models = [m for m in models if m.model_id != model.model_id]
    models.append(model)
    save_installed_models(models)


def unregister_installed_model(model_id: str) -> bool:
    models = load_installed_models()
    original_len = len(models)
    models = [m for m in models if m.model_id != model_id]
    if len(models) < original_len:
        save_installed_models(models)
        return True
    return False


def find_installed_model(shortname: str) -> InstalledModel | None:
    for m in load_installed_models():
        if m.shortname == shortname or m.model_id == shortname:
            return m
    return None
