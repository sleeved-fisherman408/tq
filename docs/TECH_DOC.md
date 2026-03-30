# tq — Technical Documentation

## Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.10+ | Entire ML ecosystem is Python; all TurboQuant implementations are Python/PyTorch |
| KV compression | `turboquant` (back2matching) | Most complete drop-in HuggingFace integration with OpenAI server |
| Hardware detection | `llmfit` (subprocess) | Best hardware scoring engine, 157+ models, Rust binary |
| Model loading | HuggingFace Transformers / llama-cpp-python | Standard model loading for GGUF and HF formats |
| API server | FastAPI + uvicorn | Async, auto-generated OpenAPI docs, OpenAI-compatible |
| Model downloads | `huggingface_hub` | GGUF model pulling with resume support |
| CLI framework | Click + Rich | Composable CLI commands + progress bars, tables, colors |
| Config storage | TOML | Human-readable, standard Python support via `tomllib` (3.11+) / `tomli` |
| Data models | Frozen dataclasses | Immutable, typed, no mutation bugs |

---

## Dependencies

### Core (`pyproject.toml`)

```toml
[project]
name = "tq-serve"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "turboquant>=0.1",
    "fastapi>=0.100",
    "uvicorn>=0.20",
    "click>=8.0",
    "rich>=13.0",
    "huggingface-hub>=0.20",
    "psutil>=5.9",
    "numpy>=1.24",
]

[project.optional-dependencies]
cuda = ["nvidia-ml-py"]
profiler = ["scipy"]
all = ["tq-serve[cuda,profiler]"]

[project.scripts]
tq = "tq.cli.main:cli"
```

### Optional

| Package | Purpose | When Needed |
|---|---|---|
| `nvidia-ml-py` | NVIDIA GPU detection via Python bindings | `pip install tq-serve[cuda]` |
| `scipy` | K/V ratio analysis statistics | `pip install tq-serve[profiler]` |
| `llmfit` | Rich hardware scoring (Rust binary) | `cargo install llmfit` or `brew install llmfit` |
| `llama-cpp-python` | GGUF model loading alternative | When using GGUF models directly |

---

## Installation

### End User

```bash
# Basic install
pip install tq-serve

# With NVIDIA GPU support
pip install tq-serve[cuda]

# With profiling tools
pip install tq-serve[all]

# Optional: install llmfit for richer hardware recommendations
cargo install llmfit
```

### Development Setup

```bash
git clone https://github.com/rohansx/tq.git
cd tq
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Install dev dependencies
pip install pytest pytest-cov pytest-asyncio httpx ruff mypy

# Run tests
pytest tests/ -v --cov=tq --cov-report=term-missing

# Run linter
ruff check tq/
ruff format tq/

# Type check
mypy tq/ --strict
```

---

## Configuration

### Global Config (`~/.tq/config.toml`)

```toml
[defaults]
port = 8000
host = "127.0.0.1"
preferred_quant = "Q4_K_M"
default_bits = 4                # TurboQuant bit-width

[storage]
models_dir = "~/.tq/models"
configs_dir = "~/.tq/configs"
codebooks_dir = "~/.tq/codebooks"

[server]
max_concurrent_requests = 4
request_timeout_seconds = 300

# Phase 3
[routing]
enabled = false
local_endpoint = "http://localhost:8000"
cloud_endpoint = ""
cloud_api_key = ""
```

### Per-Model Config (`~/.tq/configs/<model>.toml`)

```toml
[meta]
model_id = "Qwen/Qwen3-8B-Instruct"
profiled_on = "2026-04-01"
profiled_by = "tq-profile v0.1.0"
hardware = "RTX 4060 8GB"

[config]
key_bits = 4
value_bits = 4
key_method = "mse"
value_method = "mse"
outlier_channels = 0
compression_ratio = 4.9

[quality]
perplexity_baseline = 8.42
perplexity_tq = 8.45
perplexity_delta_pct = 0.4
cosine_sim = 0.997
verdict = "lossless"
```

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `TQ_DEBUG` | Enable debug logging | `0` |
| `TQ_HOME` | Override ~/.tq directory | `~/.tq` |
| `TQ_PORT` | Override default port | `8000` |
| `TQ_HOST` | Override default host | `127.0.0.1` |
| `HF_HOME` | HuggingFace cache directory | `~/.cache/huggingface` |
| `CUDA_VISIBLE_DEVICES` | Restrict GPU selection | all GPUs |

---

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures (hardware profiles, model configs)
├── test_hardware.py            # Mock GPU detection, verify profile parsing
├── test_recommend.py           # Model selection logic with various hardware configs
├── test_turbo_config.py        # Auto-configuration decision tree
├── test_vram_math.py           # VRAM budget calculations
├── test_api_models.py          # OpenAI API request/response format validation
├── test_config_db.py           # TOML config read/write
└── integration/
    ├── test_start_flow.py      # Full tq start flow with SmolLM-135M
    ├── test_api_server.py      # Start server, send requests, verify responses
    ├── test_streaming.py       # Verify SSE streaming format
    └── test_tq_quality.py      # TQ compression quality verification
```

### Test Fixtures

```python
@pytest.fixture
def rtx_4060():
    return HardwareProfile(
        gpu_name="NVIDIA RTX 4060",
        gpu_vram_mb=8192,
        system_ram_mb=24576,
        backend="cuda",
        ...
    )

@pytest.fixture
def m2_macbook():
    return HardwareProfile(
        gpu_name="Apple M2",
        gpu_vram_mb=16384,
        system_ram_mb=16384,
        backend="metal",
        ...
    )

@pytest.fixture
def low_end_gpu():
    return HardwareProfile(
        gpu_name="NVIDIA GTX 1650",
        gpu_vram_mb=4096,
        system_ram_mb=16384,
        backend="cuda",
        ...
    )
```

### Test Categories

| Category | What | How |
|---|---|---|
| Unit tests | Individual functions, VRAM math, config parsing | Mock hardware, assert outputs |
| Integration tests | Full start flow, API request/response cycle | Use SmolLM-135M (tiny model, no GPU needed) |
| API contract tests | OpenAI-compatible request/response format | Validate against OpenAI API schema |
| Quality tests | TQ compression doesn't break output | Perplexity comparison, cosine similarity |

### Coverage Target

80%+ line coverage. Critical paths (VRAM math, TQ configuration, API response formatting) must be 100%.

---

## Development Commands

```bash
# Run with a tiny model for testing (no GPU needed)
tq start smollm-135m --cpu

# Profile a model
tq profile qwen3-8b --verbose

# Start with full debug logging
TQ_DEBUG=1 tq start qwen3-8b --verbose

# Run specific test file
pytest tests/test_vram_math.py -v

# Run integration tests only
pytest tests/integration/ -v

# Check coverage
pytest tests/ --cov=tq --cov-report=html
open htmlcov/index.html
```

---

## Key External References

| Resource | URL |
|---|---|
| TurboQuant paper | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| Google Research blog | [TurboQuant blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) |
| turboquant library | [github.com/back2matching/turboquant](https://github.com/back2matching/turboquant) |
| llmfit | [github.com/AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) |
| OpenAI API reference | [platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference) |
| Lloyd-Max quantization | [Wikipedia](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) |
