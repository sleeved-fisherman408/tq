# tq — Architecture Document

## System Overview

tq is a layered system with four primary concerns: CLI interface, core engine, API server, and inference layer. Each layer has a single responsibility and communicates through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Layer (Click + Rich)            │
│  tq start | tq profile | tq list | tq pull | tq bench  │
└───────────┬─────────────────────────────────┬───────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐     ┌───────────────────────────┐
│    Core Engine         │     │     API Server            │
│                       │     │                           │
│  hardware.py          │     │  FastAPI (uvicorn)        │
│  recommend.py         │     │  /v1/chat/completions     │
│  turbo.py             │     │  /v1/models               │
│  profiler.py          │     │  /v1/completions          │
│  config.py            │     │  /health                  │
│                       │     │  /tq/status               │
└───────────┬───────────┘     └─────────────┬─────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│                    Inference Layer                        │
│                                                          │
│  Model Loading: HuggingFace Transformers / llama-cpp-py  │
│  KV Compression: turboquant (TurboQuantCache)            │
│  Codebooks: Pre-computed Lloyd-Max for d=128, d=256      │
└─────────────────────────────────────────────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐     ┌───────────────────────────┐
│  Hardware Detection    │     │  Model Storage             │
│  llmfit (subprocess)   │     │  ~/.tq/models/             │
│  nvidia-smi / sysinfo  │     │  ~/.tq/configs/            │
└───────────────────────┘     └───────────────────────────┘
```

---

## Design Principles

1. **Zero-config by default, full-config when needed.** `tq start` works with no arguments. Every auto-decision can be overridden with flags.
2. **Build on existing libraries.** Don't reimplement TurboQuant or hardware detection. Import and wrap.
3. **Fail loudly and helpfully.** If hardware can't run a model, say why and suggest alternatives.
4. **OpenAI API compatibility.** Any tool that works with OpenAI's API should work with tq unchanged.
5. **Immutable data flow.** Core engine functions take inputs and return new dataclass instances — no mutation of shared state.

---

## Component Architecture

### 1. CLI Layer (`tq/cli/`)

**Responsibility:** Parse user commands, orchestrate core engine calls, render output with Rich.

| Module | Purpose |
|---|---|
| `main.py` | Click group entry point, registers all subcommands |
| `start.py` | `tq start` — orchestrates the full startup flow |
| `profile.py` | `tq profile` — model profiling command (Phase 2) |
| `models.py` | `tq list`, `tq pull`, `tq remove` — model management |

**Key design decisions:**
- Click for CLI parsing (mature, composable, well-documented)
- Rich for terminal output (progress bars, tables, colored status)
- CLI layer is thin — all logic lives in core engine

### 2. Core Engine (`tq/core/`)

**Responsibility:** All business logic — hardware detection, model recommendation, TurboQuant configuration, profiling.

| Module | Purpose | Key Types |
|---|---|---|
| `hardware.py` | Detect GPU, VRAM, RAM, OS | `HardwareProfile` |
| `recommend.py` | Match models to hardware + use case | `ModelRecommendation` |
| `turbo.py` | TurboQuant configuration and application | `TQConfig` |
| `profiler.py` | K/V ratio analysis and benchmarking (Phase 2) | `ProfileResult`, `LayerProfile` |
| `config.py` | Config management (TOML read/write) | `TQGlobalConfig` |

**Key design decisions:**
- Pure functions where possible — take inputs, return dataclass outputs
- Hardware detection uses a priority chain: llmfit → nvidia-smi → torch.cuda → psutil
- Model database is a static TOML file (`data/model_configs.toml`), not a database
- VRAM math is deterministic: `available_vram - model_size - overhead = kv_budget`

### 3. API Server (`tq/server/`)

**Responsibility:** OpenAI-compatible REST API serving the loaded model.

| Module | Purpose |
|---|---|
| `app.py` | FastAPI application, route definitions |
| `models.py` | Pydantic models for OpenAI API request/response schemas |
| `inference.py` | Model inference with TurboQuant cache management |

**Key design decisions:**
- FastAPI for async request handling and auto-generated OpenAPI docs
- Uvicorn as ASGI server
- Global model state (model, tokenizer, tq_cache) initialized once at startup
- SSE streaming for chat completions via `StreamingResponse`

### 4. Inference Layer

**Responsibility:** Load models, manage TurboQuant-compressed KV cache, generate tokens.

**Not a separate directory** — this is the combination of:
- `transformers.AutoModelForCausalLM` or `llama-cpp-python` for model loading
- `turboquant.TurboQuantCache` for KV cache compression
- Pre-computed Lloyd-Max codebooks stored in `~/.tq/codebooks/`

---

## Data Flow

### `tq start --coding` (cold start)

```
User runs: tq start --coding
    │
    ▼
┌─────────────┐
│ 1. Detect   │ hardware.detect() → HardwareProfile
│    Hardware  │ {gpu: "RTX 4060", vram: 8192, ram: 24576, backend: "cuda"}
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 2. Recommend│ recommend(hardware, use_case="coding") → ModelRecommendation
│    Model    │ {model: "qwen3-8b", size: 4800MB, max_ctx_tq: 81920}
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 3. Download │ pull_model("qwen3-8b") → ~/.tq/models/qwen3-8b-q4km.gguf
│    Model    │ (skip if already downloaded)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 4. Configure│ auto_configure_tq(model, vram, size) → TQConfig
│    TQ       │ {key_bits: 4, value_bits: 4, compression: 4.9x}
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 5. Load     │ Model weights → VRAM, TurboQuantCache initialized
│    Model    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 6. Start    │ FastAPI + Uvicorn on http://localhost:8000
│    Server   │ OpenAI-compatible API ready
└─────────────┘
```

### Request Flow (runtime)

```
Client POST /v1/chat/completions
    │
    ▼
┌─────────────┐
│ Tokenize    │ Apply chat template → input_ids tensor
│ Input       │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Model Forward Pass  │ For each generated token:
│                     │   1. Compute K, V vectors (FP16)
│                     │   2. TurboQuantCache.quantize(K, V):
│                     │      a. Rotate with random orthogonal matrix
│                     │      b. Quantize to 4-bit via Lloyd-Max codebook
│                     │      c. Store compressed indices + norms
│                     │   3. Compute attention using compressed KV
│                     │   4. Sample next token
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ Stream      │ SSE events → client
│ Response    │ data: {"choices":[{"delta":{"content":"..."}}]}
└─────────────┘
```

---

## Data Models

### Core Dataclasses

```python
@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str              # "NVIDIA RTX 4060"
    gpu_vram_mb: int           # 8192
    gpu_compute_cap: str       # "8.9"
    system_ram_mb: int         # 24576
    cpu_name: str              # "AMD Ryzen 7 7800X"
    cpu_cores: int             # 16
    os: str                    # "linux"
    arch: str                  # "x86_64"
    backend: str               # "cuda" | "metal" | "cpu"
    available_vram_mb: int     # 7800

@dataclass(frozen=True)
class TQConfig:
    key_bits: int              # 4 or 3
    value_bits: int            # 4 or 3
    key_method: str            # "mse" or "prod"
    value_method: str          # "mse"
    outlier_channels: int      # 32
    outlier_bits: int          # 8
    compression_ratio: float   # 4.9
    estimated_quality: str     # "lossless" | "near-lossless" | "slight-degradation"

@dataclass(frozen=True)
class ModelRecommendation:
    model_id: str              # "Qwen/Qwen3-8B-Instruct"
    gguf_file: str             # "Qwen3-8B-Instruct-Q4_K_M.gguf"
    quant: str                 # "Q4_K_M"
    param_count: str           # "8B"
    model_size_mb: int         # 4800
    max_context_vanilla: int   # 16384
    max_context_tq: int        # 81920
    kv_cache_per_1k_fp16: int  # 150 MB
    kv_cache_per_1k_tq: int    # 30 MB
    tq_config: TQConfig
    estimated_tok_s: float     # 40.0
    fit_score: float           # 0.92
    alternatives: list         # Other ranked options
```

---

## Storage Layout

```
~/.tq/
├── models/                           # GGUF model files
│   ├── qwen3-8b-instruct-q4km.gguf
│   └── qwen2.5-coder-7b-q4km.gguf
├── configs/                          # TurboQuant profiles (per model)
│   ├── qwen3-8b.toml
│   └── custom-model.toml
├── codebooks/                        # Pre-computed Lloyd-Max codebooks
│   ├── d128_2bit.npy
│   ├── d128_3bit.npy
│   └── d128_4bit.npy
└── config.toml                       # Global tq settings
```

---

## Project Structure

```
tq/
├── cli/
│   ├── __init__.py
│   ├── main.py              # CLI entry point (Click group)
│   ├── start.py             # tq start command
│   ├── profile.py           # tq profile command (Phase 2)
│   └── models.py            # tq list/pull/remove commands
├── core/
│   ├── __init__.py
│   ├── hardware.py          # Hardware detection (wraps llmfit + fallbacks)
│   ├── recommend.py         # Model recommendation engine
│   ├── turbo.py             # TurboQuant configuration and application
│   ├── profiler.py          # K/V ratio analysis (Phase 2)
│   └── config.py            # Config management
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI OpenAI-compatible server
│   ├── models.py            # Pydantic models for API schemas
│   └── inference.py         # Model inference with TurboQuant cache
├── data/
│   ├── model_configs.toml   # Pre-tested model configurations
│   └── hardware_profiles.toml
├── tests/
│   ├── test_hardware.py
│   ├── test_recommend.py
│   ├── test_turbo_config.py
│   ├── test_vram_math.py
│   ├── test_api_models.py
│   ├── test_config_db.py
│   └── integration/
│       ├── test_start_flow.py
│       ├── test_api_server.py
│       ├── test_streaming.py
│       └── test_tq_quality.py
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Technology Decisions

| Component | Choice | Rationale |
|---|---|---|
| Language | Python | Entire ML ecosystem is Python; all TurboQuant implementations are Python/PyTorch |
| KV compression | `turboquant` (back2matching) | Most complete drop-in HuggingFace integration with OpenAI server |
| Hardware detection | `llmfit` (subprocess) | Best hardware scoring engine, 157+ models, Rust binary |
| Model loading | HuggingFace Transformers / llama-cpp-python | Standard model loading |
| API server | FastAPI + uvicorn | Async, auto-docs, OpenAI-compatible REST API |
| Model downloads | `huggingface_hub` | GGUF model pulling with resume support |
| CLI framework | Click + Rich | Composable CLI + beautiful terminal output |
| Config storage | TOML | Human-readable, standard Python support |
| Data models | Frozen dataclasses | Immutable, typed, no mutation bugs |

---

## Key Interfaces

### Hardware Detection Priority Chain

```
1. llmfit system --json         → Full hardware profile (if installed)
2. nvidia-smi --query-gpu=...   → NVIDIA GPU name, VRAM total/free
3. torch.cuda.get_device_props  → PyTorch CUDA detection
4. torch.backends.mps           → Apple Silicon detection
5. psutil.virtual_memory()      → System RAM fallback (CPU-only mode)
```

Each level provides progressively less information. The core engine normalizes all sources into the same `HardwareProfile` dataclass.

### TurboQuant Decision Tree

```
Given: model_id, available_vram_mb, model_size_mb, target_context

1. Calculate remaining VRAM = available_vram - model_size - 500MB overhead
2. Try turbo4 (4-bit): can it fit target_context?
   → Yes: use symmetric turbo4 (best quality)
   → No: continue
3. Try turbo3 (3-bit): can it fit target_context?
   → Yes + model >= 8B params: use symmetric turbo3
   → Yes + model < 8B params: use asymmetric (keys=4, values=3)
   → No: raise InsufficientVRAM, suggest smaller model
```

### VRAM Budget Formula

```
kv_per_token_fp16 = 2 * num_layers * num_kv_heads * head_dim * 2 bytes
compression_ratio = 16 / target_bits  (e.g., 16/4 = 4x)
kv_per_token_tq = kv_per_token_fp16 / compression_ratio
max_context = remaining_vram / kv_per_token_tq
```
