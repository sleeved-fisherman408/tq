# tq — System Design Document

## Feature Specifications

This document contains detailed specifications for every feature, including inputs, outputs, API contracts, data models, and implementation notes.

---

## F1.1: Hardware Detection

**Purpose:** Detect GPU, VRAM, system RAM, and compute capability to determine what models can run.

### Interface

```python
def detect_hardware() -> HardwareProfile:
    """Detect system hardware. No arguments — reads from system."""
    ...
```

### Output: `HardwareProfile`

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
    available_vram_mb: int     # 7800 (after subtracting current usage)
```

### Detection Strategy (priority order)

| Priority | Method | What It Provides |
|---|---|---|
| 1 | `llmfit system --json` | Full profile including model scoring |
| 2 | `nvidia-smi --query-gpu=...` | GPU name, VRAM total/free |
| 3 | `torch.cuda.get_device_properties(0)` | GPU props via PyTorch |
| 4 | `torch.backends.mps.is_available()` | Apple Silicon detection |
| 5 | `psutil.virtual_memory()` | System RAM only (CPU fallback) |

### CLI Output: `tq system`

```
Hardware Profile:
  GPU:           NVIDIA RTX 4060
  VRAM:          8,192 MB (7,800 MB available)
  System RAM:    24,576 MB
  CPU:           AMD Ryzen 7 7800X (16 cores)
  Backend:       CUDA 12.4
  Compute:       8.9

Model budget (with TurboQuant):
  Model weights: up to 5.0 GB
  KV cache:      up to 2.3 GB (turbo4)
  Max context:   ~80K tokens with 8B model
                 ~32K tokens with 14B model (partial offload)
```

### Error Handling

| Condition | Behavior |
|---|---|
| No GPU detected | Warn, offer CPU-only mode with smaller models |
| VRAM < 4 GB | Suggest 1-3B models only |
| llmfit not installed | Use direct detection, suggest `pip install llmfit` |
| nvidia-smi not found | Try torch.cuda, then CPU fallback |

---

## F1.2: Model Recommendation

**Purpose:** Given hardware and use-case, recommend the best model that fits with TurboQuant-enabled context.

### Interface

```python
def recommend(
    hardware: HardwareProfile,
    use_case: str = "general",     # "coding" | "chat" | "reasoning" | "general"
    min_context: int = 16384,      # minimum desired context length
    prefer_quality: bool = True,   # prefer larger model vs longer context
) -> ModelRecommendation:
    ...
```

### Output: `ModelRecommendation`

```python
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
    fit_score: float           # 0.92 (0-1)
    alternatives: list[ModelRecommendation]
```

### VRAM Budget Formula

```python
available_for_kv = gpu_vram - model_size - OVERHEAD_MB  # OVERHEAD_MB = 500

# Without TurboQuant (FP16 KV cache)
kv_per_token_fp16 = 2 * num_layers * num_kv_heads * head_dim * 2  # bytes
max_context_vanilla = available_for_kv / kv_per_token_fp16

# With TurboQuant
compression_ratio = 16 / tq_bits  # e.g., 16/4 = 4x
kv_per_token_tq = kv_per_token_fp16 / compression_ratio
max_context_tq = available_for_kv / kv_per_token_tq
```

### Model Database Format (`data/model_configs.toml`)

```toml
[models."Qwen/Qwen3-8B-Instruct"]
params = "8B"
use_cases = ["coding", "chat", "general"]
gguf_q4km_size_mb = 4800
num_layers = 32
num_kv_heads = 8
head_dim = 128
min_vram_mb = 5500
recommended_tq = "turbo4"
quality_tier = "high"

[models."Qwen/Qwen2.5-Coder-7B-Instruct"]
params = "7B"
use_cases = ["coding"]
gguf_q4km_size_mb = 4200
num_layers = 28
num_kv_heads = 4
head_dim = 128
min_vram_mb = 4900
recommended_tq = "turbo4"
quality_tier = "high"

[models."meta-llama/Llama-3.3-8B-Instruct"]
params = "8B"
use_cases = ["chat", "general", "reasoning"]
gguf_q4km_size_mb = 4900
num_layers = 32
num_kv_heads = 8
head_dim = 128
min_vram_mb = 5600
recommended_tq = "turbo4"
quality_tier = "high"

[models."Qwen/Qwen3.5-4B-Instruct"]
params = "4B"
use_cases = ["coding", "chat"]
gguf_q4km_size_mb = 2500
num_layers = 24
num_kv_heads = 4
head_dim = 128
min_vram_mb = 3200
recommended_tq = "turbo4"
quality_tier = "medium"
notes = "Best option for 6GB VRAM cards"
```

---

## F1.3: TurboQuant Auto-Configuration

**Purpose:** Determine optimal TurboQuant settings for a given model and hardware combination.

### Interface

```python
def auto_configure_tq(
    model_id: str,
    available_vram_mb: int,
    model_size_mb: int,
    target_context: int = 65536,
) -> TQConfig:
    ...
```

### Output: `TQConfig`

```python
@dataclass(frozen=True)
class TQConfig:
    key_bits: int              # 4 (turbo4) or 3 (turbo3)
    value_bits: int            # 4 or 3
    key_method: str            # "mse" or "prod"
    value_method: str          # "mse"
    outlier_channels: int      # 32 (channels kept at higher precision)
    outlier_bits: int          # 8
    compression_ratio: float   # 4.9
    estimated_quality: str     # "lossless" | "near-lossless" | "slight-degradation"
```

### Decision Logic

```
1. Calculate remaining VRAM after model + overhead
2. Try turbo4 (4-bit):
   - If target context fits → return symmetric turbo4 ("lossless")
3. Try turbo3 (3-bit):
   - If fits AND model >= 8B params → return symmetric turbo3 ("near-lossless")
   - If fits AND model < 8B params → return asymmetric keys=4/values=3 ("near-lossless")
   - If doesn't fit → raise InsufficientVRAM with suggestion
```

### Configuration Rules (from community research)

| Model Size | Weight Quant | Recommended TQ | Quality |
|---|---|---|---|
| 8B+ params | Q8_0 or higher | Symmetric turbo3 or turbo4 | Safe for both K and V |
| 8B+ params | Q4_K_M | Keys: q8_0, Values: turbo4 | K precision matters more with low-bit weights |
| 3-7B params | Q4_K_M | Symmetric turbo4 only | turbo3 causes noticeable degradation |
| < 3B params | Any | turbo4 with caution | Test quality carefully |

---

## F1.4: Model Download and Management

**Purpose:** Download GGUF models from HuggingFace, manage local storage.

### Commands

| Command | Purpose |
|---|---|
| `tq pull <model>` | Download a model |
| `tq list` | List installed models |
| `tq remove <model>` | Remove an installed model |

### Model Name Resolution

Users type short names (e.g., `qwen3-8b`). tq resolves to full HuggingFace identifiers via the model database. Unambiguous prefix matching is supported.

```
qwen3-8b     → Qwen/Qwen3-8B-Instruct (Q4_K_M)
llama3.3-8b  → meta-llama/Llama-3.3-8B-Instruct (Q4_K_M)
coder-7b     → Qwen/Qwen2.5-Coder-7B-Instruct (Q4_K_M)
```

### Implementation

```python
from huggingface_hub import hf_hub_download

def pull_model(model_shortname: str, quant: str = "Q4_K_M") -> Path:
    model_info = MODEL_REGISTRY[model_shortname]
    path = hf_hub_download(
        repo_id=model_info.hf_repo,
        filename=model_info.gguf_filename(quant),
        local_dir=TQ_MODELS_DIR,
        resume_download=True,
    )
    register_model(model_shortname, path, model_info)
    return path
```

### Storage

```
~/.tq/models/
├── qwen3-8b-instruct-q4km.gguf      # 5.2 GB
└── qwen2.5-coder-7b-q4km.gguf       # 4.2 GB
```

---

## F1.5: OpenAI-Compatible API Server

**Purpose:** Expose the running model as an OpenAI-compatible REST API.

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| POST | `/v1/completions` | Text completion |
| GET | `/v1/models` | List available models |
| GET | `/health` | Server health check |
| GET | `/tq/status` | TurboQuant-specific metrics |

### Chat Completion Request

```json
{
  "model": "qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to merge two sorted lists."}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": true
}
```

### Chat Completion Response (streaming)

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711234567,"model":"qwen3-8b","choices":[{"index":0,"delta":{"role":"assistant","content":"def "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711234567,"model":"qwen3-8b","choices":[{"index":0,"delta":{"content":"merge_sorted"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1711234567,"model":"qwen3-8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Chat Completion Response (non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1711234567,
  "model": "qwen3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "def merge_sorted(a, b):\n    ..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

### TQ Status Endpoint Response

```json
{
  "model": "qwen3-8b-instruct",
  "turboquant": {
    "enabled": true,
    "key_bits": 4,
    "value_bits": 4,
    "compression_ratio": 4.9,
    "kv_cache_size_mb": 384,
    "kv_cache_tokens": 12800,
    "max_tokens_available": 81920,
    "quality_estimate": "lossless"
  },
  "hardware": {
    "gpu": "NVIDIA RTX 4060",
    "vram_total_mb": 8192,
    "vram_used_mb": 5484,
    "vram_free_mb": 2708
  },
  "performance": {
    "tokens_generated": 4521,
    "avg_tok_s": 41.2,
    "uptime_seconds": 3600
  }
}
```

---

## F1.6: CLI Interface

### Command Tree

```
tq                              # Show help
tq start [model]                # Start serving a model with TurboQuant
tq start --coding               # Auto-select best coding model
tq start --chat                 # Auto-select best chat model
tq stop                         # Stop the running server
tq status                       # Show server status and TQ metrics
tq system                       # Show detected hardware
tq recommend [--coding|--chat]  # Recommend models for your hardware
tq pull <model>                 # Download a model
tq list                         # List installed models
tq remove <model>               # Remove an installed model
tq profile <model>              # [Phase 2] Profile optimal TQ config
tq bench <model>                # [Phase 2] Run quality benchmarks
```

### Global Options

```
--context <n>      Target context length (default: auto-maximize)
--bits <n>         Force TQ bit-width (3 or 4, default: auto)
--port <n>         Server port (default: 8000)
--host <addr>      Server host (default: 127.0.0.1)
--verbose          Show detailed TQ configuration
--json             Output in JSON format
--cpu              Force CPU-only mode
```

---

## Error Handling Matrix

| Error | User Message | Resolution |
|---|---|---|
| No GPU detected | "No GPU found. tq can run on CPU but will be much slower (~5 tok/s). Continue? [y/N]" | Offer CPU mode with 3B model |
| VRAM too low for model | "RTX 4060 has 8 GB VRAM, but qwen3-32b needs 20 GB. Try: `tq start qwen3-8b`" | Suggest fitting model |
| Model not found | "Model 'qwen4-8b' not found. Did you mean 'qwen3-8b'?" | Fuzzy match suggestion |
| Download interrupted | "Download interrupted. Run `tq pull qwen3-8b` to resume." | Resume download |
| Port in use | "Port 8000 is already in use. Try: `tq start --port 8001`" | Suggest alternative |
| CUDA OOM during inference | "GPU ran out of memory at 52K tokens. Reducing context to 48K and restarting." | Auto-reduce context |
| TQ quality degradation | "Warning: turbo3 on this 4B model shows 3.2% perplexity increase. Switching to turbo4." | Auto-upgrade config |
| turboquant not installed | "Required dependency 'turboquant' not found. Run: `pip install turboquant`" | Install instructions |
| Model file corrupted | "Model file checksum mismatch. Run `tq pull --force qwen3-8b` to re-download." | Force re-download |
