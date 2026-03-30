# tq

Run local LLMs with maximum context on minimum hardware. **tq implements [TurboQuant](https://arxiv.org/abs/2502.20365)** (Google Research, ICLR 2026) — a KV cache compression algorithm that gives you 4-6x more context from the same GPU, with near-zero quality loss. One command: detect hardware, pick a model, compress the KV cache, serve via an OpenAI-compatible API.

```bash
pip install tq-serve
tq start --coding
```

## What is TurboQuant?

When you chat with a local LLM, two things eat your GPU memory:

1. **Model weights** — fixed cost, loaded once (~5 GB for an 8B model)
2. **KV cache** — grows linearly with every token in the conversation. This is the model's working memory — how it remembers the start of your prompt while generating the end.

The KV cache is the real bottleneck. On an 8 GB RTX 4060, an 8B model maxes out at ~16K tokens of context — about 500 lines of code. Paste a file plus documentation, and the GPU either crashes or spills to system RAM (speed drops from 40 tok/s to 3 tok/s).

**TurboQuant** solves this by compressing the KV cache at the tensor level. It quantizes the key and value matrices to 3-4 bits (from the default 16-bit FP16) using MSE-optimal quantization with outlier channel preservation. The result:

- **4-6x compression** of the KV cache
- **< 0.5% perplexity increase** (essentially lossless at 4-bit, near-lossless at 3-bit)
- **No retraining required** — works with any transformer model at inference time
- **Zero latency overhead** — decompression is faster than the memory bandwidth savings

tq implements the full TurboQuant pipeline: automatic bit-width selection (3-bit vs 4-bit) based on your available VRAM, asymmetric key/value configuration for smaller models, outlier channel detection, and codebook generation — all configured automatically based on your hardware.

## Before / After

| | Without tq | With tq | Compression |
|---|---|---|---|
| RTX 4060 (8 GB) | 16K context | **80K context** | 4x |
| RTX 4050 (6 GB) | 8K context | **48K context** | 4x |
| RTX 4090 (24 GB) | 64K context | **320K context** | 4x |

Same model, same quality, same speed. The KV cache is just stored more efficiently.

## How It Works

```
tq start --coding
```

1. **Detects hardware** — GPU, VRAM, RAM, CPU via llmfit → nvidia-smi → torch.cuda → psutil fallback chain
2. **Recommends a model** — scores all candidates against your VRAM budget and use case
3. **Configures TurboQuant** — auto-selects 3-bit or 4-bit KV compression, generates codebooks, detects outlier channels
4. **Downloads the model** — from HuggingFace, with resume support
5. **Starts the server** — OpenAI-compatible API at `http://localhost:8000`

Point any OpenAI client at it and go.

## Quick Start

```bash
# Install
pip install tq-serve

# Auto-detect hardware, recommend best coding model, start serving
tq start --coding

# Or for general chat
tq start --chat

# Or specify a model
tq start qwen3-8b

# Check your hardware
tq system

# See what tq recommends for your hardware
tq recommend --coding
```

## Commands

| Command | Description |
|---|---|
| `tq start` | Start serving a model with TurboQuant |
| `tq stop` | Stop the running server |
| `tq status` | Show server status and TQ metrics |
| `tq system` | Display hardware profile |
| `tq recommend` | Show recommended models for your hardware |
| `tq pull <model>` | Download a model |
| `tq list` | List installed models |
| `tq remove <model>` | Remove an installed model |

## Start Options

```bash
tq start [MODEL] [OPTIONS]

Options:
  --coding           Auto-select best coding model
  --chat             Auto-select best chat model
  --context INT      Target context length (default: 16384)
  --bits INT         Force TQ bit-width (3 or 4)
  --port INT         Server port (default: 8000)
  --host TEXT        Server host (default: 127.0.0.1)
  --cpu              Force CPU-only mode
  --verbose          Show detailed TQ configuration
  --json             Output in JSON format
```

## API Endpoints

The server exposes an OpenAI-compatible API:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |
| `/tq/status` | GET | TurboQuant status and metrics |

### Example: Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain KV caches"}],
    "max_tokens": 256
  }'
```

### Example: Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python function"}],
    "stream": true
  }'
```

### Example: Use with Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="tq")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Example: Use with Cursor / Continue

Set the API base URL to `http://localhost:8000/v1` in your editor's AI settings.

## Configuration

tq stores config in `~/.tq/`:

```
~/.tq/
├── config.toml       # Global settings
├── models.json       # Installed model index
├── models/           # Downloaded GGUF files
├── configs/          # TurboQuant configs
└── codebooks/        # Compression codebooks
```

### config.toml

```toml
[defaults]
port = 8000
host = "127.0.0.1"
preferred_quant = "Q4_K_M"
default_bits = 4

[storage]
models_dir = "~/.tq/models"
```

## Supported Models

| Model | Params | VRAM Required | Max Context (tq) |
|---|---|---|---|
| Qwen3-8B-Instruct | 8B | ~5 GB | 80K |
| Qwen2.5-Coder-7B-Instruct | 7B | ~4.5 GB | 64K |
| Llama-3.3-8B-Instruct | 8B | ~5 GB | 80K |
| Qwen3.5-4B | 4B | ~3 GB | 128K |

## Requirements

- Python 3.10+
- NVIDIA GPU (CUDA) for best performance, or CPU mode for testing
- Linux (macOS Metal and Windows CUDA support planned)

## Development

```bash
git clone https://github.com/rohansx/tq.git
cd tq
pip install -e ".[dev]"

# Run unit tests
pytest

# Run integration tests (downloads SmolLM-135M)
TQ_INTEGRATION=1 pytest tests/integration/

# Lint
ruff check tq/ tests/
```

## License

MIT
