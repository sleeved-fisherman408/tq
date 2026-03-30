# tq

Run local LLMs with maximum context on minimum hardware. One command detects your GPU, picks the best model, applies TurboQuant KV cache compression (4-6x), and starts an OpenAI-compatible server.

```bash
pip install tq-serve
tq start --coding
```

## The Problem

Your GPU runs out of memory not because of the model, but because of the conversation. The KV cache (the model's working memory) grows with every token — on an 8 GB RTX 4060, an 8B model maxes out at ~16K tokens of context. That's about 500 lines of code.

TurboQuant compresses the KV cache by 4-6x with near-zero quality loss. Same model, same speed, 4-6x more context.

**RTX 4060 with tq:** 16K → 80K tokens of context.

## How It Works

```
tq start --coding
```

1. **Detects hardware** — GPU, VRAM, RAM, CPU via llmfit → nvidia-smi → torch.cuda → psutil fallback chain
2. **Recommends a model** — scores all candidates against your VRAM budget and use case
3. **Configures TurboQuant** — auto-selects 3-bit or 4-bit KV compression based on available memory
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

## Before / After

| | Without tq | With tq |
|---|---|---|
| RTX 4060 (8 GB) | 16K context | 80K context |
| RTX 4050 (6 GB) | 8K context | 48K context |
| RTX 4090 (24 GB) | 64K context | 320K context |

Same model, same quality, same speed. The KV cache is just stored more efficiently.

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
