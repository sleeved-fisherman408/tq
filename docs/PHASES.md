# tq — Phase Plan

## Timeline Overview

| Phase | Timeline | Focus | Deliverable |
|---|---|---|---|
| Phase 1 — MVP | Week 1-2 | Core functionality | `tq start`, hardware detection, model recommendation, TurboQuant auto-config, OpenAI API server |
| Phase 2 — Smart Profiling | Week 3-4 | Optimization | `tq profile`, K/V ratio analysis, asymmetric configs, config database, benchmarks |
| Phase 3 — Ecosystem | Month 2-3 | Integration | MCP server, privacy routing, multi-model, session persistence |
| Deprecation pivot | Q3 2026 | Specialization | When Ollama ships native TurboQuant, tq becomes the "advanced config" tool |

---

## Phase 1 — MVP (Week 1-2)

### Goal

A single command that detects hardware, downloads the right model, applies TurboQuant, and starts an OpenAI-compatible server. Zero configuration required.

### Features

| Feature | Command | Description |
|---|---|---|
| Hardware detection | `tq system` | Auto-detect GPU, VRAM, RAM, OS via llmfit + fallbacks |
| Model recommendation | `tq recommend` | Match model to hardware with TurboQuant-aware VRAM math |
| Model management | `tq pull/list/remove` | Download GGUF models from HuggingFace, manage local storage |
| TurboQuant auto-config | (internal) | Select optimal bit-width (turbo3/turbo4) based on model size and VRAM |
| OpenAI-compatible API | `tq start` | Drop-in replacement server at localhost:8000 |
| Server management | `tq stop/status` | Stop server, show status + TQ metrics |

### Acceptance Criteria

- [ ] `tq start --coding` works end-to-end on a Linux machine with NVIDIA GPU
- [ ] Hardware is correctly detected (GPU name, VRAM, RAM)
- [ ] Correct model is recommended for the detected hardware
- [ ] Model downloads with progress bar and resume support
- [ ] TurboQuant is applied automatically with correct bit-width
- [ ] Server responds to OpenAI-compatible chat completion requests
- [ ] Streaming (SSE) works correctly
- [ ] Context window is 4-5x larger than vanilla serving
- [ ] Speed degradation is < 5% at equivalent context lengths
- [ ] All error cases produce helpful messages with suggestions
- [ ] 80%+ test coverage

### Implementation Order

```
1. Project scaffolding (pyproject.toml, directory structure, CLI skeleton)
2. Hardware detection (core/hardware.py)
3. Model database (data/model_configs.toml)
4. VRAM math and recommendation engine (core/recommend.py)
5. TurboQuant configuration (core/turbo.py)
6. Model download (core integrated with huggingface_hub)
7. Config management (core/config.py)
8. Inference layer (server/inference.py)
9. API server (server/app.py, server/models.py)
10. CLI commands (cli/start.py, cli/models.py)
11. Integration testing
12. README and docs
```

### Key Risks

| Risk | Mitigation |
|---|---|
| `turboquant` library API changes | Pin version, wrap with thin adapter |
| llmfit not available on all systems | Multiple fallback detection methods |
| GGUF model availability varies | Start with well-known models (Qwen3, Llama3.3) |
| Different GPU architectures behave differently | Test on CUDA first, macOS Metal in Phase 2 |

---

## Phase 2 — Smart Profiling (Week 3-4)

### Goal

Move beyond one-size-fits-all TurboQuant configs. Profile each model's KV cache behavior to find the optimal compression settings, potentially different for keys vs values and different per layer.

### Features

| Feature | Command | Description |
|---|---|---|
| K/V ratio analysis | `tq profile <model>` | Analyze per-layer key vs value magnitude ratios |
| Asymmetric config | (auto) | Recommend different bit-widths for keys vs values |
| Layer-adaptive mode | (auto) | Keep quality-critical layers at higher precision |
| Quality benchmarks | `tq bench <model>` | Perplexity, needle-in-haystack, cosine similarity |
| Config database | (auto) | Community-contributed optimal configs per model |

### F2.1: K/V Ratio Analysis

**Why it matters:** Some models have K/V magnitude ratios over 100x, meaning keys need much more precision than values. Symmetric configs waste either quality or memory.

**Process:**
1. Load model
2. Run forward pass on calibration prompt (1-2K tokens)
3. Capture raw KV cache tensors at every layer
4. Compute per-layer: K norm mean, V norm mean, K/V ratio, outlier percentage

**Output example:**
```
Layer  0: K_rms=  4.21  V_rms= 0.34  ratio= 12.4x  ⚠ high
Layer 14: K_rms= 15.62  V_rms= 0.29  ratio= 53.9x  ⚠ very high
Layer 27: K_rms=  3.44  V_rms= 0.52  ratio=  6.6x

Recommendation:
  Keys:   q8_0 (8-bit)
  Values: turbo4 (4-bit TurboQuant)
  Effective compression: 3.2x
```

### F2.2: Asymmetric Configuration Modes

| Mode | Keys | Values | Compression | Use Case |
|---|---|---|---|---|
| Symmetric turbo4 | 4-bit | 4-bit | 4.9x | Default for 8B+ with Q8 weights |
| Symmetric turbo3 | 3-bit | 3-bit | 6.4x | Max compression for 8B+ |
| Asymmetric safe | q8_0 | turbo4 | 3.2x | High K/V ratio or Q4 weights |
| Asymmetric aggressive | turbo4 | turbo3 | 5.5x | Balance compression vs quality |
| Layer-adaptive | Mixed | Mixed | 3.5x | Last N layers at q8_0, rest turbo3 |

### F2.3: Quality Benchmarks (`tq bench`)

| Benchmark | Method | Pass Criteria |
|---|---|---|
| Perplexity | wikitext-103, 50 chunks, FP16 vs TQ | < 0.5% increase |
| Needle-in-a-haystack | Hidden fact at 8K/16K/32K | 100% retrieval |
| Cosine similarity | Attention scores FP16 vs TQ | > 0.995 |
| Top-1 match | Highest-attended token preserved | > 97% |

### F2.4: Config Database

Configs stored in repo under `configs/` and pulled automatically. Format:

```toml
[meta]
model_id = "Qwen/Qwen3-8B-Instruct"
profiled_on = "2026-04-01"
hardware = "RTX 4060 8GB"

[config]
key_bits = 4
value_bits = 4
compression_ratio = 4.9

[quality]
perplexity_delta_pct = 0.4
cosine_sim = 0.997
verdict = "lossless"
```

### Acceptance Criteria

- [ ] `tq profile` runs on any supported model and produces K/V analysis
- [ ] Asymmetric configs are auto-generated when K/V ratio > 10x
- [ ] `tq bench` runs 4 benchmark types and produces pass/fail verdict
- [ ] Config database has entries for top 10 popular models
- [ ] Profiling results are cached and reused

---

## Phase 3 — Ecosystem (Month 2-3)

### Goal

Integrate tq into the broader AI tooling ecosystem — let AI agents manage local inference, route between local and cloud, and support multi-model setups.

### Features

| Feature | Description |
|---|---|
| MCP server | Let AI agents (Claude Code, etc.) manage local inference |
| Privacy routing | Route sensitive queries to local, general queries to cloud |
| Multi-model | Fast small model + slow large model, route by complexity |
| Session persistence | Save/restore KV cache across restarts |

### F3.1: MCP Server

Expose tq capabilities via Model Context Protocol:

| Tool | Description |
|---|---|
| `tq_start` | Start a model with specified config |
| `tq_stop` | Stop the running model |
| `tq_status` | Get current model, VRAM usage, context usage |
| `tq_recommend` | Get model recommendations for hardware |
| `tq_switch` | Switch to a different model |
| `tq_query` | Send a prompt to the running model |

### F3.2: Privacy Routing (CloakPipe Integration)

```
User request → CloakPipe Proxy (localhost:9000)
    ├── PII detected → Route to local tq server (localhost:8000)
    └── No PII       → Route to cloud API (Claude/OpenAI)
```

Configuration:
```toml
[routing]
enabled = true
local_endpoint = "http://localhost:8000"
cloud_endpoint = "https://api.anthropic.com"

[routing.pii_patterns]
aadhaar = true
pan = true
health_keywords = true
financial_data = true
custom_patterns = ["CONFIDENTIAL", "INTERNAL ONLY"]
```

### F3.3: Multi-Model Routing

Run two models simultaneously:
- **Fast model** (3-4B): Simple questions, autocomplete, quick lookups
- **Large model** (8-14B): Complex reasoning, code generation, analysis

Router decides based on prompt complexity, length, and user hints.

### F3.4: Session Persistence

Save compressed KV cache to disk, restore on server restart. Enables:
- Resume long conversations after tq restart
- Pre-warm context with common system prompts
- Share context snapshots between machines

### Acceptance Criteria

- [ ] MCP server works with Claude Code
- [ ] Privacy routing correctly classifies PII vs non-PII
- [ ] Multi-model routing reduces average latency by 30%+
- [ ] Session save/restore preserves conversation state

---

## Deprecation Strategy

**Trigger:** Ollama ships native TurboQuant support (expected Q3 2026).

**Pivot:** tq becomes the "advanced config" tool:
- **Profiling:** `tq profile` generates optimal configs that Ollama can consume
- **Benchmarking:** `tq bench` validates quality across configs
- **Optimization:** Asymmetric K/V, layer-adaptive, and per-model tuning
- **Integration:** MCP server and privacy routing remain unique value

**Migration path:** Users running `tq start` get a message suggesting Ollama for basic usage, with tq recommended for advanced optimization workflows.
