# tq — Product Requirements Document

## Overview

**tq** is a CLI tool that lets users run local LLMs with maximum context on minimum hardware. It combines hardware detection, model recommendation, TurboQuant KV cache compression, and an OpenAI-compatible server into a single command.

**Package name:** `tq-serve` (PyPI) | **CLI command:** `tq` | **Repo:** `rohansx/tq`

---

## Problem Statement

Running LLMs locally in 2026 has a brutal, poorly-communicated bottleneck: **GPU memory runs out not because of the model, but because of the conversation.**

### The Memory Math

When you chat with a local LLM, two things consume VRAM:

1. **Model weights** — fixed cost, loaded once (e.g., 4.7 GB for an 8B model at Q4_K_M)
2. **KV cache** — grows linearly with every token in the conversation. This is the model's working memory — how it remembers the beginning of your prompt while generating the end.

On an RTX 4060 (8 GB VRAM):

```
8.0 GB total VRAM
- 4.7 GB model weights
- 0.5 GB system overhead
= 2.8 GB left for KV cache

At FP16: 2.8 GB / 0.15 GB per 1K tokens = ~16K token max context
```

16K tokens is ~500 lines of code. Paste a medium-sized file plus documentation, and the GPU either crashes or spills to system RAM (40 tok/s drops to 3 tok/s).

### Why Users Give Up

- The model is smart enough
- The GPU is fast enough
- The conversation memory is just stored wastefully
- Users go back to paying $200/month for cloud APIs

---

## Solution

TurboQuant (Google Research, ICLR 2026) compresses the KV cache by 4-6x with near-zero quality loss. The algorithm exists, but using it today requires navigating 15+ fragmented repos, understanding KV cache internals, and compiling custom forks.

**tq is the missing product layer** — gluing hardware detection + KV compression + model recommendation + serving into one command:

```bash
pip install tq-serve
tq start --coding
```

RTX 4060 goes from 16K to 80K context. Same model, same quality, same speed.

---

## Target Users

| Segment | Description | Pain Point |
|---|---|---|
| **Cloud API refugees** | Developers paying for cloud APIs who want to run locally for coding, RAG, document Q&A | Don't want to spend hours configuring |
| **Privacy-conscious orgs** | Healthcare, fintech, legal teams needing long-context local inference | Regulatory requirements prevent cloud API use |
| **Hardware-constrained users** | 8-16 GB VRAM users wanting bigger models or longer contexts | GPU "should" support more but doesn't |
| **Local-first advocates** | Open-source enthusiasts wanting maximum capability from existing hardware | Want simple tooling, not PhD-level configuration |

---

## Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Time to first inference | < 3 minutes (including model download) | CLI timer from `tq start` to first response |
| Context improvement | 4-5x over vanilla serving | Compare max in-VRAM context with and without TQ |
| Quality preservation | < 0.5% perplexity increase | Automated benchmark suite (`tq bench`) |
| Speed preservation | < 5% token/s degradation | Benchmark at equivalent context lengths |
| PyPI installs | 1K first month | PyPI download stats |
| GitHub stars | 500 first month | GitHub metrics |

---

## User Scenarios

### Scenario 1: Developer with RTX 4060

```
$ tq start --coding
# Auto-detects GPU, recommends Qwen3-8B, downloads, applies TurboQuant
# Starts OpenAI-compatible server at localhost:8000
# Developer points Cursor/Continue at localhost:8000
# Gets 80K context instead of 16K — multi-file codebase review works
```

### Scenario 2: Privacy-sensitive analyst

```
$ tq start --chat --context 64000
# Long-context chat for reviewing confidential documents
# Everything stays on-device, no data leaves the machine
```

### Scenario 3: Power user with specific model

```
$ tq start qwen2.5-coder-7b --bits 3 --context 100000
# Manual control over model choice, compression level, context target
```

---

## Competitive Landscape

| Tool | What It Does | Gap tq Fills |
|---|---|---|
| Ollama | One-command local LLM serving | No TurboQuant support (expected Q3 2026) |
| llmfit | Hardware detection + model scoring | No serving, no TurboQuant |
| turboquant (back2matching) | KV compression library + basic server | No hardware detection, no model recommendation, no auto-config |
| turboquant_plus (TheTom) | Advanced llama.cpp TurboQuant fork | Must compile from source, no auto-detection |
| LM Studio | GUI for local LLMs | No TurboQuant, closed source |

**tq's moat:** First tool to combine all four pieces (hardware detection + model recommendation + TurboQuant + serving) into a zero-config experience.

**Deprecation plan:** When Ollama ships native TurboQuant (expected Q3 2026), tq pivots to "advanced config" — profiling, asymmetric K/V optimization, and multi-model routing.

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority | Phase |
|---|---|---|---|
| FR-1 | Detect GPU type, VRAM, system RAM, OS automatically | Must | 1 |
| FR-2 | Recommend best model for hardware + use case | Must | 1 |
| FR-3 | Download GGUF models from HuggingFace | Must | 1 |
| FR-4 | Auto-configure TurboQuant (bit-width, K/V settings) | Must | 1 |
| FR-5 | Serve model via OpenAI-compatible API | Must | 1 |
| FR-6 | Support streaming and non-streaming chat completions | Must | 1 |
| FR-7 | Manage installed models (list, pull, remove) | Must | 1 |
| FR-8 | Profile K/V ratios per model layer | Should | 2 |
| FR-9 | Generate asymmetric K/V configurations | Should | 2 |
| FR-10 | Run quality benchmarks (perplexity, needle-in-haystack) | Should | 2 |
| FR-11 | Community config database | Should | 2 |
| FR-12 | MCP server for AI agent integration | Could | 3 |
| FR-13 | Privacy routing (local vs cloud) | Could | 3 |
| FR-14 | Multi-model routing by complexity | Could | 3 |
| FR-15 | Session persistence (save/restore KV cache) | Could | 3 |

### Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Zero-config experience — `tq start --coding` works with no prior setup | Mandatory |
| NFR-2 | Every auto-decision overridable with CLI flags | Mandatory |
| NFR-3 | Fail loudly with actionable suggestions | Mandatory |
| NFR-4 | Python 3.10+ compatibility | Mandatory |
| NFR-5 | Works on Linux (CUDA), macOS (Metal), Windows (CUDA) | Phase 1: Linux, Phase 2: macOS/Windows |
| NFR-6 | < 3 minute cold start (including first model download excluded) | Mandatory |
| NFR-7 | 80%+ test coverage | Mandatory |

---

## Constraints

- **TurboQuant library dependency:** The `turboquant` package (back2matching) is the most complete drop-in integration. If it breaks or is abandoned, tq needs a fallback path.
- **llmfit availability:** llmfit is a Rust binary. If not installed, tq falls back to direct GPU detection (nvidia-smi, torch.cuda, psutil).
- **GGUF ecosystem:** Model recommendations are tied to GGUF quantized models. If the ecosystem shifts, the model database needs updating.
- **GPU-only meaningful performance:** CPU mode is supported but at ~5 tok/s — useful for testing, not for real work.

---

## Out of Scope

- Training or fine-tuning models
- Model weight quantization (use GGUF/GPTQ/AWQ tools for that)
- GUI or web interface (CLI-only for Phase 1-3)
- Cloud deployment or multi-user serving
- Non-transformer architectures (Mamba, RWKV, etc.)
