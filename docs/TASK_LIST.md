# tq — Implementation Task List (Phase 1 MVP)

## Task Dependencies

```
T1 (Scaffolding) ──┬── T2 (Hardware Detection) ──┐
                   │                              │
                   ├── T3 (Model Database)────────┤
                   │                              │
                   │                              ▼
                   │                    T4 (Recommendation Engine)
                   │                              │
                   ├── T5 (TurboQuant Config) ────┤
                   │                              │
                   ├── T6 (Model Download) ───────┤
                   │                              │
                   ├── T7 (Config Management) ────┤
                   │                              ▼
                   │                    T8 (Inference Layer)
                   │                              │
                   │                              ▼
                   │                    T9 (API Server)
                   │                              │
                   │                              ▼
                   └──────────────────► T10 (CLI Commands)
                                                  │
                                                  ▼
                                        T11 (Integration Tests)
                                                  │
                                                  ▼
                                        T12 (README)
```

---

## Tasks

### T1: Project Scaffolding
**Priority:** P0 | **Blocks:** All other tasks

- [ ] Create `pyproject.toml` with dependencies, scripts entry point, metadata
- [ ] Create directory structure: `tq/cli/`, `tq/core/`, `tq/server/`, `tq/data/`, `tests/`, `tests/integration/`
- [ ] Add `__init__.py` files
- [ ] Create CLI skeleton with Click group in `tq/cli/main.py`
- [ ] Set up `ruff` config for linting/formatting
- [ ] Set up `pytest` config
- [ ] Add `.gitignore` (Python, models, codebooks)
- [ ] Add MIT `LICENSE`
- [ ] Verify `pip install -e .` works and `tq --help` runs

**Acceptance:** `tq --help` shows command list, tests can be discovered by pytest.

---

### T2: Hardware Detection (`tq/core/hardware.py`)
**Priority:** P0 | **Blocked by:** T1

- [ ] Define `HardwareProfile` frozen dataclass
- [ ] Implement `detect_hardware() -> HardwareProfile`
- [ ] Detection chain: llmfit → nvidia-smi → torch.cuda → torch.mps → psutil
- [ ] Parse llmfit JSON output via subprocess
- [ ] Parse nvidia-smi CSV output via subprocess
- [ ] Compute `available_vram_mb` (total - currently used)
- [ ] Handle no-GPU gracefully (return backend="cpu")
- [ ] Add `tq system` CLI command to display hardware profile
- [ ] Write unit tests with mocked subprocess calls

**Acceptance:** `tq system` shows correct GPU, VRAM, RAM on the dev machine. Tests pass with mocked outputs for all detection paths.

---

### T3: Model Database (`tq/data/model_configs.toml`)
**Priority:** P0 | **Blocked by:** T1

- [ ] Create TOML schema for model entries
- [ ] Add entries for initial models:
  - Qwen3-8B-Instruct
  - Qwen2.5-Coder-7B-Instruct
  - Llama-3.3-8B-Instruct
  - Qwen3.5-4B-Instruct
- [ ] Include per-model: params, use_cases, gguf_size, num_layers, num_kv_heads, head_dim, min_vram, recommended_tq, quality_tier
- [ ] Create model name alias registry (short names → full HF IDs)
- [ ] Write loader function to parse TOML into typed dataclasses
- [ ] Write unit tests for TOML parsing and alias resolution

**Acceptance:** All 4 model entries load correctly. Short name resolution works (e.g., `qwen3-8b` → `Qwen/Qwen3-8B-Instruct`).

---

### T4: Recommendation Engine (`tq/core/recommend.py`)
**Priority:** P0 | **Blocked by:** T2, T3

- [ ] Define `ModelRecommendation` frozen dataclass
- [ ] Implement VRAM budget formula:
  ```
  available_for_kv = gpu_vram - model_size - 500MB
  kv_per_token = 2 * layers * kv_heads * head_dim * 2 bytes
  max_context = available_for_kv / (kv_per_token / compression_ratio)
  ```
- [ ] Implement `recommend(hardware, use_case, min_context, prefer_quality) -> ModelRecommendation`
- [ ] Filter models by: fits in VRAM, matches use_case, meets min_context
- [ ] Sort by fit_score (combination of quality_tier, context_gain, speed_estimate)
- [ ] Return top recommendation + alternatives list
- [ ] Add `tq recommend` CLI command
- [ ] Write unit tests with various hardware profiles (RTX 4060, M2, GTX 1650, CPU-only)

**Acceptance:** `tq recommend --coding` returns correct model for the dev machine. Unit tests cover all hardware tiers.

---

### T5: TurboQuant Configuration (`tq/core/turbo.py`)
**Priority:** P0 | **Blocked by:** T1

- [ ] Define `TQConfig` frozen dataclass
- [ ] Implement `auto_configure_tq(model_id, available_vram, model_size, target_context) -> TQConfig`
- [ ] Decision tree: try turbo4 → turbo3 → asymmetric → raise InsufficientVRAM
- [ ] Apply model-size-aware rules (8B+ safe for turbo3, <8B turbo4 only)
- [ ] Calculate compression_ratio and estimated_quality
- [ ] Write unit tests for all decision branches

**Acceptance:** Correct TQ config generated for each model/hardware combination. InsufficientVRAM raised when nothing fits.

---

### T6: Model Download (`tq/core/` + `tq/cli/models.py`)
**Priority:** P0 | **Blocked by:** T1, T3

- [ ] Implement `pull_model(shortname, quant) -> Path` using `huggingface_hub.hf_hub_download`
- [ ] Support resume on interrupted downloads
- [ ] Register downloaded models in local index (`~/.tq/models.json` or similar)
- [ ] Implement `list_models() -> list[InstalledModel]`
- [ ] Implement `remove_model(shortname)` with confirmation
- [ ] Add Rich progress bar for downloads
- [ ] Implement `tq pull`, `tq list`, `tq remove` CLI commands
- [ ] Write unit tests (mock HF hub calls)

**Acceptance:** `tq pull qwen3-8b` downloads with progress. `tq list` shows installed. `tq remove` cleans up.

---

### T7: Config Management (`tq/core/config.py`)
**Priority:** P1 | **Blocked by:** T1

- [ ] Implement `~/.tq/` directory creation on first run
- [ ] Implement global config read/write (`~/.tq/config.toml`)
- [ ] Implement per-model config read/write (`~/.tq/configs/<model>.toml`)
- [ ] Provide sensible defaults for all settings
- [ ] Write unit tests for config round-trip

**Acceptance:** Config creates on first run, persists settings, loads correctly.

---

### T8: Inference Layer (`tq/server/inference.py`)
**Priority:** P0 | **Blocked by:** T5, T6

- [ ] Implement model loading with `AutoModelForCausalLM` + `AutoTokenizer`
- [ ] Initialize `TurboQuantCache` with config from T5
- [ ] Implement `generate(input_ids, max_tokens, temperature, ...) -> generator[str]`
- [ ] Support streaming token generation (yield each token)
- [ ] Support non-streaming (return complete text)
- [ ] Handle CUDA OOM with context reduction fallback
- [ ] Track generation metrics (tokens generated, speed, KV cache size)
- [ ] Write unit tests with a tiny model (SmolLM-135M or similar)

**Acceptance:** Model loads, generates text with TurboQuant cache. Streaming yields tokens correctly.

---

### T9: API Server (`tq/server/app.py`, `tq/server/models.py`)
**Priority:** P0 | **Blocked by:** T8

- [ ] Define Pydantic models for OpenAI API schemas:
  - `ChatCompletionRequest`, `ChatCompletionResponse`
  - `ChatCompletionChunk` (streaming)
  - `CompletionRequest`, `CompletionResponse`
  - `ModelList`, `ModelObject`
- [ ] Implement `POST /v1/chat/completions` (streaming + non-streaming)
- [ ] Implement `POST /v1/completions`
- [ ] Implement `GET /v1/models`
- [ ] Implement `GET /health`
- [ ] Implement `GET /tq/status` (TurboQuant metrics, VRAM usage, performance)
- [ ] SSE streaming via `StreamingResponse`
- [ ] Proper error responses (400, 404, 500 with helpful messages)
- [ ] Write unit tests for Pydantic models
- [ ] Write integration tests for API endpoints

**Acceptance:** API responds correctly to OpenAI-format requests. Streaming works. Status endpoint shows TQ metrics.

---

### T10: CLI Commands (`tq/cli/start.py`)
**Priority:** P0 | **Blocked by:** T2, T4, T5, T6, T8, T9

- [ ] Implement `tq start [model]` — full orchestration flow:
  1. Detect hardware
  2. Recommend model (if not specified)
  3. Download model (if not present)
  4. Configure TurboQuant
  5. Load model
  6. Start server
- [ ] Implement `tq start --coding` and `tq start --chat` shortcuts
- [ ] Implement `tq stop` (signal running server to shut down)
- [ ] Implement `tq status` (query running server's /tq/status)
- [ ] Rich output: step-by-step progress with spinners and status messages
- [ ] Support all CLI flags: `--context`, `--bits`, `--port`, `--host`, `--verbose`, `--json`, `--cpu`
- [ ] Write integration test for full start flow

**Acceptance:** `tq start --coding` runs the complete flow end-to-end. All flags work.

---

### T11: Integration Testing
**Priority:** P0 | **Blocked by:** T10

- [ ] Full `tq start` flow with SmolLM-135M (tiny model, CPU-only)
- [ ] Start server → send chat completion → verify response format
- [ ] Start server → send streaming request → verify SSE format
- [ ] Verify `/v1/models` returns correct model list
- [ ] Verify `/health` and `/tq/status` endpoints
- [ ] Test error cases: bad model name, port in use, insufficient VRAM (mocked)
- [ ] Verify 80%+ test coverage

**Acceptance:** All integration tests pass. Coverage >= 80%.

---

### T12: README and Documentation
**Priority:** P1 | **Blocked by:** T10

- [ ] Write README.md with:
  - One-line pitch
  - Quick start (3 lines)
  - Before/after comparison table
  - How it works section
  - Installation instructions
  - Usage examples
  - Configuration reference
  - FAQ
- [ ] Verify all CLI examples in README actually work

**Acceptance:** README is complete, all examples are tested.

---

## Estimated Effort

| Task | Effort | Notes |
|---|---|---|
| T1: Scaffolding | 2-3 hours | Boilerplate, but sets the foundation |
| T2: Hardware Detection | 4-6 hours | Multiple detection paths, parsing, fallbacks |
| T3: Model Database | 2-3 hours | Research model specs, write TOML |
| T4: Recommendation Engine | 4-6 hours | VRAM math, scoring, ranking |
| T5: TurboQuant Config | 3-4 hours | Decision tree, community rules |
| T6: Model Download | 3-4 hours | HF hub integration, progress bars |
| T7: Config Management | 2-3 hours | TOML read/write, defaults |
| T8: Inference Layer | 6-8 hours | Model loading, TQ cache, generation |
| T9: API Server | 6-8 hours | FastAPI, Pydantic, streaming |
| T10: CLI Commands | 4-6 hours | Orchestration, Rich output |
| T11: Integration Tests | 4-6 hours | End-to-end testing |
| T12: README | 2-3 hours | Documentation |
| **Total** | **~42-60 hours** | **~1.5-2 weeks** |
