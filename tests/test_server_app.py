from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tq.core.turbo import TQConfig
from tq.server.app import app, get_engine, set_engine
from tq.server.inference import GenerationMetrics


@pytest.fixture(autouse=True)
def _reset_engine():
    yield
    import tq.server.app as mod

    mod._engine = None


def _make_engine(
    model_name: str = "test-model",
    tq_config: TQConfig | None = None,
):
    engine = MagicMock()
    engine.model_name = model_name
    engine.tq_config = tq_config
    engine.get_status.return_value = {
        "model": model_name,
        "turboquant": {
            "enabled": tq_config is not None,
            "key_bits": tq_config.key_bits if tq_config else None,
            "value_bits": tq_config.value_bits if tq_config else None,
            "compression_ratio": tq_config.compression_ratio if tq_config else None,
        },
        "performance": {"tokens_generated": 42, "uptime_seconds": 10.0},
    }
    engine.generate.return_value = (
        "Hello!",
        GenerationMetrics(prompt_tokens=5, completion_tokens=1, tokens_per_second=10.0),
    )
    engine.generate_stream.return_value = iter(["Hel", "lo", "!"])
    return engine


class TestHealthEndpoint:
    def test_health(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestGetEngineErrors:
    def test_no_engine_raises(self):
        with pytest.raises(RuntimeError, match="No model loaded"):
            get_engine()

    def test_set_and_get_engine(self):
        engine = _make_engine()
        set_engine(engine)
        assert get_engine() is engine


class TestModelsEndpoint:
    def test_list_models(self):
        engine = _make_engine("Qwen/Qwen3-8B")
        set_engine(engine)
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "Qwen/Qwen3-8B"

    def test_list_models_no_engine(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        assert resp.status_code == 503


class TestChatCompletions:
    def test_non_streaming(self):
        engine = _make_engine()
        set_engine(engine)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 1
        assert data["usage"]["total_tokens"] == 6
        assert data["model"] == "test-model"
        engine.generate.assert_called_once()

    def test_streaming(self):
        engine = _make_engine()
        set_engine(engine)
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data: " in body
        assert "[DONE]" in body
        lines = [ln for ln in body.strip().split("\n") if ln.startswith("data: ")]
        assert len(lines) >= 3
        for line in lines[:-1]:
            payload = json.loads(line[len("data: ") :])
            assert payload["object"] == "chat.completion.chunk"
        assert lines[-1] == "data: [DONE]"

    def test_chat_prompt_formatting(self):
        engine = _make_engine()
        set_engine(engine)
        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there."},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        call_args = engine.generate.call_args
        prompt = (
            call_args.kwargs.get("prompt") or call_args[0][0]
            if call_args[0]
            else call_args.kwargs["prompt"]
        )
        assert "System: Be brief." in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there." in prompt
        assert "User: How are you?" in prompt


class TestCompletions:
    def test_basic(self):
        engine = _make_engine()
        set_engine(engine)
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Once upon a time", "max_tokens": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "Hello!"
        assert data["usage"]["total_tokens"] == 6

    def test_prompt_as_list(self):
        engine = _make_engine()
        set_engine(engine)
        client = TestClient(app)
        resp = client.post(
            "/v1/completions",
            json={"prompt": ["First prompt"], "max_tokens": 50},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "Hello!"

    def test_no_engine(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/completions", json={"prompt": "test"})
        assert resp.status_code == 503


class TestTQStatus:
    def test_status(self):
        tq = TQConfig(
            key_bits=4,
            value_bits=4,
            key_method="mse",
            value_method="mse",
            outlier_channels=32,
            outlier_bits=8,
            compression_ratio=4.0,
            estimated_quality="lossless",
        )
        engine = _make_engine(tq_config=tq)
        set_engine(engine)
        client = TestClient(app)
        resp = client.get("/tq/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "test-model"
        assert data["turboquant"]["enabled"] is True
        assert data["turboquant"]["key_bits"] == 4
        assert data["performance"]["tokens_generated"] == 42

    def test_status_no_tq(self):
        engine = _make_engine()
        set_engine(engine)
        client = TestClient(app)
        resp = client.get("/tq/status")
        data = resp.json()
        assert data["turboquant"]["enabled"] is False

    def test_error_handler(self):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        assert resp.status_code == 503
        assert resp.json()["error"]["type"] == "server_error"
