import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TQ_INTEGRATION") != "1",
    reason="Set TQ_INTEGRATION=1 to run integration tests (downloads a model)",
)

import json
import time

from fastapi.testclient import TestClient

from tq.server.app import app, set_engine
from tq.server.inference import InferenceEngine

MODEL_ID = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def engine():
    eng = InferenceEngine(MODEL_ID, device="cpu")
    eng.load()
    set_engine(eng)
    return eng


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(app)


class TestHealthIntegration:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsIntegration:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == MODEL_ID
        assert data["data"][0]["owned_by"] == "tq"
        assert data["data"][0]["object"] == "model"


class TestChatCompletionsIntegration:
    def test_non_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == MODEL_ID
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(data["choices"][0]["message"]["content"], str)
        assert len(data["choices"][0]["message"]["content"]) > 0
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == (
            data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
        )

    def test_streaming(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 5,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        chunks = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                payload = line[6:]
                if payload == "[DONE]":
                    chunks.append("[DONE]")
                else:
                    chunks.append(json.loads(payload))

        assert len(chunks) >= 2
        assert chunks[-1] == "[DONE]"

        for chunk in chunks[:-1]:
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["model"] == MODEL_ID
            assert len(chunk["choices"]) == 1

        first_delta = chunks[0]["choices"][0]["delta"]
        assert first_delta.get("role") == "assistant"

    def test_multi_turn(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"]


class TestCompletionsIntegration:
    def test_basic(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": "The capital of France is", "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["model"] == MODEL_ID
        assert len(data["choices"]) == 1
        assert isinstance(data["choices"][0]["text"], str)
        assert data["usage"]["total_tokens"] > 0

    def test_list_prompt(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": ["Once upon a time"], "max_tokens": 5},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["choices"][0]["text"], str)


class TestTQStatusIntegration:
    def test_status(self, client):
        resp = client.get("/tq/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == MODEL_ID
        assert "turboquant" in data
        assert "performance" in data
        assert isinstance(data["performance"]["tokens_generated"], int)
        assert data["performance"]["uptime_seconds"] >= 0
