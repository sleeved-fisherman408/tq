from __future__ import annotations

from tq.server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelList,
    ModelObject,
    UsageInfo,
)


class TestAPIModels:
    def test_chat_request_defaults(self):
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hello")])
        assert req.temperature == 0.7
        assert req.max_tokens == 2048
        assert req.stream is False
        assert req.model == "default"

    def test_chat_request_custom(self):
        req = ChatCompletionRequest(
            model="qwen3-8b",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.5,
            max_tokens=1024,
            stream=True,
        )
        assert req.model == "qwen3-8b"
        assert req.temperature == 0.5
        assert req.stream is True

    def test_chat_response(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="qwen3-8b",
            choices=[ChatCompletionChoice(message=ChatMessage(role="assistant", content="Hello!"))],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        data = resp.model_dump()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 15

    def test_model_list(self):
        ml = ModelList(data=[ModelObject(id="qwen3-8b", created=1234567890, owned_by="tq")])
        assert ml.object == "list"
        assert len(ml.data) == 1
        assert ml.data[0].id == "qwen3-8b"

    def test_completion_request(self):
        req = CompletionRequest(prompt="Once upon a time")
        assert req.model == "default"
        assert req.temperature == 0.7

    def test_completion_response(self):
        resp = CompletionResponse(
            id="cmpl-123",
            created=1234567890,
            model="qwen3-8b",
            choices=[CompletionChoice(text="there was a princess")],
            usage=UsageInfo(prompt_tokens=4, completion_tokens=5, total_tokens=9),
        )
        assert resp.choices[0].text == "there was a princess"
