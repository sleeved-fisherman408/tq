from __future__ import annotations

import time
import uuid
from collections.abc import Generator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tq.server.inference import InferenceEngine
from tq.server.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaContent,
    ModelList,
    ModelObject,
    UsageInfo,
)

app = FastAPI(title="tq", version="0.1.0")

_engine: InferenceEngine | None = None


def set_engine(engine: InferenceEngine) -> None:
    global _engine
    _engine = engine


def get_engine() -> InferenceEngine:
    if _engine is None:
        raise RuntimeError("No model loaded")
    return _engine


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> ModelList:
    engine = get_engine()
    return ModelList(
        data=[
            ModelObject(
                id=engine.model_name,
                created=int(time.time()),
                owned_by="tq",
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    engine = get_engine()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt = "\n".join(prompt_parts) + "\nAssistant:"

    if request.stream:
        return StreamingResponse(
            _stream_chat(engine, prompt, request, request_id, created),
            media_type="text/event-stream",
        )

    text, metrics = engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=engine.model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=text),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.prompt_tokens + metrics.completion_tokens,
        ),
    )


async def _stream_chat(
    engine: InferenceEngine,
    prompt: str,
    request: ChatCompletionRequest,
    request_id: str,
    created: int,
) -> Generator[str, None, None]:
    first = True
    for token in engine.generate_stream(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    ):
        delta = DeltaContent(content=token)
        if first:
            delta = DeltaContent(role="assistant", content=token)
            first = False

        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=engine.model_name,
            choices=[ChunkChoice(delta=delta)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    final = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=engine.model_name,
        choices=[ChunkChoice(delta=DeltaContent(), finish_reason="stop")],
    )
    yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    engine = get_engine()
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    text, metrics = engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    return CompletionResponse(
        id=request_id,
        created=created,
        model=engine.model_name,
        choices=[CompletionChoice(text=text)],
        usage=UsageInfo(
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=metrics.prompt_tokens + metrics.completion_tokens,
        ),
    )


@app.get("/tq/status")
async def tq_status() -> dict:
    engine = get_engine()
    return engine.get_status()


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=503, content={"error": {"message": str(exc), "type": "server_error"}}
    )
