from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str | list[str]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = False


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tq"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]
