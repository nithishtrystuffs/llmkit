"""
Core neutral types for llmkit.

These types are the contract between user code and provider adapters.
No provider's API shape is privileged here — every adapter (Anthropic,
OpenAI, Gemini, ...) translates to/from these types symmetrically.

Design notes:
- `Message.content` is a list of typed blocks (not a plain string) from
  day one, because Anthropic and Gemini are already block-based and
  OpenAI's plain-string content is the special case, not the norm.
  This avoids a breaking change later when tool calls / images / thinking
  blocks need to be added.
- `StreamChunk` is a tagged union (`type` discriminator) so new chunk
  kinds (tool_call_delta, thinking_delta, ...) can be added later without
  breaking code that pattern-matches on `type`.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TextBlock(BaseModel):
    """Plain text content block."""

    type: Literal["text"] = "text"
    text: str


# Union of content block types. Only `TextBlock` exists for now;
# `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`, `ThinkingBlock` will be
# added here later without changing this file's existing exports.
ContentBlock = TextBlock


class Message(BaseModel):
    """A single turn in the conversation, provider-agnostic."""

    role: Role
    content: list[ContentBlock]

    @classmethod
    def text(cls, role: Role, text: str) -> "Message":
        """Convenience constructor for the common case of a plain-text message."""
        return cls(role=role, content=[TextBlock(text=text)])


class Usage(BaseModel):
    """Token usage, normalized across providers."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class StopReason(str, Enum):
    """Normalized stop reasons. Providers use different vocab for this
    (Anthropic: end_turn/max_tokens/stop_sequence; OpenAI: stop/length/...).
    Adapters map their provider's reason onto this enum.
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    OTHER = "other"


class Response(BaseModel):
    """Normalized non-streaming completion result."""

    content: list[ContentBlock]
    role: Role = Role.ASSISTANT
    stop_reason: StopReason
    usage: Usage
    model: str
    raw: dict = Field(default_factory=dict, exclude=True)
    """The original, untranslated provider response. Always available as an
    escape hatch — never required for normal use, never serialized."""

    def text(self) -> str:
        """Convenience: concatenate all text blocks. Most callers just want this."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))


# --- Streaming ---


class StreamChunkType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_START = "message_start"
    MESSAGE_STOP = "message_stop"


class TextDeltaChunk(BaseModel):
    type: Literal[StreamChunkType.TEXT_DELTA] = StreamChunkType.TEXT_DELTA
    text: str


class MessageStartChunk(BaseModel):
    type: Literal[StreamChunkType.MESSAGE_START] = StreamChunkType.MESSAGE_START
    model: str


class MessageStopChunk(BaseModel):
    type: Literal[StreamChunkType.MESSAGE_STOP] = StreamChunkType.MESSAGE_STOP
    stop_reason: StopReason
    usage: Usage | None = None


StreamChunk = TextDeltaChunk | MessageStartChunk | MessageStopChunk
