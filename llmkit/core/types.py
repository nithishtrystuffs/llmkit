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
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TextBlock(BaseModel):
    """Plain text content block."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """The model is requesting a tool call. Appears in assistant Messages
    and in Response.content.

    `input` is always a dict here regardless of provider — OpenAI's
    adapter is responsible for json.loads()-ing its wire-format JSON
    string into a dict before this object is constructed, so that quirk
    never leaks past the adapter boundary.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


class ToolResultBlock(BaseModel):
    """The result of a tool call, sent back to the model. Placed inside a
    USER-role Message's content list — this matches Anthropic's and
    Gemini's mental model (tool results are content blocks within the
    conversation flow). The OpenAI adapter splits this out into its own
    role="tool" message internally, since that's how OpenAI's wire format
    requires it; that translation cost is intentionally absorbed there so
    the core schema stays unbiased toward any one provider's shape.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


# Discriminated union of content block types, keyed on the `type` field.
# Annotated + Field(discriminator=...) is required here (not just a plain
# `|` union) so Pydantic deserializes raw dicts into the correct concrete
# block class instead of guessing/erroring. Adding ImageBlock/ThinkingBlock
# later extends this Annotated union without touching existing exports.
ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")
]


class Tool(BaseModel):
    """A tool definition the model may call. `input_schema` is a plain
    JSON Schema dict (the `parameters`/`input_schema`/`parameters_json_schema`
    field name differs per provider, but the JSON Schema shape itself is
    common ground across all three — each adapter just renames the field).
    """

    name: str
    description: str
    input_schema: dict


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
    TOOL_USE = "tool_use"
    """The model stopped because it wants to call one or more tools.
    Anthropic and OpenAI report this directly (stop_reason="tool_use" /
    finish_reason="tool_calls"). Gemini has no dedicated finish reason for
    this — its adapter infers TOOL_USE by checking whether the response
    contains any function_call parts, and reports that instead of STOP.
    """
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

    def tool_calls(self) -> list[ToolUseBlock]:
        """Convenience: every tool call the model requested in this turn.
        Empty list if the model didn't call any tools."""
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


# --- Streaming ---


class StreamChunkType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_START = "message_start"
    MESSAGE_STOP = "message_stop"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"


class TextDeltaChunk(BaseModel):
    type: Literal[StreamChunkType.TEXT_DELTA] = StreamChunkType.TEXT_DELTA
    text: str


class MessageStartChunk(BaseModel):
    type: Literal[StreamChunkType.MESSAGE_START] = StreamChunkType.MESSAGE_START
    model: str


class ToolCallStartChunk(BaseModel):
    """Signals a new tool call has begun streaming. `index` lets callers
    track multiple concurrent tool calls in one turn (OpenAI streams
    multiple calls interleaved by index; Anthropic and Gemini stream one
    content block / part at a time, so index is just its position).
    """

    type: Literal[StreamChunkType.TOOL_CALL_START] = StreamChunkType.TOOL_CALL_START
    index: int
    id: str
    name: str


class ToolCallDeltaChunk(BaseModel):
    """An incremental fragment of a tool call's JSON arguments string.

    This is the lowest common denominator across providers: Anthropic
    streams partial JSON text (input_json_delta), OpenAI streams partial
    argument-string fragments per tool_call index. Callers accumulate
    `partial_json` fragments per `index` and json.loads() the full string
    once the call's stream is done.

    Gemini's SDK does not stream tool-call arguments incrementally — its
    adapter emits one ToolCallStartChunk immediately followed by a single
    ToolCallDeltaChunk carrying the complete arguments as one fragment,
    rather than true incremental deltas. This is documented here rather
    than silently faked, per the capability-gap-should-be-visible principle
    from this project's design.
    """

    type: Literal[StreamChunkType.TOOL_CALL_DELTA] = StreamChunkType.TOOL_CALL_DELTA
    index: int
    partial_json: str


class MessageStopChunk(BaseModel):
    type: Literal[StreamChunkType.MESSAGE_STOP] = StreamChunkType.MESSAGE_STOP
    stop_reason: StopReason
    usage: Usage | None = None


StreamChunk = Annotated[
    TextDeltaChunk
    | MessageStartChunk
    | MessageStopChunk
    | ToolCallStartChunk
    | ToolCallDeltaChunk,
    Field(discriminator="type"),
]