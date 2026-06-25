"""
Anthropic adapter.

Owns 100% of the translation between llmkit's neutral core types and the
`anthropic` Python SDK's request/response shapes. No other file in this
project should know what an Anthropic API request looks like — that
knowledge lives here and only here. This is the core of the adapter
pattern: if Anthropic changes their API, only this file changes.

Tool calling notes:
- Tool definitions: Anthropic wants {name, description, input_schema}.
  Our Tool type already matches this field-for-field — no renaming needed
  (this is the provider our Tool type happens to mirror most closely).
- Incoming tool_use: Anthropic puts ToolUseBlock(id, name, input) directly
  in the assistant message's `content` list, same as our core ContentBlock
  model — again, a near 1:1 match.
- Outgoing tool_result: must be a `user`-role message containing a
  `tool_result` content block (NOT its own message role, unlike OpenAI).
  Our core ToolResultBlock is designed to sit inside a USER Message's
  content list for exactly this reason.
- Streaming tool calls: arrives as content_block_start (type="tool_use",
  with id/name but empty input) followed by content_block_delta events
  with delta.type="input_json_delta" carrying partial_json fragments.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anthropic

from llmkit.adapters.base import ProviderAdapter
from llmkit.core.types import (
    Message,
    MessageStartChunk,
    MessageStopChunk,
    Response,
    Role,
    StopReason,
    StreamChunk,
    TextBlock,
    TextDeltaChunk,
    Tool,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

# Anthropic's stop_reason strings -> our normalized enum.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "tool_use": StopReason.TOOL_USE,
}


def _normalize_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return StopReason.OTHER
    return _STOP_REASON_MAP.get(reason, StopReason.OTHER)


class AnthropicAdapter(ProviderAdapter):
    """ProviderAdapter implementation backed by the official `anthropic` SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        # api_key=None lets the underlying SDK fall back to the
        # ANTHROPIC_API_KEY env var, matching standard SDK behavior.
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    # --- translation: core -> anthropic ---

    @staticmethod
    def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
        """Anthropic does NOT accept a system-role message in the `messages`
        list (unlike OpenAI) — system prompts are a separate top-level
        `system` param. Callers are expected to pass `system` separately;
        this only translates user/assistant turns.

        ToolUseBlock and ToolResultBlock both pass through close to as-is,
        since Anthropic's wire format is the closest match to our core
        schema of any of the four providers.
        """
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                raise ValueError(
                    "System messages must be passed via the `system` parameter, "
                    "not in the `messages` list, when using the Anthropic adapter."
                )

            content_blocks = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    content_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )

            result.append({"role": msg.role.value, "content": content_blocks})
        return result

    @staticmethod
    def _to_anthropic_tools(tools: list[Tool] | None) -> list[dict] | None:
        """Anthropic's tool schema matches our Tool type field-for-field —
        this is the one adapter where tool translation is nearly a no-op."""
        if not tools:
            return None
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]

    # --- translation: anthropic -> core ---

    @staticmethod
    def _from_anthropic_response(raw: anthropic.types.Message) -> Response:
        content = []
        for block in raw.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseBlock(id=block.id, name=block.name, input=block.input)
                )
        return Response(
            content=content,
            stop_reason=_normalize_stop_reason(raw.stop_reason),
            usage=Usage(
                input_tokens=raw.usage.input_tokens,
                output_tokens=raw.usage.output_tokens,
            ),
            model=raw.model,
            raw=raw.model_dump(),
        )

    # --- ProviderAdapter interface ---

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> Response:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }
        if system is not None:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        anthropic_tools = self._to_anthropic_tools(tools)
        if anthropic_tools is not None:
            kwargs["tools"] = anthropic_tools

        raw = await self._client.messages.create(**kwargs)
        return self._from_anthropic_response(raw)

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._to_anthropic_messages(messages),
        }
        if system is not None:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        anthropic_tools = self._to_anthropic_tools(tools)
        if anthropic_tools is not None:
            kwargs["tools"] = anthropic_tools

        # Tracks which content_block index is a tool_use block, so
        # content_block_delta events with input_json_delta know to emit
        # ToolCallDeltaChunk instead of being silently dropped.
        tool_use_indices: set[int] = set()

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                chunk = self._translate_stream_event(event, tool_use_indices)
                if chunk is not None:
                    yield chunk

    @staticmethod
    def _translate_stream_event(event, tool_use_indices: set[int]) -> StreamChunk | None:
        """Anthropic's stream yields typed events: message_start,
        content_block_start/delta/stop, message_delta, message_stop, etc.
        Tool calls arrive as content_block_start (type="tool_use") followed
        by content_block_delta events with delta.type="input_json_delta".
        """
        if event.type == "message_start":
            return MessageStartChunk(model=event.message.model)

        if event.type == "content_block_start" and event.content_block.type == "tool_use":
            tool_use_indices.add(event.index)
            return ToolCallStartChunk(
                index=event.index,
                id=event.content_block.id,
                name=event.content_block.name,
            )

        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                return TextDeltaChunk(text=event.delta.text)
            if event.delta.type == "input_json_delta":
                return ToolCallDeltaChunk(
                    index=event.index, partial_json=event.delta.partial_json
                )

        if event.type == "message_delta":
            usage = None
            if event.usage is not None:
                # message_delta usage only carries output_tokens; input_tokens
                # was already known from message_start in the full SDK, but
                # we keep this minimal and only report what's certain here.
                usage = Usage(
                    input_tokens=0,
                    output_tokens=event.usage.output_tokens,
                )
            return MessageStopChunk(
                stop_reason=_normalize_stop_reason(event.delta.stop_reason),
                usage=usage,
            )

        # message_stop, content_block_stop, ping: no normalized equivalent
        # needed for this pass.
        return None