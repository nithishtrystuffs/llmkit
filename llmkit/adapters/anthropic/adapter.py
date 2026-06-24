"""
Anthropic adapter.

Owns 100% of the translation between llmkit's neutral core types and the
`anthropic` Python SDK's request/response shapes. No other file in this
project should know what an Anthropic API request looks like — that
knowledge lives here and only here. This is the core of the adapter
pattern: if Anthropic changes their API, only this file changes.
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
    Usage,
)

# Anthropic's stop_reason strings -> our normalized enum.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
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
        """
        result = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                raise ValueError(
                    "System messages must be passed via the `system` parameter, "
                    "not in the `messages` list, when using the Anthropic adapter."
                )
            result.append(
                {
                    "role": msg.role.value,
                    "content": [
                        {"type": "text", "text": block.text}
                        for block in msg.content
                        if isinstance(block, TextBlock)
                    ],
                }
            )
        return result

    # --- translation: anthropic -> core ---

    @staticmethod
    def _from_anthropic_response(raw: anthropic.types.Message) -> Response:
        content = [
            TextBlock(text=block.text)
            for block in raw.content
            if block.type == "text"
        ]
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

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                chunk = self._translate_stream_event(event)
                if chunk is not None:
                    yield chunk

    @staticmethod
    def _translate_stream_event(event) -> StreamChunk | None:
        """Anthropic's stream yields typed events: message_start,
        content_block_delta (with a nested delta.type of text_delta /
        input_json_delta / ...), message_delta, message_stop, etc.
        We only handle text_delta here — tool-call streaming
        (input_json_delta) is explicitly out of scope for this first pass.
        """
        if event.type == "message_start":
            return MessageStartChunk(model=event.message.model)

        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            return TextDeltaChunk(text=event.delta.text)

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

        # message_stop, content_block_start, content_block_stop, ping:
        # no normalized equivalent needed for this minimal pass.
        return None
