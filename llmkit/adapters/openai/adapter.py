"""
OpenAI adapter.

Owns 100% of the translation between llmkit's neutral core types and the
`openai` Python SDK's request/response shapes.

Key differences from Anthropic that this file has to absorb (this is the
whole point of the adapter pattern — these differences are invisible to
everything outside this file):

- OpenAI has NO separate `system` parameter. System instructions go inside
  the `messages` list as a `{"role": "system", ...}` entry. Our
  ProviderAdapter interface still passes `system` as its own argument
  (because that's how Anthropic works), so this adapter's job is to fold
  it back into the messages list before sending.
- OpenAI's `max_tokens` chat-completions param is deprecated in favor of
  `max_completion_tokens`. We send the new one.
- Streaming chunks are `choices[0].delta.content` deltas, not typed
  content-block events like Anthropic. Usage and finish_reason arrive on
  the *last* chunk, not as a separate event type.
- finish_reason vocab differs from Anthropic's stop_reason vocab
  ("stop"/"length"/... vs "end_turn"/"max_tokens"/...).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai

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

# OpenAI's finish_reason strings -> our normalized enum.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    # OpenAI doesn't have a direct stop_sequence-vs-natural-stop distinction
    # in finish_reason the way Anthropic does — "stop" covers both. Custom
    # stop sequences also report "stop", so we can't distinguish without
    # extra bookkeeping; OTHER is reserved for tool_calls/content_filter/etc.
}


def _normalize_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return StopReason.OTHER
    return _STOP_REASON_MAP.get(reason, StopReason.OTHER)


class OpenAIAdapter(ProviderAdapter):
    """ProviderAdapter implementation backed by the official `openai` SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)

    # --- translation: core -> openai ---

    @staticmethod
    def _to_openai_messages(
        messages: list[Message], system: str | None
    ) -> list[dict]:
        """Unlike Anthropic, OpenAI wants system content INSIDE the messages
        list. We fold the adapter-level `system` argument in here as the
        first message, and also pass through any explicit system-role
        messages the caller included directly (rare, but not an error here
        the way it is for Anthropic — OpenAI's API genuinely supports this).
        """
        result = []
        if system is not None:
            result.append({"role": "system", "content": system})

        for msg in messages:
            result.append(
                {
                    "role": msg.role.value,
                    "content": "".join(
                        block.text for block in msg.content if isinstance(block, TextBlock)
                    ),
                }
            )
        return result

    # --- translation: openai -> core ---

    @staticmethod
    def _from_openai_response(raw) -> Response:
        choice = raw.choices[0]
        content = [TextBlock(text=choice.message.content or "")]
        return Response(
            content=content,
            stop_reason=_normalize_stop_reason(choice.finish_reason),
            usage=Usage(
                input_tokens=raw.usage.prompt_tokens,
                output_tokens=raw.usage.completion_tokens,
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
            "max_completion_tokens": max_tokens,
            "messages": self._to_openai_messages(messages, system),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        raw = await self._client.chat.completions.create(**kwargs)
        return self._from_openai_response(raw)

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
            "max_completion_tokens": max_tokens,
            "messages": self._to_openai_messages(messages, system),
            "stream": True,
            # Without this, the final chunk has usage=None — OpenAI requires
            # explicitly opting in to get usage on the last streamed chunk.
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        started = False
        async for chunk in await self._client.chat.completions.create(**kwargs):
            if not started:
                started = True
                yield MessageStartChunk(model=chunk.model)

            normalized = self._translate_stream_chunk(chunk)
            if normalized is not None:
                yield normalized

    @staticmethod
    def _translate_stream_chunk(chunk) -> StreamChunk | None:
        # The final usage-only chunk has an empty choices list.
        if not chunk.choices:
            if chunk.usage is not None:
                return MessageStopChunk(
                    stop_reason=StopReason.OTHER,  # finish_reason was on an earlier chunk
                    usage=Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    ),
                )
            return None

        choice = chunk.choices[0]

        if choice.finish_reason is not None:
            return MessageStopChunk(
                stop_reason=_normalize_stop_reason(choice.finish_reason),
                usage=None,  # usage arrives on the subsequent usage-only chunk
            )

        if choice.delta.content:
            return TextDeltaChunk(text=choice.delta.content)

        return None
