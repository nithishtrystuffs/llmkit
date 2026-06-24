"""
Ollama adapter.

Owns 100% of the translation between llmkit's neutral core types and the
`ollama` Python library's request/response shapes.

This adapter is the most structurally different of the four, and is the
real stress test of two assumptions baked into ProviderAdapter:

1. "Every provider needs an api_key" — false. Ollama runs locally with no
   auth by default. `api_key` here is repurposed to mean an optional
   bearer token for Ollama Cloud (https://ollama.com) rather than a local
   instance; `host` covers pointing at a non-default local/remote address.
   ProviderAdapter.__init__ was never required to take api_key by the
   interface itself (only generate()/stream() are abstract methods), so
   this is a legitimate per-adapter constructor — confirms the interface
   doesn't secretly assume hosted-API-with-key for every provider.

2. "max_tokens is a top-level param" — false here. Ollama nests it inside
   an `options` dict as `num_predict`.

Other differences absorbed here:
- System prompts ARE allowed inline in the messages list (role="system"),
  same as OpenAI — no separate system param.
- Responses are typed Pydantic models in the modern ollama lib (ChatResponse),
  but usage lives under different names: `prompt_eval_count` /
  `eval_count`, not `usage.input_tokens` / `usage.output_tokens`.
- `done_reason` is Ollama's stop-reason field, with a much smaller
  vocabulary than the hosted providers (chiefly "stop" and "length").
- Streaming chunks are also full ChatResponse objects (like Gemini), each
  carrying an incremental `message.content` delta; the final chunk has
  `done=True` plus the usage/done_reason fields populated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import ollama

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

# Ollama's done_reason strings -> our normalized enum. Smaller vocabulary
# than the hosted providers — most local models only ever report these two.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
}


def _normalize_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return StopReason.OTHER
    return _STOP_REASON_MAP.get(reason, StopReason.OTHER)


class OllamaAdapter(ProviderAdapter):
    """ProviderAdapter implementation backed by the official `ollama` Python library.

    Unlike the hosted-provider adapters, `api_key` is optional and only
    meaningful when pointing at Ollama Cloud (host="https://ollama.com").
    For a local install, leave both api_key and host as defaults.
    """

    def __init__(self, api_key: str | None = None, host: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        # host=None lets the underlying client fall back to localhost:11434,
        # or the OLLAMA_HOST env var, matching standard library behavior.
        self._client = ollama.AsyncClient(host=host, headers=headers)

    # --- translation: core -> ollama ---

    @staticmethod
    def _to_ollama_messages(messages: list[Message], system: str | None) -> list[dict]:
        """Ollama allows role="system" inline in messages, like OpenAI —
        no separate system param. We prepend it here so callers use the
        same `system` kwarg shape across all adapters."""
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

    # --- translation: ollama -> core ---

    @staticmethod
    def _from_ollama_response(raw: ollama.ChatResponse) -> Response:
        content = [TextBlock(text=raw.message.content or "")]
        return Response(
            content=content,
            stop_reason=_normalize_stop_reason(raw.done_reason),
            usage=Usage(
                input_tokens=raw.prompt_eval_count or 0,
                output_tokens=raw.eval_count or 0,
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
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        raw = await self._client.chat(
            model=model,
            messages=self._to_ollama_messages(messages, system),
            options=options,
        )
        return self._from_ollama_response(raw)

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        started = False
        async for chunk in await self._client.chat(
            model=model,
            messages=self._to_ollama_messages(messages, system),
            options=options,
            stream=True,
        ):
            if not started:
                yield MessageStartChunk(model=chunk.model)
                started = True

            translated = self._translate_stream_chunk(chunk)
            if translated is not None:
                yield translated

    @staticmethod
    def _translate_stream_chunk(chunk: ollama.ChatResponse) -> StreamChunk | None:
        if chunk.done:
            return MessageStopChunk(
                stop_reason=_normalize_stop_reason(chunk.done_reason),
                usage=Usage(
                    input_tokens=chunk.prompt_eval_count or 0,
                    output_tokens=chunk.eval_count or 0,
                ),
            )

        if chunk.message and chunk.message.content:
            return TextDeltaChunk(text=chunk.message.content)

        return None
