"""
Gemini adapter.

Owns 100% of the translation between llmkit's neutral core types and the
`google-genai` Python SDK's request/response shapes.

Key differences this adapter has to absorb (none of which leak outside
this file, per the adapter pattern):
- Gemini uses "model" as the assistant role string, not "assistant".
- Messages are `Content(role=..., parts=[Part(text=...)])` objects, not
  plain dicts — closer to Anthropic's block-based shape than OpenAI's.
- System instructions are a separate `system_instruction` field on
  `GenerateContentConfig`, like Anthropic's `system` param — NOT folded
  into the contents list like OpenAI does.
- `max_output_tokens`, not `max_tokens` / `max_completion_tokens`.
- Streaming is the biggest divergence: `generate_content_stream` yields a
  sequence of FULL `GenerateContentResponse` objects, not delta-only
  chunks. Each one has its own `.text` already-accumulated-for-that-chunk
  content and its own usage_metadata/finish_reason (which only become
  meaningful on the final chunk). We treat `.text` per chunk as the
  incremental delta, matching how the SDK actually emits it in practice.
- finish_reason is a FinishReason enum, not a string — values like STOP
  and MAX_TOKENS map directly onto our existing StopReason members,
  confirming (again) that the enum was modeled generally enough to not
  need new members for a third provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import types as gtypes

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

# Gemini's FinishReason enum values -> our normalized enum.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "STOP": StopReason.END_TURN,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    # SAFETY, RECITATION, PROHIBITED_CONTENT, MALFORMED_FUNCTION_CALL, etc.
    # have no clean equivalent yet — fall through to OTHER.
}

# Our Role -> Gemini's role string. Gemini calls the assistant role "model".
_ROLE_MAP: dict[Role, str] = {
    Role.USER: "user",
    Role.ASSISTANT: "model",
}


def _normalize_stop_reason(reason) -> StopReason:
    if reason is None:
        return StopReason.OTHER
    return _STOP_REASON_MAP.get(str(reason), StopReason.OTHER)


class GeminiAdapter(ProviderAdapter):
    """ProviderAdapter implementation backed by the official `google-genai` SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        # api_key=None lets the underlying SDK fall back to the
        # GOOGLE_API_KEY / GEMINI_API_KEY env var, matching standard SDK behavior.
        self._client = genai.Client(api_key=api_key)

    # --- translation: core -> gemini ---

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> list[gtypes.Content]:
        """Gemini has no system role inside contents — a SYSTEM-role
        Message here is a caller error, same as the Anthropic adapter,
        since system prompts belong in system_instruction instead.
        """
        contents = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                raise ValueError(
                    "System messages must be passed via the `system` parameter, "
                    "not in the `messages` list, when using the Gemini adapter."
                )
            contents.append(
                gtypes.Content(
                    role=_ROLE_MAP[msg.role],
                    parts=[
                        gtypes.Part(text=block.text)
                        for block in msg.content
                        if isinstance(block, TextBlock)
                    ],
                )
            )
        return contents

    @staticmethod
    def _build_config(
        max_tokens: int,
        system: str | None,
        temperature: float | None,
    ) -> gtypes.GenerateContentConfig:
        kwargs: dict = {"max_output_tokens": max_tokens}
        if system is not None:
            kwargs["system_instruction"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        return gtypes.GenerateContentConfig(**kwargs)

    # --- translation: gemini -> core ---

    @staticmethod
    def _from_gemini_response(raw: gtypes.GenerateContentResponse, model: str) -> Response:
        candidate = raw.candidates[0]
        content = [
            TextBlock(text=part.text)
            for part in (candidate.content.parts or [])
            if part.text is not None
        ]
        usage = Usage(
            input_tokens=raw.usage_metadata.prompt_token_count or 0,
            output_tokens=raw.usage_metadata.candidates_token_count or 0,
        )
        return Response(
            content=content,
            stop_reason=_normalize_stop_reason(candidate.finish_reason),
            usage=usage,
            model=model,
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
        raw = await self._client.aio.models.generate_content(
            model=model,
            contents=self._to_gemini_contents(messages),
            config=self._build_config(max_tokens, system, temperature),
        )
        return self._from_gemini_response(raw, model)

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        started = False
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=model,
            contents=self._to_gemini_contents(messages),
            config=self._build_config(max_tokens, system, temperature),
        ):
            if not started:
                yield MessageStartChunk(model=model)
                started = True

            translated = self._translate_stream_chunk(chunk)
            if translated is not None:
                yield translated

    @staticmethod
    def _translate_stream_chunk(chunk: gtypes.GenerateContentResponse) -> StreamChunk | None:
        """Each chunk is a full GenerateContentResponse. In practice the
        SDK emits incremental text per chunk (not the full accumulated
        text each time) — we surface that as a text_delta. The final
        chunk additionally carries a real finish_reason and usage_metadata,
        which we surface as message_stop.
        """
        if not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        text = "".join(
            part.text for part in (candidate.content.parts or []) if part.text is not None
        )

        if candidate.finish_reason is not None:
            usage = None
            if chunk.usage_metadata is not None:
                usage = Usage(
                    input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                    output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                )
            return MessageStopChunk(
                stop_reason=_normalize_stop_reason(candidate.finish_reason),
                usage=usage,
            )

        if text:
            return TextDeltaChunk(text=text)

        return None
