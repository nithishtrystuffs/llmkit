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

Tool calling notes — this is the adapter with the most genuine capability
gaps, documented rather than silently faked:

- Tool definitions: Gemini's FunctionDeclaration takes
  `parameters_json_schema` for a plain JSON Schema dict — a closer match
  to our Tool.input_schema than the Schema-object-based `parameters`
  field would be, so we use that field specifically.
- Incoming tool_use: arrives as a `Part(function_call=FunctionCall(id,
  name, args))` — `args` is already a dict (unlike OpenAI's JSON string),
  so no parsing step is needed here.
- No dedicated finish_reason for tool calls: Gemini's FinishReason enum
  has no TOOL_USE-equivalent member. We detect tool use by checking
  whether the response contains any function_call parts and report
  StopReason.TOOL_USE ourselves rather than trusting finish_reason, which
  would otherwise just say STOP.
- Outgoing tool_result: Gemini's FunctionResponse requires the function's
  `name`, not just an id (id is documented as merely optional metadata).
  Our core ToolResultBlock only carries `tool_use_id`, so this adapter
  maintains an internal id->name map populated from ToolUseBlocks it
  encounters while translating messages, and looks up the name when it
  later sees the matching ToolResultBlock. This requires the full
  conversation history (including the assistant's prior tool_use turn)
  to be passed in `messages` each call — which is already required by
  this library's stateless-per-call design, so no new constraint on
  callers.
- No incremental argument streaming: unlike Anthropic/OpenAI, the SDK
  does not stream tool-call arguments fragment-by-fragment. This adapter
  emits one ToolCallStartChunk immediately followed by exactly one
  ToolCallDeltaChunk carrying the COMPLETE arguments as a single
  fragment. Callers using the same accumulate-then-parse pattern across
  all adapters get correct behavior either way, but should not assume
  multiple fragments will arrive for a Gemini-sourced tool call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import json

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
    Tool,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
    ToolResultBlock,
    ToolUseBlock,
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


def _normalize_stop_reason(reason, has_tool_calls: bool = False) -> StopReason:
    if has_tool_calls:
        return StopReason.TOOL_USE
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

        Also builds an id->name map from any ToolUseBlocks encountered,
        since FunctionResponse needs `name` but our ToolResultBlock only
        carries the id.
        """
        id_to_name: dict[str, str] = {}
        contents = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                raise ValueError(
                    "System messages must be passed via the `system` parameter, "
                    "not in the `messages` list, when using the Gemini adapter."
                )

            parts = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(gtypes.Part(text=block.text))
                elif isinstance(block, ToolUseBlock):
                    id_to_name[block.id] = block.name
                    parts.append(
                        gtypes.Part(
                            function_call=gtypes.FunctionCall(
                                id=block.id, name=block.name, args=block.input
                            )
                        )
                    )
                elif isinstance(block, ToolResultBlock):
                    name = id_to_name.get(block.tool_use_id)
                    if name is None:
                        raise ValueError(
                            f"Gemini adapter could not find a tool name for "
                            f"tool_use_id={block.tool_use_id!r}. The matching "
                            f"ToolUseBlock must appear earlier in `messages` "
                            f"for Gemini's FunctionResponse, which requires "
                            f"`name` rather than just an id."
                        )
                    parts.append(
                        gtypes.Part(
                            function_response=gtypes.FunctionResponse(
                                id=block.tool_use_id,
                                name=name,
                                response={"result": block.content}
                                if not block.is_error
                                else {"error": block.content},
                            )
                        )
                    )

            contents.append(gtypes.Content(role=_ROLE_MAP[msg.role], parts=parts))

        return contents

    @staticmethod
    def _build_config(
        max_tokens: int,
        system: str | None,
        temperature: float | None,
        tools: list[Tool] | None,
    ) -> gtypes.GenerateContentConfig:
        kwargs: dict = {"max_output_tokens": max_tokens}
        if system is not None:
            kwargs["system_instruction"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                gtypes.Tool(
                    function_declarations=[
                        gtypes.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters_json_schema=t.input_schema,
                        )
                        for t in tools
                    ]
                )
            ]
        return gtypes.GenerateContentConfig(**kwargs)

    # --- translation: gemini -> core ---

    @staticmethod
    def _from_gemini_response(raw: gtypes.GenerateContentResponse, model: str) -> Response:
        candidate = raw.candidates[0]
        content = []
        has_tool_calls = False
        for part in candidate.content.parts or []:
            if part.text is not None:
                content.append(TextBlock(text=part.text))
            elif part.function_call is not None:
                has_tool_calls = True
                fc = part.function_call
                content.append(
                    ToolUseBlock(id=fc.id or fc.name, name=fc.name, input=fc.args or {})
                )

        usage = Usage(
            input_tokens=raw.usage_metadata.prompt_token_count or 0,
            output_tokens=raw.usage_metadata.candidates_token_count or 0,
        )
        return Response(
            content=content,
            stop_reason=_normalize_stop_reason(candidate.finish_reason, has_tool_calls),
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
        tools: list[Tool] | None = None,
    ) -> Response:
        raw = await self._client.aio.models.generate_content(
            model=model,
            contents=self._to_gemini_contents(messages),
            config=self._build_config(max_tokens, system, temperature, tools),
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
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        started = False
        # Gemini doesn't stream tool-call args incrementally, so each
        # function_call part we see is brand new — no index-tracking
        # needed across chunks the way OpenAI's adapter requires.
        next_index = 0

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=model,
            contents=self._to_gemini_contents(messages),
            config=self._build_config(max_tokens, system, temperature, tools),
        ):
            if not started:
                yield MessageStartChunk(model=model)
                started = True

            for translated in self._translate_stream_chunk(chunk, next_index):
                yield translated
                if translated.type == "tool_call_start":
                    next_index += 1

    @staticmethod
    def _translate_stream_chunk(
        chunk: gtypes.GenerateContentResponse, next_index: int
    ) -> list[StreamChunk]:
        """Each chunk is a full GenerateContentResponse. In practice the
        SDK emits incremental text per chunk (not the full accumulated
        text each time) — we surface that as a text_delta. Function calls
        arrive complete in a single chunk (no incremental arg streaming),
        so each one becomes a ToolCallStartChunk + one ToolCallDeltaChunk
        carrying the full arguments. The final chunk additionally carries
        a real finish_reason and usage_metadata, surfaced as message_stop.
        """
        results: list[StreamChunk] = []
        if not chunk.candidates:
            return results

        candidate = chunk.candidates[0]
        text = ""
        has_tool_calls = False
        index = next_index

        for part in candidate.content.parts or []:
            if part.text is not None:
                text += part.text
            elif part.function_call is not None:
                has_tool_calls = True
                fc = part.function_call
                call_id = fc.id or fc.name
                results.append(ToolCallStartChunk(index=index, id=call_id, name=fc.name))
                results.append(
                    ToolCallDeltaChunk(index=index, partial_json=json.dumps(fc.args or {}))
                )
                index += 1

        if text:
            results.append(TextDeltaChunk(text=text))

        if candidate.finish_reason is not None:
            usage = None
            if chunk.usage_metadata is not None:
                usage = Usage(
                    input_tokens=chunk.usage_metadata.prompt_token_count or 0,
                    output_tokens=chunk.usage_metadata.candidates_token_count or 0,
                )
            results.append(
                MessageStopChunk(
                    stop_reason=_normalize_stop_reason(candidate.finish_reason, has_tool_calls),
                    usage=usage,
                )
            )

        return results