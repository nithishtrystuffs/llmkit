"""
OpenAI adapter.

Owns 100% of the translation between llmkit's neutral core types and the
`openai` Python SDK's request/response shapes.

Key differences from Anthropic that this file has to absorb (this is the
whole point of the adapter pattern — these differences are invisible to
everything outside this file):

- OpenAI has NO separate `system` parameter. System instructions go inside
  the `messages` list as a `{"role": "system", ...}` entry.
- `max_completion_tokens`, not `max_tokens`.
- Streaming chunks are `choices[0].delta.content` deltas; usage and
  finish_reason arrive split across the last two chunks, not one event.
- finish_reason vocab differs from Anthropic's stop_reason vocab.

Tool calling notes (the part that makes this adapter structurally
different from Anthropic's, not just renamed fields):

- Tool definitions: OpenAI wants {"type": "function", "function": {name,
  description, parameters}} — note `parameters`, not `input_schema`.
- Incoming tool_use: OpenAI puts calls in `message.tool_calls`, a list
  separate from `message.content` (Anthropic interleaves them in one
  content list). Each call's arguments arrive as a JSON **string**
  (`function.arguments`), not a dict — this adapter is responsible for
  json.loads()-ing it before constructing a ToolUseBlock, so that quirk
  never leaks past this file.
- Outgoing tool_result: OpenAI requires its OWN message with
  role="tool" and a top-level `tool_call_id`, not a content block nested
  inside a user message. This means a single core `Message` containing
  one or more ToolResultBlocks can expand into MULTIPLE OpenAI messages
  — one role="tool" message per ToolResultBlock — which is the most
  structurally significant translation this adapter performs.
- Streaming tool calls: arrive as `delta.tool_calls`, a list of partial
  tool-call objects keyed by `index` (NOT id — id may be absent on
  continuation chunks), each carrying a fragment of `function.arguments`
  as a string. We track index -> id/name from the first chunk that
  introduces each call.
"""

from __future__ import annotations

import json
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
    Tool,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

# OpenAI's finish_reason strings -> our normalized enum.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    # content_filter has no equivalent yet — falls through to OTHER.
}


def _normalize_stop_reason(reason: str | None) -> StopReason:
    if reason is None:
        return StopReason.OTHER
    return _STOP_REASON_MAP.get(reason, StopReason.OTHER)


class OpenAIAdapter(ProviderAdapter):
    """ProviderAdapter implementation backed by the official `openai` SDK."""

    def __init__(self, api_key: str | None = None, azure_endpoint: str | None = None, api_version: str | None = None ) -> None:
        if azure_endpoint is not None:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-02-01"
            )
        else:
            self._client = openai.AsyncOpenAI(api_key=api_key)
    # --- translation: core -> openai ---

    @staticmethod
    def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict]:
        """Folds the adapter-level `system` argument into the messages
        list as OpenAI requires, and expands ToolUseBlock/ToolResultBlock
        content into OpenAI's separate tool_calls field / role="tool"
        messages respectively.
        """
        result = []
        if system is not None:
            result.append({"role": "system", "content": system})

        for msg in messages:
            text_parts = []
            tool_calls = []
            tool_result_messages = []

            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                # OpenAI's wire format wants arguments as a
                                # JSON string, not a dict.
                                "arguments": json.dumps(block.input),
                            },
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    # Tool results are their OWN message in OpenAI's format,
                    # not a content block nested in this one — collect them
                    # separately and append as standalone messages below.
                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        }
                    )

            # Only emit a primary message if there's text and/or tool_calls;
            # a Message containing *only* ToolResultBlocks produces zero
            # "primary" messages and N tool-role messages instead.
            if text_parts or tool_calls:
                primary: dict = {"role": msg.role.value, "content": "".join(text_parts)}
                if tool_calls:
                    primary["tool_calls"] = tool_calls
                result.append(primary)

            result.extend(tool_result_messages)

        return result

    @staticmethod
    def _to_openai_tools(tools: list[Tool] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    # --- translation: openai -> core ---

    @staticmethod
    def _from_openai_response(raw) -> Response:
        choice = raw.choices[0]
        content = []
        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))
        for call in choice.message.tool_calls or []:
            content.append(
                ToolUseBlock(
                    id=call.id,
                    name=call.function.name,
                    # json.loads() here is the mirror of json.dumps() in
                    # _to_openai_messages — the JSON-string quirk is fully
                    # contained within this file.
                    input=json.loads(call.function.arguments),
                )
            )
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
        tools: list[Tool] | None = None,
    ) -> Response:
        kwargs: dict = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": self._to_openai_messages(messages, system),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        openai_tools = self._to_openai_tools(tools)
        if openai_tools is not None:
            kwargs["tools"] = openai_tools

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
        tools: list[Tool] | None = None,
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
        openai_tools = self._to_openai_tools(tools)
        if openai_tools is not None:
            kwargs["tools"] = openai_tools

        started = False
        # Maps OpenAI's tool_call list `index` -> True once we've emitted
        # a ToolCallStartChunk for it, since id/name only appear on the
        # first chunk that introduces each call — later chunks for the
        # same index carry only argument fragments, keyed by index alone.
        seen_indices: set[int] = set()

        async for chunk in await self._client.chat.completions.create(**kwargs):
            if not started:
                started = True
                yield MessageStartChunk(model=chunk.model)

            for normalized in self._translate_stream_chunk(chunk, seen_indices):
                yield normalized

    @staticmethod
    def _translate_stream_chunk(chunk, seen_indices: set[int]) -> list[StreamChunk]:
        results: list[StreamChunk] = []

        # The final usage-only chunk has an empty choices list.
        if not chunk.choices:
            if chunk.usage is not None:
                results.append(
                    MessageStopChunk(
                        stop_reason=StopReason.OTHER,  # finish_reason was on an earlier chunk
                        usage=Usage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        ),
                    )
                )
            return results

        choice = chunk.choices[0]

        if choice.delta.content:
            results.append(TextDeltaChunk(text=choice.delta.content))

        for tc in choice.delta.tool_calls or []:
            if tc.index not in seen_indices and tc.id is not None:
                seen_indices.add(tc.index)
                results.append(
                    ToolCallStartChunk(
                        index=tc.index,
                        id=tc.id,
                        name=tc.function.name if tc.function else "",
                    )
                )
            if tc.function is not None and tc.function.arguments:
                results.append(
                    ToolCallDeltaChunk(index=tc.index, partial_json=tc.function.arguments)
                )

        if choice.finish_reason is not None:
            results.append(
                MessageStopChunk(
                    stop_reason=_normalize_stop_reason(choice.finish_reason),
                    usage=None,  # usage arrives on the subsequent usage-only chunk
                )
            )

        return results