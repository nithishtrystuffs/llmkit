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

2. "max_tokens is a top-level param" — false here. Ollama nests it inside
   an `options` dict as `num_predict`.

Other differences absorbed here:
- System prompts ARE allowed inline in the messages list (role="system"),
  same as OpenAI — no separate system param.
- Responses are typed Pydantic models (ChatResponse), but usage lives
  under different names: `prompt_eval_count` / `eval_count`.
- `done_reason` is Ollama's stop-reason field, with a much smaller
  vocabulary than the hosted providers (chiefly "stop" and "length").
- Streaming chunks are also full ChatResponse objects (like Gemini), each
  carrying an incremental `message.content` delta; the final chunk has
  `done=True` plus the usage/done_reason fields populated.

Tool calling notes — Ollama has the weakest tool-calling model of the
four, and this adapter has to compensate for a genuine missing feature
rather than just a naming difference:

- Tool definitions: passed as plain dicts (the library accepts
  Mapping[str, Any] alongside its own strict Tool/Function/Parameters
  models) in the same {"type": "function", "function": {name,
  description, parameters}} shape OpenAI uses — Ollama's chat API is
  OpenAI-derived for this part.
- Incoming tool_use: Ollama's Message.tool_calls entries have NO id field
  at all — just {function: {name, arguments: dict}}. Unlike OpenAI/Gemini,
  there's no call identifier to round-trip. This adapter synthesizes a
  deterministic id (f"{name}_{position}") so our ToolUseBlock.id is never
  empty, but this id only has meaning within this library — it is NOT
  something the Ollama API itself recognizes or expects back.
- Outgoing tool_result: Ollama's Message supports `tool_name` (the
  function's name) rather than a tool_call_id-keyed role="tool" message.
  Since our synthesized id is f"{name}_{position}", this adapter recovers
  the name by stripping the position suffix — fragile in the abstract,
  but correct for the synthesis scheme this same adapter controls on both
  ends, since both translation directions are owned by this one file.
- No incremental argument streaming, same limitation as Gemini: arguments
  arrive complete in one chunk. This adapter emits one ToolCallStartChunk
  immediately followed by one ToolCallDeltaChunk with the full arguments.
- Tool support also depends on the local model itself supporting tools
  (not every model pulled into Ollama does) — a model that doesn't
  support tools will simply never emit tool_calls, which surfaces to
  callers as an ordinary text-only Response rather than an error.
"""

from __future__ import annotations

import json
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
    Tool,
    ToolCallDeltaChunk,
    ToolCallStartChunk,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

# Ollama's done_reason strings -> our normalized enum. Smaller vocabulary
# than the hosted providers — most local models only ever report these two.
_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
}


def _normalize_stop_reason(reason: str | None, has_tool_calls: bool = False) -> StopReason:
    if has_tool_calls:
        return StopReason.TOOL_USE
    if reason is None:
        return StopReason.OTHER
    return _STOP_REASON_MAP.get(reason, StopReason.OTHER)


def _synthesize_tool_call_id(name: str, position: int) -> str:
    """Ollama gives no call id at all. This synthesized id is only ever
    interpreted by this adapter (see _to_ollama_messages's reverse lookup)
    — it is never sent to or expected by the Ollama API itself."""
    return f"{name}_{position}"


def _name_from_synthesized_id(tool_use_id: str) -> str:
    """Reverses _synthesize_tool_call_id. Strips the trailing _<position>
    suffix. Relies on tool names not containing underscores followed only
    by digits at the very end, which holds for the synthesis scheme this
    same file controls end-to-end."""
    name, _, _position = tool_use_id.rpartition("_")
    return name or tool_use_id


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
        no separate system param. ToolUseBlock/ToolResultBlock both expand
        into Ollama's own shapes, with the id<->name synthesis described
        in this module's docstring.
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
                        {"function": {"name": block.name, "arguments": block.input}}
                    )
                elif isinstance(block, ToolResultBlock):
                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_name": _name_from_synthesized_id(block.tool_use_id),
                            "content": block.content,
                        }
                    )

            if text_parts or tool_calls:
                primary: dict = {"role": msg.role.value, "content": "".join(text_parts)}
                if tool_calls:
                    primary["tool_calls"] = tool_calls
                result.append(primary)

            result.extend(tool_result_messages)

        return result

    @staticmethod
    def _to_ollama_tools(tools: list[Tool] | None) -> list[dict] | None:
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

    # --- translation: ollama -> core ---

    @staticmethod
    def _from_ollama_response(raw: ollama.ChatResponse) -> Response:
        content = []
        if raw.message.content:
            content.append(TextBlock(text=raw.message.content))

        has_tool_calls = bool(raw.message.tool_calls)
        for i, call in enumerate(raw.message.tool_calls or []):
            content.append(
                ToolUseBlock(
                    id=_synthesize_tool_call_id(call.function.name, i),
                    name=call.function.name,
                    input=dict(call.function.arguments),
                )
            )

        return Response(
            content=content,
            stop_reason=_normalize_stop_reason(raw.done_reason, has_tool_calls),
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
        tools: list[Tool] | None = None,
    ) -> Response:
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        kwargs: dict = {
            "model": model,
            "messages": self._to_ollama_messages(messages, system),
            "options": options,
        }
        ollama_tools = self._to_ollama_tools(tools)
        if ollama_tools is not None:
            kwargs["tools"] = ollama_tools

        raw = await self._client.chat(**kwargs)
        return self._from_ollama_response(raw)

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
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        kwargs: dict = {
            "model": model,
            "messages": self._to_ollama_messages(messages, system),
            "options": options,
            "stream": True,
        }
        ollama_tools = self._to_ollama_tools(tools)
        if ollama_tools is not None:
            kwargs["tools"] = ollama_tools

        started = False
        next_index = 0
        async for chunk in await self._client.chat(**kwargs):
            if not started:
                yield MessageStartChunk(model=chunk.model)
                started = True

            for translated in self._translate_stream_chunk(chunk, next_index):
                yield translated
                if translated.type == "tool_call_start":
                    next_index += 1

    @staticmethod
    def _translate_stream_chunk(
        chunk: ollama.ChatResponse, next_index: int
    ) -> list[StreamChunk]:
        results: list[StreamChunk] = []
        index = next_index
        has_tool_calls = False

        for call in chunk.message.tool_calls or []:
            has_tool_calls = True
            call_id = _synthesize_tool_call_id(call.function.name, index)
            results.append(ToolCallStartChunk(index=index, id=call_id, name=call.function.name))
            results.append(
                ToolCallDeltaChunk(index=index, partial_json=json.dumps(dict(call.function.arguments)))
            )
            index += 1

        if chunk.message and chunk.message.content:
            results.append(TextDeltaChunk(text=chunk.message.content))

        if chunk.done:
            results.append(
                MessageStopChunk(
                    stop_reason=_normalize_stop_reason(chunk.done_reason, has_tool_calls),
                    usage=Usage(
                        input_tokens=chunk.prompt_eval_count or 0,
                        output_tokens=chunk.eval_count or 0,
                    ),
                )
            )

        return results