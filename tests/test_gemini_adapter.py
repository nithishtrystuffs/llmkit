"""
Tests for GeminiAdapter, mirroring the Anthropic/OpenAI test structure.
The key thing being proven here: Gemini's "each stream chunk is a full
response object" shape and "model" role naming don't leak outside the
adapter — Client/Message usage stays identical to the other providers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llmkit.adapters.gemini.adapter import GeminiAdapter
from llmkit.core.client import Client
from llmkit.core.types import Message, Role, StopReason


def _fake_part(text: str):
    return SimpleNamespace(text=text)


def _fake_candidate(text: str, finish_reason=None):
    return SimpleNamespace(
        content=SimpleNamespace(parts=[_fake_part(text)] if text else []),
        finish_reason=finish_reason,
    )


def _fake_usage(prompt=10, candidates=5):
    return SimpleNamespace(prompt_token_count=prompt, candidates_token_count=candidates)


def _fake_gemini_response(text: str = "hi there", finish_reason: str = "STOP"):
    return SimpleNamespace(
        candidates=[_fake_candidate(text, finish_reason)],
        usage_metadata=_fake_usage(),
        model_dump=lambda: {"fake": "raw_response"},
    )


@pytest.mark.asyncio
async def test_generate_translates_response_correctly():
    adapter = GeminiAdapter(api_key="fake-key-not-used")

    fake_response = _fake_gemini_response(text="Hello, world!")
    with patch.object(
        adapter._client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=fake_response),
    ) as mock_create:
        client = Client(adapter)
        result = await client.generate(
            [Message.text(Role.USER, "say hello")],
            model="gemini-2.5-flash",
            max_tokens=100,
            system="be friendly",
        )

    assert result.text() == "Hello, world!"
    assert result.stop_reason == StopReason.END_TURN
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.model == "gemini-2.5-flash"

    # system_instruction is set on config, NOT folded into contents.
    _, kwargs = mock_create.call_args
    assert kwargs["config"].system_instruction == "be friendly"
    assert len(kwargs["contents"]) == 1
    assert kwargs["contents"][0].role == "user"  # not "model" for a user turn


@pytest.mark.asyncio
async def test_assistant_role_maps_to_model():
    """Gemini calls the assistant role 'model', not 'assistant' — this
    must not leak into the public Message/Role API."""
    adapter = GeminiAdapter(api_key="fake-key-not-used")
    contents = adapter._to_gemini_contents(
        [
            Message.text(Role.USER, "hi"),
            Message.text(Role.ASSISTANT, "hello"),
        ]
    )
    assert contents[0].role == "user"
    assert contents[1].role == "model"


@pytest.mark.asyncio
async def test_generate_maps_max_tokens_stop_reason():
    adapter = GeminiAdapter(api_key="fake-key-not-used")
    fake_response = _fake_gemini_response(finish_reason="MAX_TOKENS")

    with patch.object(
        adapter._client.aio.models,
        "generate_content",
        new=AsyncMock(return_value=fake_response),
    ):
        result = await adapter.generate(
            [Message.text(Role.USER, "go on forever")],
            model="gemini-2.5-flash",
            max_tokens=10,
        )

    assert result.stop_reason == StopReason.MAX_TOKENS


def test_system_message_in_messages_list_raises():
    """Like Anthropic, Gemini requires system prompts via a separate
    field, not in the contents list."""
    msgs = [Message.text(Role.SYSTEM, "you are a pirate")]
    with pytest.raises(ValueError, match="system"):
        GeminiAdapter._to_gemini_contents(msgs)


@pytest.mark.asyncio
async def test_stream_yields_normalized_text_deltas():
    """Each Gemini stream chunk is a FULL response object — the adapter
    must still surface only incremental text_delta chunks to callers."""
    adapter = GeminiAdapter(api_key="fake-key-not-used")

    fake_chunks = [
        _fake_gemini_response_for_stream(text="Hel"),
        _fake_gemini_response_for_stream(text="lo!"),
        _fake_gemini_response_for_stream(text="", finish_reason="STOP"),
    ]

    class FakeAsyncStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for c in fake_chunks:
                yield c

    with patch.object(
        adapter._client.aio.models,
        "generate_content_stream",
        new=AsyncMock(return_value=FakeAsyncStream()),
    ):
        chunks = [
            c
            async for c in adapter.stream(
                [Message.text(Role.USER, "hi")],
                model="gemini-2.5-flash",
                max_tokens=50,
            )
        ]

    types = [c.type for c in chunks]
    assert types == ["message_start", "text_delta", "text_delta", "message_stop"]
    assert chunks[1].text == "Hel"
    assert chunks[2].text == "lo!"
    assert chunks[3].stop_reason == StopReason.END_TURN


def _fake_gemini_response_for_stream(text: str, finish_reason=None):
    return SimpleNamespace(
        candidates=[_fake_candidate(text, finish_reason)],
        usage_metadata=_fake_usage() if finish_reason else None,
    )
