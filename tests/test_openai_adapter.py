"""
Tests for OpenAIAdapter, mirroring the Anthropic adapter test style:
mocked SDK objects, no network access required.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llmkit.adapters.openai.adapter import OpenAIAdapter
from llmkit.core.client import Client
from llmkit.core.types import Message, Role, StopReason


def _fake_openai_response(text: str = "hi there", finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6),
        model="gpt-5",
        model_dump=lambda: {"fake": "raw_response"},
    )


@pytest.mark.asyncio
async def test_generate_translates_response_correctly():
    adapter = OpenAIAdapter(api_key="fake-key-not-used")
    fake_response = _fake_openai_response(text="Hello, world!")

    with patch.object(
        adapter._client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response),
    ) as mock_create:
        client = Client(adapter)
        result = await client.generate(
            [Message.text(Role.USER, "say hello")],
            model="gpt-5",
            max_tokens=100,
            system="be friendly",
        )

    assert result.text() == "Hello, world!"
    assert result.stop_reason == StopReason.END_TURN
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 6
    assert result.model == "gpt-5"

    # Request correctly translated: system folded INTO messages (unlike
    # Anthropic, which uses a separate top-level param), and max_tokens
    # mapped onto the non-deprecated max_completion_tokens field.
    _, kwargs = mock_create.call_args
    assert kwargs["max_completion_tokens"] == 100
    assert "max_tokens" not in kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "be friendly"},
        {"role": "user", "content": "say hello"},
    ]


@pytest.mark.asyncio
async def test_generate_maps_length_stop_reason():
    adapter = OpenAIAdapter(api_key="fake-key-not-used")
    fake_response = _fake_openai_response(finish_reason="length")

    with patch.object(
        adapter._client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_response),
    ):
        result = await adapter.generate(
            [Message.text(Role.USER, "go on forever")],
            model="gpt-5",
            max_tokens=10,
        )

    assert result.stop_reason == StopReason.MAX_TOKENS


@pytest.mark.asyncio
async def test_stream_yields_normalized_text_deltas():
    adapter = OpenAIAdapter(api_key="fake-key-not-used")

    fake_chunks = [
        SimpleNamespace(
            model="gpt-5",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"), finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-5",
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo!"), finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-5",
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")],
            usage=None,
        ),
        SimpleNamespace(
            model="gpt-5",
            choices=[],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        ),
    ]

    async def fake_stream():
        for c in fake_chunks:
            yield c

    with patch.object(
        adapter._client.chat.completions,
        "create",
        new=AsyncMock(return_value=fake_stream()),
    ):
        chunks = [
            c
            async for c in adapter.stream(
                [Message.text(Role.USER, "hi")],
                model="gpt-5",
                max_tokens=50,
            )
        ]

    types = [c.type for c in chunks]
    assert types == [
        "message_start",
        "text_delta",
        "text_delta",
        "message_stop",  # finish_reason chunk
        "message_stop",  # usage-only chunk
    ]
    assert chunks[1].text == "Hel"
    assert chunks[2].text == "lo!"
    assert chunks[3].stop_reason == StopReason.END_TURN
    assert chunks[4].usage.input_tokens == 3
    assert chunks[4].usage.output_tokens == 2
