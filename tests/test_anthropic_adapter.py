"""
Tests for AnthropicAdapter, using mocked `anthropic` SDK objects so no
network access or API key is required. This is the "contract test" style
from the project design: verify the adapter correctly translates in both
directions without depending on a live API.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llmkit.adapters.anthropic.adapter import AnthropicAdapter
from llmkit.core.client import Client
from llmkit.core.types import Message, Role, StopReason


def _fake_anthropic_message(text: str = "hi there", stop_reason: str = "end_turn"):
    """Builds a fake object shaped like anthropic.types.Message, just
    enough to exercise our translation code."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-sonnet-4-6",
        model_dump=lambda: {"fake": "raw_response"},
    )


@pytest.mark.asyncio
async def test_generate_translates_response_correctly():
    adapter = AnthropicAdapter(api_key="fake-key-not-used")

    fake_response = _fake_anthropic_message(text="Hello, world!")
    with patch.object(
        adapter._client.messages, "create", new=AsyncMock(return_value=fake_response)
    ) as mock_create:
        client = Client(adapter)
        result = await client.generate(
            [Message.text(Role.USER, "say hello")],
            model="claude-sonnet-4-6",
            max_tokens=100,
            system="be friendly",
        )

    # Response correctly normalized
    assert result.text() == "Hello, world!"
    assert result.stop_reason == StopReason.END_TURN
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.model == "claude-sonnet-4-6"
    assert result.raw == {"fake": "raw_response"}  # escape hatch present

    # Request correctly translated — system passed separately, not in messages
    _, kwargs = mock_create.call_args
    assert kwargs["system"] == "be friendly"
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "say hello"}]}
    ]


@pytest.mark.asyncio
async def test_generate_maps_max_tokens_stop_reason():
    adapter = AnthropicAdapter(api_key="fake-key-not-used")
    fake_response = _fake_anthropic_message(stop_reason="max_tokens")

    with patch.object(
        adapter._client.messages, "create", new=AsyncMock(return_value=fake_response)
    ):
        result = await adapter.generate(
            [Message.text(Role.USER, "go on forever")],
            model="claude-sonnet-4-6",
            max_tokens=10,
        )

    assert result.stop_reason == StopReason.MAX_TOKENS


def test_system_message_in_messages_list_raises():
    """Anthropic requires system prompts via a separate param, not in the
    messages list. Adapter should fail loudly, not silently drop it."""
    msgs = [Message.text(Role.SYSTEM, "you are a pirate")]
    with pytest.raises(ValueError, match="system"):
        AnthropicAdapter._to_anthropic_messages(msgs)


@pytest.mark.asyncio
async def test_stream_yields_normalized_text_deltas():
    adapter = AnthropicAdapter(api_key="fake-key-not-used")

    fake_events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(model="claude-sonnet-4-6")),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hel"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="lo!"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=2),
        ),
    ]

    class FakeStreamCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for e in fake_events:
                yield e

    with patch.object(adapter._client.messages, "stream", return_value=FakeStreamCtx()):
        chunks = [c async for c in adapter.stream(
            [Message.text(Role.USER, "hi")],
            model="claude-sonnet-4-6",
            max_tokens=50,
        )]

    types = [c.type for c in chunks]
    assert types == ["message_start", "text_delta", "text_delta", "message_stop"]
    assert chunks[1].text == "Hel"
    assert chunks[2].text == "lo!"
    assert chunks[3].stop_reason == StopReason.END_TURN
