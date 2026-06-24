"""
Tests for OllamaAdapter, mirroring the other providers' test structure.
The key thing being proven here: a provider with no API key concept and
a different usage-field naming scheme (prompt_eval_count / eval_count
instead of usage.input_tokens / output_tokens) still satisfies the same
ProviderAdapter contract and produces identical Response/StreamChunk shapes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llmkit.adapters.ollama.adapter import OllamaAdapter
from llmkit.core.client import Client
from llmkit.core.types import Message, Role, StopReason


def _fake_ollama_response(text: str = "hi there", done_reason: str = "stop"):
    return SimpleNamespace(
        message=SimpleNamespace(content=text),
        done_reason=done_reason,
        prompt_eval_count=10,
        eval_count=5,
        model="gpt-oss:120b-cloud",
        model_dump=lambda: {"fake": "raw_response"},
    )


@pytest.mark.asyncio
async def test_generate_translates_response_correctly():
    adapter = OllamaAdapter()  # no api_key needed — local by default

    fake_response = _fake_ollama_response(text="Hello, world!")
    with patch.object(
        adapter._client, "chat", new=AsyncMock(return_value=fake_response)
    ) as mock_chat:
        client = Client(adapter)
        result = await client.generate(
            [Message.text(Role.USER, "say hello")],
            model="gpt-oss:120b-cloud",
            max_tokens=100,
            system="be friendly",
        )

    assert result.text() == "Hello, world!"
    assert result.stop_reason == StopReason.END_TURN
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.model == "gpt-oss:120b-cloud"

    # system role goes inline in messages, like OpenAI — and max_tokens
    # gets translated into options.num_predict, not a top-level kwarg.
    _, kwargs = mock_chat.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "be friendly"}
    assert kwargs["options"]["num_predict"] == 100


def test_no_api_key_required():
    """Confirms the ProviderAdapter contract never assumed every provider
    needs an api_key — Ollama works with zero arguments."""
    adapter = OllamaAdapter()
    assert adapter is not None


@pytest.mark.asyncio
async def test_generate_maps_length_stop_reason():
    adapter = OllamaAdapter()
    fake_response = _fake_ollama_response(done_reason="length")

    with patch.object(adapter._client, "chat", new=AsyncMock(return_value=fake_response)):
        result = await adapter.generate(
            [Message.text(Role.USER, "go on forever")],
            model="gpt-oss:120b-cloud",
            max_tokens=10,
        )

    assert result.stop_reason == StopReason.MAX_TOKENS


@pytest.mark.asyncio
async def test_stream_yields_normalized_text_deltas():
    adapter = OllamaAdapter()

    fake_chunks = [
        SimpleNamespace(model="gpt-oss:120b-cloud", done=False, message=SimpleNamespace(content="Hel")),
        SimpleNamespace(model="gpt-oss:120b-cloud", done=False, message=SimpleNamespace(content="lo!")),
        SimpleNamespace(
            model="gpt-oss:120b-cloud",
            done=True,
            done_reason="stop",
            message=SimpleNamespace(content=""),
            prompt_eval_count=10,
            eval_count=2,
        ),
    ]

    class FakeAsyncStream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for c in fake_chunks:
                yield c

    with patch.object(
        adapter._client, "chat", new=AsyncMock(return_value=FakeAsyncStream())
    ):
        chunks = [
            c
            async for c in adapter.stream(
                [Message.text(Role.USER, "hi")],
                model="gpt-oss:120b-cloud",
                max_tokens=50,
            )
        ]

    types = [c.type for c in chunks]
    assert types == ["message_start", "text_delta", "text_delta", "message_stop"]
    assert chunks[1].text == "Hel"
    assert chunks[2].text == "lo!"
    assert chunks[3].stop_reason == StopReason.END_TURN
    assert chunks[3].usage.input_tokens == 10
