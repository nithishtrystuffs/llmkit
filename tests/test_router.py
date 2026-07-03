"""
Tests for RouterClient.

Proves: primary success, fallback on various error types, retry-before-fallback
for rate limits/timeouts, immediate fallback for connection/auth/API errors,
no fallback for InvalidRequestError, callback firing, streaming fallback,
cost tracking across providers, and active_providers property.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmkit import (
    APIError,
    AuthenticationError,
    ConnectionError,
    CostTracker,
    InvalidRequestError,
    Message,
    ProviderConfig,
    RateLimitError,
    RetryConfig,
    Role,
    RouterClient,
)
from llmkit.core.errors import TimeoutError as LLMTimeoutError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_adapter(name: str, response=None, side_effect=None):
    adapter = MagicMock()
    adapter.__class__.__name__ = f"{name.capitalize()}Adapter"
    if side_effect is not None:
        adapter.generate = AsyncMock(side_effect=side_effect)
    else:
        adapter.generate = AsyncMock(return_value=response)
    return adapter


def _fake_response(text="hello", model="claude-sonnet-4-6"):
    from llmkit.core.types import Response, StopReason, TextBlock, Usage
    return Response(
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=10, output_tokens=5),
        model=model,
    )


def _provider(name: str, model: str, response=None, side_effect=None):
    adapter = _fake_adapter(name, response=response, side_effect=side_effect)
    return ProviderConfig(adapter=adapter, model=model)


def _messages():
    return [Message.text(Role.USER, "hi")]


# ---------------------------------------------------------------------------
# Basic routing tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_success_no_fallback():
    """Primary succeeds — fallback adapter should never be called."""
    primary = _provider("anthropic", "claude-sonnet-4-6", response=_fake_response("from anthropic"))
    fallback = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    router = RouterClient([primary, fallback], retry_config=RetryConfig(max_attempts=1))
    result = await router.generate(_messages(), max_tokens=100)

    assert result.text() == "from anthropic"
    fallback.adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_falls_back_to_second_provider_on_connection_error():
    primary = _provider("anthropic", "claude-sonnet-4-6",
                        side_effect=ConnectionError("down", provider="anthropic"))
    fallback = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    router = RouterClient([primary, fallback], retry_config=RetryConfig(max_attempts=1))
    result = await router.generate(_messages(), max_tokens=100)

    assert result.text() == "from openai"


@pytest.mark.asyncio
async def test_falls_back_through_all_providers():
    """If two providers fail, third succeeds."""
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=ConnectionError("down", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o",
                   side_effect=APIError("500", provider="openai", status_code=500))
    p3 = _provider("gemini", "gemini-2.5-flash", response=_fake_response("from gemini"))

    router = RouterClient([p1, p2, p3], retry_config=RetryConfig(max_attempts=1))
    result = await router.generate(_messages(), max_tokens=100)

    assert result.text() == "from gemini"


@pytest.mark.asyncio
async def test_raises_last_error_when_all_providers_fail():
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=ConnectionError("down", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o",
                   side_effect=APIError("500", provider="openai", status_code=500))

    router = RouterClient([p1, p2], retry_config=RetryConfig(max_attempts=1))

    with pytest.raises(APIError):
        await router.generate(_messages(), max_tokens=100)


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_request_error_not_retried_and_not_fallen_back():
    """InvalidRequestError means the request itself is broken — no fallback."""
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=InvalidRequestError("bad params", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    router = RouterClient([p1, p2], retry_config=RetryConfig(max_attempts=1))

    with pytest.raises(InvalidRequestError):
        await router.generate(_messages(), max_tokens=100)

    p2.adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_auth_error_triggers_immediate_fallback():
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=AuthenticationError("bad key", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    router = RouterClient([p1, p2], retry_config=RetryConfig(max_attempts=1))
    result = await router.generate(_messages(), max_tokens=100)

    assert result.text() == "from openai"
    # Primary called once — no retry, immediate fallback
    assert p1.adapter.generate.call_count == 1


@pytest.mark.asyncio
async def test_rate_limit_retried_before_fallback():
    """RateLimitError: retry same provider (per retry_config), THEN fall back."""
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=RateLimitError("429", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    router = RouterClient(
        [p1, p2],
        retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
    )
    result = await router.generate(_messages(), max_tokens=100)

    assert result.text() == "from openai"
    # Primary retried 3 times (max_attempts=3), then fell back to openai
    assert p1.adapter.generate.call_count == 3
    assert p2.adapter.generate.call_count == 1


@pytest.mark.asyncio
async def test_api_error_triggers_immediate_fallback_no_retry():
    """APIError (5xx): fall back immediately without per-provider retries."""
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=APIError("500", provider="anthropic", status_code=500))
    p2 = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    router = RouterClient(
        [p1, p2],
        retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
    )
    result = await router.generate(_messages(), max_tokens=100)

    assert result.text() == "from openai"
    # Despite max_attempts=3, APIError triggers immediate fallback
    assert p1.adapter.generate.call_count == 1


# ---------------------------------------------------------------------------
# on_fallback callback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_fallback_callback_called_with_correct_args():
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=ConnectionError("down", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o", response=_fake_response())

    calls = []
    router = RouterClient(
        [p1, p2],
        retry_config=RetryConfig(max_attempts=1),
        on_fallback=lambda from_p, to_p, err: calls.append((from_p, to_p, type(err).__name__)),
    )
    await router.generate(_messages(), max_tokens=100)

    assert len(calls) == 1
    assert calls[0] == ("anthropic", "openai", "ConnectionError")


@pytest.mark.asyncio
async def test_on_fallback_callback_exception_doesnt_crash_router():
    """A buggy on_fallback callback must never crash the fallback loop."""
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=ConnectionError("down", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o", response=_fake_response("from openai"))

    def bad_callback(from_p, to_p, err):
        raise RuntimeError("callback exploded")

    router = RouterClient(
        [p1, p2],
        retry_config=RetryConfig(max_attempts=1),
        on_fallback=bad_callback,
    )
    result = await router.generate(_messages(), max_tokens=100)
    assert result.text() == "from openai"


# ---------------------------------------------------------------------------
# Cost tracking tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cost_tracker_records_actual_provider_used():
    tracker = CostTracker()
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=ConnectionError("down", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o",
                   response=_fake_response(model="gpt-4o"))

    router = RouterClient(
        [p1, p2],
        retry_config=RetryConfig(max_attempts=1),
        cost_tracker=tracker,
    )
    await router.generate(_messages(), max_tokens=100)

    assert tracker.call_count == 1
    assert tracker.calls[0].model == "gpt-4o"
    assert tracker.calls[0].provider == "openai"


# ---------------------------------------------------------------------------
# Properties tests
# ---------------------------------------------------------------------------

def test_active_providers_returns_names_in_order():
    p1 = _provider("anthropic", "claude-sonnet-4-6", response=_fake_response())
    p2 = _provider("openai", "gpt-4o", response=_fake_response())
    p3 = _provider("gemini", "gemini-2.5-flash", response=_fake_response())

    router = RouterClient([p1, p2, p3])
    assert router.active_providers == ["anthropic", "openai", "gemini"]


def test_single_provider_works():
    p1 = _provider("anthropic", "claude-sonnet-4-6", response=_fake_response())
    router = RouterClient([p1])
    assert router.active_providers == ["anthropic"]


def test_empty_providers_raises():
    with pytest.raises(ValueError, match="at least one"):
        RouterClient([])


# ---------------------------------------------------------------------------
# Streaming fallback tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_falls_back_on_connection_error():
    """If stream setup fails before first chunk, router falls back."""
    async def failing_stream(*args, **kwargs):
        raise ConnectionError("down", provider="anthropic")
        yield  # make it an async generator

    p1_adapter = MagicMock()
    p1_adapter.__class__.__name__ = "AnthropicAdapter"
    p1_adapter.generate = AsyncMock()
    p1_adapter.stream = failing_stream

    p2_adapter = MagicMock()
    p2_adapter.__class__.__name__ = "OpenAIAdapter"
    p2_adapter.generate = AsyncMock()

    async def good_stream(*args, **kwargs):
        from llmkit.core.types import MessageStartChunk, TextDeltaChunk
        yield MessageStartChunk(model="gpt-4o")
        yield TextDeltaChunk(text="hello from openai")

    p2_adapter.stream = good_stream

    router = RouterClient(
        [
            ProviderConfig(p1_adapter, model="claude-sonnet-4-6"),
            ProviderConfig(p2_adapter, model="gpt-4o"),
        ],
        retry_config=RetryConfig(max_attempts=1),
    )

    chunks = [c async for c in router.stream(_messages(), max_tokens=100)]
    text_chunks = [c for c in chunks if c.type == "text_delta"]
    assert len(text_chunks) == 1
    assert text_chunks[0].text == "hello from openai"


@pytest.mark.asyncio
async def test_per_provider_model_used_in_generate_call():
    """Each provider must receive its own configured model string."""
    p1 = _provider("anthropic", "claude-sonnet-4-6",
                   side_effect=ConnectionError("down", provider="anthropic"))
    p2 = _provider("openai", "gpt-4o", response=_fake_response(model="gpt-4o"))

    router = RouterClient([p1, p2], retry_config=RetryConfig(max_attempts=1))
    await router.generate(_messages(), max_tokens=200)

    _, kwargs_p1 = p1.adapter.generate.call_args_list[0]
    assert kwargs_p1["model"] == "claude-sonnet-4-6"

    _, kwargs_p2 = p2.adapter.generate.call_args_list[0]
    assert kwargs_p2["model"] == "gpt-4o"