"""
Tests for retry logic, timeout handling, error normalization, and cost tracking.
All mocked — no network access or real API keys required.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmkit import (
    APIError,
    AuthenticationError,
    Client,
    ConnectionError,
    CostTracker,
    LLMKitError,
    Message,
    RateLimitError,
    RetryConfig,
    Role,
    TimeoutError,
    UnknownError,
)
from llmkit.core.error_map import normalize_error
from llmkit.core.errors import InvalidRequestError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_adapter(response=None, side_effect=None):
    """Returns a mock ProviderAdapter whose generate() returns `response`
    or raises `side_effect`."""
    adapter = MagicMock()
    adapter.__class__.__name__ = "AnthropicAdapter"
    if side_effect is not None:
        adapter.generate = AsyncMock(side_effect=side_effect)
    else:
        adapter.generate = AsyncMock(return_value=response)
    return adapter


def _fake_response(input_tokens=10, output_tokens=5, model="claude-sonnet-4-6"):
    from llmkit.core.types import Response, StopReason, TextBlock, Usage
    return Response(
        content=[TextBlock(text="hello")],
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
    )


# ---------------------------------------------------------------------------
# Error normalization tests
# ---------------------------------------------------------------------------

def test_normalize_already_llmkit_error_passes_through():
    """LLMKitErrors must not be double-wrapped."""
    original = RateLimitError("too fast", provider="anthropic")
    result = normalize_error(original, "anthropic")
    assert result is original


def test_normalize_anthropic_rate_limit():
    import anthropic
    exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    exc.args = ("rate limited",)
    result = normalize_error(exc, "anthropic")
    assert isinstance(result, RateLimitError)
    assert result.provider == "anthropic"
    assert result.status_code == 429
    assert result.cause is exc


def test_normalize_anthropic_auth_error():
    import anthropic
    exc = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
    exc.args = ("bad key",)
    result = normalize_error(exc, "anthropic")
    assert isinstance(result, AuthenticationError)
    assert result.provider == "anthropic"


def test_normalize_openai_rate_limit():
    import openai
    exc = openai.RateLimitError.__new__(openai.RateLimitError)
    exc.args = ("rate limited",)
    result = normalize_error(exc, "openai")
    assert isinstance(result, RateLimitError)
    assert result.provider == "openai"


def test_normalize_ollama_connection_refused():
    exc = ConnectionRefusedError("connection refused")
    result = normalize_error(exc, "ollama")
    assert isinstance(result, ConnectionError)
    assert result.provider == "ollama"
    assert "ollama serve" in result.message


def test_normalize_unknown_provider_returns_unknown_error():
    exc = ValueError("something weird")
    result = normalize_error(exc, "someprovider")
    assert isinstance(result, UnknownError)
    assert result.provider == "someprovider"
    assert result.cause is exc


@pytest.mark.asyncio
async def test_client_normalizes_raw_sdk_exception():
    """Raw SDK exceptions must be wrapped in a LLMKitError before surfacing."""
    import anthropic
    raw_exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    raw_exc.args = ("rate limited",)

    adapter = _fake_adapter(side_effect=raw_exc)
    client = Client(adapter, retry_config=RetryConfig(max_attempts=1))

    with pytest.raises(RateLimitError):
        await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    """First call raises a retryable error; second call succeeds."""
    import anthropic
    rate_exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    rate_exc.args = ("rate limited",)

    good_response = _fake_response()
    adapter = _fake_adapter(side_effect=[rate_exc, good_response])

    client = Client(
        adapter,
        retry_config=RetryConfig(max_attempts=2, base_delay=0.0),
    )
    result = await client.generate(
        [Message.text(Role.USER, "hi")], model="claude-sonnet-4-6"
    )
    assert result.text() == "hello"
    assert adapter.generate.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted_raises_last_error():
    """All attempts fail — last error is raised."""
    import anthropic
    rate_exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    rate_exc.args = ("rate limited",)

    adapter = _fake_adapter(side_effect=rate_exc)
    client = Client(
        adapter,
        retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
    )
    with pytest.raises(RateLimitError):
        await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")

    assert adapter.generate.call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_error_not_retried():
    """AuthenticationError is not retryable — must fail immediately, not retry."""
    import anthropic
    auth_exc = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
    auth_exc.args = ("bad key",)

    adapter = _fake_adapter(side_effect=auth_exc)
    client = Client(
        adapter,
        retry_config=RetryConfig(max_attempts=3, base_delay=0.0),
    )
    with pytest.raises(AuthenticationError):
        await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")

    # Must have only tried once despite max_attempts=3
    assert adapter.generate.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_retry_callback_called():
    """on_retry callback is called once before each retry."""
    import anthropic
    rate_exc = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    rate_exc.args = ("rate limited",)

    good_response = _fake_response()
    adapter = _fake_adapter(side_effect=[rate_exc, rate_exc, good_response])

    calls = []
    def on_retry(attempt, error, wait):
        calls.append((attempt, type(error).__name__, wait))

    client = Client(
        adapter,
        retry_config=RetryConfig(max_attempts=3, base_delay=0.0, on_retry=on_retry),
    )
    await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")
    assert len(calls) == 2
    assert calls[0][1] == "RateLimitError"


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_raises_timeout_error():
    """If the adapter call exceeds timeout_seconds, TimeoutError is raised."""
    import asyncio as aio

    async def slow_generate(*args, **kwargs):
        await aio.sleep(10)  # longer than timeout

    adapter = MagicMock()
    adapter.__class__.__name__ = "AnthropicAdapter"
    adapter.generate = slow_generate

    client = Client(
        adapter,
        retry_config=RetryConfig(max_attempts=1),
        timeout_seconds=0.01,
    )
    with pytest.raises(TimeoutError) as exc_info:
        await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")

    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_timeout_retried_if_retryable():
    """TimeoutError is retryable — a second attempt that succeeds should work."""
    import asyncio as aio

    call_count = 0

    async def sometimes_slow(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await aio.sleep(10)
        return _fake_response()

    adapter = MagicMock()
    adapter.__class__.__name__ = "AnthropicAdapter"
    adapter.generate = sometimes_slow

    client = Client(
        adapter,
        retry_config=RetryConfig(max_attempts=2, base_delay=0.0),
        timeout_seconds=0.01,
    )
    result = await client.generate(
        [Message.text(Role.USER, "hi")], model="claude-sonnet-4-6"
    )
    assert result.text() == "hello"
    assert call_count == 2


# ---------------------------------------------------------------------------
# Cost tracking tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cost_tracker_records_per_call():
    tracker = CostTracker()
    adapter = _fake_adapter(response=_fake_response(input_tokens=100, output_tokens=50))
    client = Client(adapter, cost_tracker=tracker)

    await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")

    assert tracker.call_count == 1
    assert tracker.total_input_tokens == 100
    assert tracker.total_output_tokens == 50
    assert tracker.total_tokens == 150


@pytest.mark.asyncio
async def test_cost_tracker_accumulates_across_calls():
    tracker = CostTracker()
    adapter = _fake_adapter(response=_fake_response(input_tokens=10, output_tokens=5))
    client = Client(adapter, cost_tracker=tracker)

    await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")
    await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")
    await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6")

    assert tracker.call_count == 3
    assert tracker.total_input_tokens == 30
    assert tracker.total_output_tokens == 15


@pytest.mark.asyncio
async def test_cost_tracker_calculates_cost_correctly():
    """claude-sonnet-4-6: $3.00/1M input, $15.00/1M output."""
    tracker = CostTracker()
    adapter = _fake_adapter(
        response=_fake_response(input_tokens=1_000_000, output_tokens=1_000_000)
    )
    client = Client(adapter, cost_tracker=tracker)

    await client.generate(
        [Message.text(Role.USER, "hi")], model="claude-sonnet-4-6"
    )

    assert abs(tracker.total_cost_usd - 18.0) < 0.001  # $3 in + $15 out


@pytest.mark.asyncio
async def test_cost_tracker_unknown_model_returns_zero_cost():
    """Unknown models should record 0 cost, not raise."""
    tracker = CostTracker()
    adapter = _fake_adapter(
        response=_fake_response(input_tokens=1000, output_tokens=500, model="mystery-model-9000")
    )
    client = Client(adapter, cost_tracker=tracker)
    await client.generate([Message.text(Role.USER, "hi")], model="mystery-model-9000")

    assert tracker.total_cost_usd == 0.0
    assert tracker.call_count == 1


@pytest.mark.asyncio
async def test_cost_tracker_custom_price_table():
    """Custom price table values are merged over defaults."""
    tracker = CostTracker(price_table={"my-model": {"input": 10.0, "output": 20.0}})
    adapter = _fake_adapter(
        response=_fake_response(input_tokens=1_000_000, output_tokens=1_000_000, model="my-model")
    )
    client = Client(adapter, cost_tracker=tracker)
    await client.generate([Message.text(Role.USER, "hi")], model="my-model")

    assert abs(tracker.total_cost_usd - 30.0) < 0.001  # $10 in + $20 out


def test_cost_tracker_summary():
    tracker = CostTracker()
    from llmkit.core.cost import CallRecord
    tracker.calls.append(CallRecord(
        model="claude-sonnet-4-6", provider="anthropic",
        input_tokens=100, output_tokens=50, cost_usd=0.001,
    ))
    summary = tracker.summary()
    assert "1" in summary
    assert "150" in summary


def test_cost_tracker_by_model():
    tracker = CostTracker()
    from llmkit.core.cost import CallRecord
    tracker.calls.append(CallRecord("gpt-4o", "openai", 100, 50, 0.001))
    tracker.calls.append(CallRecord("gpt-4o", "openai", 200, 100, 0.002))
    tracker.calls.append(CallRecord("claude-sonnet-4-6", "anthropic", 50, 25, 0.0005))
    by_model = tracker.by_model()
    assert by_model["gpt-4o"]["calls"] == 2
    assert by_model["gpt-4o"]["input_tokens"] == 300
    assert "claude-sonnet-4-6" in by_model


def test_cost_tracker_reset():
    tracker = CostTracker()
    from llmkit.core.cost import CallRecord
    tracker.calls.append(CallRecord("gpt-4o", "openai", 100, 50, 0.001))
    tracker.reset()
    assert tracker.call_count == 0
    assert tracker.total_cost_usd == 0.0


def test_client_cost_tracker_property():
    tracker = CostTracker()
    adapter = _fake_adapter(response=_fake_response())
    client = Client(adapter, cost_tracker=tracker)
    assert client.cost_tracker is tracker


def test_client_no_cost_tracker_property_is_none():
    adapter = _fake_adapter(response=_fake_response())
    client = Client(adapter)
    assert client.cost_tracker is None
