"""
Public-facing client. This is what user code imports and calls.

Now supports:
- Retry with exponential backoff (optional RetryConfig)
- Per-attempt timeout (optional timeout_seconds)
- Normalized error exceptions (always — raw SDK errors are never surfaced)
- Cost tracking (optional CostTracker)

Minimal usage (same as before):
    client = Client(AnthropicAdapter())
    response = await client.generate(...)

With all features:
    from llmkit.core.retry import RetryConfig
    from llmkit.core.cost import CostTracker

    tracker = CostTracker()
    client = Client(
        AnthropicAdapter(),
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        timeout_seconds=30.0,
        cost_tracker=tracker,
    )
    response = await client.generate(...)
    print(tracker.summary())
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from llmkit.adapters.base import ProviderAdapter
from llmkit.core.cost import CostTracker
from llmkit.core.error_map import normalize_error
from llmkit.core.errors import LLMKitError
from llmkit.core.retry import RetryConfig, with_retry
from llmkit.core.types import Message, Response, StreamChunk, Tool


def _provider_name(adapter: ProviderAdapter) -> str:
    """Derive a provider name string from the adapter's class name.
    AnthropicAdapter -> 'anthropic', OpenAIAdapter -> 'openai', etc.
    Falls back to the full class name if the convention isn't followed.
    """
    name = type(adapter).__name__.lower()
    for suffix in ("adapter",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


class Client:
    def __init__(
        self,
        adapter: ProviderAdapter,
        *,
        retry_config: RetryConfig | None = None,
        timeout_seconds: float | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self._adapter = adapter
        self._provider = _provider_name(adapter)
        self._retry = retry_config or RetryConfig(max_attempts=1)  # default: no retries
        self._timeout = timeout_seconds
        self._cost_tracker = cost_tracker

    async def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> Response:
        def make_call():
            return self._adapter.generate(
                messages,
                model=model,
                max_tokens=max_tokens,
                system=system,
                temperature=temperature,
                tools=tools,
            )

        try:
            response: Response = await with_retry(
                make_call,
                retry_config=self._retry,
                timeout=self._timeout,
                provider=self._provider,
            )
        except LLMKitError:
            raise
        except Exception as exc:
            raise normalize_error(exc, self._provider) from exc

        if self._cost_tracker is not None:
            self._cost_tracker.record(
                model=response.model,
                provider=self._provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        return response

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks from the adapter, normalizing errors as they surface.

        Note: retry and timeout are NOT applied to streaming calls here.
        Streaming introduces state (partial chunks already yielded to the
        caller) that can't be safely replayed on retry — restarting a stream
        mid-flight would cause duplicate output. Apply retry logic at the
        level of your own application code if you want to retry a failed
        stream from scratch.

        Cost tracking for streams: use the `message_stop` chunk's `usage`
        field to record cost manually if needed, since the stream doesn't
        return a Response object.
        """
        try:
            async for chunk in self._adapter.stream(
                messages,
                model=model,
                max_tokens=max_tokens,
                system=system,
                temperature=temperature,
                tools=tools,
            ):
                yield chunk
        except LLMKitError:
            raise
        except Exception as exc:
            raise normalize_error(exc, self._provider) from exc

    @property
    def cost_tracker(self) -> CostTracker | None:
        """The CostTracker instance attached to this client, if any."""
        return self._cost_tracker
