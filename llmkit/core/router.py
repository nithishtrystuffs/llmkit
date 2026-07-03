"""
RouterClient — automatic provider fallback for llmkit.

Wraps multiple (adapter, model) pairs behind the same generate()/stream()
interface as the regular Client. If the primary provider fails, the router
tries the next one automatically, with configurable per-provider retry before
deciding to fall back.

Usage:

    from llmkit import RouterClient, ProviderConfig, RetryConfig, CostTracker
    from llmkit.adapters.anthropic import AnthropicAdapter
    from llmkit.adapters.openai import OpenAIAdapter
    from llmkit.adapters.gemini import GeminiAdapter

    tracker = CostTracker()

    router = RouterClient(
        providers=[
            ProviderConfig(AnthropicAdapter(), model="claude-sonnet-4-6"),
            ProviderConfig(OpenAIAdapter(),    model="gpt-4o"),
            ProviderConfig(GeminiAdapter(),    model="gemini-2.5-flash"),
        ],
        retry_config=RetryConfig(max_attempts=2, base_delay=1.0),
        timeout_seconds=30.0,
        cost_tracker=tracker,
        on_fallback=lambda from_p, to_p, err:
            print(f"Falling back from {from_p} to {to_p}: {err}"),
    )

    response = await router.generate(
        [Message.text(Role.USER, "hi")],
        max_tokens=200,
    )

Retry vs. fallback logic:
    RateLimitError / TimeoutError   -> retry same provider (up to max_attempts),
                                       then fall back to next if still failing
    APIError / ConnectionError /
    AuthenticationError / Unknown   -> fall back immediately, no retry on this provider
    InvalidRequestError             -> raise immediately, no fallback
                                       (request is broken; another provider won't help)

Note on streaming: the router applies fallback logic for the initial stream
setup, but cannot fall back mid-stream once chunks have started arriving.
If the stream breaks after the first chunk, the error surfaces to the caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Callable

from llmkit.adapters.base import ProviderAdapter
from llmkit.core.client import Client
from llmkit.core.cost import CostTracker
from llmkit.core.errors import (
    IMMEDIATE_FALLBACK_ERRORS,
    RETRY_BEFORE_FALLBACK_ERRORS,
    InvalidRequestError,
    LLMKitError,
)
from llmkit.core.retry import RetryConfig
from llmkit.core.types import Message, Response, StreamChunk, Tool


@dataclass
class ProviderConfig:
    """One provider slot in the router: an adapter + the model to use with it.

    `retry_config` and `timeout_seconds` on the ProviderConfig override the
    router-level defaults for this specific provider — useful when one provider
    needs a longer timeout or more retries than the others.
    """

    adapter: ProviderAdapter
    model: str
    retry_config: RetryConfig | None = None
    timeout_seconds: float | None = None

    @property
    def name(self) -> str:
        n = type(self.adapter).__name__.lower()
        return n[: -len("adapter")] if n.endswith("adapter") else n


FallbackCallback = Callable[[str, str, LLMKitError], None]


class RouterClient:
    """Wraps multiple provider Clients, falling back automatically on failure.

    Args:
        providers:        Ordered list of ProviderConfig. First entry is the
                          primary; the rest are fallbacks tried in order.
        retry_config:     Default retry config applied to each provider.
                          Override per-provider via ProviderConfig.retry_config.
        timeout_seconds:  Default per-attempt timeout. Override per-provider
                          via ProviderConfig.timeout_seconds.
        cost_tracker:     Optional CostTracker — shared across all providers,
                          so totals reflect the real provider that handled each
                          call, not just the primary.
        on_fallback:      Optional callback called whenever the router falls back
                          to a new provider: (from_provider, to_provider, error).
                          Useful for logging/alerting without coupling to a logger.
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        retry_config: RetryConfig | None = None,
        timeout_seconds: float | None = None,
        cost_tracker: CostTracker | None = None,
        on_fallback: FallbackCallback | None = None,
    ) -> None:
        if not providers:
            raise ValueError("RouterClient requires at least one ProviderConfig.")

        self._configs = providers
        self._default_retry = retry_config or RetryConfig(max_attempts=1)
        self._default_timeout = timeout_seconds
        self._cost_tracker = cost_tracker
        self._on_fallback = on_fallback

        # Build a Client for each provider, sharing the cost tracker so
        # costs are attributed to whichever provider actually handled the call.
        # Each per-provider Client only retries RETRY_BEFORE_FALLBACK_ERRORS
        # (rate limits and timeouts) — IMMEDIATE_FALLBACK_ERRORS are caught
        # by the router and fall through to the next provider without retry.
        self._clients: list[Client] = [
            Client(
                cfg.adapter,
                retry_config=self._router_retry_config(cfg.retry_config),
                timeout_seconds=cfg.timeout_seconds
                if cfg.timeout_seconds is not None
                else self._default_timeout,
                cost_tracker=cost_tracker,
            )
            for cfg in providers
        ]

    def _router_retry_config(self, override: RetryConfig | None) -> RetryConfig:
        """Build the per-provider RetryConfig for use inside the router.

        Uses the provider's own override if set, otherwise derives from the
        router's default — but always restricts retryable_errors to
        RETRY_BEFORE_FALLBACK_ERRORS so that APIError/ConnectionError/
        AuthenticationError fall through to the router's fallback logic
        rather than being retried per-provider.
        """
        from llmkit.core.errors import RETRY_BEFORE_FALLBACK_ERRORS

        base = override or self._default_retry
        return RetryConfig(
            max_attempts=base.max_attempts,
            base_delay=base.base_delay,
            max_delay=base.max_delay,
            retryable_errors=RETRY_BEFORE_FALLBACK_ERRORS,
            on_retry=base.on_retry,
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> Response:
        """Generate a response, falling back through providers on failure.

        Note: `model` is not a parameter here — it's set per-provider in
        ProviderConfig and picked up automatically.
        """
        last_error: LLMKitError | None = None

        for i, (client, config) in enumerate(zip(self._clients, self._configs)):
            try:
                return await client.generate(
                    messages,
                    model=config.model,
                    max_tokens=max_tokens,
                    system=system,
                    temperature=temperature,
                    tools=tools,
                )

            except InvalidRequestError:
                # Request is malformed — another provider won't help. Raise now.
                raise

            except LLMKitError as exc:
                last_error = exc
                is_last = i == len(self._clients) - 1

                if is_last:
                    break

                next_config = self._configs[i + 1]

                if isinstance(exc, RETRY_BEFORE_FALLBACK_ERRORS):
                    # Already retried by the Client (up to retry_config.max_attempts).
                    # The Client exhausted its retries and still failed — now fall back.
                    pass
                elif isinstance(exc, IMMEDIATE_FALLBACK_ERRORS):
                    # Don't retry — move to next provider immediately.
                    pass
                else:
                    # Unclassified error — fall back rather than guess.
                    pass

                if self._on_fallback is not None:
                    try:
                        self._on_fallback(config.name, next_config.name, exc)
                    except Exception:
                        pass  # never let a callback crash the fallback loop

        raise last_error  # type: ignore[misc]

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a response, falling back through providers on setup failure.

        Fallback only applies during stream initialization. Once the first
        chunk has been yielded, errors surface directly to the caller — the
        router cannot rewind and restart from a different provider mid-stream
        without producing duplicate output.
        """
        last_error: LLMKitError | None = None

        for i, (client, config) in enumerate(zip(self._clients, self._configs)):
            try:
                # Collect the first chunk to confirm the stream opened
                # successfully before yielding anything to the caller.
                # If setup fails (connection error, auth error, etc.) we catch
                # it here and fall back — once we yield the first chunk, we're
                # committed and fallback is no longer possible.
                stream = client.stream(
                    messages,
                    model=config.model,
                    max_tokens=max_tokens,
                    system=system,
                    temperature=temperature,
                    tools=tools,
                )

                first_chunk = None
                async for chunk in stream:
                    first_chunk = chunk
                    break

                if first_chunk is None:
                    # Empty stream — treat as an error and try next provider.
                    from llmkit.core.errors import APIError
                    raise APIError(
                        f"Provider {config.name} returned an empty stream.",
                        provider=config.name,
                    )

                # Stream opened successfully — yield first chunk and the rest.
                yield first_chunk
                async for chunk in stream:
                    yield chunk
                return

            except InvalidRequestError:
                raise

            except LLMKitError as exc:
                last_error = exc
                is_last = i == len(self._clients) - 1
                if is_last:
                    break

                next_config = self._configs[i + 1]
                if self._on_fallback is not None:
                    try:
                        self._on_fallback(config.name, next_config.name, exc)
                    except Exception:
                        pass

        raise last_error  # type: ignore[misc]

    @property
    def cost_tracker(self) -> CostTracker | None:
        return self._cost_tracker

    @property
    def active_providers(self) -> list[str]:
        """Names of all configured providers in priority order."""
        return [c.name for c in self._configs]