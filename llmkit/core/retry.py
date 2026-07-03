"""
Retry logic with exponential backoff and jitter, plus async timeout handling.

Kept separate from Client so it can be tested independently and reused if
a router/fallback layer is added later.

Backoff formula:
    wait = min(base_delay * (2 ** attempt) + jitter, max_delay)
    jitter = random value in [0, base_delay] — prevents thundering herd
    when many clients hit a rate limit at the same time.

Only RETRYABLE_ERRORS are retried. Authentication errors, bad requests, and
unknown errors are re-raised immediately since retrying won't help.
"""

from __future__ import annotations

import asyncio
import random

from llmkit.core.errors import RETRYABLE_ERRORS, LLMKitError


class RetryConfig:
    """Configuration for retry behaviour. Pass an instance to Client().

    Args:
        max_attempts:    Total attempts (1 = no retries, 3 = initial + 2 retries).
        base_delay:      Base wait in seconds before the first retry.
        max_delay:       Cap on wait time between retries.
        retryable_errors: Tuple of LLMKitError subclasses that will be retried.
                         Defaults to RETRYABLE_ERRORS. Override to restrict which
                         errors trigger retries — e.g. RouterClient uses this to
                         ensure per-provider retry only fires for rate limits and
                         timeouts, while API errors fall back immediately.
        on_retry:        Optional callback called before each retry with
                         (attempt: int, error: LLMKitError, wait: float).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        retryable_errors: "tuple[type[LLMKitError], ...] | None" = None,
        on_retry: "RetryCallback | None" = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._retryable_errors = retryable_errors  # None means use default
        self.on_retry = on_retry

    @property
    def retryable_errors(self) -> "tuple[type[LLMKitError], ...]":
        from llmkit.core.errors import RETRYABLE_ERRORS
        return self._retryable_errors if self._retryable_errors is not None else RETRYABLE_ERRORS

    def wait_for_attempt(self, attempt: int) -> float:
        """Seconds to wait before `attempt` (0-indexed). Attempt 0 = first retry."""
        jitter = random.uniform(0, self.base_delay)
        return min(self.base_delay * (2**attempt) + jitter, self.max_delay)


RetryCallback = "callable[[int, LLMKitError, float], None]"


async def with_retry(
    coro_factory,
    *,
    retry_config: RetryConfig,
    timeout: float | None = None,
    provider: str = "unknown",
) -> object:
    """Execute `coro_factory()` with retry logic and optional timeout.

    Args:
        coro_factory:   A zero-argument callable that returns a new coroutine
                        each time it's called. Must be a factory (not the
                        coroutine itself) because coroutines can only be
                        awaited once — retrying requires creating a fresh one.
        retry_config:   RetryConfig instance controlling backoff behaviour.
        timeout:        Per-attempt timeout in seconds. None = no timeout.
        provider:       Provider name, used only for error messages.

    Returns:
        The result of the first successful attempt.

    Raises:
        The last LLMKitError raised if all attempts are exhausted.
    """
    last_error: LLMKitError | None = None

    for attempt in range(retry_config.max_attempts):
        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro_factory(), timeout=timeout)
            else:
                result = await coro_factory()
            return result

        except asyncio.TimeoutError as exc:
            from llmkit.core.errors import TimeoutError as LLMTimeoutError

            last_error = LLMTimeoutError(
                f"Request to {provider} timed out after {timeout}s",
                provider=provider,
                cause=exc,
            )

        except LLMKitError as exc:
            last_error = exc

        except Exception as exc:
            # Raw SDK exception — normalize it before deciding retryability.
            from llmkit.core.error_map import normalize_error
            last_error = normalize_error(exc, provider)

        # Only retry if this error class is in the retryable set.
        if not isinstance(last_error, retry_config.retryable_errors):
            raise last_error

        # Don't sleep after the last attempt — just raise.
        if attempt + 1 >= retry_config.max_attempts:
            break

        wait = retry_config.wait_for_attempt(attempt)
        if retry_config.on_retry is not None:
            try:
                retry_config.on_retry(attempt, last_error, wait)
            except Exception:
                pass  # never let the callback crash the retry loop

        await asyncio.sleep(wait)

    raise last_error  # type: ignore[misc]