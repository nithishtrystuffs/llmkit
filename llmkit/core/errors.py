"""
llmkit normalized exception hierarchy.

Every provider SDK raises its own exception types — anthropic.RateLimitError,
openai.AuthenticationError, google.api_core.exceptions.ResourceExhausted,
ConnectionRefusedError for Ollama, etc. Application code shouldn't need to
import from four different SDKs just to write a try/except.

This module defines a provider-neutral exception hierarchy that the Client
maps all raw SDK exceptions onto before they surface to callers. The mapping
lives in core/error_map.py per provider; this file is purely the types.

Hierarchy:

    LLMKitError                   ← base for all llmkit exceptions
    ├── RateLimitError            ← 429 / quota exceeded — retryable
    ├── AuthenticationError       ← 401 / bad key — not retryable
    ├── TimeoutError              ← asyncio timeout exceeded — retryable
    ├── ConnectionError           ← server unreachable (e.g. Ollama not running)
    ├── InvalidRequestError       ← 400 / bad prompt/params — not retryable
    ├── APIError                  ← 5xx / unexpected server error — retryable
    └── UnknownError              ← anything we didn't recognize — not retryable

All normalized exceptions carry:
    - `message`: human-readable description
    - `provider`: which provider raised it (e.g. "anthropic")
    - `cause`: the original raw SDK exception, always available for debugging
    - `status_code`: HTTP status code if known, else None
"""

from __future__ import annotations


class LLMKitError(Exception):
    """Base class for all llmkit exceptions."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        cause: BaseException | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.cause = cause
        self.status_code = status_code

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"provider={self.provider!r}, "
            f"status_code={self.status_code!r})"
        )


class RateLimitError(LLMKitError):
    """The provider returned a rate-limit or quota-exceeded response (HTTP 429).
    This error is retryable — back off and try again."""


class AuthenticationError(LLMKitError):
    """The API key is missing, invalid, or doesn't have permission for the
    requested resource (HTTP 401/403). Not retryable — fix the key first."""


class TimeoutError(LLMKitError):
    """The request exceeded the configured timeout. Retryable depending on
    context — may indicate a slow model or an overloaded provider."""


class ConnectionError(LLMKitError):
    """The provider's server was unreachable — DNS failure, refused connection,
    or (most commonly for Ollama) the local server isn't running. Retryable
    if the connectivity issue is transient."""


class InvalidRequestError(LLMKitError):
    """The request was malformed or contained invalid parameters (HTTP 400).
    Not retryable — fix the request before trying again."""


class APIError(LLMKitError):
    """The provider returned an unexpected server-side error (HTTP 5xx).
    Retryable — likely a transient infrastructure issue."""


class UnknownError(LLMKitError):
    """A raw SDK exception we didn't recognize. The original exception is
    always available in `cause`. Not retried by default."""


# Which error types are considered retryable by the retry handler.
RETRYABLE_ERRORS: tuple[type[LLMKitError], ...] = (
    RateLimitError,
    TimeoutError,
    ConnectionError,
    APIError,
)
