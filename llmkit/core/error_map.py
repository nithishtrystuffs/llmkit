"""
Per-provider exception mapping.

Each function takes a raw SDK exception and returns a normalized LLMKitError.
These are called by the Client's retry/error-handling layer — nothing else
should be catching raw SDK exceptions. The goal is that application code
only ever sees llmkit exceptions, never needs to import from anthropic/openai/
google/ollama directly to write a try/except.

Adding a new provider: add a new `normalize_<provider>_error` function here
and register it in `_NORMALIZERS` at the bottom.
"""

from __future__ import annotations

from llmkit.core.errors import (
    APIError,
    AuthenticationError,
    ConnectionError,
    InvalidRequestError,
    LLMKitError,
    RateLimitError,
    UnknownError,
)


def normalize_anthropic_error(exc: BaseException) -> LLMKitError:
    try:
        import anthropic
    except ImportError:
        return UnknownError(str(exc), provider="anthropic", cause=exc)

    provider = "anthropic"
    status = getattr(exc, "status_code", None)

    if isinstance(exc, anthropic.RateLimitError):
        return RateLimitError(str(exc), provider=provider, cause=exc, status_code=429)
    if isinstance(exc, anthropic.AuthenticationError):
        return AuthenticationError(str(exc), provider=provider, cause=exc, status_code=401)
    if isinstance(exc, anthropic.BadRequestError):
        return InvalidRequestError(str(exc), provider=provider, cause=exc, status_code=400)
    if isinstance(exc, anthropic.APIStatusError):
        # Catches 5xx and any other HTTP status errors not handled above.
        return APIError(str(exc), provider=provider, cause=exc, status_code=status)
    if isinstance(exc, anthropic.APIConnectionError):
        return ConnectionError(str(exc), provider=provider, cause=exc)
    if isinstance(exc, anthropic.APIError):
        # Broad base catch for anything else in the anthropic SDK.
        return APIError(str(exc), provider=provider, cause=exc, status_code=status)

    return UnknownError(str(exc), provider=provider, cause=exc)


def normalize_openai_error(exc: BaseException) -> LLMKitError:
    try:
        import openai
    except ImportError:
        return UnknownError(str(exc), provider="openai", cause=exc)

    provider = "openai"
    status = getattr(exc, "status_code", None)

    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(str(exc), provider=provider, cause=exc, status_code=429)
    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(str(exc), provider=provider, cause=exc, status_code=401)
    if isinstance(exc, openai.BadRequestError):
        return InvalidRequestError(str(exc), provider=provider, cause=exc, status_code=400)
    if isinstance(exc, openai.APIStatusError):
        return APIError(str(exc), provider=provider, cause=exc, status_code=status)
    if isinstance(exc, openai.APIConnectionError):
        return ConnectionError(str(exc), provider=provider, cause=exc)
    if isinstance(exc, openai.APIError):
        return APIError(str(exc), provider=provider, cause=exc, status_code=status)

    return UnknownError(str(exc), provider=provider, cause=exc)


def normalize_gemini_error(exc: BaseException) -> LLMKitError:
    provider = "gemini"

    # google-api-core exceptions are used by the google-genai SDK.
    try:
        from google.api_core import exceptions as google_exc

        if isinstance(exc, google_exc.ResourceExhausted):
            return RateLimitError(str(exc), provider=provider, cause=exc, status_code=429)
        if isinstance(exc, (google_exc.Unauthenticated, google_exc.PermissionDenied)):
            return AuthenticationError(str(exc), provider=provider, cause=exc, status_code=401)
        if isinstance(exc, google_exc.InvalidArgument):
            return InvalidRequestError(str(exc), provider=provider, cause=exc, status_code=400)
        if isinstance(exc, google_exc.ServiceUnavailable):
            return APIError(str(exc), provider=provider, cause=exc, status_code=503)
        if isinstance(exc, google_exc.InternalServerError):
            return APIError(str(exc), provider=provider, cause=exc, status_code=500)
        if isinstance(exc, google_exc.GoogleAPICallError):
            return APIError(str(exc), provider=provider, cause=exc)
    except ImportError:
        pass

    # Fallback for connectivity issues not wrapped by google-api-core.
    if isinstance(exc, (OSError, ConnectionRefusedError)):
        return ConnectionError(str(exc), provider=provider, cause=exc)

    return UnknownError(str(exc), provider=provider, cause=exc)


def normalize_ollama_error(exc: BaseException) -> LLMKitError:
    provider = "ollama"

    # Ollama's Python library wraps most errors in ResponseError.
    try:
        from ollama import ResponseError

        if isinstance(exc, ResponseError):
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg:
                return RateLimitError(str(exc), provider=provider, cause=exc, status_code=429)
            if "unauthorized" in msg or "401" in msg or "forbidden" in msg:
                return AuthenticationError(str(exc), provider=provider, cause=exc, status_code=401)
            return APIError(str(exc), provider=provider, cause=exc)
    except ImportError:
        pass

    # The most common Ollama error: `ollama serve` isn't running.
    if isinstance(exc, (ConnectionRefusedError, OSError)):
        return ConnectionError(
            "Could not connect to Ollama. Is `ollama serve` running?",
            provider=provider,
            cause=exc,
        )

    return UnknownError(str(exc), provider=provider, cause=exc)


# Registry: provider name -> normalizer function.
# Used by the Client to pick the right mapper based on adapter type.
_NORMALIZERS = {
    "anthropic": normalize_anthropic_error,
    "openai": normalize_openai_error,
    "gemini": normalize_gemini_error,
    "ollama": normalize_ollama_error,
}


def normalize_error(exc: BaseException, provider: str) -> LLMKitError:
    """Normalize a raw SDK exception for the given provider name.
    If the provider has no registered normalizer, wraps in UnknownError."""
    if isinstance(exc, LLMKitError):
        return exc  # already normalized — don't double-wrap
    normalizer = _NORMALIZERS.get(provider)
    if normalizer is None:
        return UnknownError(str(exc), provider=provider, cause=exc)
    return normalizer(exc)
