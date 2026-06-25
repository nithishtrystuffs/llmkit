"""
Public-facing client. This is what user code imports and calls.

The Client is deliberately dumb — it holds an adapter and forwards calls
to it. All provider-specific logic lives in the adapter, not here. This
keeps the door open for adding a router/fallback layer on top of Client
later (e.g. "try Anthropic, fall back to OpenAI on error") without
touching adapters at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from llmkit.adapters.base import ProviderAdapter
from llmkit.core.types import Message, Response, StreamChunk, Tool


class Client:
    def __init__(self, adapter: ProviderAdapter) -> None:
        self._adapter = adapter

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
        return await self._adapter.generate(
            messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            tools=tools,
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        return self._adapter.stream(
            messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            tools=tools,
        )