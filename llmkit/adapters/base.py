"""
The ProviderAdapter contract.

Every provider (Anthropic, OpenAI, Gemini, ...) implements this interface.
Core code (Client) only ever talks to this interface — it never knows or
cares which concrete provider it's calling. This is what makes the
adapter pattern work: adding a new provider means writing a new class
that satisfies this contract, with zero changes to existing adapters or
to core/client.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from llmkit.core.types import Message, Response, StreamChunk


class ProviderAdapter(ABC):
    """Abstract interface all provider adapters must implement."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Non-streaming chat completion. Must return a fully-normalized Response."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion. Must yield normalized StreamChunk objects."""
        ...
