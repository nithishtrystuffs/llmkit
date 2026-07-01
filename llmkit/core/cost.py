"""
Cost tracking for llmkit.

Token prices change frequently — the default price table is a reasonable
starting point as of mid-2026, but treat it as a reference, not a guarantee.
Pass your own `price_table` to CostTracker to override specific models or
add models not listed here.

Price table format:
    {
        "model-name": {"input": <USD per 1M tokens>, "output": <USD per 1M tokens>},
    }

Usage:
    tracker = CostTracker()
    client = Client(AnthropicAdapter(), cost_tracker=tracker)

    await client.generate(...)          # cost recorded automatically

    print(tracker.total_cost_usd)       # accumulated total
    print(tracker.total_input_tokens)
    print(tracker.calls[-1].cost_usd)   # per-call cost
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# Default price table: USD per 1,000,000 tokens (input / output separately).
# Sources: provider pricing pages, mid-2026. Override via CostTracker(price_table={...}).
DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-5":       {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":     {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":      {"input":  0.80, "output":  4.00},
    # OpenAI
    "gpt-4o":                {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":           {"input":  0.15, "output":  0.60},
    "o3":                    {"input": 10.00, "output": 40.00},
    "o4-mini":               {"input":  1.10, "output":  4.40},
    # Gemini
    "gemini-2.5-pro":        {"input":  1.25, "output": 10.00},
    "gemini-2.5-flash":      {"input":  0.075,"output":  0.30},
    # Ollama — local, no token cost
    "llama3.2":              {"input":  0.00, "output":  0.00},
    "llama3.1":              {"input":  0.00, "output":  0.00},
    "mistral":               {"input":  0.00, "output":  0.00},
}


@dataclass
class CallRecord:
    """Record of a single generate() call's cost and usage."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostTracker:
    """Tracks per-call and accumulated token costs across all generate() calls.

    Thread-safety: not thread-safe for concurrent async tasks sharing one
    tracker — if you run multiple generate() calls concurrently, each should
    use its own tracker instance or you should protect access externally.

    Usage:
        tracker = CostTracker()
        client = Client(adapter, cost_tracker=tracker)
        await client.generate(...)
        print(tracker.summary())
    """

    def __init__(self, price_table: dict[str, dict[str, float]] | None = None) -> None:
        # Merge the provided price_table over the defaults so partial overrides
        # work (you don't have to re-specify every model to change one price).
        self._prices: dict[str, dict[str, float]] = {
            **DEFAULT_PRICE_TABLE,
            **(price_table or {}),
        }
        self.calls: list[CallRecord] = []

    def _cost_for(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for a given model and token counts.
        Returns 0.0 if the model isn't in the price table rather than raising,
        since unknown models are common (new models ship faster than price tables
        update) and a missing price should never crash the application.
        """
        prices = self._prices.get(model)
        if prices is None:
            return 0.0
        return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

    def record(self, model: str, provider: str, input_tokens: int, output_tokens: int) -> CallRecord:
        """Record a completed call. Called automatically by Client — you don't
        normally need to call this directly."""
        cost = self._cost_for(model, input_tokens, output_tokens)
        record = CallRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self.calls.append(record)
        return record

    # --- Accumulated totals ---

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        """Clear all recorded calls and reset totals."""
        self.calls.clear()

    def summary(self) -> str:
        """Human-readable summary of all calls so far."""
        if not self.calls:
            return "No calls recorded yet."
        lines = [
            f"Calls:          {self.call_count}",
            f"Total tokens:   {self.total_tokens:,} "
            f"({self.total_input_tokens:,} in / {self.total_output_tokens:,} out)",
            f"Total cost:     ${self.total_cost_usd:.6f} USD",
        ]
        if self.call_count > 1:
            lines.append(f"Avg cost/call:  ${self.total_cost_usd / self.call_count:.6f} USD")
        return "\n".join(lines)

    def by_model(self) -> dict[str, dict]:
        """Cost and token breakdown grouped by model name."""
        result: dict[str, dict] = {}
        for call in self.calls:
            if call.model not in result:
                result[call.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            r = result[call.model]
            r["calls"] += 1
            r["input_tokens"] += call.input_tokens
            r["output_tokens"] += call.output_tokens
            r["cost_usd"] += call.cost_usd
        return result
