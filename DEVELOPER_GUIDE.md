# llmkit — Developer Guide

This document is for people **modifying or extending** llmkit itself — adding a new
provider, fixing an adapter, or understanding why something is built the way it is. If
you just want to *use* the library in an application, see `USER_GUIDE.md` instead.

---

## Table of contents

1. [Architecture overview](#1-architecture-overview)
2. [Project layout](#2-project-layout)
3. [The core types](#3-the-core-types)
4. [The ProviderAdapter contract](#4-the-provideradapter-contract)
5. [Adding a new provider](#5-adding-a-new-provider)
6. [Provider-specific gotchas (read before touching an adapter)](#6-provider-specific-gotchas-read-before-touching-an-adapter)
7. [Testing](#7-testing)
8. [Local development setup](#8-local-development-setup)
9. [Release / tagging process](#9-release--tagging-process)
10. [Design principles to preserve](#10-design-principles-to-preserve)

---

## 1. Architecture overview

llmkit solves vendor lock-in with the **adapter pattern**: one neutral core schema, and
one independent adapter per provider that translates to/from it. This is a deliberate
choice over the alternative — a single central translator that secretly favors one
provider's wire format (the approach taken by, e.g., LiteLLM's SDK mode, which models
everything as OpenAI-shaped).

```
Your application code
        |
        v
   llmkit.Client  ------  holds adapter + optional retry/timeout/cost config
        |
        v
  Resilience layer        RetryConfig  ->  with_retry() + exponential backoff
  (inside Client)         timeout      ->  asyncio.wait_for per attempt
                          error_map    ->  raw SDK exc -> normalized LLMKitError
                          CostTracker  ->  records Usage per call
        |
        v
ProviderAdapter (abstract interface)
        |
   +----+----+---------+---------+
   v    v    v         v         v
Anthropic OpenAI    Gemini    Ollama   <- each owns 100% of its own translation
```

**Why this matters for you as a maintainer:** adding a 5th provider should require
writing exactly one new file and touching zero existing files. If you ever find yourself
needing to modify `core/types.py` or another adapter just to add a provider, something
about the new provider doesn't fit the abstraction cleanly — that's worth a design
conversation, not a quiet workaround.

The interface has been validated against real, meaningfully different providers:
Anthropic (separate `system` param, typed streaming events), OpenAI (system folded into
messages, two-chunk streaming usage), Gemini (no system-role messages, full-response-per-
streaming-chunk), and Ollama (no API key concept, no tool-call IDs at all). Adding tool
calling required real structural work in every adapter but **zero changes** to the
`ProviderAdapter` interface itself beyond one new `tools` parameter — that's the signal
the abstraction is sound.

---

## 2. Project layout

```
llmkit/
├── pyproject.toml              # package metadata, optional extras per provider
├── llmkit/
│   ├── __init__.py             # public API surface — what `from llmkit import X` exposes
│   ├── core/
│   │   ├── __init__.py         # empty, just marks the package
│   │   ├── types.py            # the neutral schema: Message, Tool, Response, StreamChunk...
│   │   ├── client.py           # Client — wires adapter + resilience layer together
│   │   ├── errors.py           # normalized exception hierarchy (LLMKitError and subtypes)
│   │   ├── error_map.py        # per-provider raw SDK exc -> LLMKitError translation
│   │   ├── retry.py            # RetryConfig + with_retry() async helper
│   │   └── cost.py             # CostTracker, CallRecord, DEFAULT_PRICE_TABLE
│   └── adapters/
│       ├── __init__.py         # empty
│       ├── base.py             # ProviderAdapter — the abstract contract
│       ├── anthropic/
│       │   ├── __init__.py     # exposes AnthropicAdapter
│       │   └── adapter.py
│       ├── openai/
│       ├── gemini/
│       └── ollama/
├── tests/
│   ├── test_anthropic_adapter.py
│   ├── test_openai_adapter.py
│   ├── test_gemini_adapter.py
│   ├── test_ollama_adapter.py
│   └── test_resilience.py      # retry, timeout, error normalization, cost tracking
├── run_compare.py              # live script: same Client/Message, 4 providers
└── run_tool_calling.py         # live script: full tool-calling loop, 4 providers
```

Each adapter folder is self-contained. Nothing outside `adapters/<provider>/adapter.py`
should ever need to know what that provider's wire format looks like.

---

## 3. The core types

All defined in `core/types.py`. This file is the contract — every adapter translates
to/from it, and it should remain provider-neutral (no field names or shapes borrowed
disproportionately from any one provider's API).

| Type | Purpose |
|---|---|
| `Role` | `SYSTEM`, `USER`, `ASSISTANT` |
| `TextBlock` | Plain text content |
| `ToolUseBlock` | Model is requesting a tool call: `id`, `name`, `input: dict` |
| `ToolResultBlock` | Result sent back to the model: `tool_use_id`, `content`, `is_error` |
| `ContentBlock` | Discriminated union of the three blocks above, keyed on `type` |
| `Tool` | A tool definition: `name`, `description`, `input_schema: dict` (plain JSON Schema) |
| `Message` | `role` + `content: list[ContentBlock]` |
| `Usage` | `input_tokens`, `output_tokens`, `total_tokens` (property) |
| `StopReason` | `END_TURN`, `MAX_TOKENS`, `STOP_SEQUENCE`, `TOOL_USE`, `OTHER` |
| `Response` | Non-streaming result: `content`, `stop_reason`, `usage`, `model`, `raw` (escape hatch) |
| `StreamChunk` | Discriminated union: `TextDeltaChunk`, `MessageStartChunk`, `MessageStopChunk`, `ToolCallStartChunk`, `ToolCallDeltaChunk` |

### Why `ContentBlock` and `StreamChunk` are `Annotated` discriminated unions

```python
ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")
]
```

A plain `|` union isn't enough — Pydantic needs the `discriminator="type"` hint to know
which concrete class to deserialize a raw dict into. Every block/chunk class carries a
`Literal[...]` `type` field for exactly this reason. **When adding a new block or chunk
type, give it a unique `type` literal and add it to the relevant union** — both
`ContentBlock` and `StreamChunk` are designed to grow this way without breaking existing
code that pattern-matches on `.type`.

### Why `Response.raw` exists

`raw: dict = Field(default_factory=dict, exclude=True)` is the deliberate escape hatch.
It holds the original, untranslated provider response (via `.model_dump()` on the
provider SDK's response object) so that callers who need a provider-specific field
llmkit hasn't normalized yet can still get at it, without that field polluting the
neutral schema. It's excluded from serialization on purpose.

---

## 3b. The resilience modules

Four files in `core/` handle everything between the public `Client` and the raw adapter
call. None of these touch adapters — they only operate on the normalized types.

### `core/errors.py` — the exception hierarchy

```
LLMKitError                   <- base; always carry .provider, .status_code, .cause
├── RateLimitError            <- 429 / quota exceeded; retryable
├── AuthenticationError       <- 401 / bad key; not retryable
├── TimeoutError              <- asyncio timeout; retryable
├── ConnectionError           <- server unreachable; retryable
├── InvalidRequestError       <- 400 / bad params; not retryable
├── APIError                  <- 5xx server error; retryable
└── UnknownError              <- unrecognised SDK exception; not retried
```

`RETRYABLE_ERRORS` is a tuple of the retryable subtypes, used by `retry.py` to decide
whether to retry or raise immediately. When adding a new error subtype, decide
whether it belongs in `RETRYABLE_ERRORS` at definition time — the decision should be
encoded in the type, not scattered in calling code.

### `core/error_map.py` — per-provider normalization

One function per provider (`normalize_anthropic_error`, `normalize_openai_error`, etc.)
plus a registry dict and a single public entry point:

```python
from llmkit.core.error_map import normalize_error
normalized = normalize_error(raw_sdk_exc, "anthropic")
```

**When adding a new provider:** add a `normalize_<provider>_error(exc)` function and
register it in `_NORMALIZERS`. The function should handle the provider SDK's specific
exception hierarchy and fall through to `UnknownError` for anything it doesn't recognise.
Never raise from inside a normalizer — always return a `LLMKitError`.

### `core/retry.py` — RetryConfig and with_retry()

`with_retry(coro_factory, *, retry_config, timeout, provider)` is the core helper. It
takes a **factory** (a zero-argument callable returning a fresh coroutine each time),
not a coroutine directly — coroutines can only be awaited once, so retrying requires
creating a fresh one per attempt.

Backoff formula: `min(base_delay * 2^attempt + jitter, max_delay)` where jitter is a
random value in `[0, base_delay]`, preventing thundering-herd when many clients hit a
rate limit simultaneously.

`Client` calls `with_retry` for `generate()` only — not for `stream()`. See the
"Streaming" design note in section 10 for why.

### `core/cost.py` — CostTracker and DEFAULT_PRICE_TABLE

`DEFAULT_PRICE_TABLE` is a plain dict of `model_name -> {input: float, output: float}`
in USD per 1M tokens. It's a reasonable starting point as of mid-2026 but will go stale
— treat it as a reference, not a contract.

`CostTracker.record()` is called automatically by `Client` after every successful
`generate()` call. It reads `response.usage` (which is already normalized across all
providers) and looks up the model price. Unknown models return `$0.00` silently — a
missing price never crashes an application.

When token prices change, update `DEFAULT_PRICE_TABLE` in `cost.py` and bump the
library version — callers who need accurate prices can override specific entries via
`CostTracker(price_table={...})` without waiting for a library update.

---

## 4. The ProviderAdapter contract

Defined in `adapters/base.py`:

```python
class ProviderAdapter(ABC):
    @abstractmethod
    async def generate(self, messages, *, model, max_tokens,
                        system=None, temperature=None, tools=None) -> Response: ...

    @abstractmethod
    def stream(self, messages, *, model, max_tokens,
               system=None, temperature=None, tools=None) -> AsyncIterator[StreamChunk]: ...
```

This is the **entire** interface. `Client` only ever calls these two methods — it has no
knowledge of any concrete adapter. Every adapter implementation is responsible for:

1. Translating `list[Message]` (+ `system`, `tools`) into that provider's wire format.
2. Calling the provider's SDK.
3. Translating the provider's response back into `Response` / a stream of `StreamChunk`.

Adapter constructors are **not** part of the abstract contract — `__init__` is free to
take whatever each provider actually needs (Anthropic/OpenAI/Gemini take `api_key`;
Ollama takes optional `api_key` *and* `host`, since "API key" doesn't really apply to a
local install). Don't force every adapter into the same constructor shape; let the
contract live entirely in `generate()`/`stream()`.

---

## 5. Adding a new provider

This is the core workflow. Using the real steps taken for Gemini and Ollama as the
template:

### Step 0 — Inspect the real SDK before writing any translation code

Don't translate from memory or assumption. Install the provider's SDK and inspect the
actual method signatures and response field names:

```python
import inspect
sig = inspect.signature(some_sdk.SomeClient.some_method)
print(sig)
print(list(SomeResponseType.model_fields.keys()))  # if it's a Pydantic model
```

This caught real discrepancies during development — e.g. confirming OpenAI's
`max_completion_tokens` vs the deprecated `max_tokens`, and Gemini's `parameters_json_schema`
field existing specifically for plain JSON Schema dicts (as opposed to `parameters`,
which wants a structured `Schema` object).

### Step 1 — Create the folder

```
llmkit/adapters/myprovider/
    __init__.py
    adapter.py
```

```python
# __init__.py
from llmkit.adapters.myprovider.adapter import MyProviderAdapter

__all__ = ["MyProviderAdapter"]
```

### Step 2 — Implement the four translation directions

Every adapter needs:

| Direction | Method |
|---|---|
| Core `Message`s -> provider request | a `_to_<provider>_messages` static helper |
| Core `Tool`s -> provider tool definitions | a `_to_<provider>_tools` static helper |
| Provider response -> core `Response` | a `_from_<provider>_response` static helper |
| Provider stream event(s) -> core `StreamChunk`(s) | a `_translate_stream_event` / `_translate_stream_chunk` static helper |

```python
from llmkit.adapters.base import ProviderAdapter
from llmkit.core.types import Message, Response, StreamChunk, Tool

class MyProviderAdapter(ProviderAdapter):
    def __init__(self, api_key: str | None = None) -> None:
        self._client = myprovider.Client(api_key=api_key)  # api_key=None should fall
                                                              # back to the provider's own
                                                              # standard env var

    async def generate(self, messages, *, model, max_tokens,
                        system=None, temperature=None, tools=None) -> Response:
        # 1. translate messages/system/tools -> provider's request shape
        # 2. call the provider SDK
        # 3. translate the response -> Response
        ...

    async def stream(self, messages, *, model, max_tokens,
                      system=None, temperature=None, tools=None):
        # same, but yield StreamChunk objects as they arrive
        ...
```

### Step 3 — Map the provider's stop/finish-reason vocabulary onto `StopReason`

Build a small dict like every existing adapter does:

```python
_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    # anything unmapped falls through to StopReason.OTHER -- never crash on
    # an unrecognized value, and never guess
}
```

If the provider has **no dedicated value** for tool-use (Gemini's case), don't force a
mapping — instead detect tool use from the response content itself and override the
reported value. See `GeminiAdapter._normalize_stop_reason`'s `has_tool_calls` parameter
for the pattern.

### Step 4 — Write mocked contract tests

Every adapter has a test file using `unittest.mock` to fake the SDK's response objects
(`SimpleNamespace` is used throughout — see existing tests for the pattern), so the suite
never needs network access or real API keys. At minimum, cover:

- Basic text generation, request translation correctness
- Stop reason mapping for at least two distinct reasons
- Streaming text deltas
- Tool definition translation
- Tool-call response parsing
- Tool-result message translation
- Streaming tool-call chunks

### Step 5 — Add the optional dependency

In `pyproject.toml`:

```toml
[project.optional-dependencies]
myprovider = ["myprovider-sdk>=1.0"]
all = ["llmkit[anthropic,openai,gemini,ollama,myprovider]"]
```

### Step 6 — Export it from the public API if appropriate

Most providers are imported directly (`from llmkit.adapters.myprovider import MyProviderAdapter`)
rather than from the top-level `llmkit` package, to avoid forcing every optional SDK to be
importable just to `import llmkit`. Don't add provider adapters to `llmkit/__init__.py`.

---

## 6. Provider-specific gotchas (read before touching an adapter)

These are non-obvious workarounds. If you're modifying one of these adapters, understand
*why* the workaround exists before changing it — it's very easy to "simplify" these into
something that looks cleaner but silently reintroduces a bug.

### OpenAI: JSON string vs. dict

OpenAI sends tool-call arguments as a **JSON string** (`function.arguments: str`), unlike
Anthropic and Gemini, which give you a dict directly. `OpenAIAdapter` does `json.loads()`
on the way in and `json.dumps()` on the way out — this conversion must stay fully
contained inside `openai/adapter.py`. `ToolUseBlock.input` is documented as always being
a `dict` regardless of provider; if you see a JSON string anywhere outside this file,
something has gone wrong.

### OpenAI: one core Message can expand into multiple OpenAI messages

OpenAI requires tool results to be their own `role="tool"` message with a `tool_call_id`
— it has no concept of a tool result nested inside another message's content, unlike
Anthropic/Gemini. A core `Message` containing one or more `ToolResultBlock`s therefore
expands into **N separate OpenAI messages**, and may expand into **zero** "primary"
messages if it contains *only* `ToolResultBlock`s. See `OpenAIAdapter._to_openai_messages`
— this asymmetry (1 core message -> 0..N wire messages) is the most structurally
significant translation in the whole codebase. Don't assume a 1:1 message count when
modifying this method.

### Gemini: FunctionResponse needs a name, not just an id

Gemini's `FunctionResponse` is documented as requiring `name` (the function's name),
while `id` is merely optional metadata. Our `ToolResultBlock` only carries `tool_use_id`.
`GeminiAdapter._to_gemini_contents` solves this by building an `id -> name` map from any
`ToolUseBlock`s it encounters earlier in the same `messages` list, and raises a clear
`ValueError` if a `ToolResultBlock` shows up with no matching prior `ToolUseBlock`. **This
means the full conversation history (including the assistant's tool-call turn) must
always be passed to Gemini calls that include tool results** — don't truncate history in
a way that could drop the original tool-call message.

### Gemini & Ollama: no incremental tool-call argument streaming

Unlike Anthropic and OpenAI, neither SDK streams a tool call's arguments fragment-by-
fragment. Both adapters emit exactly one `ToolCallStartChunk` immediately followed by
exactly one `ToolCallDeltaChunk` containing the complete arguments JSON. This is
documented in `ToolCallDeltaChunk`'s docstring and should **not** be "fixed" by trying to
fake incremental streaming — callers are expected to accumulate fragments per `index`
regardless of how many there turn out to be, so genuinely emitting one fragment is
correct and forward-compatible if the underlying SDKs ever add real incremental support.

### Ollama: no tool-call IDs at all

Ollama's tool calls have no `id` field whatsoever — just `{function: {name, arguments}}`.
`OllamaAdapter` synthesizes a deterministic id (`f"{name}_{position}"`) so
`ToolUseBlock.id` is never empty, and reverses it (`_name_from_synthesized_id`) when
translating a `ToolResultBlock` back. **This synthesized id has no meaning to the Ollama
API itself** — it exists purely so llmkit's own `id`-based round-trip (assistant calls ->
your code executes -> you send a result keyed by `id`) works consistently across all four
providers. If you change the synthesis scheme, both directions (`_synthesize_tool_call_id`
and `_name_from_synthesized_id`) must change together, in the same file.

### Ollama: no API key concept

`OllamaAdapter.__init__` takes an optional `api_key` *and* `host`, unlike the three hosted
adapters which only take `api_key`. `api_key` here is repurposed to mean a bearer token
for Ollama Cloud, not a local-install credential. This is intentional, not an oversight —
`ProviderAdapter`'s contract never required every adapter to need an API key (only
`generate()`/`stream()` are abstract), so don't "fix" this by forcing a uniform
constructor signature across adapters.

### Anthropic & Gemini: system messages in `messages` raise, not silently drop

Both adapters raise `ValueError` if a `Role.SYSTEM` message appears in `messages` instead
of being passed via the `system=` parameter. This is intentional — silently dropping a
system message would be a much worse failure mode (the model loses instructions and
nobody notices) than a loud, immediate error. Don't change this to a silent skip.

---

## 7. Testing

```bash
pip install -e ".[anthropic,openai,gemini,ollama,dev]"
python -m pytest tests/ -v
```

All adapter tests use mocked SDK objects (`unittest.mock.patch.object` + `SimpleNamespace`
fakes) — no network access or real API keys required to run the suite. This is
deliberate: it lets CI run on every PR without secrets, and lets contributors validate an
adapter change without needing accounts on all four providers.

`test_resilience.py` covers the retry, timeout, error normalization, and cost tracking
layers independently — it uses a generic `MagicMock` adapter rather than a real provider
adapter, so it stays valid even if adapter internals change. When modifying `core/retry.py`,
`core/errors.py`, `core/error_map.py`, or `core/cost.py`, this is the test file to run
first.

**When you change a core type** (`core/types.py`), expect every adapter's tests to
potentially break if their mocked fakes don't have the new fields your code now reads —
this is the test suite doing its job, not a sign something is wrong. Update the fakes,
don't work around the tests.

**Live, end-to-end verification** (requires real API keys / a running Ollama instance):

```bash
python run_compare.py        # basic generate + stream, all 4 providers
python run_tool_calling.py   # full tool-calling loop, all 4 providers
```

These aren't part of the automated test suite (they need live credentials and a network),
but should be run manually after any adapter change before tagging a release.

---

## 8. Local development setup

```bash
git clone <repo-url>
cd llmkit
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[all,dev]"
python -m pytest tests/ -v
```

A common setup mistake: running scripts or pip commands from *inside* the inner
`llmkit/` package folder instead of the project root (the folder containing
`pyproject.toml`). If `from llmkit import Client` fails with an import error after a
fresh clone, check `pwd` and confirm `pyproject.toml` is in the current directory before
debugging anything else.

---

## 9. Release / tagging process

Since this is distributed via a private GitHub repo rather than PyPI, releases are git
tags that `pip install package[extra] @ git+https://...@TAG` resolves against.

1. Bump `version` in `pyproject.toml` to match the tag you're about to create — keep
   these in sync. Mismatches don't break the install, but confuse anyone who runs
   `pip show llmkit` after installing a specific tag and sees a different version number.
2. Run the full test suite and at least one live script (`run_tool_calling.py` is the
   most comprehensive) against all four providers.
3. Commit, then tag and push:
   ```bash
   git tag v0.1.4
   git push origin v0.1.4
   ```
4. Confirm the tag resolves correctly before announcing it to the team:
   ```bash
   git ls-remote --tags origin   # confirms it's actually on GitHub, not just local
   pip install "llmkit[all] @ git+https://github.com/YOUR_ORG/llmkit.git@v0.1.4" --dry-run
   ```

---

## 10. Design principles to preserve

These were deliberate decisions made over the course of building this library. Preserve
them when extending it — each one exists because of a specific failure mode it avoids.

- **Adapter pattern over central translator.** Each provider's translation logic lives
  entirely in its own file. Resist the temptation to extract "shared" translation helpers
  across adapters unless two providers' wire formats are *genuinely* identical — most
  apparent similarities (e.g. OpenAI's and Ollama's inline system messages) still diverge
  in some other dimension (max_tokens naming, tool-call id presence) and sharing code
  across them tends to produce awkward special-casing rather than real reuse.
- **Neutral core schema, not OpenAI-shaped.** `core/types.py` should never silently drift
  toward modeling one provider's API more closely than the others.
- **Capability gaps are surfaced, never silently faked.** Gemini/Ollama's lack of
  incremental tool-call streaming, Gemini's lack of a TOOL_USE finish reason — all
  documented and handled explicitly, never papered over with something that looks
  consistent but loses information.
- **Fail loudly on ambiguous input.** System messages in the wrong place, missing
  tool-name lookups — these raise clear errors rather than guessing or silently dropping
  data.
- **`raw` is always available, never required.** Every `Response` carries the
  untranslated provider response as an escape hatch, excluded from normal serialization.
- **Error normalization is unconditional, not opt-in.** Raw SDK exceptions are always
  mapped to `LLMKitError` subtypes before surfacing to callers, regardless of whether
  retry is configured. Application code should never need to import from provider SDKs
  just to write a `try/except`.
- **Retry is never applied to streaming.** `with_retry` wraps `generate()` only.
  Streaming introduces stateful output (chunks already yielded) that can't be safely
  replayed on retry — restarting mid-stream would produce duplicate output. This is a
  deliberate design boundary, not an oversight to fill in later.