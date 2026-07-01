# llmkit — User Guide

`llmkit` is a provider-agnostic LLM SDK for Python. Write your code once against one
interface, and switch between Anthropic, OpenAI, Gemini, and Ollama by changing a single
line — no rewrites, no provider-specific branching in your application code.

This guide covers everything you need to *use* llmkit effectively. If you're looking to
modify or extend the library itself (add a provider, fix an adapter), see the developer
documentation instead.

---

## Table of contents

1. [Installation](#1-installation)
2. [Quickstart](#2-quickstart)
3. [Setting API keys](#3-setting-api-keys)
4. [System prompts](#4-system-prompts)
5. [Multi-turn conversations](#5-multi-turn-conversations)
6. [Streaming](#6-streaming)
7. [Tool calling](#7-tool-calling)
8. [The Response object](#8-the-response-object)
9. [Resilience — retry, timeout, errors, and cost tracking](#9-resilience--retry-timeout-errors-and-cost-tracking)
10. [All parameters reference](#10-all-parameters-reference)
11. [Provider-specific notes](#11-provider-specific-notes)
12. [Capabilities & limitations](#12-capabilities--limitations)
13. [Quick reference](#13-quick-reference)

---

## 1. Installation

Install only the provider(s) you actually need — each one is an optional extra, so you
never pull in SDKs you won't use.

```bash
pip install "llmkit[anthropic]"   # just Anthropic
pip install "llmkit[openai]"      # just OpenAI
pip install "llmkit[gemini]"      # just Gemini
pip install "llmkit[ollama]"      # just Ollama (local, no API key)
pip install "llmkit[all]"         # everything
```

If your team distributes this via a private GitHub repo rather than PyPI, install a
specific tagged version like this instead:

```bash
pip install "llmkit[all] @ git+https://github.com/YOUR_ORG/llmkit.git@v0.1.3"
```

> **Note on private repos:** the command above only works if your machine already has
> git authenticated against GitHub (the same way `git clone` would work without a
> password prompt). If it doesn't, ask whoever manages the repo for either an SSH-based
> install URL or a personal access token to use instead.

---

## 2. Quickstart

```python
import asyncio
from llmkit import Client, Message, Role
from llmkit.adapters.anthropic import AnthropicAdapter

async def main():
    client = Client(AnthropicAdapter())  # swap this one line to change provider

    response = await client.generate(
        messages=[Message.text(Role.USER, "What is the capital of France?")],
        model="claude-sonnet-4-6",
        max_tokens=100,
    )

    print(response.text())        # "Paris is the capital of France."
    print(response.stop_reason)   # StopReason.END_TURN
    print(response.usage)         # Usage(input_tokens=14, output_tokens=8)

asyncio.run(main())
```

**Swapping providers — only the adapter and model string change. Everything else in your
code stays identical:**

```python
from llmkit.adapters.anthropic import AnthropicAdapter
from llmkit.adapters.openai import OpenAIAdapter
from llmkit.adapters.gemini import GeminiAdapter
from llmkit.adapters.ollama import OllamaAdapter

# Anthropic
client, model = Client(AnthropicAdapter()), "claude-sonnet-4-6"

# OpenAI
client, model = Client(OpenAIAdapter()), "gpt-4o"

# Gemini
client, model = Client(GeminiAdapter()), "gemini-2.5-flash"

# Ollama (local, no API key needed)
client, model = Client(OllamaAdapter()), "llama3.2"
```

This is the entire point of the library: `Client.generate()` and `Client.stream()` behave
identically no matter which adapter is plugged in.

---

## 3. Setting API keys

Pass the key directly, or set an environment variable — the SDK picks it up automatically
if you don't pass one.

```python
# Option A: pass directly
client = Client(AnthropicAdapter(api_key="sk-ant-..."))
client = Client(OpenAIAdapter(api_key="sk-..."))
client = Client(GeminiAdapter(api_key="AIza..."))

# Option B: environment variable (recommended — keep keys out of source code)
# export ANTHROPIC_API_KEY=sk-ant-...
# export OPENAI_API_KEY=sk-...
# export GEMINI_API_KEY=AIza...
client = Client(AnthropicAdapter())  # picks up ANTHROPIC_API_KEY automatically
```

**Ollama needs no key at all** — it talks to a local (or self-hosted) server:

```python
# Default: connects to localhost:11434
client = Client(OllamaAdapter())

# Custom host (e.g. a remote Ollama server, or Ollama Cloud)
client = Client(OllamaAdapter(host="http://192.168.1.10:11434"))

# Ollama Cloud (https://ollama.com) does use a bearer token via api_key
client = Client(OllamaAdapter(api_key="...", host="https://ollama.com"))
```

| Provider | Env var | Required? |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | Yes |
| OpenAI | `OPENAI_API_KEY` | Yes |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Yes |
| Ollama | none | No — local by default |

---

## 4. System prompts

```python
response = await client.generate(
    messages=[Message.text(Role.USER, "Tell me a joke.")],
    model="claude-sonnet-4-6",
    max_tokens=200,
    system="You are a pirate. Respond only in pirate dialect.",
)

print(response.text())  # "Arrr, why do pirates make great singers? ..."
```

The `system` parameter works identically across all four providers — llmkit absorbs the
differences internally (Anthropic and Gemini take it as a separate field; OpenAI and
Ollama fold it into the messages list as a `system`-role entry). You never need to know
or care which one your chosen provider does.

---

## 5. Multi-turn conversations

llmkit is stateless per call — there's no hidden session. You build the conversation
yourself by appending each turn to a list and passing the whole history every time.

```python
from llmkit import Client, Message, Role
from llmkit.adapters.ollama import OllamaAdapter

async def chat():
    client = Client(OllamaAdapter())
    history = []

    # Turn 1
    history.append(Message.text(Role.USER, "My name is Arun."))
    response = await client.generate(history, model="llama3.2", max_tokens=100)
    print("Assistant:", response.text())

    # Append the assistant's reply before the next turn
    history.append(Message(role=Role.ASSISTANT, content=response.content))

    # Turn 2 — model remembers the context
    history.append(Message.text(Role.USER, "What is my name?"))
    response = await client.generate(history, model="llama3.2", max_tokens=100)
    print("Assistant:", response.text())  # "Your name is Arun."

asyncio.run(chat())
```

`response.content` is exactly the right shape to append back into history as the next
`Message` — this works whether the previous turn was plain text or included tool calls
(see [Tool calling](#7-tool-calling) below).

---

## 6. Streaming

```python
async def stream_example():
    client = Client(AnthropicAdapter())

    async for chunk in client.stream(
        messages=[Message.text(Role.USER, "Write a short poem about the sea.")],
        model="claude-sonnet-4-6",
        max_tokens=200,
    ):
        if chunk.type == "text_delta":
            print(chunk.text, end="", flush=True)  # prints word by word

    print()

asyncio.run(stream_example())
```

**All chunk types you may receive:**

```python
async for chunk in client.stream(...):
    if chunk.type == "message_start":
        print(f"Model: {chunk.model}")

    elif chunk.type == "text_delta":
        print(chunk.text, end="", flush=True)

    elif chunk.type == "tool_call_start":
        print(f"\nCalling tool: {chunk.name} (id={chunk.id})")

    elif chunk.type == "tool_call_delta":
        print(chunk.partial_json, end="")  # accumulate these per chunk.index

    elif chunk.type == "message_stop":
        print(f"\nStop reason: {chunk.stop_reason}")
        if chunk.usage:
            print(f"Tokens: {chunk.usage.input_tokens} in, {chunk.usage.output_tokens} out")
```

> **Accumulating streamed tool calls:** `tool_call_delta` chunks carry a fragment of the
> tool's arguments as a JSON string, keyed by `index`. Concatenate every delta sharing the
> same `index`, then `json.loads()` the full string once the tool call's stream segment
> ends. Note that Gemini and Ollama don't stream arguments incrementally — you'll get
> exactly one `tool_call_delta` per call, already containing the complete JSON. Anthropic
> and OpenAI may split it across several deltas. Writing your accumulation logic to handle
> "one or more fragments" covers both cases correctly.

---

## 7. Tool calling

llmkit normalizes tool/function calling across all four providers behind one shape:
define a `Tool`, get back `ToolUseBlock`s when the model wants to call one, execute it
yourself, and send the result back as a `ToolResultBlock`.

### Defining a tool

```python
from llmkit import Tool

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)
```

`input_schema` is plain [JSON Schema](https://json-schema.org/) — the same schema works
unchanged across Anthropic, OpenAI, Gemini, and Ollama. llmkit handles each provider's
own field-naming differences (`input_schema` vs `parameters` vs `parameters_json_schema`)
internally.

### The full ask → execute → answer loop

```python
from llmkit import Client, Message, Role, ToolResultBlock
from llmkit.adapters.anthropic import AnthropicAdapter
import json

def get_weather(city: str) -> dict:
    # Your real implementation — call an API, query a database, etc.
    return {"city": city, "temp_c": 31, "description": "sunny"}

async def run():
    client = Client(AnthropicAdapter())
    messages = [Message.text(Role.USER, "What's the weather in Chennai?")]

    response = await client.generate(
        messages, model="claude-sonnet-4-6", max_tokens=300, tools=[weather_tool]
    )

    if response.tool_calls():
        # Append the assistant's tool-call turn to history
        messages.append(Message(role=Role.ASSISTANT, content=response.content))

        # Execute every requested call and append the result
        for call in response.tool_calls():
            result = get_weather(**call.input)
            messages.append(Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_use_id=call.id, content=json.dumps(result))],
            ))

        # Ask again with the tool result in context — model gives the final answer
        response = await client.generate(
            messages, model="claude-sonnet-4-6", max_tokens=300, tools=[weather_tool]
        )

    print(response.text())

asyncio.run(run())
```

This exact loop works unchanged for OpenAI, Gemini, and Ollama — just swap the adapter.

### Checking whether the model called a tool

```python
response = await client.generate(messages, model=..., max_tokens=300, tools=[weather_tool])

if response.stop_reason == StopReason.TOOL_USE:
    for call in response.tool_calls():
        print(call.id, call.name, call.input)   # input is always a dict, never a JSON string
```

`response.tool_calls()` returns an empty list if the model answered directly without
calling anything — always safe to check unconditionally.

### Handling a tool error

If your tool execution fails, tell the model so it can recover gracefully instead of
treating the failure as a real answer:

```python
messages.append(Message(
    role=Role.USER,
    content=[ToolResultBlock(
        tool_use_id=call.id,
        content="Error: city not found",
        is_error=True,
    )],
))
```

---

## 8. The Response object

```python
response = await client.generate(...)

# Get the text (most common)
text = response.text()

# Get any tool calls the model requested
calls = response.tool_calls()  # [] if none

# Check why the model stopped
from llmkit import StopReason

if response.stop_reason == StopReason.END_TURN:
    print("Finished naturally")
elif response.stop_reason == StopReason.MAX_TOKENS:
    print("Hit the token limit — increase max_tokens if needed")
elif response.stop_reason == StopReason.TOOL_USE:
    print("Model wants to call a tool")

# Token usage
print(response.usage.input_tokens)
print(response.usage.output_tokens)
print(response.usage.total_tokens)

# Access the raw, untranslated provider response (escape hatch for
# provider-specific fields llmkit doesn't normalize)
print(response.raw)
```

---

## 9. Resilience — retry, timeout, errors, and cost tracking

All four features are optional and configured on `Client` at construction time. Omitting
them gives you the same behaviour as before — no retries, no timeout, raw exceptions
normalised to llmkit errors, no cost tracking.

### Retry with exponential backoff

```python
from llmkit import Client, RetryConfig
from llmkit.adapters.anthropic import AnthropicAdapter

client = Client(
    AnthropicAdapter(),
    retry_config=RetryConfig(
        max_attempts=3,    # 1 initial attempt + 2 retries
        base_delay=1.0,    # seconds before first retry; doubles each attempt
        max_delay=60.0,    # cap on wait between retries
    ),
)
```

Only retryable errors are retried — rate limits, timeouts, connection errors, and server
errors (5xx). Authentication errors and bad requests are raised immediately since retrying
them won't help.

**Logging retries** via the `on_retry` callback:

```python
import logging

def log_retry(attempt: int, error, wait: float):
    logging.warning(f"Retry {attempt + 1} after {wait:.1f}s — {error}")

client = Client(
    AnthropicAdapter(),
    retry_config=RetryConfig(max_attempts=3, base_delay=1.0, on_retry=log_retry),
)
```

### Per-call timeout

```python
client = Client(
    AnthropicAdapter(),
    timeout_seconds=30.0,   # raises TimeoutError if any single attempt takes longer
)
```

Timeout applies per attempt — if you also set `max_attempts=3`, each of the three
attempts gets its own 30-second window.

> **Streaming:** retry and timeout are **not** applied to `client.stream()` calls.
> Streaming introduces state (chunks already yielded to your code) that can't be safely
> replayed — restarting mid-stream would produce duplicate output. If you need retry
> behaviour around streams, wrap the entire `async for` loop in your own try/except.

### Normalised error exceptions

Whether or not you use retry, all raw SDK exceptions are mapped to llmkit's own
exception hierarchy before they surface to your code. You never need to import from
`anthropic`, `openai`, or `google` just to write a `try/except`.

```python
from llmkit import (
    LLMKitError,           # base class — catch-all
    RateLimitError,        # 429 / quota exceeded — retryable
    AuthenticationError,   # 401 / bad key — not retryable
    TimeoutError,          # request exceeded timeout — retryable
    ConnectionError,       # server unreachable (e.g. Ollama not running)
    InvalidRequestError,   # 400 / bad prompt or params
    APIError,              # 5xx server error — retryable
    UnknownError,          # unrecognized exception
)

try:
    response = await client.generate(...)
except RateLimitError as e:
    print(f"Rate limited by {e.provider} (HTTP {e.status_code})")
    print(f"Original error: {e.cause}")
except AuthenticationError:
    print("Check your API key")
except LLMKitError as e:
    print(f"Something went wrong: {e}")
```

Every exception carries `provider` (which provider raised it), `status_code` (HTTP code
if known, else `None`), and `cause` (the original raw SDK exception, always available for
debugging).

### Cost tracking

```python
from llmkit import CostTracker

tracker = CostTracker()
client = Client(AnthropicAdapter(), cost_tracker=tracker)

await client.generate([Message.text(Role.USER, "hi")], model="claude-sonnet-4-6", max_tokens=100)
await client.generate([Message.text(Role.USER, "hello")], model="claude-sonnet-4-6", max_tokens=100)

# Per-call records
for call in tracker.calls:
    print(f"{call.model}: {call.total_tokens} tokens, ${call.cost_usd:.6f}")

# Accumulated totals
print(tracker.total_tokens)
print(tracker.total_cost_usd)
print(tracker.call_count)

# Human-readable summary
print(tracker.summary())

# Breakdown by model
print(tracker.by_model())

# Reset between sessions
tracker.reset()
```

**Custom price table** — override specific models or add models not in the defaults:

```python
tracker = CostTracker(price_table={
    "my-fine-tuned-model": {"input": 5.00, "output": 25.00},  # USD per 1M tokens
    "gpt-4o": {"input": 2.00, "output": 8.00},                # override a default
})
```

Unknown models return `$0.00` rather than raising — a missing price never crashes your
application. Access the tracker from the client at any time via `client.cost_tracker`.

### Putting it all together

```python
from llmkit import Client, RetryConfig, CostTracker, RateLimitError
from llmkit.adapters.anthropic import AnthropicAdapter
import logging

tracker = CostTracker()
client = Client(
    AnthropicAdapter(),
    retry_config=RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        on_retry=lambda attempt, err, wait:
            logging.warning(f"Retry {attempt + 1} in {wait:.1f}s: {err}"),
    ),
    timeout_seconds=30.0,
    cost_tracker=tracker,
)

try:
    response = await client.generate(
        [Message.text(Role.USER, "Summarise this document...")],
        model="claude-sonnet-4-6",
        max_tokens=500,
    )
    print(response.text())
except RateLimitError:
    print("All retries exhausted — try again later")

print(tracker.summary())
```

---

## 10. All parameters reference

```python
response = await client.generate(
    messages=[Message.text(Role.USER, "Hello")],
    model="claude-sonnet-4-6",   # required: model string, specific to the chosen provider
    max_tokens=1024,              # required: max output tokens
    system="Be concise.",         # optional: system prompt
    temperature=0.7,              # optional: 0.0 = deterministic, 1.0 = more creative
    tools=[weather_tool],         # optional: list of Tool definitions
)
```

`client.stream(...)` accepts the exact same parameters and returns an async iterator of
chunks instead of a single `Response`.

**Client constructor parameters:**

```python
client = Client(
    adapter,                          # required: any ProviderAdapter instance
    retry_config=RetryConfig(...),    # optional: retry + backoff config
    timeout_seconds=30.0,            # optional: per-attempt timeout in seconds
    cost_tracker=CostTracker(),      # optional: token cost tracking
)
```

---

## 11. Provider-specific notes

These are real differences in the underlying provider, not bugs — llmkit surfaces them
honestly rather than papering over them with something that looks consistent but quietly
loses information.

**Anthropic & OpenAI**
- Stream tool-call arguments incrementally, across multiple `tool_call_delta` chunks.
- Have the richest, most reliable tool-calling support of the four.

**Gemini**
- Does not stream tool-call arguments incrementally — each call arrives as exactly one
  `tool_call_delta` with the complete JSON already in it.
- Sending a tool result back requires the matching tool-call's `Message` to still be
  present earlier in your `messages` list (llmkit looks up the tool's name from it
  internally). Don't drop earlier turns from history when tools are involved.

**Ollama**
- Needs `ollama serve` running locally, with the model already pulled
  (e.g. `ollama pull llama3.1`) — `client.generate()` will raise a connection error if
  the server isn't reachable.
- Tool-calling support depends entirely on the specific local model — not every model
  supports it, and support quality varies. If a model doesn't support tools, it will
  simply answer in plain text instead of calling anything; this is the model's
  limitation, not an llmkit error.
- Like Gemini, does not stream tool-call arguments incrementally.

---

## 12. Capabilities & limitations

What llmkit supports today:

- Text generation (single-turn and multi-turn), streaming, system prompts
- Tool/function calling, including streaming tool calls
- Anthropic, OpenAI, Gemini, and Ollama, behind one identical interface
- Retry with exponential backoff (configurable, retryable errors only)
- Per-attempt timeout
- Normalised error exceptions across all four providers
- Token cost tracking — per-call and accumulated totals

What llmkit does **not** support yet — don't assume these work:

- **Provider-hosted tools** (e.g. Anthropic's/OpenAI's built-in web search, code
  execution). These run server-side and use a different mechanism than the
  `Tool`/`ToolUseBlock` flow described above. Not yet integrated.
- **MCP (Model Context Protocol)** servers. llmkit's `Tool` type is structurally
  compatible with tools discovered via MCP, but there's no built-in bridge yet — you'd
  need to convert an MCP server's tool list into `Tool` objects yourself.
- **Vision / image inputs.** Only text content blocks exist today.
- **Structured/JSON-schema-constrained outputs** as a first-class feature (you can ask a
  model to return JSON via your prompt, but there's no dedicated validation layer yet).
- **Automatic fallback between providers.** `Client` wraps one adapter at a time and
  retries against the same provider — it does not automatically switch to a different
  provider if one is down.

If your use case needs any of these, check with whoever maintains this library before
assuming the gap doesn't matter for your task.

---

## 13. Quick reference

| Task | Code |
|---|---|
| Install for one provider | `pip install "llmkit[anthropic]"` |
| Create a client | `Client(AnthropicAdapter())` |
| Single-turn generate | `await client.generate([Message.text(Role.USER, "hi")], model=..., max_tokens=...)` |
| Add a system prompt | `system="You are ..."` kwarg |
| Stream the response | `async for chunk in client.stream(...)` |
| Define a tool | `Tool(name=..., description=..., input_schema={...})` |
| Pass tools to a call | `tools=[my_tool]` kwarg |
| Check for tool calls | `response.tool_calls()` |
| Send a tool result back | `ToolResultBlock(tool_use_id=call.id, content=...)` |
| Get text from response | `response.text()` |
| Check stop reason | `response.stop_reason == StopReason.END_TURN` |
| Check token usage | `response.usage.total_tokens` |
| Enable retry | `Client(adapter, retry_config=RetryConfig(max_attempts=3))` |
| Set a timeout | `Client(adapter, timeout_seconds=30.0)` |
| Catch any llmkit error | `except LLMKitError` |
| Catch rate limits | `except RateLimitError` |
| Track costs | `tracker = CostTracker(); Client(adapter, cost_tracker=tracker)` |
| Print cost summary | `tracker.summary()` |
| Cost by model | `tracker.by_model()` |
| Use Ollama locally | `Client(OllamaAdapter())` — no key needed |
| Change provider | Swap the adapter, keep everything else the same |