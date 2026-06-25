# llmkit

A provider-agnostic LLM SDK. Write your code once, swap between Anthropic, OpenAI, Gemini, and Ollama by changing one line.

---

## Installation

Install only the provider(s) you need directly from Git:

```bash
# Just Anthropic
pip install "llmkit[anthropic] @ git+https://github.com/nithishtrystuffs/llmkit.git@v0.1.3"

# Just OpenAI
pip install "llmkit[openai] @ git+https://github.com/nithishtrystuffs/llmkit.git@v0.1.3"

# Just Gemini
pip install "llmkit[gemini] @ git+https://github.com/nithishtrystuffs/llmkit.git@v0.1.3"

# Just Ollama (local, no API key needed)
pip install "llmkit[ollama] @ git+https://github.com/nithishtrystuffs/llmkit.git@v0.1.3"

# Everything
pip install "llmkit[all] @ git+https://github.com/nithishtrystuffs/llmkit.git@v0.1.3"
```

To upgrade to a newer version, change the tag and add `--upgrade`:

```bash
pip install "llmkit[all] @ git+https://github.com/nithishtrystuffs/llmkit.git@v0.1.4" --upgrade
```

---

## Supported providers

| Provider | Adapter | API key env var | Notes |
|---|---|---|---|
| Anthropic (Claude) | `AnthropicAdapter` | `ANTHROPIC_API_KEY` | |
| OpenAI | `OpenAIAdapter` | `OPENAI_API_KEY` | Also supports Azure OpenAI |
| Google Gemini | `GeminiAdapter` | `GEMINI_API_KEY` | |
| Ollama | `OllamaAdapter` | — | Runs locally, no key needed |

---

## 1. Basic usage

```python
import asyncio
from llmkit import Client, Message, Role
from llmkit.adapters.anthropic import AnthropicAdapter

async def main():
    client = Client(AnthropicAdapter())

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

**Swapping providers — only the adapter and model string change:**

```python
from llmkit.adapters.anthropic import AnthropicAdapter
from llmkit.adapters.openai import OpenAIAdapter
from llmkit.adapters.gemini import GeminiAdapter
from llmkit.adapters.ollama import OllamaAdapter

# Anthropic
client = Client(AnthropicAdapter())
model  = "claude-sonnet-4-6"

# OpenAI
client = Client(OpenAIAdapter())
model  = "gpt-4o"

# Gemini
client = Client(GeminiAdapter())
model  = "gemini-2.5-flash"

# Ollama (local, no API key needed)
client = Client(OllamaAdapter())
model  = "llama3.2"
```

Everything else — `generate()`, `stream()`, `Response`, `Message` — stays identical.

---

## 2. API keys

Pass the key directly or set the environment variable — the SDK picks it up automatically.

```python
# Option A: pass directly
client = Client(AnthropicAdapter(api_key="sk-ant-..."))
client = Client(OpenAIAdapter(api_key="sk-..."))
client = Client(GeminiAdapter(api_key="AIza..."))

# Option B: environment variable (recommended — keep keys out of code)
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=AIza...
client = Client(AnthropicAdapter())  # picks up env var automatically
```

**Ollama needs no key** — it runs locally:

```python
# Default: connects to localhost:11434
client = Client(OllamaAdapter())

# Custom host
client = Client(OllamaAdapter(host="http://192.168.1.10:11434"))
```

**Azure OpenAI** — pass the extra Azure params:

```python
client = Client(OpenAIAdapter(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-01",   # optional, defaults to 2024-02-01
))
# model= takes your Azure deployment name, not the model name
response = await client.generate(..., model="my-gpt4-deployment")
```

---

## 3. System prompts

```python
response = await client.generate(
    messages=[Message.text(Role.USER, "Tell me a joke.")],
    model="claude-sonnet-4-6",
    max_tokens=200,
    system="You are a pirate. Respond only in pirate dialect.",
)

print(response.text())
```

The `system` parameter works identically across all providers — llmkit handles the differences internally.

---

## 4. Multi-turn conversations

Build the conversation by appending each turn to the messages list:

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

    # Append assistant reply to history and continue
    history.append(Message(role=Role.ASSISTANT, content=response.content))
    history.append(Message.text(Role.USER, "What is my name?"))

    response = await client.generate(history, model="llama3.2", max_tokens=100)
    print("Assistant:", response.text())  # "Your name is Arun."

asyncio.run(chat())
```

---

## 5. Streaming

```python
async def stream_example():
    client = Client(AnthropicAdapter())

    async for chunk in client.stream(
        messages=[Message.text(Role.USER, "Write a short poem about the sea.")],
        model="claude-sonnet-4-6",
        max_tokens=200,
    ):
        if chunk.type == "text_delta":
            print(chunk.text, end="", flush=True)

    print()

asyncio.run(stream_example())
```

**All chunk types:**

```python
async for chunk in client.stream(...):
    if chunk.type == "message_start":
        print(f"Model: {chunk.model}")

    elif chunk.type == "text_delta":
        print(chunk.text, end="", flush=True)

    elif chunk.type == "tool_call_start":
        print(f"Tool call: {chunk.name} (id={chunk.id})")

    elif chunk.type == "tool_call_delta":
        print(chunk.partial_json, end="")   # streaming tool arguments

    elif chunk.type == "message_stop":
        print(f"\nStop reason: {chunk.stop_reason}")
        if chunk.usage:
            print(f"Tokens: {chunk.usage.input_tokens} in, {chunk.usage.output_tokens} out")
```

---

## 6. Tool calling

Tools let the LLM request actions from your code — your code defines the tool, implements the logic, and runs the result back to the LLM.

### Step 1 — Define the tool (what the LLM knows about it)

```python
from llmkit.core.types import Tool

weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a city",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
)
```

### Step 2 — Implement the tool (your actual logic)

```python
def get_weather(city: str) -> str:
    # your own code — call a real API, query a DB, anything
    return f"{city}: 36°C, Sunny"
```

### Step 3 — Run the loop

```python
from llmkit.core.types import StopReason, ToolUseBlock, ToolResultBlock

async def main():
    client = Client(AnthropicAdapter())

    messages = [Message.text(Role.USER, "What's the weather in Chennai?")]

    # First call — LLM decides to use the tool
    response = await client.generate(
        messages=messages,
        model="claude-sonnet-4-6",
        tools=[weather_tool],
        max_tokens=500,
    )

    if response.stop_reason == StopReason.TOOL_USE:
        # LLM said "call this tool" — you run your function
        tool_results = []
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                result = get_weather(block.input["city"])  # your function
                tool_results.append(
                    ToolResultBlock(
                        tool_use_id=block.id,
                        content=result,
                    )
                )

        # Send result back to LLM for final answer
        messages.append(Message(role=Role.ASSISTANT, content=response.content))
        messages.append(Message(role=Role.USER, content=tool_results))

        final = await client.generate(
            messages=messages,
            model="claude-sonnet-4-6",
            tools=[weather_tool],
            max_tokens=500,
        )
        print(final.text())  # "The weather in Chennai is 36°C and Sunny."

asyncio.run(main())
```

Tool calling works identically across all 4 providers — llmkit handles the differences internally.

---

## 7. All parameters

```python
response = await client.generate(
    messages=[Message.text(Role.USER, "Hello")],
    model="claude-sonnet-4-6",   # required: model string for the chosen provider
    max_tokens=1024,              # required: max output tokens
    system="Be concise.",         # optional: system prompt
    temperature=0.7,              # optional: 0.0 = deterministic, 1.0 = creative
    tools=[my_tool],              # optional: list of Tool definitions
)
```

---

## 8. Working with the Response object

```python
response = await client.generate(...)

# Get the text
text = response.text()

# Check why the model stopped
from llmkit.core.types import StopReason

if response.stop_reason == StopReason.END_TURN:
    print("Finished naturally")
elif response.stop_reason == StopReason.MAX_TOKENS:
    print("Hit the token limit — increase max_tokens")
elif response.stop_reason == StopReason.TOOL_USE:
    print("LLM wants to call a tool")

# Token usage
print(response.usage.input_tokens)
print(response.usage.output_tokens)
print(response.usage.total_tokens)

# Raw provider response (escape hatch)
print(response.raw)
```

---

## 9. Adding a new provider

Adding a provider touches zero existing code. Create one file:

```
llmkit/adapters/myprovider/
    __init__.py
    adapter.py
```

```python
# adapter.py
from llmkit.adapters.base import ProviderAdapter
from llmkit.core.types import Message, Response, StreamChunk, Tool

class MyProviderAdapter(ProviderAdapter):

    async def generate(self, messages, *, model, max_tokens,
                       system=None, temperature=None, tools=None) -> Response:
        # translate llmkit types → your provider's SDK
        # call your provider's API
        # translate response → llmkit Response
        ...

    async def stream(self, messages, *, model, max_tokens,
                     system=None, temperature=None, tools=None):
        # same, but yield StreamChunk objects
        ...
```

Then use it exactly like the built-in adapters:

```python
client = Client(MyProviderAdapter())
```

---

## Quick reference

| Task | Code |
|---|---|
| Install | `pip install "llmkit[anthropic] @ git+https://github.com/your-org/llmkit.git@v0.1.3"` |
| Create a client | `Client(AnthropicAdapter())` |
| Generate | `await client.generate([Message.text(Role.USER, "hi")], model=..., max_tokens=...)` |
| System prompt | `system="You are ..."` kwarg |
| Stream | `async for chunk in client.stream(...)` |
| Tool calling | pass `tools=[Tool(...)]` to `generate()`, handle `StopReason.TOOL_USE` in response |
| Get text | `response.text()` |
| Check stop reason | `response.stop_reason == StopReason.END_TURN` |
| Token usage | `response.usage.total_tokens` |
| Azure OpenAI | `OpenAIAdapter(api_key=..., azure_endpoint=..., api_version=...)` |
| Ollama locally | `Client(OllamaAdapter())` — no key needed |
| Change provider | Swap the adapter, keep everything else the same |