# llmkit — Usage Guide

A provider-agnostic LLM SDK. Write your code once, swap between Anthropic, OpenAI, Gemini, and Ollama by changing one line.

---

## Installation

Install only the provider(s) you need:

```bash
# Just Anthropic
pip install -e ".[anthropic]"

# Just OpenAI
pip install -e ".[openai]"

# Just Gemini
pip install -e ".[gemini]"

# Just Ollama (local, no API key)
pip install -e ".[ollama]"

# Everything
pip install -e ".[all]"
```

---

## 1. Basic usage — generate a response

```python
import asyncio
from llmkit import Client, Message, Role
from llmkit.adapters.anthropic import AnthropicAdapter

async def main():
    client = Client(AnthropicAdapter())  # swap adapter to change provider

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

---

## 2. Setting API keys

Pass the key directly, or set the environment variable — the SDK picks it up automatically.

```python
# Option A: pass directly
client = Client(AnthropicAdapter(api_key="sk-ant-..."))
client = Client(OpenAIAdapter(api_key="sk-..."))
client = Client(GeminiAdapter(api_key="AIza..."))

# Option B: environment variable (recommended — keep keys out of code)
# export ANTHROPIC_API_KEY=sk-ant-...
# export OPENAI_API_KEY=sk-...
# export GEMINI_API_KEY=AIza...
client = Client(AnthropicAdapter())  # picks up env var automatically
```

**Ollama needs no key** — it runs locally:

```python
# Default: connects to localhost:11434
client = Client(OllamaAdapter())

# Custom host (e.g. remote Ollama server)
client = Client(OllamaAdapter(host="http://192.168.1.10:11434"))
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

print(response.text())  # "Arrr, why do pirates make great singers? ..."
```

The `system` parameter works identically across all providers — llmkit handles the differences internally (Anthropic and Gemini use a separate field; OpenAI and Ollama fold it into the messages list).

---

## 4. Multi-turn conversations

Build the conversation by appending each turn to the messages list:

```python
from llmkit import Client, Message, Role
from llmkit.adapters.ollama import OllamaAdapter
from llmkit.core.types import TextBlock

async def chat():
    client = Client(OllamaAdapter())
    history = []

    # Turn 1
    history.append(Message.text(Role.USER, "My name is Arun."))
    response = await client.generate(history, model="llama3.2", max_tokens=100)
    print("Assistant:", response.text())

    # Append the assistant reply to history
    history.append(Message(role=Role.ASSISTANT, content=response.content))

    # Turn 2 — model remembers the context
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
            print(chunk.text, end="", flush=True)  # prints word by word

    print()  # newline at the end

asyncio.run(stream_example())
```

**All chunk types:**

```python
async for chunk in client.stream(...):
    if chunk.type == "message_start":
        print(f"Model: {chunk.model}")

    elif chunk.type == "text_delta":
        print(chunk.text, end="", flush=True)  # the actual text, streamed

    elif chunk.type == "message_stop":
        print(f"\nStop reason: {chunk.stop_reason}")
        if chunk.usage:
            print(f"Tokens: {chunk.usage.input_tokens} in, {chunk.usage.output_tokens} out")
```

---

## 6. All parameters

```python
response = await client.generate(
    messages=[Message.text(Role.USER, "Hello")],
    model="claude-sonnet-4-6",   # required: model string for the chosen provider
    max_tokens=1024,              # required: max output tokens
    system="Be concise.",         # optional: system prompt
    temperature=0.7,              # optional: 0.0 = deterministic, 1.0 = creative
)
```

---

## 7. Working with the Response object

```python
response = await client.generate(...)

# Get the text (most common)
text = response.text()

# Check why the model stopped
from llmkit.core.types import StopReason

if response.stop_reason == StopReason.END_TURN:
    print("Finished naturally")
elif response.stop_reason == StopReason.MAX_TOKENS:
    print("Hit the token limit — increase max_tokens if needed")

# Token usage
print(response.usage.input_tokens)   # prompt tokens
print(response.usage.output_tokens)  # completion tokens
print(response.usage.total_tokens)   # sum

# Access raw provider response (escape hatch — provider-specific)
print(response.raw)   # original dict from the provider SDK
```

---

## 8. Adding a new provider

The whole point of the adapter pattern is that adding a provider touches zero existing code. Create one file:

```
llmkit/adapters/myprovider/
    __init__.py
    adapter.py          ← implement ProviderAdapter here
```

```python
# adapter.py
from llmkit.adapters.base import ProviderAdapter
from llmkit.core.types import Message, Response, StreamChunk

class MyProviderAdapter(ProviderAdapter):

    async def generate(self, messages, *, model, max_tokens,
                       system=None, temperature=None) -> Response:
        # translate llmkit types → your provider's SDK
        # call your provider's API
        # translate response → llmkit Response
        ...

    async def stream(self, messages, *, model, max_tokens,
                     system=None, temperature=None):
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
| Install for one provider | `pip install -e ".[anthropic]"` |
| Create a client | `Client(AnthropicAdapter())` |
| Single-turn generate | `await client.generate([Message.text(Role.USER, "hi")], model=..., max_tokens=...)` |
| Add a system prompt | `system="You are ..."` kwarg |
| Stream the response | `async for chunk in client.stream(...)` |
| Get text from response | `response.text()` |
| Check stop reason | `response.stop_reason == StopReason.END_TURN` |
| Check token usage | `response.usage.total_tokens` |
| Use Ollama locally | `Client(OllamaAdapter())` — no key needed |
| Change provider | Swap the adapter, keep everything else the same |
