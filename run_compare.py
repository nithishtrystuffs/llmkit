"""
Proves the adapter pattern works across providers: identical Client and
Message usage, only the adapter passed in changes.

Run from the project root (where pyproject.toml lives):

    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...          # or GOOGLE_API_KEY
    # Ollama needs no key — just `ollama serve` running locally with a
    # model pulled, e.g. `ollama pull gpt-oss:120b-cloud`
    python run_compare.py

On Windows (cmd):
    set ANTHROPIC_API_KEY=sk-ant-...
    set OPENAI_API_KEY=sk-...
    set GEMINI_API_KEY=...
    py run_compare.py
"""

import asyncio
import os
from llmkit import Client, Message, Role
from llmkit.adapters.anthropic import AnthropicAdapter
from llmkit.adapters.gemini import GeminiAdapter
from llmkit.adapters.ollama import OllamaAdapter
from llmkit.adapters.openai import OpenAIAdapter
from dotenv import load_dotenv

load_dotenv()

PROMPT = "Say hello in exactly 5 words."


async def run(label: str, client: Client, model: str) -> None:
    print(f"\n--- {label} ({model}) ---")
    try:
        resp = await client.generate(
            [Message.text(Role.USER, PROMPT)],
            model=model,
            max_tokens=50,
        )
        print("Text:       ", resp.text())
        print("Stop reason:", resp.stop_reason)
        print("Usage:      ", resp.usage)
    except Exception as e:
        print(f"generate() failed: {e!r}")
        return

    print(f"--- {label} streaming ---")
    try:
        async for chunk in client.stream(
            [Message.text(Role.USER, PROMPT)], model=model, max_tokens=50
        ):
            if chunk.type == "text_delta":
                print(chunk.text, end="", flush=True)
        print()  # newline after stream
    except Exception as e:
        print(f"stream() failed: {e!r}")


async def main() -> None:
    # Note: same `Client`, same `Message`, same call shape — only the
    # adapter instance and model string differ. This is the whole point.
    await run("Anthropic", Client(AnthropicAdapter()), model="claude-sonnet-4-6")
    await run("OpenAI", Client(OpenAIAdapter()), model="gpt-4o")
    await run("Gemini", Client(GeminiAdapter()), model="gemini-2.5-flash")
    await run("Ollama (local)", Client(OllamaAdapter()), model="gpt-oss:120b-cloud")


if __name__ == "__main__":
    asyncio.run(main())
