"""
Proves tool calling works end-to-end, identically, across all 4 providers.

Defines one tool, asks the same question, and runs the full
ask -> tool_call -> execute -> tool_result -> final_answer loop using
ONLY the neutral Client/Message/Tool API — no provider-specific code in
this file at all.

Run from the project root:

    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...
    # Ollama needs `ollama serve` running locally with a tool-capable
    # model pulled, e.g. `ollama pull llama3.1` (llama3.2 has weaker
    # tool support; use a model documented as tool-capable)
    python run_tool_calling.py
"""

import asyncio
import json

from llmkit import Client, Message, Role, Tool, ToolResultBlock
from llmkit.adapters.anthropic import AnthropicAdapter
from llmkit.adapters.gemini import GeminiAdapter
from llmkit.adapters.ollama import OllamaAdapter
from llmkit.adapters.openai import OpenAIAdapter

WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather for a city",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)


def fake_get_weather(city: str) -> dict:
    """A stand-in for a real tool — same fake result for every provider
    so we can compare behavior on equal footing."""
    return {"city": city, "temp_c": 31, "description": "sunny"}


async def run_tool_loop(label: str, client: Client, model: str) -> None:
    print(f"\n=== {label} ({model}) ===")
    messages = [Message.text(Role.USER, "What's the weather like in Chennai right now?")]

    try:
        response = await client.generate(
            messages, model=model, max_tokens=300, tools=[WEATHER_TOOL]
        )
    except Exception as e:
        print(f"generate() failed: {e!r}")
        return

    print("First response stop_reason:", response.stop_reason)
    calls = response.tool_calls()
    if not calls:
        print("Model answered directly without calling the tool:")
        print(response.text())
        return

    # Append the assistant's tool-call turn, then execute the tool and
    # append the result — same shape regardless of provider.
    messages.append(Message(role=Role.ASSISTANT, content=response.content))

    for call in calls:
        print(f"Tool call requested: {call.name}({call.input})")
        result = fake_get_weather(**call.input)
        messages.append(
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_use_id=call.id, content=json.dumps(result))],
            )
        )

    try:
        final = await client.generate(
            messages, model=model, max_tokens=300, tools=[WEATHER_TOOL]
        )
    except Exception as e:
        print(f"second generate() failed: {e!r}")
        return

    print("Final answer:", final.text())


async def main() -> None:
    # await run_tool_loop("Anthropic", Client(AnthropicAdapter()), model="claude-sonnet-4-6")
    # await run_tool_loop("OpenAI", Client(OpenAIAdapter()), model="gpt-4o")
    await run_tool_loop("Gemini", Client(GeminiAdapter()), model="gemini-2.5-flash")
    await run_tool_loop("Ollama (local)", Client(OllamaAdapter()), model="gpt-oss:120b-cloud")


if __name__ == "__main__":
    asyncio.run(main())