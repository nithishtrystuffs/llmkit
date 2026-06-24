import asyncio
from llmkit import Client, Message, Role
from llmkit.adapters.anthropic import AnthropicAdapter
from pprint import pprint

async def main():
    client = Client(AnthropicAdapter())
    resp = await client.generate(
        [Message.text(Role.USER, "say hi in 5 words")],
        model="gpt-oss:120b-cloud",
        max_tokens=50,
    )
    pprint(resp.text())

asyncio.run(main())