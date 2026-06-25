"""
Top-level package for llmkit.

Provides convenient imports for core classes.
"""

# from .adapters import ProviderAdapter
from .core.client import Client

from .core.types import (
    Message,
    Response,
    Role,
    StopReason,
    StreamChunk,
    TextBlock,
    Tool,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

__all__ = [
    "Client",
    "Message",
    "Response",
    "Role",
    "StopReason",
    "StreamChunk",
    "TextBlock",
    "Tool",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
    # "ProviderAdapter",
]
