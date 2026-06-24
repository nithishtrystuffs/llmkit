"""
Top-level package for llmkit.

Provides convenient imports for core classes.
"""

from .core.client import Client
from .core.types import Message, Role 
from .adapters import ProviderAdapter

__all__ = ["Client", "Message", "Role", "ProviderAdapter"]
