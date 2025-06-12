"""Configuration module for AngelaMCP."""

from .settings import settings, Settings, Environment, LogLevel

# Import prompts if they exist
try:
    from . import prompts
    __all__ = ["settings", "Settings", "Environment", "LogLevel", "prompts"]
except ImportError:
    __all__ = ["settings", "Settings", "Environment", "LogLevel"]
