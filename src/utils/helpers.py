"""
Helper utilities for AngelaMCP.

Common utility functions used across the platform.
"""

import json
import asyncio
from typing import Any, Dict, Optional


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON response string safely."""
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None


def format_cost(cost_usd: float) -> str:
    """Format cost in USD for display."""
    return f"${cost_usd:.4f}"


def format_tokens(tokens: int) -> str:
    """Format token count for display."""
    if tokens < 1000:
        return str(tokens)
    elif tokens < 1000000:
        return f"{tokens/1000:.1f}K"
    else:
        return f"{tokens/1000000:.1f}M"


async def safe_async_call(coro, default_value=None):
    """Safely call an async function with error handling."""
    try:
        return await coro
    except Exception:
        return default_value


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_timestamp(timestamp) -> str:
    """Format timestamp for display."""
    if hasattr(timestamp, 'strftime'):
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return str(timestamp)
