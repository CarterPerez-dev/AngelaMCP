"""
Prompt templates for AngelaMCP agents.

This module contains all the prompt templates used by different agents
for various collaboration scenarios.
"""

# System prompts
try:
    from .system_prompts import (
        SYSTEM_PROMPT_CLAUDE,
        SYSTEM_PROMPT_OPENAI,
        SYSTEM_PROMPT_GEMINI,
        COLLABORATION_SYSTEM_PROMPT
    )
    _system_imports = [
        "SYSTEM_PROMPT_CLAUDE",
        "SYSTEM_PROMPT_OPENAI", 
        "SYSTEM_PROMPT_GEMINI",
        "COLLABORATION_SYSTEM_PROMPT"
    ]
except ImportError:
    _system_imports = []

# Claude-specific prompts
try:
    from .claude_prompts import (
        CLAUDE_CODE_GENERATION_PROMPT,
        CLAUDE_REVIEW_PROMPT,
        CLAUDE_PROJECT_PROMPT
    )
    _claude_imports = [
        "CLAUDE_CODE_GENERATION_PROMPT",
        "CLAUDE_REVIEW_PROMPT",
        "CLAUDE_PROJECT_PROMPT"
    ]
except ImportError:
    _claude_imports = []

# Debate prompts
try:
    from .debate_prompts import (
        DEBATE_MODERATOR_PROMPT,
        DEBATE_PARTICIPANT_PROMPT,
        DEBATE_SYNTHESIS_PROMPT,
        VOTING_PROMPT
    )
    _debate_imports = [
        "DEBATE_MODERATOR_PROMPT",
        "DEBATE_PARTICIPANT_PROMPT", 
        "DEBATE_SYNTHESIS_PROMPT",
        "VOTING_PROMPT"
    ]
except ImportError:
    _debate_imports = []

__all__ = _system_imports + _claude_imports + _debate_imports
