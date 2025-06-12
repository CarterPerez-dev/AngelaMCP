"""Agent implementations for AngelaMCP."""

# Base agent classes and enums
from .base import (
    BaseAgent,
    AgentResponse, 
    AgentType,
    TaskType,
    AgentRole,
    TaskContext,
    agent_registry
)

# Specific agent implementations
from .claude_agent import ClaudeCodeAgent
from .openai_agent import OpenAIAgent
from .gemini_agent import GeminiAgent

# Registry for easy access
AVAILABLE_AGENTS = {
    AgentType.CLAUDE: ClaudeCodeAgent,
    AgentType.OPENAI: OpenAIAgent,
    AgentType.GEMINI: GeminiAgent
}

def create_agent(agent_type: AgentType, **kwargs) -> BaseAgent:
    """Factory function to create agents."""
    agent_class = AVAILABLE_AGENTS.get(agent_type)
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return agent_class(**kwargs)

def get_all_agent_types() -> list[AgentType]:
    """Get list of all available agent types."""
    return list(AVAILABLE_AGENTS.keys())

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentResponse", 
    "AgentType",
    "TaskType",
    "AgentRole",
    "TaskContext",
    "agent_registry",
    
    # Agent implementations
    "ClaudeCodeAgent",
    "OpenAIAgent", 
    "GeminiAgent",
    
    # Utilities
    "AVAILABLE_AGENTS",
    "create_agent",
    "get_all_agent_types"
]
