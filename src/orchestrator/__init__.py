"""Orchestration engine for multi-agent collaboration."""

# Main orchestrator (unified from manager.py)
from .manager import (
    TaskOrchestrator,
    CollaborationStrategy,
    TaskComplexity,
    CollaborationResult
)

# Debate system
from .debate import (
    DebateProtocol,
    DebateResult,
    DebateRound,
    DebateError
)

# Voting system  
from .voting import (
    VotingSystem,
    VotingResult,
    VotingMethod,
    VoteType,
    VotingError
)

# Task queue for async operations
try:
    from .task_queue import AsyncTaskQueue, TaskStatus, QueueError
    _queue_imports = ["AsyncTaskQueue", "TaskStatus", "QueueError"]
except ImportError:
    _queue_imports = []

# Factory function for easy orchestrator creation
def create_orchestrator(claude_agent, openai_agent, gemini_agent, db_manager) -> TaskOrchestrator:
    """Factory function to create a fully configured orchestrator."""
    return TaskOrchestrator(
        claude_agent=claude_agent,
        openai_agent=openai_agent, 
        gemini_agent=gemini_agent,
        db_manager=db_manager
    )

__all__ = [
    # Main orchestrator
    "TaskOrchestrator",
    "CollaborationStrategy", 
    "TaskComplexity",
    "CollaborationResult",
    
    # Debate system
    "DebateProtocol",
    "DebateResult",
    "DebateRound", 
    "DebateError",
    
    # Voting system
    "VotingSystem",
    "VotingResult",
    "VotingMethod",
    "VoteType",
    "VotingError",
    
    # Utilities
    "create_orchestrator"
] + _queue_imports

