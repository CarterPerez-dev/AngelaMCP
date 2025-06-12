"""
Orchestration package for AngelaMCP.

This package contains the orchestration engine that manages multi-agent collaboration,
task routing, debate protocols, and consensus building across Claude Code, OpenAI, and Gemini agents.
"""

from .orchestrator import TaskOrchestrator, OrchestrationEngine
from .debate import DebateProtocol, DebateRound, DebateResult
from .voting import VotingSystem, VoteResult, ConsensusBuilder

__all__ = [
    "TaskOrchestrator",
    "OrchestrationEngine", 
    "DebateProtocol",
    "DebateRound",
    "DebateResult",
    "VotingSystem",
    "VoteResult",
    "ConsensusBuilder"
]