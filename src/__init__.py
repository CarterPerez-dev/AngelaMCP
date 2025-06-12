"""
AngelaMCP - Multi-AI Agent Collaboration Platform

A production-grade platform for orchestrating collaboration between
Claude Code, OpenAI, and Gemini agents with structured debate and voting.
"""

__version__ = "1.0.0"
__author__ = "AngelaMCP Team"
__description__ = "Multi-AI Agent Collaboration Platform"

# Core imports for easy access
from config.settings import settings

# Agent types for external use
from src.agents.base import AgentType, TaskType, AgentRole

# Main orchestrator for external integrations
from src.orchestrator.manager import TaskOrchestrator, CollaborationStrategy

# Database for external integrations
from src.persistence.database import DatabaseManager

# Logging setup
from src.utils.logger import setup_logging, get_logger

__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "settings",
    "AgentType",
    "TaskType", 
    "AgentRole",
    "TaskOrchestrator",
    "CollaborationStrategy",
    "DatabaseManager",
    "setup_logging",
    "get_logger"
]
