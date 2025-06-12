"""Utility functions and helpers."""

# Logging system
from .logger import (
    setup_logging,
    get_logger,
    log_context,
    AsyncPerformanceLogger,
    AgentLogger,
    DebateLogger,
    VotingLogger,
    monitor_performance,
    log_agent_interaction,
    log_collaboration_event
)

# Exception classes
from .exceptions import (
    AngelaMCPError,
    AgentError,
    OrchestrationError,
    DatabaseError,
    ConfigurationError,
    ValidationError
)

# Helper functions
from .helpers import (
    parse_json_response,
    format_timestamp,
    truncate_text,
    clean_agent_response,
    extract_code_blocks,
    validate_file_path,
    calculate_text_similarity,
    format_cost,
    safe_filename,
    merge_dicts
)

# Logging convenience functions
def setup_app_logging():
    """Setup application logging with default configuration."""
    setup_logging()
    logger = get_logger("app")
    logger.info("AngelaMCP logging initialized")
    return logger

# Error handling utilities
def handle_agent_error(error: Exception, agent_name: str, operation: str) -> AgentError:
    """Convert generic exception to AgentError with context."""
    return AgentError(f"Agent {agent_name} failed during {operation}: {str(error)}")

def handle_orchestration_error(error: Exception, operation: str) -> OrchestrationError:
    """Convert generic exception to OrchestrationError with context."""
    return OrchestrationError(f"Orchestration failed during {operation}: {str(error)}")

__all__ = [
    # Logging
    "setup_logging",
    "get_logger", 
    "log_context",
    "AsyncPerformanceLogger",
    "AgentLogger",
    "DebateLogger",
    "VotingLogger",
    "monitor_performance",
    "log_agent_interaction",
    "log_collaboration_event",
    "setup_app_logging",
    
    # Exceptions
    "AngelaMCPError",
    "AgentError",
    "OrchestrationError", 
    "DatabaseError",
    "ConfigurationError",
    "ValidationError",
    "OrchestratorError",
    "handle_agent_error",
    "handle_orchestration_error",
    
    # Helpers
    "parse_json_response",
    "format_timestamp",
    "truncate_text",
    "clean_agent_response",
    "extract_code_blocks", 
    "validate_file_path",
    "calculate_text_similarity",
    "format_cost",
    "safe_filename",
    "merge_dicts"
]
