"""
Exception classes for AngelaMCP.

Custom exceptions for the multi-agent collaboration platform.
"""


class AngelaMCPError(Exception):
    """Base exception for AngelaMCP platform."""
    pass


class AgentError(AngelaMCPError):
    """Exception for agent-related errors."""
    pass


class OrchestrationError(AngelaMCPError):
    """Exception for orchestration-related errors."""
    pass


class DatabaseError(AngelaMCPError):
    """Exception for database-related errors."""
    pass


class ConfigurationError(AngelaMCPError):
    """Exception for configuration-related errors."""
    pass


class ValidationError(AngelaMCPError):
    """Exception for validation-related errors."""
    pass


class OrchestratorError(AngelaMCPError):
    """Exception for orchestrator-related errors."""
    pass
