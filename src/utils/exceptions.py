"""
Exception classes for AngelaMCP.

Complete exception hierarchy for the multi-agent collaboration platform.
I'm implementing a comprehensive exception system with proper error categorization,
context tracking, and debugging support.
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    AGENT = "agent"
    ORCHESTRATION = "orchestration"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    API = "api"
    PARSING = "parsing"
    DEBATE = "debate"
    VOTING = "voting"
    MCP = "mcp"
    CLI = "cli"
    SYSTEM = "system"


class AngelaMCPError(Exception):
    """
    Base exception for AngelaMCP platform.
    
    All platform-specific exceptions inherit from this base class.
    Provides consistent error handling with context tracking.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = True,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        self.original_exception = original_exception
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


# Agent-related exceptions
class AgentError(AngelaMCPError):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AGENT)
        if agent_name:
            kwargs.setdefault('context', {})['agent_name'] = agent_name
        super().__init__(message, **kwargs)


class AgentInitializationError(AgentError):
    """Exception raised when agent initialization fails."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)
        kwargs.setdefault('suggestions', [
            "Check API keys and configuration",
            "Verify network connectivity",
            "Review agent-specific setup requirements"
        ])
        super().__init__(message, agent_name, **kwargs)


class AgentCommunicationError(AgentError):
    """Exception raised when agent communication fails."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('suggestions', [
            "Check network connectivity",
            "Verify API endpoints are accessible",
            "Review rate limiting settings"
        ])
        super().__init__(message, agent_name, **kwargs)


class AgentTimeoutError(AgentError):
    """Exception raised when agent operations timeout."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, timeout_seconds: Optional[float] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('suggestions', [
            "Increase timeout settings",
            "Break down complex tasks into smaller parts",
            "Check agent response times"
        ])
        if timeout_seconds:
            kwargs.setdefault('context', {})['timeout_seconds'] = timeout_seconds
        super().__init__(message, agent_name, **kwargs)


class AgentRateLimitError(AgentError):
    """Exception raised when agent rate limits are exceeded."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, retry_after: Optional[float] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('category', ErrorCategory.RATE_LIMIT)
        kwargs.setdefault('suggestions', [
            "Implement exponential backoff",
            "Reduce request frequency",
            "Consider upgrading API plan"
        ])
        if retry_after:
            kwargs.setdefault('context', {})['retry_after_seconds'] = retry_after
        super().__init__(message, agent_name, **kwargs)


class AgentAuthenticationError(AgentError):
    """Exception raised when agent authentication fails."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('recoverable', False)
        kwargs.setdefault('suggestions', [
            "Verify API key is correct and active",
            "Check API key permissions",
            "Ensure API key is not expired"
        ])
        super().__init__(message, agent_name, **kwargs)


# Orchestration-related exceptions
class OrchestrationError(AngelaMCPError):
    """Base exception for orchestration-related errors."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.ORCHESTRATION)
        if task_id:
            kwargs.setdefault('context', {})['task_id'] = task_id
        super().__init__(message, **kwargs)


class TaskExecutionError(OrchestrationError):
    """Exception raised when task execution fails."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, strategy: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Review task requirements and constraints",
            "Try different collaboration strategy",
            "Check agent availability and status"
        ])
        if strategy:
            kwargs.setdefault('context', {})['strategy'] = strategy
        super().__init__(message, task_id, **kwargs)


class StrategySelectionError(OrchestrationError):
    """Exception raised when strategy selection fails."""
    
    def __init__(self, message: str, available_strategies: Optional[List[str]] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Specify strategy explicitly",
            "Review task complexity and requirements",
            "Check agent capabilities"
        ])
        if available_strategies:
            kwargs.setdefault('context', {})['available_strategies'] = available_strategies
        super().__init__(message, **kwargs)


# Debate-related exceptions
class DebateError(AngelaMCPError):
    """Base exception for debate-related errors."""
    
    def __init__(self, message: str, debate_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DEBATE)
        if debate_id:
            kwargs.setdefault('context', {})['debate_id'] = debate_id
        super().__init__(message, **kwargs)


class DebateTimeoutError(DebateError):
    """Exception raised when debate operations timeout."""
    
    def __init__(self, message: str, debate_id: Optional[str] = None, round_number: Optional[int] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('suggestions', [
            "Increase debate timeout settings",
            "Reduce debate round complexity",
            "Check agent response times"
        ])
        if round_number:
            kwargs.setdefault('context', {})['round_number'] = round_number
        super().__init__(message, debate_id, **kwargs)


class ConsensusError(DebateError):
    """Exception raised when consensus cannot be reached."""
    
    def __init__(self, message: str, consensus_score: Optional[float] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Increase maximum debate rounds",
            "Review consensus threshold settings",
            "Consider alternative collaboration strategies"
        ])
        if consensus_score:
            kwargs.setdefault('context', {})['consensus_score'] = consensus_score
        super().__init__(message, **kwargs)


# Voting-related exceptions
class VotingError(AngelaMCPError):
    """Base exception for voting-related errors."""
    
    def __init__(self, message: str, voting_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VOTING)
        if voting_id:
            kwargs.setdefault('context', {})['voting_id'] = voting_id
        super().__init__(message, **kwargs)


class VotingTimeoutError(VotingError):
    """Exception raised when voting operations timeout."""
    
    def __init__(self, message: str, voting_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('suggestions', [
            "Increase voting timeout settings",
            "Simplify voting criteria",
            "Check agent availability"
        ])
        super().__init__(message, voting_id, **kwargs)


class VetoError(VotingError):
    """Exception raised when a proposal is vetoed."""
    
    def __init__(self, message: str, vetoed_by: Optional[str] = None, veto_reason: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('suggestions', [
            "Review and address veto concerns",
            "Modify proposal based on feedback",
            "Consider alternative approaches"
        ])
        if vetoed_by:
            kwargs.setdefault('context', {})['vetoed_by'] = vetoed_by
        if veto_reason:
            kwargs.setdefault('context', {})['veto_reason'] = veto_reason
        super().__init__(message, **kwargs)


# Database-related exceptions
class DatabaseError(AngelaMCPError):
    """Base exception for database-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATABASE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DatabaseConnectionError(DatabaseError):
    """Exception raised when database connection fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('recoverable', False)
        kwargs.setdefault('suggestions', [
            "Check database server status",
            "Verify connection string and credentials",
            "Ensure database service is running"
        ])
        super().__init__(message, **kwargs)


class DatabaseMigrationError(DatabaseError):
    """Exception raised when database migration fails."""
    
    def __init__(self, message: str, migration_version: Optional[str] = None, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recoverable', False)
        kwargs.setdefault('suggestions', [
            "Review migration scripts",
            "Check database permissions",
            "Backup and restore database if needed"
        ])
        if migration_version:
            kwargs.setdefault('context', {})['migration_version'] = migration_version
        super().__init__(message, **kwargs)


# Configuration-related exceptions
class ConfigurationError(AngelaMCPError):
    """Base exception for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)
        if config_key:
            kwargs.setdefault('context', {})['config_key'] = config_key
        super().__init__(message, **kwargs)


class MissingConfigurationError(ConfigurationError):
    """Exception raised when required configuration is missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Check .env file exists and is complete",
            "Verify environment variables are set",
            "Review configuration documentation"
        ])
        super().__init__(message, config_key, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Exception raised when configuration values are invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, expected_type: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Verify configuration value format",
            "Check configuration documentation",
            "Validate against expected data types"
        ])
        if expected_type:
            kwargs.setdefault('context', {})['expected_type'] = expected_type
        super().__init__(message, config_key, **kwargs)


# Validation-related exceptions
class ValidationError(AngelaMCPError):
    """Base exception for validation-related errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        if field_name:
            kwargs.setdefault('context', {})['field_name'] = field_name
        super().__init__(message, **kwargs)


class InputValidationError(ValidationError):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Check input format and requirements",
            "Verify all required fields are provided",
            "Review input validation rules"
        ])
        super().__init__(message, field_name, **kwargs)


class SchemaValidationError(ValidationError):
    """Exception raised when schema validation fails."""
    
    def __init__(self, message: str, schema_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Check data structure matches schema",
            "Verify all required fields are present",
            "Review schema documentation"
        ])
        if schema_name:
            kwargs.setdefault('context', {})['schema_name'] = schema_name
        super().__init__(message, **kwargs)


# MCP-related exceptions
class MCPError(AngelaMCPError):
    """Base exception for MCP-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MCP)
        super().__init__(message, **kwargs)


class MCPServerError(MCPError):
    """Exception raised when MCP server operations fail."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('suggestions', [
            "Check MCP server configuration",
            "Verify Claude Code connection",
            "Review MCP protocol compatibility"
        ])
        super().__init__(message, **kwargs)


class MCPProtocolError(MCPError):
    """Exception raised when MCP protocol violations occur."""
    
    def __init__(self, message: str, protocol_version: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Check MCP protocol version compatibility",
            "Review message format and structure",
            "Verify tool definitions are correct"
        ])
        if protocol_version:
            kwargs.setdefault('context', {})['protocol_version'] = protocol_version
        super().__init__(message, **kwargs)


# CLI-related exceptions
class CLIError(AngelaMCPError):
    """Base exception for CLI-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CLI)
        super().__init__(message, **kwargs)


class CLIInitializationError(CLIError):
    """Exception raised when CLI initialization fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('suggestions', [
            "Check terminal compatibility",
            "Verify required dependencies are installed",
            "Review CLI configuration settings"
        ])
        super().__init__(message, **kwargs)


class CLICommandError(CLIError):
    """Exception raised when CLI command execution fails."""
    
    def __init__(self, message: str, command: Optional[str] = None, **kwargs):
        kwargs.setdefault('suggestions', [
            "Check command syntax and arguments",
            "Review available commands with 'help'",
            "Verify system state and prerequisites"
        ])
        if command:
            kwargs.setdefault('context', {})['command'] = command
        super().__init__(message, **kwargs)


# Network and API-related exceptions
class NetworkError(AngelaMCPError):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('suggestions', [
            "Check internet connectivity",
            "Verify firewall and proxy settings",
            "Review API endpoint accessibility"
        ])
        super().__init__(message, **kwargs)


class APIError(AngelaMCPError):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.API)
        if api_name:
            kwargs.setdefault('context', {})['api_name'] = api_name
        if status_code:
            kwargs.setdefault('context', {})['status_code'] = status_code
        super().__init__(message, **kwargs)


# Parsing-related exceptions
class ParsingError(AngelaMCPError):
    """Exception raised when parsing operations fail."""
    
    def __init__(self, message: str, content_type: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.PARSING)
        kwargs.setdefault('suggestions', [
            "Check content format and structure",
            "Verify content type is supported",
            "Review parsing rules and expectations"
        ])
        if content_type:
            kwargs.setdefault('context', {})['content_type'] = content_type
        super().__init__(message, **kwargs)


class JSONParsingError(ParsingError):
    """Exception raised when JSON parsing fails."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('suggestions', [
            "Verify JSON syntax is valid",
            "Check for missing quotes or brackets",
            "Use JSON validator to identify issues"
        ])
        super().__init__(message, content_type="json", **kwargs)


# Utility functions for exception handling
def wrap_exception(
    original_exception: Exception,
    message: str,
    exception_class: type = AngelaMCPError,
    **kwargs
) -> AngelaMCPError:
    """
    Wrap an existing exception in an AngelaMCP exception.
    
    Preserves the original exception while providing AngelaMCP-specific context.
    """
    kwargs['original_exception'] = original_exception
    return exception_class(message, **kwargs)


def format_exception_for_user(exception: Exception) -> str:
    """
    Format exception message for user display.
    
    Provides user-friendly error messages without technical details.
    """
    if isinstance(exception, AngelaMCPError):
        message = exception.message
        if exception.suggestions:
            suggestions = "\n".join(f"• {suggestion}" for suggestion in exception.suggestions)
            message += f"\n\nSuggestions:\n{suggestions}"
        return message
    else:
        return str(exception)


def is_recoverable_error(exception: Exception) -> bool:
    """
    Check if an exception represents a recoverable error.
    
    Returns True if the operation can be retried or has alternative approaches.
    """
    if isinstance(exception, AngelaMCPError):
        return exception.recoverable
    
    # Common recoverable exception types
    recoverable_types = (
        ConnectionError,
        TimeoutError,
        OSError  # Network-related OS errors
    )
    
    return isinstance(exception, recoverable_types)


def get_error_severity(exception: Exception) -> ErrorSeverity:
    """
    Get the severity level of an exception.
    
    Returns the severity if it's an AngelaMCP exception, otherwise estimates based on type.
    """
    if isinstance(exception, AngelaMCPError):
        return exception.severity
    
    # Estimate severity for standard exceptions
    if isinstance(exception, (KeyboardInterrupt, SystemExit)):
        return ErrorSeverity.CRITICAL
    elif isinstance(exception, (MemoryError, OSError)):
        return ErrorSeverity.HIGH
    elif isinstance(exception, (ValueError, TypeError)):
        return ErrorSeverity.MEDIUM
    else:
        return ErrorSeverity.LOW
