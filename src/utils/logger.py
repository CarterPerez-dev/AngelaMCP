"""
Structured logging system for AngelaMCP.

This module provides comprehensive logging capabilities with structured output,
file rotation, correlation IDs, and performance tracking. I'm implementing
production-grade logging suitable for debugging multi-agent interactions.
"""

import asyncio
import json
import logging
import logging.handlers
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar

from config.settings import settings

# Context variable for tracking correlation IDs across async operations
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
agent_name: ContextVar[Optional[str]] = ContextVar('agent_name', default=None)


class CorrelationFilter(logging.Filter):
    """
    Logging filter that adds correlation ID, session ID, and agent name to log records.
    
    I'm implementing contextual logging to track requests and agent operations
    across the entire system for better debugging and monitoring.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation context to the log record."""
        record.correlation_id = correlation_id.get()
        record.session_id = session_id.get()
        record.agent_name = agent_name.get()
        record.timestamp_ms = int(time.time() * 1000)
        return True


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    I'm creating structured logs that can be easily parsed by log aggregation
    systems and provide rich context for debugging agent interactions.
    """
    
    def __init__(self):
        super().__init__()
        self.hostname = self._get_hostname()
    
    def _get_hostname(self) -> str:
        """Get the hostname for log context."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "timestamp_ms": getattr(record, "timestamp_ms", int(time.time() * 1000)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add correlation context if available
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id
        
        if hasattr(record, "session_id") and record.session_id:
            log_entry["session_id"] = record.session_id
            
        if hasattr(record, "agent_name") and record.agent_name:
            log_entry["agent_name"] = record.agent_name
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add any extra fields from the log call
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'exc_info', 'exc_text',
                'stack_info', 'correlation_id', 'session_id', 'agent_name',
                'timestamp_ms'
            }:
                # Only include serializable values
                try:
                    json.dumps(value)
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for development environments.
    
    I'm providing readable colored output for development while maintaining
    structured information for debugging.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Build context information
        context_parts = []
        
        if hasattr(record, "correlation_id") and record.correlation_id:
            context_parts.append(f"corr:{record.correlation_id[:8]}")
        
        if hasattr(record, "session_id") and record.session_id:
            context_parts.append(f"sess:{record.session_id[:8]}")
            
        if hasattr(record, "agent_name") and record.agent_name:
            context_parts.append(f"agent:{record.agent_name}")
        
        context_str = f"[{', '.join(context_parts)}] " if context_parts else ""
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        # Format message
        formatted = (
            f"{color}{timestamp}{reset} "
            f"{color}{record.levelname:8}{reset} "
            f"{context_str}"
            f"{record.name}: {record.getMessage()}"
        )
        
        # Add exception information if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class PerformanceLogger:
    """
    Performance logging utility for tracking operation durations.
    
    I'm implementing performance tracking to monitor agent response times,
    database operations, and API calls for optimization insights.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"Starting operation: {self.operation}",
            extra={
                "operation": self.operation,
                "operation_status": "started",
                **self.context
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            # Success
            self.logger.info(
                f"Completed operation: {self.operation} in {duration:.3f}s",
                extra={
                    "operation": self.operation,
                    "operation_status": "completed",
                    "duration_seconds": duration,
                    **self.context
                }
            )
        else:
            # Error
            self.logger.error(
                f"Failed operation: {self.operation} after {duration:.3f}s: {exc_val}",
                extra={
                    "operation": self.operation,
                    "operation_status": "failed",
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__ if exc_type else None,
                    "error_message": str(exc_val) if exc_val else None,
                    **self.context
                },
                exc_info=True
            )


class AsyncPerformanceLogger:
    """
    Async version of PerformanceLogger for async operations.
    
    I'm providing async context manager support for tracking async operations
    like API calls and database queries.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.end_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"Starting async operation: {self.operation}",
            extra={
                "operation": self.operation,
                "operation_status": "started",
                **self.context
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            # Success
            self.logger.info(
                f"Completed async operation: {self.operation} in {duration:.3f}s",
                extra={
                    "operation": self.operation,
                    "operation_status": "completed",
                    "duration_seconds": duration,
                    **self.context
                }
            )
        else:
            # Error
            self.logger.error(
                f"Failed async operation: {self.operation} after {duration:.3f}s: {exc_val}",
                extra={
                    "operation": self.operation,
                    "operation_status": "failed",
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__ if exc_type else None,
                    "error_message": str(exc_val) if exc_val else None,
                    **self.context
                },
                exc_info=True
            )


@contextmanager
def log_context(corr_id: Optional[str] = None, sess_id: Optional[str] = None, 
                agent: Optional[str] = None):
    """
    Context manager for setting logging context variables.
    
    This allows tracking operations across function calls and async boundaries
    by setting correlation IDs, session IDs, and agent names.
    """
    # Generate correlation ID if not provided
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    
    # Store previous values
    prev_corr_id = correlation_id.get()
    prev_sess_id = session_id.get()
    prev_agent = agent_name.get()
    
    # Set new values
    correlation_id.set(corr_id)
    if sess_id is not None:
        session_id.set(sess_id)
    if agent is not None:
        agent_name.set(agent)
    
    try:
        yield corr_id
    finally:
        # Restore previous values
        correlation_id.set(prev_corr_id)
        session_id.set(prev_sess_id)
        agent_name.set(prev_agent)


def setup_logging(
    log_file: Optional[Path] = None,
    log_level: Union[str, int] = logging.INFO,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = True,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging configuration for AngelaMCP.
    
    I'm configuring both file and console logging with structured output,
    file rotation, and correlation tracking for production deployment.
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add correlation filter to all handlers
    correlation_filter = CorrelationFilter()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ColoredConsoleFormatter())
        
        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)
    
    # Set up specific logger levels for external libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('google').setLevel(logging.INFO)
    
    # Create application logger
    app_logger = logging.getLogger('angelamcp')
    app_logger.info(
        "Logging system initialized",
        extra={
            "log_file": str(log_file) if log_file else None,
            "log_level": logging.getLevelName(log_level),
            "json_format": json_format,
            "console_output": console_output,
            "max_file_size": max_file_size,
            "backup_count": backup_count
        }
    )
    
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f"angelamcp.{name}")


def log_agent_interaction(
    logger: logging.Logger,
    agent_name: str,
    operation: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log agent interactions with structured data.
    
    I'm creating a standardized way to log agent operations for debugging
    multi-agent collaboration and tracking agent performance.
    """
    log_data = {
        "agent_name": agent_name,
        "operation": operation,
        "interaction_type": "agent_operation"
    }
    
    if input_data is not None:
        # Truncate large inputs for logging
        if isinstance(input_data, str) and len(input_data) > 1000:
            log_data["input_preview"] = input_data[:1000] + "..."
            log_data["input_length"] = len(input_data)
        else:
            log_data["input"] = input_data
    
    if output_data is not None:
        # Truncate large outputs for logging
        if isinstance(output_data, str) and len(output_data) > 1000:
            log_data["output_preview"] = output_data[:1000] + "..."
            log_data["output_length"] = len(output_data)
        else:
            log_data["output"] = output_data
    
    if metadata:
        log_data["metadata"] = metadata
    
    logger.info(
        f"Agent interaction: {agent_name} - {operation}",
        extra=log_data
    )


def log_performance_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    unit: str,
    tags: Optional[Dict[str, str]] = None
):
    """
    Log performance metrics in a structured format.
    
    I'm providing a way to track system performance metrics that can be
    easily aggregated and monitored in production.
    """
    metric_data = {
        "metric_name": metric_name,
        "metric_value": value,
        "metric_unit": unit,
        "metric_type": "performance"
    }
    
    if tags:
        metric_data["tags"] = tags
    
    logger.info(
        f"Performance metric: {metric_name} = {value} {unit}",
        extra=metric_data
    )


# Initialize logging if settings are available
def init_logging() -> logging.Logger:
    """Initialize logging based on application settings."""
    log_file = None
    if settings.log_file:
        log_file = Path(settings.log_file)
    
    return setup_logging(
        log_file=log_file,
        log_level=getattr(logging, settings.log_level.upper()),
        max_file_size=settings.log_max_size,
        backup_count=settings.log_backup_count,
        json_format=not settings.debug,  # Use colored format in debug mode
        console_output=True
    )
