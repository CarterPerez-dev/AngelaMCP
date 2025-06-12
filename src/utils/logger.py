"""
Logging configuration for AngelaMCP.

Complete logging system with structured logging, performance monitoring,
and context tracking for multi-agent collaboration.
"""

import asyncio
import contextlib
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar
from datetime import datetime

from loguru import logger as loguru_logger
from config import settings


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
agent_name_var: ContextVar[Optional[str]] = ContextVar('agent_name', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with context information."""
        
        # Get context information
        request_id = request_id_var.get()
        session_id = session_id_var.get()
        agent_name = agent_name_var.get()
        
        # Add context to record
        if request_id:
            record.request_id = request_id
        if session_id:
            record.session_id = session_id
        if agent_name:
            record.agent_name = agent_name
        
        # Format timestamp
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        return super().format(record)


class AsyncPerformanceLogger:
    """Context manager for performance logging of async operations."""
    
    def __init__(
        self, 
        logger: logging.Logger, 
        operation: str, 
        level: int = logging.INFO,
        **context_data
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.context_data = context_data
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.logger.log(
            self.level,
            f"Started {self.operation}",
            extra={
                "operation": self.operation,
                "event": "start",
                **self.context_data
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End performance monitoring and log results."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            if exc_type is None:
                self.logger.log(
                    self.level,
                    f"Completed {self.operation} in {duration:.3f}s",
                    extra={
                        "operation": self.operation,
                        "event": "complete",
                        "duration_seconds": duration,
                        "success": True,
                        **self.context_data
                    }
                )
            else:
                self.logger.error(
                    f"Failed {self.operation} after {duration:.3f}s: {exc_val}",
                    extra={
                        "operation": self.operation,
                        "event": "error",
                        "duration_seconds": duration,
                        "success": False,
                        "error_type": exc_type.__name__ if exc_type else None,
                        "error_message": str(exc_val) if exc_val else None,
                        **self.context_data
                    }
                )


@contextlib.contextmanager
def log_context(request_id: Optional[str] = None, session_id: Optional[str] = None, agent_name: Optional[str] = None):
    """Context manager for setting logging context variables."""
    
    # Store current values
    current_request_id = request_id_var.get()
    current_session_id = session_id_var.get()
    current_agent_name = agent_name_var.get()
    
    # Set new values
    token_request = request_id_var.set(request_id) if request_id else None
    token_session = session_id_var.set(session_id) if session_id else None
    token_agent = agent_name_var.set(agent_name) if agent_name else None
    
    try:
        yield
    finally:
        # Reset to previous values
        if token_request:
            request_id_var.reset(token_request)
        if token_session:
            session_id_var.reset(token_session)
        if token_agent:
            agent_name_var.reset(token_agent)


def setup_logging() -> None:
    """Set up logging configuration for the application."""
    
    # Ensure log directory exists
    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure loguru for structured logging
    loguru_logger.remove()  # Remove default handler
    
    # Console handler with colors
    if settings.ui_enable_colors:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        console_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    # Add console handler
    loguru_logger.add(
        sys.stdout,
        format=console_format,
        level=settings.log_level.value,
        colorize=settings.ui_enable_colors,
        backtrace=settings.debug,
        diagnose=settings.debug
    )
    
    # Add file handler if configured
    if settings.log_file:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "req_id:{extra[request_id]} | "
            "session_id:{extra[session_id]} | "
            "agent:{extra[agent_name]} | "
            "{message}"
        )
        
        loguru_logger.add(
            str(settings.log_file),
            format=file_format,
            level=settings.log_level.value,
            rotation=settings.log_max_size,
            retention=settings.log_backup_count,
            compression="gz",
            backtrace=settings.debug,
            diagnose=settings.debug,
            enqueue=True,  # Async logging
            serialize=False
        )
    
    # Configure standard library logging to use loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set levels for specific loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING if not settings.database_echo else logging.INFO)


class InterceptHandler(logging.Handler):
    """Intercept standard library logs and redirect to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record through loguru."""
        
        # Get corresponding loguru level
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        # Add context information
        extra = {
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "agent_name": agent_name_var.get()
        }
        
        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(), **extra
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Add custom methods for convenience
    def log_with_context(level: int, msg: str, **context):
        """Log message with additional context."""
        extra = {
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "agent_name": agent_name_var.get(),
            **context
        }
        logger.log(level, msg, extra=extra)
    
    def debug_context(msg: str, **context):
        """Debug log with context."""
        log_with_context(logging.DEBUG, msg, **context)
    
    def info_context(msg: str, **context):
        """Info log with context."""
        log_with_context(logging.INFO, msg, **context)
    
    def warning_context(msg: str, **context):
        """Warning log with context."""
        log_with_context(logging.WARNING, msg, **context)
    
    def error_context(msg: str, **context):
        """Error log with context."""
        log_with_context(logging.ERROR, msg, **context)
    
    # Monkey patch methods onto logger
    logger.debug_context = debug_context
    logger.info_context = info_context
    logger.warning_context = warning_context
    logger.error_context = error_context
    
    return logger


class AgentLogger:
    """Specialized logger for agent operations."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = get_logger(f"agents.{agent_name}")
    
    def log_request(self, message: str, **context):
        """Log an agent request."""
        with log_context(agent_name=self.agent_name):
            self.logger.info(f"[REQUEST] {message}", extra=context)
    
    def log_response(self, message: str, **context):
        """Log an agent response."""
        with log_context(agent_name=self.agent_name):
            self.logger.info(f"[RESPONSE] {message}", extra=context)
    
    def log_error(self, message: str, error: Exception = None, **context):
        """Log an agent error."""
        with log_context(agent_name=self.agent_name):
            self.logger.error(f"[ERROR] {message}", exc_info=error, extra=context)
    
    def log_performance(self, operation: str, duration: float, **context):
        """Log performance metrics."""
        with log_context(agent_name=self.agent_name):
            self.logger.info(
                f"[PERFORMANCE] {operation} completed in {duration:.3f}s",
                extra={"operation": operation, "duration_seconds": duration, **context}
            )


class DebateLogger:
    """Specialized logger for debate operations."""
    
    def __init__(self):
        self.logger = get_logger("orchestrator.debate")
    
    def log_debate_start(self, debate_id: str, topic: str, participants: list):
        """Log start of debate."""
        self.logger.info(
            f"🎭 Starting debate {debate_id[:8]}: {topic}",
            extra={
                "debate_id": debate_id,
                "topic": topic,
                "participants": participants,
                "event": "debate_start"
            }
        )
    
    def log_debate_round(self, debate_id: str, round_num: int, phase: str):
        """Log debate round progress."""
        self.logger.info(
            f"⚡ Debate {debate_id[:8]} Round {round_num}: {phase}",
            extra={
                "debate_id": debate_id,
                "round_number": round_num,
                "phase": phase,
                "event": "debate_round"
            }
        )
    
    def log_debate_end(self, debate_id: str, success: bool, consensus_score: float):
        """Log end of debate."""
        emoji = "🏆" if success else "❌"
        self.logger.info(
            f"{emoji} Debate {debate_id[:8]} {'completed' if success else 'failed'} "
            f"(consensus: {consensus_score:.2f})",
            extra={
                "debate_id": debate_id,
                "success": success,
                "consensus_score": consensus_score,
                "event": "debate_end"
            }
        )


class VotingLogger:
    """Specialized logger for voting operations."""
    
    def __init__(self):
        self.logger = get_logger("orchestrator.voting")
    
    def log_voting_start(self, voting_id: str, proposals_count: int):
        """Log start of voting."""
        self.logger.info(
            f"🗳️ Starting voting {voting_id[:8]} on {proposals_count} proposals",
            extra={
                "voting_id": voting_id,
                "proposals_count": proposals_count,
                "event": "voting_start"
            }
        )
    
    def log_vote_cast(self, voting_id: str, agent: str, vote_type: str, confidence: float):
        """Log individual vote."""
        self.logger.info(
            f"✅ Vote cast by {agent}: {vote_type} (confidence: {confidence:.2f})",
            extra={
                "voting_id": voting_id,
                "agent": agent,
                "vote_type": vote_type,
                "confidence": confidence,
                "event": "vote_cast"
            }
        )
    
    def log_voting_end(self, voting_id: str, winner: Optional[str], consensus_reached: bool):
        """Log end of voting."""
        emoji = "🏆" if winner else "🤷"
        self.logger.info(
            f"{emoji} Voting {voting_id[:8]} completed: Winner is {winner or 'None'} "
            f"(consensus: {'Yes' if consensus_reached else 'No'})",
            extra={
                "voting_id": voting_id,
                "winner": winner,
                "consensus_reached": consensus_reached,
                "event": "voting_end"
            }
        )


# Performance monitoring decorator
def monitor_performance(operation_name: str = None):
    """Decorator for monitoring async function performance."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)
            
            async with AsyncPerformanceLogger(logger, op_name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Convenience functions
def log_agent_interaction(agent_name: str, action: str, details: Dict[str, Any] = None):
    """Log agent interaction with structured data."""
    logger = get_logger("agents.interactions")
    
    with log_context(agent_name=agent_name):
        logger.info(
            f"Agent {agent_name}: {action}",
            extra={
                "agent": agent_name,
                "action": action,
                "details": details or {},
                "event": "agent_interaction"
            }
        )


def log_collaboration_event(event_type: str, details: Dict[str, Any] = None):
    """Log collaboration events."""
    logger = get_logger("orchestrator.collaboration")
    
    logger.info(
        f"Collaboration event: {event_type}",
        extra={
            "event_type": event_type,
            "details": details or {},
            "event": "collaboration"
        }
    )
