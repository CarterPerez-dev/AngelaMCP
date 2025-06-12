"""
Logging configuration for AngelaMCP.

I'm setting up structured logging with proper formatting and file rotation.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import asyncio
from datetime import datetime

from config.settings import settings


def setup_logging() -> None:
    """Setup application-wide logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(levelname)-8s | %(name)s | %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "angelamcp.log",
        maxBytes=settings.log_max_size,
        backupCount=settings.log_backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "errors.log",
        maxBytes=settings.log_max_size,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger("setup")
    logger.info(f"Logging initialized - Level: {settings.log_level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


class AsyncPerformanceLogger:
    """Logger for tracking async operation performance."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
        self.start_time: Optional[float] = None
    
    async def __aenter__(self):
        self.start_time = asyncio.get_event_loop().time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = asyncio.get_event_loop().time() - self.start_time
            self.logger.debug(f"Operation completed in {duration:.3f}s")


def log_context(operation: str):
    """Decorator to log function entry/exit with context."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"Starting {operation}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Completed {operation}")
                return result
            except Exception as e:
                logger.error(f"Failed {operation}: {e}")
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"Starting {operation}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {operation}")
                return result
            except Exception as e:
                logger.error(f"Failed {operation}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
