"""Persistence layer for AngelaMCP."""

# Database manager
from .database import DatabaseManager

# Database models
from .models import (
    Base,
    Conversation,
    Message, 
    TaskExecution,
)

# Database utilities
def create_database_manager() -> DatabaseManager:
    """Factory function to create database manager."""
    return DatabaseManager()

async def init_database() -> DatabaseManager:
    """Initialize database and return manager."""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    return db_manager

__all__ = [
    # Database manager
    "DatabaseManager",
    
    # Models
    "Base",
    "Conversation",
    "Message",
    "TaskExecution",
    
    # Utilities
    "create_database_manager",
    "init_database"
]
