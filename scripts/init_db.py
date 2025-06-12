#!/usr/bin/env python3
"""
Database initialization script for AngelaMCP.
Creates database tables and sets up initial data.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.persistence.database import DatabaseManager
from src.logging_config import setup_logging


async def init_database():
    """Initialize the database with all tables."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing AngelaMCP database...")
        
        # Create database manager
        db_manager = DatabaseManager()
        
        # Initialize database and create tables
        await db_manager.initialize()
        await db_manager.create_tables()
        
        # Optional: Create initial data
        await create_initial_data(db_manager)
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        if 'db_manager' in locals():
            await db_manager.close()


async def create_initial_data(db_manager: DatabaseManager):
    """Create initial data for the application."""
    logger = logging.getLogger(__name__)
    
    try:
        # You can add initial data here if needed
        # For example, default configuration, admin users, etc.
        logger.info("Initial data creation completed")
        
    except Exception as e:
        logger.error(f"Failed to create initial data: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database())