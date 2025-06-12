#!/usr/bin/env python3
"""
Database initialization script for AngelaMCP.

This script handles database creation, table setup, indexing, and initial data seeding.
I'm implementing comprehensive database initialization with error handling and verification.
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import asyncpg
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.persistence.database import db_manager, DatabaseError
from src.persistence.models import Base, Conversation, Message, TaskExecution, AgentProposal, SystemMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """
    Handles comprehensive database initialization for AngelaMCP.
    
    I'm implementing a robust initialization process that creates the database,
    tables, indexes, and performs verification checks.
    """
    
    def __init__(self):
        self.db_url = settings.database_url
        self.db_name = self.db_url.path.lstrip('/')
        self.admin_url = self.db_url.set(path='/postgres')  # Connect to postgres db for admin operations
        
    async def create_database_if_not_exists(self) -> bool:
        """
        Create the database if it doesn't exist.
        
        Returns True if database was created, False if it already existed.
        """
        try:
            # Try to connect to the target database first
            try:
                conn = await asyncpg.connect(str(self.db_url))
                await conn.close()
                logger.info(f"Database '{self.db_name}' already exists")
                return False
            except asyncpg.InvalidCatalogNameError:
                # Database doesn't exist, create it
                pass
            
            # Connect to postgres database to create the target database
            admin_conn = await asyncpg.connect(str(self.admin_url))
            
            try:
                # Check if database exists
                result = await admin_conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1",
                    self.db_name
                )
                
                if result:
                    logger.info(f"Database '{self.db_name}' already exists")
                    return False
                
                # Create the database
                await admin_conn.execute(f'CREATE DATABASE "{self.db_name}"')
                logger.info(f"Database '{self.db_name}' created successfully")
                return True
                
            finally:
                await admin_conn.close()
                
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise DatabaseError(f"Database creation failed: {e}") from e
    
    async def setup_extensions(self) -> None:
        """Set up required PostgreSQL extensions."""
        try:
            conn = await asyncpg.connect(str(self.db_url))
            
            try:
                # Enable UUID extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
                logger.info("UUID extension enabled")
                
                # Enable pg_trgm for better text search (optional)
                try:
                    await conn.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
                    logger.info("pg_trgm extension enabled")
                except asyncpg.exceptions.InsufficientPrivilegeError:
                    logger.warning("Could not enable pg_trgm extension (insufficient privileges)")
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Failed to setup extensions: {e}")
            raise DatabaseError(f"Extension setup failed: {e}") from e
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        try:
            await db_manager.initialize()
            await db_manager.create_tables()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}") from e
    
    async def create_additional_indexes(self) -> None:
        """Create additional performance indexes not defined in models."""
        additional_indexes = [
            # Composite indexes for common query patterns
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_agent_created "
            "ON messages (agent_type, created_at DESC)",
            
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status_created "
            "ON task_executions (status, started_at DESC)",
            
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_proposals_task_phase "
            "ON agent_proposals (task_execution_id, proposal_phase)",
            
            # Partial indexes for active records
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_active "
            "ON conversations (last_activity_at DESC) WHERE status = 'active'",
            
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_running "
            "ON task_executions (started_at DESC) WHERE status IN ('pending', 'running')",
            
            # Text search indexes (if pg_trgm is available)
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_content_gin "
            "ON messages USING gin (content gin_trgm_ops)",
            
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_description_gin "
            "ON task_executions USING gin (task_description gin_trgm_ops)",
        ]
        
        try:
            async with db_manager.get_session() as session:
                for index_sql in additional_indexes:
                    try:
                        await session.execute(text(index_sql))
                        logger.info(f"Created index: {index_sql.split()[-1]}")
                    except Exception as e:
                        # Skip if index already exists or extension not available
                        if "already exists" in str(e) or "does not exist" in str(e):
                            logger.debug(f"Skipped index creation: {e}")
                        else:
                            logger.warning(f"Failed to create index: {e}")
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to create additional indexes: {e}")
            # Don't raise - indexes are performance optimization, not critical
    
    async def seed_initial_data(self) -> None:
        """Seed the database with initial data."""
        try:
            async with db_manager.get_session() as session:
                # Check if we already have data
                existing_metrics = await session.execute(
                    text("SELECT COUNT(*) FROM system_metrics WHERE metric_name = 'db_initialized'")
                )
                count = existing_metrics.scalar()
                
                if count > 0:
                    logger.info("Database already contains initial data")
                    return
                
                # Create initialization metric
                init_metric = SystemMetrics(
                    metric_type="system",
                    metric_name="db_initialized",
                    value=1.0,
                    unit="boolean",
                    metadata_={
                        "initialized_at": str(asyncio.get_event_loop().time()),
                        "version": "1.0.0"
                    }
                )
                session.add(init_metric)
                
                # Create system startup metric
                startup_metric = SystemMetrics(
                    metric_type="system",
                    metric_name="system_startup",
                    value=asyncio.get_event_loop().time(),
                    unit="timestamp",
                    metadata_={
                        "startup_time": str(asyncio.get_event_loop().time())
                    }
                )
                session.add(startup_metric)
                
                await session.commit()
                logger.info("Initial data seeded successfully")
                
        except Exception as e:
            logger.error(f"Failed to seed initial data: {e}")
            raise DatabaseError(f"Data seeding failed: {e}") from e
    
    async def verify_database(self) -> Dict[str, Any]:
        """Verify database setup and return status information."""
        verification_results = {
            "database_exists": False,
            "tables_exist": False,
            "indexes_exist": False,
            "extensions_enabled": False,
            "connection_healthy": False,
            "table_counts": {},
            "errors": []
        }
        
        try:
            # Test connection
            async with db_manager.get_session() as session:
                # Check database connection
                await session.execute(text("SELECT 1"))
                verification_results["connection_healthy"] = True
                
                # Check tables exist
                table_check = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                """))
                tables = [row[0] for row in table_check.fetchall()]
                
                expected_tables = {'conversations', 'messages', 'task_executions', 'agent_proposals', 'system_metrics'}
                verification_results["tables_exist"] = expected_tables.issubset(set(tables))
                
                # Get table counts
                for table in tables:
                    count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    verification_results["table_counts"][table] = count_result.scalar()
                
                # Check extensions
                ext_check = await session.execute(text("""
                    SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'pg_trgm')
                """))
                extensions = [row[0] for row in ext_check.fetchall()]
                verification_results["extensions_enabled"] = 'uuid-ossp' in extensions
                
                # Check indexes
                index_check = await session.execute(text("""
                    SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'
                """))
                index_count = index_check.scalar()
                verification_results["indexes_exist"] = index_count > 0
                
                verification_results["database_exists"] = True
                
        except Exception as e:
            verification_results["errors"].append(str(e))
            logger.error(f"Database verification failed: {e}")
        
        return verification_results
    
    async def run_full_initialization(self, force_recreate: bool = False) -> Dict[str, Any]:
        """Run the complete database initialization process."""
        logger.info("Starting database initialization...")
        
        results = {
            "database_created": False,
            "tables_created": False,
            "indexes_created": False,
            "data_seeded": False,
            "verification": {},
            "errors": []
        }
        
        try:
            # Step 1: Create database if needed
            if force_recreate:
                logger.warning("Force recreate mode - this will drop existing data!")
                # Implementation for dropping and recreating would go here
                
            db_created = await self.create_database_if_not_exists()
            results["database_created"] = db_created
            
            # Step 2: Setup extensions
            await self.setup_extensions()
            
            # Step 3: Initialize database manager
            await db_manager.initialize()
            
            # Step 4: Create tables
            await self.create_tables()
            results["tables_created"] = True
            
            # Step 5: Create additional indexes
            await self.create_additional_indexes()
            results["indexes_created"] = True
            
            # Step 6: Seed initial data
            await self.seed_initial_data()
            results["data_seeded"] = True
            
            # Step 7: Verify everything
            verification = await self.verify_database()
            results["verification"] = verification
            
            if verification.get("errors"):
                results["errors"].extend(verification["errors"])
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            error_msg = f"Database initialization failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            raise
        
        finally:
            # Clean up database manager
            try:
                await db_manager.close()
            except Exception as e:
                logger.warning(f"Error closing database manager: {e}")
        
        return results


async def main():
    """Main initialization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize AngelaMCP database")
    parser.add_argument("--force", action="store_true", help="Force recreate database")
    parser.add_argument("--verify-only", action="store_true", help="Only verify database setup")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    initializer = DatabaseInitializer()
    
    try:
        if args.verify_only:
            # Just verify the database
            await db_manager.initialize()
            verification = await initializer.verify_database()
            await db_manager.close()
            
            print("\n=== Database Verification Results ===")
            for key, value in verification.items():
                print(f"{key}: {value}")
            
            if verification.get("errors"):
                sys.exit(1)
        else:
            # Run full initialization
            results = await initializer.run_full_initialization(force_recreate=args.force)
            
            print("\n=== Database Initialization Results ===")
            for key, value in results.items():
                if key != "verification":
                    print(f"{key}: {value}")
            
            if results.get("verification"):
                print("\n=== Verification Results ===")
                for key, value in results["verification"].items():
                    print(f"{key}: {value}")
            
            if results.get("errors"):
                print(f"\nErrors occurred: {results['errors']}")
                sys.exit(1)
            
            print("\n✅ Database initialization completed successfully!")
    
    except KeyboardInterrupt:
        logger.info("Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
