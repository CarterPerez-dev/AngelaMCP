#!/usr/bin/env python3
"""
Database initialization script for AngelaMCP.

This creates all database tables and sets up the initial schema.
I'm making this robust with proper error handling and verification.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import OperationalError
import asyncpg
from config.settings import settings
from src.persistence.models import Base
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("db_init")


async def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    try:

        db_name = settings.database_url.path.strip("/")
        
        postgres_db_url = settings.database_url.with_path("/postgres")
        
        logger.info(f"Checking if database '{db_name}' exists...")

        try:
            conn = await asyncpg.connect(str(postgres_db_url))
            
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )

            if not exists:
                logger.info(f"Creating database '{db_name}'...")
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"✅ Database '{db_name}' created successfully")
            else:
                logger.info(f"✅ Database '{db_name}' already exists")

            await conn.close()
        except Exception as e:
            logger.error(f"Failed to create or check database: {e}")
            raise
    except Exception as e:
        logger.error(f"Database creation check failed: {e}")
        pass 


async def create_tables():
    """Create all database tables asynchronously."""
    try:
        logger.info("Creating database tables asynchronously...")

        engine = create_async_engine(str(settings.database_url))

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ All database tables created successfully")

        async with engine.connect() as conn:
            tables_query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
            result = await conn.execute(tables_query)
            tables = result.fetchall()

            logger.info("Verified tables in database:")
            for table in tables:
                logger.info(f"  - {table[0]}")

        await engine.dispose()

    except OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("Make sure PostgreSQL is running in Docker and connection details in .env are correct.")
        raise
    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        raise

async def verify_database_setup():
    """Verify database setup is working correctly."""
    try:
        logger.info("Verifying database setup...")

        from src.persistence.database import DatabaseManager

        db_manager = DatabaseManager()
        await db_manager.initialize()

        async with db_manager.get_session() as session:
            result = await session.execute(text("SELECT current_timestamp"))
            timestamp = result.scalar()
            logger.info(f"✅ Database verification successful - Current time: {timestamp}")

        await db_manager.close()

    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        raise


async def main():
    """Main initialization function."""
    try:
        logger.info("🗄️ Starting AngelaMCP database initialization...")

        await create_database_if_not_exists()
        await create_tables()
        await verify_database_setup()

        logger.info("👻 Database initialization completed successfully!")

        print("\n" + "="*60)
        print("✅ AngelaMCP Database Setup Complete!")
        print("="*60)
        
        print("Database URL:", settings.database_url)
        
        print("Next steps:")
        print("1. Run: make verify")
        print("2. Run: make run")
        print("="*60)


    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print("\n" + "="*60)
        print("❌ Database Setup Failed!")
        print("="*60)
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Is Docker running? (docker ps)")
        print("2. Did 'make docker-up' complete successfully?")
        print("3. Are database credentials in .env correct?")
        print("4. Is your DATABASE_URL in the format: postgresql+asyncpg://... ?")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
