"""
Database connection and session management for AngelaMCP.

This module provides async database connectivity with connection pooling,
health checks, and proper resource management using SQLAlchemy 2.0.
I'm implementing production-grade database patterns with comprehensive error handling.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text, select, func
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy import event

from config.settings import settings
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class ConnectionPoolError(DatabaseError):
    """Exception for connection pool-related issues."""
    pass


class DatabaseManager:
    """
    Manages async database connections with pooling and health monitoring.
    
    I'm implementing a production-ready database manager that handles connection
    pooling, graceful shutdowns, health checks, and connection recovery.
    """
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_healthy = False
        
    async def initialize(self) -> None:
        """
        Initialize the database engine and connection pool.
        
        I'm setting up the async engine with appropriate pool settings
        based on the environment configuration.
        """
        try:
            # Build connection URL
            database_url = str(settings.database_url)
            
            # Configure pool settings based on environment
            pool_size = settings.database_pool_size
            max_overflow = settings.database_max_overflow
            pool_timeout = settings.database_pool_timeout
            
            # Handle different database types
            if database_url.startswith("sqlite"):
                # Convert to aiosqlite for async support
                if not database_url.startswith("sqlite+aiosqlite"):
                    database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
                
                # Create SQLite async engine
                self._engine = create_async_engine(
                    database_url,
                    echo=settings.database_echo,
                    poolclass=NullPool,  # SQLite doesn't need connection pooling
                )
            else:
                # Create PostgreSQL async engine with connection pooling
                self._engine = create_async_engine(
                    database_url,
                    echo=settings.database_echo,
                    pool_size=pool_size,
                    max_overflow=max_overflow,
                    pool_timeout=pool_timeout,
                    pool_pre_ping=True,  # Verify connections before use
                    pool_recycle=3600,   # Recycle connections every hour
                    poolclass=QueuePool,
                    connect_args={
                        "command_timeout": 60,
                        "server_settings": {
                            "application_name": "AngelaMCP",
                            "jit": "off"  # Disable JIT for better connection performance
                        }
                    }
                )
            
            # Set up connection pool event handlers (skip for SQLite)
            if not database_url.startswith("sqlite"):
                self._setup_pool_events()
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test the connection
            await self._test_connection()
            self._is_healthy = True
            
            # Start health check monitoring
            await self._start_health_monitoring()
            
            logger.info(
                f"Database initialized successfully with pool_size={pool_size}, "
                f"max_overflow={max_overflow}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}") from e
    
    def _setup_pool_events(self) -> None:
        """Set up event handlers for connection pool monitoring."""
        if not self._engine:
            return
            
        @event.listens_for(self._engine.sync_engine.pool, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new connections."""
            logger.debug("New database connection established")
            
        @event.listens_for(self._engine.sync_engine.pool, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            logger.debug("Connection checked out from pool")
            
        @event.listens_for(self._engine.sync_engine.pool, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            """Handle connection checkin to pool."""
            logger.debug("Connection checked back into pool")
            
        @event.listens_for(self._engine.sync_engine.pool, "invalidate")
        def on_invalidate(dbapi_connection, connection_record, exception):
            """Handle connection invalidation."""
            logger.warning(f"Connection invalidated: {exception}")
    
    async def _test_connection(self) -> None:
        """Test database connectivity."""
        if not self._engine:
            raise DatabaseError("Engine not initialized")
            
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()  # fetchone() is not async
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise DatabaseError(f"Connection test failed: {e}") from e
    
    async def _start_health_monitoring(self) -> None:
        """Start background health check monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            return
            
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Database health monitoring started")
    
    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                if self._engine:
                    # Perform health check
                    start_time = asyncio.get_event_loop().time()
                    await self._test_connection()
                    end_time = asyncio.get_event_loop().time()
                    
                    response_time = end_time - start_time
                    self._is_healthy = True
                    
                    logger.debug(f"Health check passed in {response_time:.3f}s")
                    
                    # Log pool status
                    pool = self._engine.pool
                    logger.debug(
                        f"Pool status: size={pool.size()}, "
                        f"checked_in={pool.checkedin()}, "
                        f"checked_out={pool.checkedout()}, "
                        f"overflow={pool.overflow()}"
                    )
                    
            except Exception as e:
                self._is_healthy = False
                logger.error(f"Health check failed: {e}")
                # Continue monitoring even if health check fails
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self._engine:
            raise DatabaseError("Engine not initialized")
            
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}") from e
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        if not self._engine:
            raise DatabaseError("Engine not initialized")
            
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table drop failed: {e}") from e
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with automatic cleanup.
        
        This context manager ensures proper session management with
        automatic rollback on errors and cleanup on exit.
        """
        if not self._session_factory:
            raise DatabaseError("Session factory not initialized")
            
        session = self._session_factory()
        try:
            logger.debug("Database session created")
            yield session
            await session.commit()
            logger.debug("Database session committed")
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session rolled back due to error: {e}")
            raise
        finally:
            await session.close()
            logger.debug("Database session closed")
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a raw SQL query with parameters."""
        async with self.get_session() as session:
            try:
                result = await session.execute(text(query), params or {})
                return result
            except SQLAlchemyError as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseError(f"Query failed: {e}") from e
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get current database connection information."""
        if not self._engine:
            return {"status": "not_initialized"}
            
        pool = self._engine.pool
        return {
            "status": "healthy" if self._is_healthy else "unhealthy",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "engine_url": str(self._engine.url).replace(self._engine.url.password or "", "***")
        }
    
    async def close(self) -> None:
        """Gracefully close all database connections."""
        try:
            # Stop health monitoring
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Close engine
            if self._engine:
                await self._engine.dispose()
                logger.info("Database engine disposed")
                
            self._engine = None
            self._session_factory = None
            self._is_healthy = False
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            raise DatabaseError(f"Database cleanup failed: {e}") from e
    
    @property
    def is_healthy(self) -> bool:
        """Check if the database connection is healthy."""
        return self._is_healthy
    
    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the database engine."""
        return self._engine


# Global database manager instance
db_manager = DatabaseManager()


async def init_database() -> None:
    """Initialize the global database manager."""
    await db_manager.initialize()


async def close_database() -> None:
    """Close the global database manager."""
    await db_manager.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session from the global manager."""
    async with db_manager.get_session() as session:
        yield session


async def health_check() -> Dict[str, Any]:
    """Perform a database health check."""
    try:
        info = await db_manager.get_connection_info()
        
        # Additional health metrics
        if db_manager.engine:
            async with db_manager.get_session() as session:
                # Test query performance
                start_time = asyncio.get_event_loop().time()
                await session.execute(select(func.now()))
                end_time = asyncio.get_event_loop().time()
                
                info["query_response_time_ms"] = (end_time - start_time) * 1000
        
        return info
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
