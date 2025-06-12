"""
Database connection management for AngelaMCP.

This handles both PostgreSQL and Redis connections with proper async support.
I'm implementing production-grade connection pooling and error handling.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from src.persistence.models import Base
from src.utils.logger import get_logger
from src.utils.exceptions import DatabaseError
from config.settings import settings


class DatabaseManager:
    """
    Manages database connections and sessions.
    
    Handles both PostgreSQL (main storage) and Redis (caching/sessions).
    I'm implementing proper connection pooling and error handling.
    """
    
    def __init__(self):
        self.logger = get_logger("database.manager")
        
        # PostgreSQL
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        
        # Redis
        self.redis_client: Optional[redis.Redis] = None
        
        # Connection status
        self._initialized = False
        self._postgres_healthy = False
        self._redis_healthy = False
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        if self._initialized:
            self.logger.warning("Database manager already initialized")
            return
        
        try:
            self.logger.info("Initializing database connections...")
            
            # Initialize PostgreSQL
            await self._initialize_postgres()
            
            # Initialize Redis
            await self._initialize_redis()
            
            # Verify connections
            await self._verify_connections()
            
            self._initialized = True
            self.logger.info("Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}", exc_info=True)
            await self.close()
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL connection."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                str(settings.database_url).replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_timeout=settings.database_pool_timeout,
                echo=settings.database_echo,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections every hour
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.logger.info("PostgreSQL engine initialized")
            
        except Exception as e:
            raise DatabaseError(f"Failed to initialize PostgreSQL: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                str(settings.redis_url),
                max_connections=settings.redis_max_connections,
                decode_responses=settings.redis_decode_responses,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_connection_timeout,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.logger.info("Redis client initialized")
            
        except Exception as e:
            raise DatabaseError(f"Failed to initialize Redis: {e}")
    
    async def _verify_connections(self) -> None:
        """Verify database connections are working."""
        # Test PostgreSQL
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
            self._postgres_healthy = True
            self.logger.info("✅ PostgreSQL connection verified")
        except Exception as e:
            self._postgres_healthy = False
            raise DatabaseError(f"PostgreSQL connection verification failed: {e}")
        
        # Test Redis
        try:
            await self.redis_client.ping()
            self._redis_healthy = True
            self.logger.info("✅ Redis connection verified")
        except Exception as e:
            self._redis_healthy = False
            raise DatabaseError(f"Redis connection verification failed: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup."""
        if not self._initialized or not self.session_factory:
            raise DatabaseError("Database not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            await session.close()
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client."""
        if not self._initialized or not self.redis_client:
            raise DatabaseError("Redis not initialized")
        
        return self.redis_client
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")
        
        try:
            self.logger.info("Creating database tables...")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("✅ Database tables created")
        except Exception as e:
            raise DatabaseError(f"Failed to create tables: {e}")
    
    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")
        
        try:
            self.logger.warning("Dropping all database tables...")
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            self.logger.info("Database tables dropped")
        except Exception as e:
            raise DatabaseError(f"Failed to drop tables: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all database connections."""
        health_status = {
            "overall": "healthy",
            "postgres": {"status": "unknown"},
            "redis": {"status": "unknown"}
        }
        
        # Check PostgreSQL
        try:
            if self.engine:
                async with self.engine.begin() as conn:
                    start_time = asyncio.get_event_loop().time()
                    result = await conn.execute(text("SELECT version()"))
                    response_time = asyncio.get_event_loop().time() - start_time
                    
                    health_status["postgres"] = {
                        "status": "healthy",
                        "response_time": response_time,
                        "version": result.scalar()[:50] + "...",
                        "pool_size": self.engine.pool.size(),
                        "checked_out": self.engine.pool.checkedout()
                    }
                    self._postgres_healthy = True
            else:
                health_status["postgres"] = {"status": "not_initialized"}
        except Exception as e:
            health_status["postgres"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            self._postgres_healthy = False
        
        # Check Redis
        try:
            if self.redis_client:
                start_time = asyncio.get_event_loop().time()
                await self.redis_client.ping()
                response_time = asyncio.get_event_loop().time() - start_time
                
                info = await self.redis_client.info()
                health_status["redis"] = {
                    "status": "healthy",
                    "response_time": response_time,
                    "version": info.get("redis_version", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory_human", "unknown")
                }
                self._redis_healthy = True
            else:
                health_status["redis"] = {"status": "not_initialized"}
        except Exception as e:
            health_status["redis"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            self._redis_healthy = False
        
        # Overall status
        if not self._postgres_healthy or not self._redis_healthy:
            health_status["overall"] = "degraded"
        
        if not self._postgres_healthy and not self._redis_healthy:
            health_status["overall"] = "unhealthy"
        
        return health_status
    
    async def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a raw SQL query."""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(query), params or {})
                return result
        except SQLAlchemyError as e:
            raise DatabaseError(f"Query execution failed: {e}")
    
    async def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about database tables."""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")
        
        try:
            stats = {}
            async with self.engine.begin() as conn:
                # Get table row counts
                tables = ["conversations", "messages", "task_executions", "debate_rounds", "agent_responses"]
                
                for table in tables:
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[table] = result.scalar()
                
                # Get database size
                result = await conn.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))"))
                stats["database_size"] = result.scalar()
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get table stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days_old: int = 30) -> Dict[str, int]:
        """Clean up old data from the database."""
        if not self.engine:
            raise DatabaseError("Database engine not initialized")
        
        try:
            cleanup_stats = {}
            
            async with self.get_session() as session:
                # Clean up old conversations
                result = await session.execute(text("""
                    DELETE FROM conversations 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                    AND status = 'completed'
                """), {"days": days_old})
                cleanup_stats["conversations_deleted"] = result.rowcount
                
                # Clean up old metrics
                result = await session.execute(text("""
                    DELETE FROM session_metrics 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                """), {"days": days_old})
                cleanup_stats["metrics_deleted"] = result.rowcount
            
            self.logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            raise DatabaseError(f"Data cleanup failed: {e}")
    
    async def close(self) -> None:
        """Close all database connections."""
        try:
            # Close PostgreSQL
            if self.engine:
                await self.engine.dispose()
                self.engine = None
                self.session_factory = None
                self.logger.info("PostgreSQL connections closed")
            
            # Close Redis
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
                self.logger.info("Redis connections closed")
            
            self._initialized = False
            self._postgres_healthy = False
            self._redis_healthy = False
            
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
    
    def is_healthy(self) -> bool:
        """Check if database connections are healthy."""
        return self._initialized and self._postgres_healthy and self._redis_healthy
    
    @property
    def postgres_healthy(self) -> bool:
        """Check if PostgreSQL is healthy."""
        return self._postgres_healthy
    
    @property
    def redis_healthy(self) -> bool:
        """Check if Redis is healthy."""
        return self._redis_healthy
