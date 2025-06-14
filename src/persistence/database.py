"""
Database connection management for AngelaMCP.

This handles both PostgreSQL and Redis connections with proper async support.
I'm implementing production-grade connection pooling and error handling.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator, Union, List
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy import text, select, func
from sqlalchemy.pool import NullPool

from src.persistence.models import Base, Conversation, Message, TaskExecution
from src.utils import get_logger, monitor_performance
from src.utils import DatabaseError
from config import settings


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
    
    @monitor_performance("database_initialization")
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
            self.logger.info("✅ Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
            await self.close()
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL connection."""
        try:
            self.logger.info("Connecting to PostgreSQL...")
            
            # Create async engine
            self.engine = create_async_engine(
                str(settings.database_url),
                echo=settings.database_echo,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                pool_size=20,
                max_overflow=0,
                poolclass=NullPool if settings.app_env.value == "testing" else None
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self._postgres_healthy = True
            self.logger.info("✅ PostgreSQL connection established")
            
        except Exception as e:
            self.logger.error(f"❌ PostgreSQL initialization failed: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            self.logger.info("Connecting to Redis...")
            
            # Parse Redis URL
            redis_url = str(settings.redis_url)
            
            # Create Redis client
            self.redis_client = redis.from_url(
                redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=settings.redis_decode_responses,
                socket_timeout=settings.redis_socket_timeout,
                socket_connect_timeout=settings.redis_connection_timeout,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self._redis_healthy = True
            self.logger.info("✅ Redis connection established")
            
        except Exception as e:
            self.logger.error(f"❌ Redis initialization failed: {e}")
            # Redis is optional, don't fail if unavailable
            self._redis_healthy = False
            self.redis_client = None
            self.logger.warning("Redis unavailable, continuing without caching")
    
    async def _verify_connections(self) -> None:
        """Verify database connections are working."""
        
        # Verify PostgreSQL
        if self.engine:
            try:
                async with self.engine.connect() as conn:
                    result = await conn.execute(text("SELECT current_timestamp"))
                    timestamp = result.scalar()
                    self.logger.info(f"PostgreSQL verification successful - Current time: {timestamp}")
            except Exception as e:
                self.logger.error(f"PostgreSQL verification failed: {e}")
                raise
        
        # Verify Redis (if available)
        if self.redis_client:
            try:
                pong = await self.redis_client.ping()
                if pong:
                    self.logger.info("Redis verification successful")
            except Exception as e:
                self.logger.warning(f"Redis verification failed: {e}")
                self._redis_healthy = False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session context manager."""
        if not self._initialized or not self.session_factory:
            raise DatabaseError("Database not initialized")
        
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()
    
    @monitor_performance("database_transaction")
    async def execute_transaction(self, operation_func, *args, **kwargs):
        """Execute a database operation within a transaction."""
        async with self.get_session() as session:
            try:
                result = await operation_func(session, *args, **kwargs)
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Transaction failed: {e}")
                raise
    
    # ============================================
    # Conversation Management
    # ============================================
    
    async def create_conversation(
        self, 
        title: str = None, 
        description: str = None,
        strategy: str = None,
        participants: List[str] = None
    ) -> Conversation:
        """Create a new conversation."""
        async with self.get_session() as session:
            conversation = Conversation(
                title=title,
                description=description,
                collaboration_strategy=strategy,
                participants=participants or []
            )
            session.add(conversation)
            await session.commit()
            await session.refresh(conversation)
            return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            return result.scalar_one_or_none()
    
    async def list_conversations(
        self, 
        limit: int = 50, 
        offset: int = 0,
        status: str = None
    ) -> List[Conversation]:
        """List conversations with pagination."""
        async with self.get_session() as session:
            query = select(Conversation)
            
            if status:
                query = query.where(Conversation.status == status)
            
            query = query.order_by(Conversation.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def update_conversation(
        self, 
        conversation_id: str, 
        **updates
    ) -> Optional[Conversation]:
        """Update conversation."""
        async with self.get_session() as session:
            conversation = await session.get(Conversation, conversation_id)
            if conversation:
                for key, value in updates.items():
                    if hasattr(conversation, key):
                        setattr(conversation, key, value)
                await session.commit()
                await session.refresh(conversation)
            return conversation
    
    # ============================================
    # Message Management
    # ============================================
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        agent_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """Add message to conversation."""
        async with self.get_session() as session:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                agent_type=agent_type,
                metadata_json=metadata or {}
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message
    
    async def get_conversation_messages(
        self, 
        conversation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
    
    # ============================================
    # Task Execution Management
    # ============================================
    
    async def save_task_execution(
        self,
        task_id: str,
        task_description: str,
        strategy: str,
        success: bool,
        final_solution: str,
        execution_time_ms: float,
        consensus_score: float = None,
        agent_responses: List[Dict[str, Any]] = None,
        cost_breakdown: Dict[str, float] = None,
        metadata: Dict[str, Any] = None,
        conversation_id: str = None
    ) -> TaskExecution:
        """Save task execution record."""
        async with self.get_session() as session:
            execution = TaskExecution(
                id=task_id,
                task_description=task_description,
                strategy=strategy,
                success=success,
                final_solution=final_solution,
                execution_time_ms=execution_time_ms,
                consensus_score=consensus_score,
                agent_responses=agent_responses or [],
                cost_breakdown=cost_breakdown or {},
                metadata_json=metadata or {},
                conversation_id=conversation_id
            )
            session.add(execution)
            await session.commit()
            await session.refresh(execution)
            return execution
    
    async def get_task_execution(self, task_id: str) -> Optional[TaskExecution]:
        """Get task execution by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                select(TaskExecution).where(TaskExecution.id == task_id)
            )
            return result.scalar_one_or_none()
    
    async def list_task_executions(
        self,
        limit: int = 50,
        offset: int = 0,
        conversation_id: str = None,
        strategy: str = None,
        success: bool = None
    ) -> List[TaskExecution]:
        """List task executions with filters."""
        async with self.get_session() as session:
            query = select(TaskExecution)
            
            if conversation_id:
                query = query.where(TaskExecution.conversation_id == conversation_id)
            if strategy:
                query = query.where(TaskExecution.strategy == strategy)
            if success is not None:
                query = query.where(TaskExecution.success == success)
            
            query = query.order_by(TaskExecution.created_at.desc()).limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    # ============================================
    # Analytics and Metrics
    # ============================================
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        async with self.get_session() as session:
            # Total conversations
            total_result = await session.execute(
                select(func.count(Conversation.id))
            )
            total_conversations = total_result.scalar()
            
            # Active conversations
            active_result = await session.execute(
                select(func.count(Conversation.id))
                .where(Conversation.status == 'active')
            )
            active_conversations = active_result.scalar()
            
            # Messages count
            messages_result = await session.execute(
                select(func.count(Message.id))
            )
            total_messages = messages_result.scalar()
            
            return {
                "total_conversations": total_conversations,
                "active_conversations": active_conversations,
                "total_messages": total_messages
            }
    
    async def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        async with self.get_session() as session:
            # Total executions
            total_result = await session.execute(
                select(func.count(TaskExecution.id))
            )
            total_executions = total_result.scalar()
            
            # Successful executions
            success_result = await session.execute(
                select(func.count(TaskExecution.id))
                .where(TaskExecution.success == True)
            )
            successful_executions = success_result.scalar()
            
            # Average execution time
            avg_time_result = await session.execute(
                select(func.avg(TaskExecution.execution_time_ms))
            )
            avg_execution_time = avg_time_result.scalar() or 0
            
            # Strategy distribution
            strategy_result = await session.execute(
                select(TaskExecution.strategy, func.count(TaskExecution.id))
                .group_by(TaskExecution.strategy)
            )
            strategy_stats = dict(strategy_result.fetchall())
            
            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "success_rate": successful_executions / max(total_executions, 1),
                "avg_execution_time_ms": float(avg_execution_time),
                "strategy_distribution": strategy_stats
            }
    
    # ============================================
    # Redis Operations (Caching/Sessions)
    # ============================================
    
    async def redis_get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self.redis_client:
            return None
        
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            self.logger.warning(f"Redis get failed for key {key}: {e}")
            return None
    
    async def redis_set(
        self, 
        key: str, 
        value: Union[str, Dict, List], 
        expire: int = None
    ) -> bool:
        """Set value in Redis."""
        if not self.redis_client:
            return False
        
        try:
            # Serialize if necessary
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            result = await self.redis_client.set(key, value, ex=expire)
            return bool(result)
        except Exception as e:
            self.logger.warning(f"Redis set failed for key {key}: {e}")
            return False
    
    async def redis_delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return bool(result)
        except Exception as e:
            self.logger.warning(f"Redis delete failed for key {key}: {e}")
            return False
    
    async def redis_exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            self.logger.warning(f"Redis exists check failed for key {key}: {e}")
            return False
    
    # ============================================
    # Session Management
    # ============================================
    
    async def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Save session data."""
        return await self.redis_set(
            f"session:{session_id}",
            session_data,
            expire=settings.session_timeout
        )
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        data = await self.redis_get(f"session:{session_id}")
        if data:
            try:
                return json.loads(data) if isinstance(data, str) else data
            except json.JSONDecodeError:
                return None
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        return await self.redis_delete(f"session:{session_id}")
    
    # ============================================
    # Health Check and Maintenance
    # ============================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all database connections."""
        health_status = {
            "postgres": {
                "status": "unknown",
                "details": {}
            },
            "redis": {
                "status": "unknown", 
                "details": {}
            }
        }
        
        # Check PostgreSQL
        try:
            if self.engine:
                async with self.engine.connect() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    result.scalar()
                    
                    # Get connection pool info
                    pool = self.engine.pool
                    health_status["postgres"] = {
                        "status": "healthy",
                        "details": {
                            "pool_size": pool.size() if hasattr(pool, 'size') else 'unknown',
                            "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else 'unknown',
                            "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else 'unknown'
                        }
                    }
            else:
                health_status["postgres"]["status"] = "not_initialized"
                
        except Exception as e:
            health_status["postgres"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
        
        # Check Redis
        try:
            if self.redis_client:
                pong = await self.redis_client.ping()
                if pong:
                    info = await self.redis_client.info()
                    health_status["redis"] = {
                        "status": "healthy",
                        "details": {
                            "connected_clients": info.get("connected_clients", "unknown"),
                            "used_memory_human": info.get("used_memory_human", "unknown"),
                            "uptime_in_seconds": info.get("uptime_in_seconds", "unknown")
                        }
                    }
                else:
                    health_status["redis"]["status"] = "no_response"
            else:
                health_status["redis"]["status"] = "not_available"
                
        except Exception as e:
            health_status["redis"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
        
        return health_status
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions (Redis handles this automatically, but we can track)."""
        # Redis TTL handles expiration automatically
        # This method could be used for additional cleanup if needed
        return 0
    
    async def optimize_database(self) -> None:
        """Run database optimization tasks."""
        if not self.engine:
            return
        
        try:
            async with self.engine.connect() as conn:
                # Analyze tables for query optimization
                await conn.execute(text("ANALYZE;"))
                self.logger.info("Database optimization completed")
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")
    
    async def close(self) -> None:
        """Close all database connections."""
        self.logger.info("Closing database connections...")
        
        # Close Redis
        if self.redis_client:
            try:
                await self.redis_client.aclose()
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Redis: {e}")
        
        # Close PostgreSQL
        if self.engine:
            try:
                await self.engine.dispose()
                self.logger.info("PostgreSQL connection closed")
            except Exception as e:
                self.logger.error(f"Error closing PostgreSQL: {e}")
        
        self._initialized = False
        self._postgres_healthy = False
        self._redis_healthy = False
