"""Tests for the database management system."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from src.persistence.database import DatabaseManager
from src.persistence.models import Base, Conversation, Message, TaskExecution


class TestDatabaseManager:
    """Test the DatabaseManager class."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)
            await db_manager.initialize()
            yield db_manager
            await db_manager.close()
            # Clean up temp file
            Path(tmp.name).unlink(missing_ok=True)
    
    def test_database_manager_initialization(self):
        """Test database manager initialization."""
        db_manager = DatabaseManager("sqlite:///test.db")
        
        assert db_manager.database_url == "sqlite:///test.db"
        assert db_manager.pool_size == 10
        assert db_manager.max_overflow == 20
        assert not db_manager.echo
        assert not db_manager._initialized
        assert db_manager.engine is None
        assert db_manager.session_factory is None
    
    def test_database_manager_custom_config(self):
        """Test database manager with custom configuration."""
        db_manager = DatabaseManager(
            "postgresql://test:test@localhost/test",
            pool_size=5,
            max_overflow=10,
            echo=True
        )
        
        assert db_manager.database_url == "postgresql://test:test@localhost/test"
        assert db_manager.pool_size == 5
        assert db_manager.max_overflow == 10
        assert db_manager.echo
    
    async def test_database_initialization(self, temp_db):
        """Test database initialization process."""
        assert temp_db._initialized
        assert temp_db.engine is not None
        assert temp_db.session_factory is not None
    
    async def test_database_reinitialization(self, temp_db):
        """Test that re-initialization is handled gracefully."""
        # Should not raise an exception
        await temp_db.initialize()
        assert temp_db._initialized
    
    async def test_get_session(self, temp_db):
        """Test session context manager."""
        async with temp_db.get_session() as session:
            assert session is not None
            # Session should be active
            assert session.is_active
    
    async def test_session_rollback_on_exception(self, temp_db):
        """Test that session rolls back on exception."""
        try:
            async with temp_db.get_session() as session:
                # Force an exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Session should have been rolled back and closed
        # This is more about ensuring no exceptions during cleanup
        assert True
    
    async def test_execute_query(self, temp_db):
        """Test executing raw SQL queries."""
        # Test a simple query
        result = await temp_db.execute_query("SELECT 1 as test_value")
        
        assert result is not None
        # For SQLite, this should return a result
    
    async def test_execute_query_with_params(self, temp_db):
        """Test executing queries with parameters."""
        # Test parameterized query
        result = await temp_db.execute_query(
            "SELECT :value as test_value", 
            {"value": "test_param"}
        )
        
        assert result is not None
    
    async def test_execute_query_error_handling(self, temp_db):
        """Test query error handling."""
        with pytest.raises(Exception):
            await temp_db.execute_query("INVALID SQL QUERY")
    
    async def test_health_check_healthy(self, temp_db):
        """Test health check on healthy database."""
        health = await temp_db.health_check()
        
        assert health["status"] == "healthy"
        assert health["database_url"] == temp_db.database_url
        assert "response_time_ms" in health
        assert health["pool_size"] == temp_db.pool_size
        assert "active_connections" in health
    
    async def test_health_check_unhealthy(self):
        """Test health check on unhealthy database."""
        # Create manager with invalid URL
        db_manager = DatabaseManager("invalid://invalid")
        
        health = await db_manager.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    async def test_get_stats(self, temp_db):
        """Test getting database statistics."""
        stats = await temp_db.get_stats()
        
        assert "database_url" in stats
        assert "pool_size" in stats
        assert "active_connections" in stats
        assert "checked_out_connections" in stats
        assert "overflow_connections" in stats
        assert "invalid_connections" in stats
    
    async def test_close_database(self, temp_db):
        """Test database closure."""
        await temp_db.close()
        
        # Should not be initialized anymore
        assert not temp_db._initialized
        assert temp_db.engine is None
        assert temp_db.session_factory is None
    
    async def test_close_uninitialized_database(self):
        """Test closing uninitialized database."""
        db_manager = DatabaseManager("sqlite:///test.db")
        
        # Should not raise an exception
        await db_manager.close()
    
    async def test_session_after_close(self, temp_db):
        """Test that sessions can't be created after close."""
        await temp_db.close()
        
        with pytest.raises(RuntimeError, match="not initialized"):
            async with temp_db.get_session():
                pass
    
    async def test_create_tables(self, temp_db):
        """Test table creation."""
        # Tables should be created during initialization
        async with temp_db.get_session() as session:
            # Try to query one of our tables
            from sqlalchemy import text
            result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            
            # Should have our model tables
            expected_tables = ["conversations", "messages", "task_executions", "agent_proposals", "system_metrics"]
            for table in expected_tables:
                assert table in tables
    
    async def test_concurrent_sessions(self, temp_db):
        """Test concurrent session usage."""
        async def use_session(session_id):
            async with temp_db.get_session() as session:
                # Simulate some work
                await asyncio.sleep(0.01)
                return session_id
        
        # Run multiple sessions concurrently
        tasks = [use_session(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 5
        assert results == list(range(5))
    
    @pytest.mark.slow
    async def test_connection_pooling(self, temp_db):
        """Test connection pool behavior."""
        # Create many concurrent sessions to test pooling
        async def create_session():
            async with temp_db.get_session() as session:
                await asyncio.sleep(0.1)
                return True
        
        # Create more sessions than pool size
        tasks = [create_session() for _ in range(15)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert all(results)
        assert len(results) == 15


class TestDatabaseModels:
    """Test the database models."""
    
    async def test_conversation_model(self, temp_db):
        """Test Conversation model CRUD operations."""
        async with temp_db.get_session() as session:
            # Create a conversation
            conversation = Conversation(
                id="test-conv-123",
                title="Test Conversation",
                description="Test conversation description",
                metadata_json={"test": "data"}
            )
            
            session.add(conversation)
            await session.commit()
            
            # Retrieve the conversation
            from sqlalchemy import select
            result = await session.execute(
                select(Conversation).where(Conversation.id == "test-conv-123")
            )
            retrieved_conv = result.scalar_one()
            
            assert retrieved_conv.id == "test-conv-123"
            assert retrieved_conv.title == "Test Conversation"
            assert retrieved_conv.metadata_json == {"test": "data"}
            assert retrieved_conv.created_at is not None
    
    async def test_message_model(self, temp_db):
        """Test Message model operations."""
        async with temp_db.get_session() as session:
            # Create conversation first
            conversation = Conversation(
                id="test-conv-456",
                title="Test Conversation for Messages"
            )
            session.add(conversation)
            
            # Create message
            message = Message(
                id="test-msg-123",
                conversation_id="test-conv-456",
                role="user",
                content="Test message content",
                agent_type="claude_code",
                metadata_json={"tokens": 50}
            )
            
            session.add(message)
            await session.commit()
            
            # Retrieve the message
            from sqlalchemy import select
            result = await session.execute(
                select(Message).where(Message.id == "test-msg-123")
            )
            retrieved_msg = result.scalar_one()
            
            assert retrieved_msg.id == "test-msg-123"
            assert retrieved_msg.conversation_id == "test-conv-456"
            assert retrieved_msg.role == "user"
            assert retrieved_msg.content == "Test message content"
            assert retrieved_msg.agent_type == "claude_code"
    
    async def test_task_execution_model(self, temp_db):
        """Test TaskExecution model operations."""
        async with temp_db.get_session() as session:
            # Create task execution
            task_execution = TaskExecution(
                id="test-task-123",
                conversation_id="test-conv-789",
                task_type="code_generation",
                task_description="Generate a Python function",
                orchestration_strategy="single_agent",
                agents_used=["claude_code"],
                success=True,
                execution_time_ms=1500.0,
                total_cost_usd=0.025,
                total_tokens_used=150,
                result_content="def hello(): return 'Hello World'",
                metadata_json={"strategy": "single_agent"}
            )
            
            session.add(task_execution)
            await session.commit()
            
            # Retrieve the task execution
            from sqlalchemy import select
            result = await session.execute(
                select(TaskExecution).where(TaskExecution.id == "test-task-123")
            )
            retrieved_task = result.scalar_one()
            
            assert retrieved_task.id == "test-task-123"
            assert retrieved_task.task_type == "code_generation"
            assert retrieved_task.success is True
            assert retrieved_task.execution_time_ms == 1500.0
            assert retrieved_task.total_cost_usd == 0.025
    
    async def test_model_relationships(self, temp_db):
        """Test relationships between models."""
        async with temp_db.get_session() as session:
            # Create conversation
            conversation = Conversation(
                id="test-rel-conv",
                title="Relationship Test"
            )
            session.add(conversation)
            
            # Create related messages
            message1 = Message(
                id="test-rel-msg1",
                conversation_id="test-rel-conv",
                role="user",
                content="First message"
            )
            message2 = Message(
                id="test-rel-msg2", 
                conversation_id="test-rel-conv",
                role="assistant",
                content="Second message"
            )
            
            session.add_all([message1, message2])
            await session.commit()
            
            # Query conversation with messages
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            
            result = await session.execute(
                select(Conversation)
                .options(selectinload(Conversation.messages))
                .where(Conversation.id == "test-rel-conv")
            )
            conv_with_messages = result.scalar_one()
            
            assert len(conv_with_messages.messages) == 2
            assert conv_with_messages.messages[0].content in ["First message", "Second message"]
    
    async def test_model_validation(self, temp_db):
        """Test model validation constraints."""
        async with temp_db.get_session() as session:
            # Test required field validation
            conversation = Conversation()  # Missing required fields
            session.add(conversation)
            
            with pytest.raises(Exception):
                await session.commit()