#!/usr/bin/env python3
"""Working database tests that match the actual implementation."""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Set environment for testing
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_database_manager():
    """Test DatabaseManager with actual API."""
    print("🧪 Testing DatabaseManager...")
    
    try:
        from src.persistence.database import DatabaseManager
        
        # Create a temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"
            
        # Set the database URL in environment
        original_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = db_url
        
        # Create and initialize manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Test session creation
        async with db_manager.get_session() as session:
            assert session is not None
            assert session.is_active
            print("✅ Session creation works")
        
        # Test health status
        assert db_manager._is_healthy
        print("✅ Health check works")
        
        # Test that we can create tables
        from src.persistence.models import Base
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Table creation works")
        
        # Test database models
        import uuid
        from src.persistence.models import Conversation, Message
        async with db_manager.get_session() as session:
            # Create a conversation
            conv_id = uuid.uuid4()
            conversation = Conversation(
                id=conv_id,
                session_id=uuid.uuid4(),
                status="active",
                metadata_={"test": "data"}
            )
            session.add(conversation)
            await session.commit()
            
            # Create a message
            message = Message(
                id=uuid.uuid4(),
                conversation_id=conv_id,
                agent_type="claude_code",
                role="user",
                content="Test message content",
                metadata_={"tokens": 50}
            )
            session.add(message)
            await session.commit()
            
            print("✅ Database models work")
        
        # Clean up
        await db_manager.close()
        
        # Restore original URL
        if original_url:
            os.environ["DATABASE_URL"] = original_url
        
        # Clean up temp file
        Path(tmp.name).unlink(missing_ok=True)
        
        print("✅ DatabaseManager tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run database tests."""
    print("🚀 Running DatabaseManager Tests\n")
    
    result = await test_database_manager()
    
    if result:
        print("\n🎉 All database tests passed!")
        return 0
    else:
        print("\n💥 Database tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)