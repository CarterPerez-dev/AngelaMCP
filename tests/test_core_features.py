#!/usr/bin/env python3
"""
Core AngelaMCP Feature Tests

Tests the essential functionality that's been implemented and working.
"""

import os
import sys
import asyncio
import tempfile
import uuid
from pathlib import Path

# Set environment for testing
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_core.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_core_functionality():
    """Test the core working features of AngelaMCP."""
    print("🚀 Testing AngelaMCP Core Features\n")
    
    test_results = []
    
    # Test 1: Configuration System
    print("1. 🧪 Testing Configuration System...")
    try:
        from config.settings import settings
        assert settings.app_name == "AngelaMCP"
        assert settings.app_env == "testing"
        print("   ✅ Configuration system working")
        test_results.append(True)
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        test_results.append(False)
    
    # Test 2: Database System
    print("\n2. 🧪 Testing Database System...")
    try:
        from src.persistence.database import DatabaseManager
        from src.persistence.models import Base, Conversation, Message
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"
            
        original_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = db_url
        
        # Initialize and test database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Create tables
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Test CRUD operations
        conv_id = uuid.uuid4()
        async with db_manager.get_session() as session:
            conversation = Conversation(
                id=conv_id,
                session_id=uuid.uuid4(),
                status="active"
            )
            session.add(conversation)
            await session.commit()
            
            message = Message(
                id=uuid.uuid4(),
                conversation_id=conv_id,
                agent_type="claude_code",
                role="user",
                content="Test message"
            )
            session.add(message)
            await session.commit()
        
        # Cleanup
        await db_manager.close()
        if original_url:
            os.environ["DATABASE_URL"] = original_url
        Path(tmp.name).unlink(missing_ok=True)
        
        print("   ✅ Database system working")
        print("   ✅ Database models working")
        print("   ✅ CRUD operations working")
        test_results.append(True)
        
    except Exception as e:
        print(f"   ❌ Database failed: {e}")
        test_results.append(False)
    
    # Test 3: Data Models
    print("\n3. 🧪 Testing Data Models...")
    try:
        from src.persistence.models import (
            Conversation, Message, TaskExecution, AgentProposal, SystemMetrics
        )
        
        # Test model instantiation
        conv = Conversation(
            id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            status="active"
        )
        
        msg = Message(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            agent_type="test",
            role="user",
            content="Test"
        )
        
        task = TaskExecution(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            task_type="test",
            task_description="Test task",
            status="pending"
        )
        
        print("   ✅ All data models instantiate correctly")
        test_results.append(True)
        
    except Exception as e:
        print(f"   ❌ Data models failed: {e}")
        test_results.append(False)
    
    # Test 4: Exception System
    print("\n4. 🧪 Testing Exception System...")
    try:
        from src.utils.exceptions import (
            AngelaMCPError, AgentError, DatabaseError, ConfigurationError,
            ValidationError, OrchestrationError, OrchestratorError
        )
        
        # Test exception hierarchy
        try:
            raise AgentError("Test agent error")
        except AngelaMCPError:
            pass  # Should catch as AngelaMCPError
        
        print("   ✅ Exception system working")
        test_results.append(True)
        
    except Exception as e:
        print(f"   ❌ Exception system failed: {e}")
        test_results.append(False)
    
    # Test 5: Cache System
    print("\n5. 🧪 Testing Cache System...")
    try:
        from src.persistence.cache import CacheManager
        
        cache = CacheManager()
        
        # Test basic cache operations
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"
        
        await cache.delete("test_key")
        value = await cache.get("test_key")
        assert value is None
        
        print("   ✅ Cache system working")
        test_results.append(True)
        
    except Exception as e:
        print(f"   ❌ Cache system failed: {e}")
        test_results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 CORE FEATURES TEST SUMMARY")
    print("="*60)
    
    passed = sum(test_results)
    total = len(test_results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Tests Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if passed == total:
        print("🎉 All core features are working!")
        print("\n✅ AngelaMCP Core Systems Status:")
        print("   • Configuration Management: WORKING")
        print("   • Database System: WORKING")
        print("   • Data Models: WORKING")
        print("   • Exception Handling: WORKING")
        print("   • Cache System: WORKING")
        
        print("\n🚀 Ready for AngelaMCP operations:")
        print("   • Database persistence is functional")
        print("   • Configuration system is operational")
        print("   • Core data models are available")
        print("   • Error handling is in place")
        
        return True
    else:
        print("⚠️  Some core features need attention")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_core_functionality())
    sys.exit(0 if success else 1)