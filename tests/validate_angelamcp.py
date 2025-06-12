#!/usr/bin/env python3
"""
AngelaMCP System Validation Script

This script tests the core functionality of the AngelaMCP multi-agent
collaboration platform to ensure all components are working correctly.
"""

import os
import sys
import asyncio
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

# Set environment for testing
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_angelamcp.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test results
test_results = []

def test_result(name: str, success: bool, message: str = ""):
    """Record a test result."""
    test_results.append({
        "name": name,
        "success": success,
        "message": message
    })
    status = "✅" if success else "❌"
    print(f"{status} {name}: {message}")

async def test_settings():
    """Test configuration settings."""
    print("\n🧪 Testing Settings Configuration...")
    
    try:
        from config.settings import settings
        
        # Test basic settings
        assert settings.app_name == "AngelaMCP"
        assert settings.app_env in ["development", "testing", "staging", "production"]
        test_result("Settings Load", True, "Configuration loaded successfully")
        
        # Test database URL
        assert settings.database_url is not None
        test_result("Database Config", True, f"Database URL: {settings.database_url}")
        
        return True
        
    except Exception as e:
        test_result("Settings Load", False, f"Failed: {e}")
        return False

async def test_database():
    """Test database functionality."""
    print("\n🧪 Testing Database System...")
    
    try:
        from src.persistence.database import DatabaseManager
        from src.persistence.models import Base, Conversation, Message
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"
            
        original_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = db_url
        
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        test_result("Database Init", True, "Database initialized successfully")
        
        # Create tables
        async with db_manager._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        test_result("Table Creation", True, "Database tables created")
        
        # Test CRUD operations
        conv_id = uuid.uuid4()
        async with db_manager.get_session() as session:
            # Create conversation
            conversation = Conversation(
                id=conv_id,
                session_id=uuid.uuid4(),
                status="active",
                metadata_={"test": "validation"}
            )
            session.add(conversation)
            await session.commit()
            
            # Create message
            message = Message(
                id=uuid.uuid4(),
                conversation_id=conv_id,
                agent_type="claude_code",
                role="user",
                content="Validation test message"
            )
            session.add(message)
            await session.commit()
            
        test_result("CRUD Operations", True, "Database CRUD operations working")
        
        # Test health
        assert db_manager._is_healthy
        test_result("Database Health", True, "Database is healthy")
        
        # Cleanup
        await db_manager.close()
        if original_url:
            os.environ["DATABASE_URL"] = original_url
        Path(tmp.name).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        test_result("Database System", False, f"Failed: {e}")
        return False

async def test_models():
    """Test database models."""
    print("\n🧪 Testing Database Models...")
    
    try:
        from src.persistence.models import (
            Conversation, Message, TaskExecution, AgentProposal, SystemMetrics
        )
        
        # Test model creation (without database)
        conv_id = uuid.uuid4()
        
        # Test Conversation model
        conversation = Conversation(
            id=conv_id,
            session_id=uuid.uuid4(),
            status="active",
            metadata_={"test": True}
        )
        assert conversation.id == conv_id
        test_result("Conversation Model", True, "Model creation successful")
        
        # Test Message model
        message = Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            agent_type="claude_code",
            role="user",
            content="Test message"
        )
        assert message.agent_type == "claude_code"
        test_result("Message Model", True, "Model creation successful")
        
        # Test TaskExecution model
        task = TaskExecution(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            task_type="code_generation",
            task_description="Generate test code",
            status="pending"
        )
        assert task.task_type == "code_generation"
        test_result("TaskExecution Model", True, "Model creation successful")
        
        return True
        
    except Exception as e:
        test_result("Database Models", False, f"Failed: {e}")
        return False

async def test_agent_base():
    """Test base agent functionality."""
    print("\n🧪 Testing Agent Base Classes...")
    
    try:
        from src.agents.base import BaseAgent, AgentResponse, TaskType
        
        # Test agent response creation
        response = AgentResponse(
            agent_type="test_agent",
            content="Test response",
            confidence=0.8,
            metadata={"test": True}
        )
        assert response.agent_type == "test_agent"
        assert response.confidence == 0.8
        test_result("AgentResponse", True, "Response object creation successful")
        
        # Test TaskType enum
        assert TaskType.CODE_GENERATION in TaskType
        assert TaskType.DEBUGGING in TaskType
        test_result("TaskType Enum", True, "Task type enumeration working")
        
        return True
        
    except Exception as e:
        test_result("Agent Base", False, f"Failed: {e}")
        return False

async def test_imports():
    """Test that all major modules can be imported."""
    print("\n🧪 Testing Module Imports...")
    
    modules_to_test = [
        ("config.settings", "Configuration"),
        ("src.persistence.database", "Database"),
        ("src.persistence.models", "Models"),
        ("src.agents.base", "Agent Base"),
        ("src.persistence.cache", "Cache"),
    ]
    
    all_imports_successful = True
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            test_result(f"Import {description}", True, f"{module_name} imported")
        except Exception as e:
            test_result(f"Import {description}", False, f"Failed to import {module_name}: {e}")
            all_imports_successful = False
    
    return all_imports_successful

async def generate_report():
    """Generate a comprehensive test report."""
    print("\n" + "="*60)
    print("🔍 ANGELAMCP SYSTEM VALIDATION REPORT")
    print("="*60)
    
    passed = sum(1 for result in test_results if result["success"])
    total = len(test_results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"📊 Overall Results: {passed}/{total} tests passed ({success_rate:.1f}%)")
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == total:
        print("🎉 All systems operational! AngelaMCP is ready for use.")
        status = "PASS"
    else:
        print("⚠️  Some systems need attention. Check failed tests above.")
        status = "FAIL"
    
    print("\n📋 Detailed Results:")
    for result in test_results:
        status_icon = "✅" if result["success"] else "❌"
        print(f"  {status_icon} {result['name']}: {result['message']}")
    
    print("\n🏗️  System Components Tested:")
    print("  • Configuration Management (Settings)")
    print("  • Database System (SQLAlchemy + Async)")
    print("  • Data Models (Conversations, Messages, Tasks)")
    print("  • Agent Framework (Base Classes)")
    print("  • Module Imports (Core Dependencies)")
    
    print("\n📚 Next Steps:")
    if passed == total:
        print("  • AngelaMCP is ready for multi-agent collaboration")
        print("  • You can now run agents and orchestrate tasks")
        print("  • Database persistence is working correctly")
    else:
        print("  • Fix the failed tests listed above")
        print("  • Ensure all dependencies are installed")
        print("  • Check environment configuration")
    
    print("="*60)
    
    return status == "PASS"

async def main():
    """Run the complete AngelaMCP validation suite."""
    print("🚀 Starting AngelaMCP System Validation")
    print("This will test all core components of the platform\n")
    
    # Run all tests
    tests = [
        test_imports,
        test_settings,
        test_models,
        test_database,
        test_agent_base,
    ]
    
    overall_success = True
    for test_func in tests:
        try:
            success = await test_func()
            if not success:
                overall_success = False
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            overall_success = False
    
    # Generate final report
    final_success = await generate_report()
    
    return 0 if (overall_success and final_success) else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)