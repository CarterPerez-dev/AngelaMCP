#!/usr/bin/env python3
"""Simple test runner to validate core functionality."""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Set environment for testing
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_database_basic():
    """Test basic database functionality."""
    print("🧪 Testing database functionality...")
    
    try:
        from src.persistence.database import DatabaseManager
        
        # Create a temporary database and set URL in environment
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"
            os.environ["DATABASE_URL"] = db_url
            
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Test basic operations
        async with db_manager.get_session() as session:
            assert session is not None
            print("✅ Database session created successfully")
        
        # Test health check
        assert db_manager._is_healthy
        print("✅ Database health check passed")
        
        await db_manager.close()
        
        # Clean up
        Path(tmp.name).unlink(missing_ok=True)
        
        print("✅ Database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_settings():
    """Test settings loading."""
    print("🧪 Testing settings...")
    
    try:
        from config.settings import settings
        
        assert settings.app_name == "AngelaMCP"
        assert settings.app_env in ["development", "testing", "staging", "production"]
        
        print("✅ Settings loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🚀 Running AngelaMCP Basic Tests\n")
    
    tests = [
        test_settings,
        test_database_basic,
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)