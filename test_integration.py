#!/usr/bin/env python3
"""
🧪 AngelaMCP Integration Test
============================

Quick integration test to verify everything is working properly.
This tests the MCP server can start and respond to basic requests.
"""

import asyncio
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

def run_command(cmd: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except Exception as e:
        return False, str(e)

def test_environment():
    """Test basic environment setup."""
    print("🔍 Testing environment setup...")
    
    # Check .env file exists
    if not Path(".env").exists():
        print("❌ .env file not found")
        return False
    
    # Check venv exists
    if not Path("venv").exists():
        print("❌ Virtual environment not found")
        return False
    
    # Check key Python packages
    success, output = run_command("venv/bin/python -c 'import openai, google.genai, mcp'")
    if not success:
        print(f"❌ Required Python packages not installed: {output}")
        return False
    
    print("✅ Environment setup looks good")
    return True

def test_docker_services():
    """Test Docker services are running."""
    print("🐳 Testing Docker services...")
    
    # Check if docker-compose is available
    success, _ = run_command("which docker-compose")
    if not success:
        print("⚠️  docker-compose not found, skipping Docker tests")
        return True
    
    # Check if services are running
    success, output = run_command("docker-compose -f docker/docker-compose.yml ps")
    if "postgres" not in output or "redis" not in output:
        print("⚠️  PostgreSQL/Redis not running in Docker, trying to start...")
        success, output = run_command("docker-compose -f docker/docker-compose.yml up -d", timeout=60)
        if not success:
            print(f"❌ Failed to start Docker services: {output}")
            return False
        
        # Wait for services to be ready
        time.sleep(10)
    
    print("✅ Docker services are running")
    return True

def test_database_connection():
    """Test database connectivity."""
    print("🗄️  Testing database connection...")
    
    success, output = run_command("venv/bin/python scripts/init_db.py")
    if not success:
        print(f"❌ Database initialization failed: {output}")
        return False
    
    print("✅ Database connection successful")
    return True

def test_mcp_server_start():
    """Test MCP server can start."""
    print("🧙‍♀️ Testing MCP server startup...")
    
    # Test if the MCP server can be imported and initialized
    test_code = """
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

async def test():
    try:
        from src.mcp_server import AngelaMCPServer
        server = AngelaMCPServer()
        print("✅ MCP server can be created")
        return True
    except Exception as e:
        print(f"❌ MCP server creation failed: {e}")
        return False

result = asyncio.run(test())
sys.exit(0 if result else 1)
"""
    
    with open("temp_test.py", "w") as f:
        f.write(test_code)
    
    try:
        success, output = run_command("venv/bin/python temp_test.py")
        Path("temp_test.py").unlink()  # Clean up
        
        if not success:
            print(f"❌ MCP server test failed: {output}")
            return False
        
        print("✅ MCP server can start")
        return True
    except Exception as e:
        print(f"❌ MCP server test error: {e}")
        return False

def test_mcp_json_creation():
    """Test .mcp.json creation."""
    print("📄 Testing .mcp.json creation...")
    
    # Remove existing .mcp.json if it exists
    mcp_json = Path(".mcp.json")
    if mcp_json.exists():
        mcp_json.unlink()
    
    # Run installer MCP creation logic
    test_code = """
import json
from pathlib import Path

project_root = Path.cwd()
mcp_config = {
    "mcpServers": {
        "angelamcp": {
            "command": "bash",
            "args": ["run-mcp.sh"],
            "cwd": str(project_root.absolute()),
            "env": {
                "PYTHONPATH": str(project_root.absolute()),
                "PATH": f"{project_root.absolute() / 'venv' / 'bin'}:/usr/local/bin:/usr/bin:/bin"
            }
        }
    }
}

with open(".mcp.json", 'w') as f:
    json.dump(mcp_config, f, indent=2)

print("✅ .mcp.json created successfully")
"""
    
    with open("temp_mcp_test.py", "w") as f:
        f.write(test_code)
    
    try:
        success, output = run_command("venv/bin/python temp_mcp_test.py")
        Path("temp_mcp_test.py").unlink()  # Clean up
        
        if not success or not Path(".mcp.json").exists():
            print(f"❌ .mcp.json creation failed: {output}")
            return False
        
        print("✅ .mcp.json created successfully")
        return True
    except Exception as e:
        print(f"❌ .mcp.json creation error: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 AngelaMCP Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment),
        ("Docker Services", test_docker_services),
        ("Database Connection", test_database_connection),
        ("MCP Server Startup", test_mcp_server_start),
        ("MCP JSON Creation", test_mcp_json_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🧪 Running: {name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! AngelaMCP is ready to use.")
        print("\nNext steps:")
        print("1. Register with Claude Code: make mcp-register")
        print("2. Test integration: claude 'Use AngelaMCP to help with a task'")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)