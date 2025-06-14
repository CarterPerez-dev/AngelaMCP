#!/usr/bin/env python3
"""
MCP Server Test Script
Tests the AngelaMCP MCP server functionality to verify it works correctly.
"""

import json
import subprocess
import sys
import asyncio
import time
from pathlib import Path

def test_mcp_server():
    """Test the MCP server by sending JSON-RPC requests."""
    
    # Get the path to the MCP server
    script_dir = Path(__file__).parent
    server_path = script_dir / "src" / "mcp_server.py"
    
    if not server_path.exists():
        print(f"❌ MCP server not found at {server_path}")
        return False
    
    print("🧪 Testing AngelaMCP MCP Server...")
    
    try:
        # Start the MCP server process
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Test 1: Initialize
        print("\n1️⃣ Testing initialization...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        if response:
            init_response = json.loads(response.strip())
            if init_response.get("result"):
                print("✅ Initialization successful")
            else:
                print(f"❌ Initialization failed: {init_response}")
                return False
        else:
            print("❌ No response to initialization")
            return False
        
        # Test 2: List tools
        print("\n2️⃣ Testing tools list...")
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        process.stdin.write(json.dumps(list_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        if response:
            list_response = json.loads(response.strip())
            tools = list_response.get("result", {}).get("tools", [])
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool['name']}: {tool['description']}")
        else:
            print("❌ No response to tools list")
            return False
        
        # Test 3: Call get_agent_status tool
        print("\n3️⃣ Testing get_agent_status tool...")
        status_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_agent_status",
                "arguments": {}
            }
        }
        
        process.stdin.write(json.dumps(status_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        if response:
            status_response = json.loads(response.strip())
            if status_response.get("result"):
                content = status_response["result"]["content"][0]["text"]
                print("✅ Agent status retrieved:")
                print(content[:200] + "..." if len(content) > 200 else content)
            else:
                print(f"❌ Agent status failed: {status_response}")
        else:
            print("❌ No response to agent status")
        
        # Test 4: Call analyze_task_complexity tool
        print("\n4️⃣ Testing analyze_task_complexity tool...")
        analyze_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "analyze_task_complexity",
                "arguments": {
                    "task_description": "Build a simple calculator app with basic arithmetic operations"
                }
            }
        }
        
        process.stdin.write(json.dumps(analyze_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        if response:
            analyze_response = json.loads(response.strip())
            if analyze_response.get("result"):
                content = analyze_response["result"]["content"][0]["text"]
                print("✅ Task analysis completed:")
                print(content[:200] + "..." if len(content) > 200 else content)
            else:
                print(f"❌ Task analysis failed: {analyze_response}")
        else:
            print("❌ No response to task analysis")
        
        # Test 5: Test error handling with invalid tool
        print("\n5️⃣ Testing error handling...")
        error_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            }
        }
        
        process.stdin.write(json.dumps(error_request) + "\n")
        process.stdin.flush()
        
        response = process.stdout.readline()
        if response:
            error_response = json.loads(response.strip())
            if error_response.get("error"):
                print("✅ Error handling works correctly")
            else:
                print(f"❌ Expected error but got: {error_response}")
        else:
            print("❌ No response to error test")
        
        print("\n🎉 MCP Server tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    finally:
        # Clean up the process
        if 'process' in locals():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
