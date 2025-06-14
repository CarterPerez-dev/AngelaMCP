#!/usr/bin/env python3
"""
Quick test script to verify MCP server functionality.
"""

import asyncio
import json
import sys

async def test_mcp_server():
    """Test the MCP server connection and tools."""
    try:
        print("🧪 Testing AngelaMCP MCP Server...")
        
        # This is a basic test - in production, Claude Code would handle the connection
        # For now, let's just verify the server can be imported and basic functionality works
        from src.mcp_server import AngelaMCPServer
        
        print("✅ MCP server module imported successfully")
        
        # Create server instance
        server = AngelaMCPServer()
        print("✅ MCP server instance created")
        
        # Test initialization (without running the server)
        await server.initialize()
        print("✅ MCP server initialized successfully")
        
        # Test health check if available
        if server.orchestrator:
            health = await server.orchestrator.health_check()
            print("✅ Health check completed:")
            for component, status in health.items():
                print(f"   - {component}: {status}")
        
        print("\n🎉 MCP Server test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)