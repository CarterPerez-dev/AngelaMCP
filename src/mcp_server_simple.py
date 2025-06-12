#!/usr/bin/env python3
"""
Simple MCP Server test for AngelaMCP.
"""

import asyncio
from mcp.server import Server
from mcp.server.models import ServerCapabilities

async def main():
    """Simple test server."""
    server = Server("angelamcp-test")
    
    @server.call_tool()
    async def hello_world(arguments: dict) -> str:
        """Test tool."""
        return "Hello from AngelaMCP!"
    
    print("Starting test MCP server...")
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())