#!/usr/bin/env python3
"""
Main application entry point for AngelaMCP.

This handles both standalone CLI mode and MCP server mode.
I'm implementing a production-ready system that can run either way.
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configuration first
from config.settings import settings
from src.utils.logger import setup_logging, get_logger

# Set up logging early
setup_logging()
logger = get_logger("main")


async def run_mcp_server():
    """Run as MCP server for Claude Code integration."""
    try:
        logger.info("Starting AngelaMCP as MCP Server...")
        from src.mcp_server import AngelaMCPServer
        
        server = AngelaMCPServer()
        await server.run()
        
    except Exception as e:
        logger.error(f"MCP server failed: {e}", exc_info=True)
        sys.exit(1)


async def run_standalone_cli():
    """Run as standalone CLI application."""
    try:
        logger.info("Starting AngelaMCP as standalone CLI...")
        from src.cli import CLI
        from src.orchestrator.manager import TaskOrchestrator
        from src.persistence.database import DatabaseManager
        from src.agents.claude_agent import ClaudeCodeAgent
        from src.agents.openai_agent import OpenAIAgent
        from src.agents.gemini_agent import GeminiAgent
        
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Initialize agents
        claude_agent = ClaudeCodeAgent()
        openai_agent = OpenAIAgent()
        gemini_agent = GeminiAgent()
        
        # Initialize orchestrator
        orchestrator = TaskOrchestrator(
            claude_agent=claude_agent,
            openai_agent=openai_agent,
            gemini_agent=gemini_agent,
            db_manager=db_manager
        )
        
        # Initialize CLI
        cli = CLI(orchestrator=orchestrator)
        
        # Run CLI
        await cli.run()
        
    except KeyboardInterrupt:
        logger.info("CLI shutdown requested")
    except Exception as e:
        logger.error(f"CLI failed: {e}", exc_info=True)
        sys.exit(1)


async def main():
    """Main application entry point."""
    try:
        # Check if running as MCP server
        if len(sys.argv) > 1 and sys.argv[1] == "mcp-server":
            await run_mcp_server()
        else:
            await run_standalone_cli()
            
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)


def cli_main():
    """Entry point for CLI scripts (macp command)."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"CLI main failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
