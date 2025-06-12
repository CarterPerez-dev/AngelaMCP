"""
Main application entry point for AngelaMCP.
Handles application lifecycle, configuration loading, and graceful shutdown.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from config.settings import settings
from src.logging_config import setup_logging
from src.cli import CLI
from src.persistence.database import DatabaseManager
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent
from src.orchestration.orchestrator import TaskOrchestrator
from src.utils.metrics import MetricsCollector


class AngelaMCPApplication:
    """Main application class for AngelaMCP."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager: Optional[DatabaseManager] = None
        self.orchestrator: Optional[TaskOrchestrator] = None
        self.cli: Optional[CLI] = None
        self.metrics: Optional[MetricsCollector] = None
        self._shutdown_event = asyncio.Event()
        
    async def startup(self) -> None:
        """Initialize all application components."""
        try:
            self.logger.info("Starting AngelaMCP application...")
            
            # Initialize metrics collection
            self.logger.info("Initializing metrics collector...")
            self.metrics = MetricsCollector()
            
            # Initialize database
            self.logger.info("Initializing database connection...")
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize agents
            self.logger.info("Initializing AI agents...")
            claude_agent = ClaudeCodeAgent()
            openai_agent = OpenAIAgent()
            gemini_agent = GeminiAgent()
            
            # Initialize orchestrator
            self.logger.info("Initializing task orchestrator...")
            self.orchestrator = TaskOrchestrator(
                claude_agent=claude_agent,
                openai_agent=openai_agent,
                gemini_agent=gemini_agent,
                db_manager=self.db_manager,
                metrics=self.metrics
            )
            
            # Initialize CLI
            self.logger.info("Initializing CLI interface...")
            self.cli = CLI(orchestrator=self.orchestrator)
            
            self.logger.info("AngelaMCP application started successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to start application: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all application components."""
        self.logger.info("Shutting down AngelaMCP application...")
        
        try:
            # Shutdown CLI
            if self.cli:
                await self.cli.cleanup()
                
            # Shutdown orchestrator
            if self.orchestrator:
                await self.orchestrator.cleanup()
                
            # Close database connections
            if self.db_manager:
                await self.db_manager.close()
                
            # Flush metrics
            if self.metrics:
                await self.metrics.flush()
                
            self.logger.info("AngelaMCP application shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def run(self) -> None:
        """Main application run loop."""
        try:
            await self.startup()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Run CLI in main loop
            cli_task = asyncio.create_task(self.cli.run())
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            
            # Wait for either CLI completion or shutdown signal
            done, pending = await asyncio.wait(
                [cli_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
        finally:
            await self.shutdown()


@asynccontextmanager
async def app_lifespan():
    """Context manager for application lifespan."""
    app = AngelaMCPApplication()
    try:
        await app.startup()
        yield app
    finally:
        await app.shutdown()


async def main() -> None:
    """Main entry point."""
    # Setup logging first
    setup_logging()
    
    # Validate critical configuration
    try:
        # Verify required API keys are present
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required")
        if not settings.redis_url:
            raise ValueError("REDIS_URL is required")
            
        # Verify Claude Code is available
        if not settings.claude_code_path.exists():
            raise ValueError(f"Claude Code not found at {settings.claude_code_path}")
            
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file and ensure all required settings are configured.")
        sys.exit(1)
    
    # Run the application
    app = AngelaMCPApplication()
    await app.run()


def cli_main() -> None:
    """CLI entry point for console scripts."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()