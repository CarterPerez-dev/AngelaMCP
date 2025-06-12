#!/usr/bin/env python3
"""
Setup verification script for AngelaMCP.
Verifies that all required components are properly configured.
"""

import asyncio
import logging
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.persistence.database import DatabaseManager
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent


class SetupVerifier:
    """Verifies AngelaMCP setup and configuration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors = []
        self.warnings = []
    
    def check_python_version(self):
        """Check Python version."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.errors.append(f"Python 3.10+ required, found {version.major}.{version.minor}")
        else:
            self.logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    def check_claude_code(self):
        """Check Claude Code installation."""
        try:
            if settings.claude_code_path.exists():
                result = subprocess.run([str(settings.claude_code_path), "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.logger.info(f"✅ Claude Code: {result.stdout.strip()}")
                else:
                    self.errors.append("Claude Code found but not responding correctly")
            else:
                # Try to find in PATH
                claude_path = shutil.which("claude")
                if claude_path:
                    self.logger.info(f"✅ Claude Code found in PATH: {claude_path}")
                else:
                    self.errors.append("Claude Code not found. Please install from https://claude.ai/code")
        except Exception as e:
            self.errors.append(f"Error checking Claude Code: {e}")
    
    def check_environment_variables(self):
        """Check required environment variables."""
        required_vars = [
            ("OPENAI_API_KEY", "OpenAI API key"),
            ("GOOGLE_API_KEY", "Google Gemini API key"),
            ("DATABASE_URL", "Database URL"),
            ("REDIS_URL", "Redis URL")
        ]
        
        for var_name, description in required_vars:
            value = getattr(settings, var_name.lower(), None)
            if not value or (hasattr(value, 'get_secret_value') and not value.get_secret_value()):
                self.errors.append(f"Missing {description} ({var_name})")
            else:
                # Don't log actual API keys
                if "key" in var_name.lower():
                    self.logger.info(f"✅ {description}: configured")
                else:
                    self.logger.info(f"✅ {description}: {value}")
    
    async def check_database_connection(self):
        """Check database connectivity."""
        try:
            db_manager = DatabaseManager()
            await db_manager.initialize()
            
            # Test connection
            session = await db_manager.get_session()
            await session.execute("SELECT 1")
            await db_manager.close()
            
            self.logger.info("✅ Database connection: successful")
        except Exception as e:
            self.errors.append(f"Database connection failed: {e}")
    
    async def check_redis_connection(self):
        """Check Redis connectivity."""
        try:
            import redis.asyncio as redis
            
            r = redis.from_url(settings.redis_url)
            await r.ping()
            await r.close()
            
            self.logger.info("✅ Redis connection: successful")
        except Exception as e:
            self.errors.append(f"Redis connection failed: {e}")
    
    async def check_ai_agents(self):
        """Check AI agent configurations."""
        # Check OpenAI
        try:
            openai_agent = OpenAIAgent()
            # Don't actually make API calls during verification
            self.logger.info("✅ OpenAI agent: configured")
        except Exception as e:
            self.errors.append(f"OpenAI agent configuration error: {e}")
        
        # Check Gemini
        try:
            gemini_agent = GeminiAgent()
            self.logger.info("✅ Gemini agent: configured")
        except Exception as e:
            self.errors.append(f"Gemini agent configuration error: {e}")
        
        # Check Claude Code agent
        try:
            claude_agent = ClaudeCodeAgent()
            self.logger.info("✅ Claude Code agent: configured")
        except Exception as e:
            self.errors.append(f"Claude Code agent configuration error: {e}")
    
    def check_directories(self):
        """Check required directories exist."""
        dirs_to_check = [
            settings.workspace_dir,
            settings.claude_session_dir,
            settings.log_file.parent
        ]
        
        for directory in dirs_to_check:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✅ Directory: {directory}")
            except Exception as e:
                self.errors.append(f"Cannot create directory {directory}: {e}")
    
    def check_system_resources(self):
        """Check system resources."""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.total < 2 * 1024 * 1024 * 1024:  # 2GB
                self.warnings.append("Low system memory (< 2GB)")
            else:
                self.logger.info(f"✅ System memory: {memory.total // (1024**3)}GB")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
                self.warnings.append("Low disk space (< 1GB free)")
            else:
                self.logger.info(f"✅ Disk space: {disk.free // (1024**3)}GB free")
                
        except ImportError:
            self.warnings.append("psutil not available for system resource checking")
        except Exception as e:
            self.warnings.append(f"Error checking system resources: {e}")
    
    async def run_verification(self):
        """Run all verification checks."""
        self.logger.info("🔍 Starting AngelaMCP setup verification...")
        self.logger.info("=" * 60)
        
        # Run all checks
        self.check_python_version()
        self.check_claude_code()
        self.check_environment_variables()
        await self.check_database_connection()
        await self.check_redis_connection()
        await self.check_ai_agents()
        self.check_directories()
        self.check_system_resources()
        
        # Summary
        self.logger.info("=" * 60)
        
        if self.errors:
            self.logger.error("❌ Verification failed with errors:")
            for error in self.errors:
                self.logger.error(f"  • {error}")
            return False
        
        if self.warnings:
            self.logger.warning("⚠️  Verification completed with warnings:")
            for warning in self.warnings:
                self.logger.warning(f"  • {warning}")
        
        self.logger.info("✅ All checks passed! AngelaMCP is ready to run.")
        self.logger.info("\nTo start AngelaMCP, run: make run")
        return True


async def main():
    """Main verification function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    verifier = SetupVerifier()
    success = await verifier.run_verification()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())