#!/usr/bin/env python3
"""
Test script for AngelaMCP with Docker databases.

This tests the connection to Docker-hosted PostgreSQL and Redis from local Python code.
I'm making this comprehensive to catch any connection issues early.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after adding to path
from config.settings import settings
from src.utils.logger import setup_logging, get_logger
from src.persistence.database import DatabaseManager
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent
from src.orchestrator.manager import TaskOrchestrator

# Setup logging
setup_logging()
logger = get_logger("test_local_docker")


class LocalDockerTester:
    """Test AngelaMCP components with Docker databases."""
    
    def __init__(self):
        self.db_manager = None
        self.orchestrator = None
        self.test_results = {}
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return success status."""
        logger.info("🧪 Starting AngelaMCP Local Docker Tests")
        
        tests = [
            ("Environment Variables", self.test_environment),
            ("Database Connection", self.test_database_connection),
            ("Redis Connection", self.test_redis_connection),
            ("Agent Initialization", self.test_agent_initialization),
            ("Basic Orchestration", self.test_basic_orchestration),
            ("Health Checks", self.test_health_checks),
        ]
        
        success_count = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"🔍 Running test: {test_name}")
                result = await test_func()
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                    self.test_results[test_name] = "PASSED"
                    success_count += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    self.test_results[test_name] = "FAILED"
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}", exc_info=True)
                self.test_results[test_name] = f"ERROR: {e}"
        
        # Print summary
        self.print_test_summary(success_count, len(tests))
        
        # Cleanup
        await self.cleanup()
        
        return success_count == len(tests)
    
    async def test_environment(self) -> bool:
        """Test environment variables are set."""
        try:
            required_vars = [
                ("DATABASE_URL", settings.database_url),
                ("REDIS_URL", settings.redis_url),
                ("OPENAI_API_KEY", settings.openai_api_key),
                ("GOOGLE_API_KEY", settings.google_api_key),
            ]
            
            missing_vars = []
            for var_name, var_value in required_vars:
                if not var_value:
                    missing_vars.append(var_name)
                else:
                    logger.info(f"  ✅ {var_name}: configured")
            
            if missing_vars:
                logger.error(f"  ❌ Missing variables: {missing_vars}")
                return False
            
            # Test Docker database URLs
            db_url = str(settings.database_url)
            redis_url = str(settings.redis_url)
            
            if "localhost" in db_url:
                logger.info("  ✅ Database URL points to localhost (Docker)")
            else:
                logger.warning(f"  ⚠️ Database URL: {db_url}")
            
            if "localhost" in redis_url:
                logger.info("  ✅ Redis URL points to localhost (Docker)")
            else:
                logger.warning(f"  ⚠️ Redis URL: {redis_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Environment test failed: {e}")
            return False
    
    async def test_database_connection(self) -> bool:
        """Test PostgreSQL connection (Docker)."""
        try:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Test basic query
            async with self.db_manager.get_session() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT current_timestamp, version()"))
                row = result.fetchone()
                
                logger.info(f"  ✅ Connected to PostgreSQL")
                logger.info(f"  ✅ Server time: {row[0]}")
                logger.info(f"  ✅ Version: {row[1][:50]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def test_redis_connection(self) -> bool:
        """Test Redis connection (Docker)."""
        try:
            if not self.db_manager:
                return False
            
            redis_client = await self.db_manager.get_redis()
            
            # Test basic operations
            await redis_client.ping()
            logger.info("  ✅ Redis ping successful")
            
            # Test set/get
            test_key = "angelamcp:test"
            test_value = "docker_test_value"
            
            await redis_client.set(test_key, test_value, ex=60)  # Expire in 60s
            retrieved_value = await redis_client.get(test_key)
            
            if retrieved_value == test_value:
                logger.info("  ✅ Redis set/get successful")
                await redis_client.delete(test_key)  # Cleanup
                return True
            else:
                logger.error(f"  ❌ Redis value mismatch: {retrieved_value} != {test_value}")
                return False
            
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            return False
    
    async def test_agent_initialization(self) -> bool:
        """Test agent initialization."""
        try:
            # Test Claude Code agent
            try:
                claude_agent = ClaudeCodeAgent()
                logger.info("  ✅ Claude Code agent initialized")
            except Exception as e:
                logger.warning(f"  ⚠️ Claude Code agent failed: {e}")
                # Create a mock for testing
                claude_agent = None
            
            # Test OpenAI agent
            try:
                openai_agent = OpenAIAgent()
                logger.info("  ✅ OpenAI agent initialized")
            except Exception as e:
                logger.error(f"  ❌ OpenAI agent failed: {e}")
                return False
            
            # Test Gemini agent  
            try:
                gemini_agent = GeminiAgent()
                logger.info("  ✅ Gemini agent initialized")
            except Exception as e:
                logger.error(f"  ❌ Gemini agent failed: {e}")
                return False
            
            # Store agents for orchestrator test
            self.agents = {
                'claude': claude_agent,
                'openai': openai_agent,
                'gemini': gemini_agent
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Agent initialization test failed: {e}")
            return False
    
    async def test_basic_orchestration(self) -> bool:
        """Test basic orchestration functionality."""
        try:
            if not self.db_manager or not hasattr(self, 'agents'):
                logger.error("  ❌ Prerequisites not met")
                return False
            
            # Initialize orchestrator
            self.orchestrator = TaskOrchestrator(
                claude_agent=self.agents['claude'],
                openai_agent=self.agents['openai'],
                gemini_agent=self.agents['gemini'],
                db_manager=self.db_manager
            )
            logger.info("  ✅ Task orchestrator initialized")
            
            # Test task complexity analysis
            analysis = await self.orchestrator.analyze_task_complexity(
                "Create a simple calculator function"
            )
            
            if analysis and 'complexity_score' in analysis:
                logger.info(f"  ✅ Task analysis: {analysis['complexity_score']:.1f}/10")
                logger.info(f"  ✅ Recommended strategy: {analysis['recommended_strategy']}")
                return True
            else:
                logger.error("  ❌ Task analysis failed")
                return False
            
        except Exception as e:
            logger.error(f"Orchestration test failed: {e}")
            return False
    
    async def test_health_checks(self) -> bool:
        """Test health check functionality."""
        try:
            if not self.db_manager:
                return False
            
            # Database health check
            db_health = await self.db_manager.health_check()
            
            if db_health['overall'] in ['healthy', 'degraded']:
                logger.info(f"  ✅ Database health: {db_health['overall']}")
                logger.info(f"  ✅ PostgreSQL: {db_health['postgres']['status']}")
                logger.info(f"  ✅ Redis: {db_health['redis']['status']}")
            else:
                logger.error(f"  ❌ Database health: {db_health['overall']}")
                return False
            
            # Agent health checks
            if hasattr(self, 'agents'):
                for agent_name, agent in self.agents.items():
                    if agent:
                        try:
                            health = await agent.health_check()
                            status = health.get('status', 'unknown')
                            logger.info(f"  ✅ {agent_name.title()} agent: {status}")
                        except Exception as e:
                            logger.warning(f"  ⚠️ {agent_name.title()} agent health check failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Health check test failed: {e}")
            return False
    
    def print_test_summary(self, success_count: int, total_tests: int) -> None:
        """Print test summary."""
        print("\n" + "="*60)
        print("🧪 ANGELAMCP LOCAL DOCKER TEST SUMMARY")
        print("="*60)
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == "PASSED" else "❌"
            print(f"{status_emoji} {test_name}: {result}")
        
        print(f"\nPassed: {success_count}/{total_tests}")
        success_rate = (success_count / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_count == total_tests:
            print("\n🎉 ALL TESTS PASSED - AngelaMCP is ready to use!")
            print("\nNext steps:")
            print("1. Run: python -m src.main")
            print("2. Or register MCP: make mcp-register")
        else:
            print(f"\n⚠️ {total_tests - success_count} test(s) failed")
            print("Check the logs above for specific issues")
        
        print("="*60)
    
    async def cleanup(self) -> None:
        """Cleanup test resources."""
        try:
            if self.db_manager:
                await self.db_manager.close()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main test function."""
    try:
        tester = LocalDockerTester()
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
