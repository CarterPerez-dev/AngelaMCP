#!/usr/bin/env python3
"""
Agent connectivity test script for AngelaMCP.
Tests connectivity and basic functionality of all AI agents.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent


class AgentTester:
    """Tests all AI agents for connectivity and basic functionality."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    async def test_claude_agent(self):
        """Test Claude Code agent."""
        self.logger.info("Testing Claude Code agent...")
        
        try:
            agent = ClaudeCodeAgent()
            
            # Simple test query
            test_query = "What is 2 + 2? Please respond with just the number."
            response = await agent.generate(test_query, {})
            
            if response and response.content:
                self.logger.info(f"✅ Claude Code: {response.content[:50]}...")
                self.results['claude'] = True
            else:
                self.logger.error("❌ Claude Code: No response received")
                self.results['claude'] = False
                
        except Exception as e:
            self.logger.error(f"❌ Claude Code: {e}")
            self.results['claude'] = False
    
    async def test_openai_agent(self):
        """Test OpenAI agent."""
        self.logger.info("Testing OpenAI agent...")
        
        try:
            agent = OpenAIAgent()
            
            # Simple test query
            test_query = "What is 2 + 2? Please respond with just the number."
            response = await agent.generate(test_query, {})
            
            if response and response.content:
                self.logger.info(f"✅ OpenAI: {response.content[:50]}...")
                self.results['openai'] = True
            else:
                self.logger.error("❌ OpenAI: No response received")
                self.results['openai'] = False
                
        except Exception as e:
            self.logger.error(f"❌ OpenAI: {e}")
            self.results['openai'] = False
    
    async def test_gemini_agent(self):
        """Test Gemini agent."""
        self.logger.info("Testing Gemini agent...")
        
        try:
            agent = GeminiAgent()
            
            # Simple test query
            test_query = "What is 2 + 2? Please respond with just the number."
            response = await agent.generate(test_query, {})
            
            if response and response.content:
                self.logger.info(f"✅ Gemini: {response.content[:50]}...")
                self.results['gemini'] = True
            else:
                self.logger.error("❌ Gemini: No response received")
                self.results['gemini'] = False
                
        except Exception as e:
            self.logger.error(f"❌ Gemini: {e}")
            self.results['gemini'] = False
    
    async def test_collaborative_scenario(self):
        """Test a simple collaborative scenario."""
        if not all(self.results.values()):
            self.logger.warning("⚠️  Skipping collaborative test due to agent failures")
            return False
        
        self.logger.info("Testing collaborative scenario...")
        
        try:
            claude_agent = ClaudeCodeAgent()
            openai_agent = OpenAIAgent()
            gemini_agent = GeminiAgent()
            
            # Simple collaborative task
            task = "Suggest a simple Python function to calculate the factorial of a number."
            
            # Get responses from all agents
            claude_response = await claude_agent.generate(task, {})
            openai_response = await openai_agent.generate(task, {})
            gemini_response = await gemini_agent.generate(task, {})
            
            responses = [claude_response, openai_response, gemini_response]
            successful_responses = [r for r in responses if r and r.content]
            
            if len(successful_responses) >= 2:
                self.logger.info(f"✅ Collaborative test: {len(successful_responses)}/3 agents responded")
                return True
            else:
                self.logger.error(f"❌ Collaborative test: Only {len(successful_responses)}/3 agents responded")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Collaborative test failed: {e}")
            return False
    
    async def run_tests(self):
        """Run all agent tests."""
        self.logger.info("🧪 Starting AngelaMCP agent connectivity tests...")
        self.logger.info("=" * 60)
        
        # Test individual agents
        await self.test_claude_agent()
        await self.test_openai_agent()
        await self.test_gemini_agent()
        
        # Test collaboration
        collaborative_success = await self.test_collaborative_scenario()
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("Test Results:")
        
        total_passed = sum(self.results.values())
        total_tests = len(self.results)
        
        for agent, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.logger.info(f"  {agent.capitalize()}: {status}")
        
        if collaborative_success:
            self.logger.info("  Collaboration: ✅ PASS")
        else:
            self.logger.info("  Collaboration: ❌ FAIL")
        
        self.logger.info(f"\nOverall: {total_passed}/{total_tests} agents working")
        
        if total_passed == total_tests and collaborative_success:
            self.logger.info("🎉 All tests passed! AngelaMCP is ready for multi-agent collaboration.")
            return True
        else:
            self.logger.error("⚠️  Some tests failed. Check your configuration and API keys.")
            return False


async def main():
    """Main test function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    tester = AgentTester()
    success = await tester.run_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())