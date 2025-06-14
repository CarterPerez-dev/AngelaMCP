#!/usr/bin/env python3
"""
Simple MCP Server Integration using the fixed async components.
Replaces the problematic orchestrator with properly managed async operations.
"""

import asyncio
import json
import sys
import os
import time
import signal
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixed components
from .async_task_manager import (
    get_task_manager, 
    cleanup_async_resources,
    timeout_protection
)
from .fixed_debate_orchestrator import FixedDebateProtocol

__version__ = "1.0.0"


class SimpleMCPServer:
    """
    Simplified MCP server that doesn't hang.
    Uses the fixed async components for proper resource management.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("simple_mcp")
        self.agents: Dict[str, Any] = {}
        self.debate_protocol: Optional[FixedDebateProtocol] = None
        self.initialized = False
        self.shutdown_requested = False
        
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            stream=sys.stderr
        )
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        self.logger.info("SimpleMCPServer initialized")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, requesting shutdown...")
        self.shutdown_requested = True
    
    async def initialize_agents(self) -> bool:
        """Initialize AI agents with timeout protection."""
        if self.initialized:
            return True
        
        try:
            async with timeout_protection(30.0, "agent_initialization"):
                self.logger.info("🤖 Initializing AI agents...")
                
                # Try to initialize each agent with individual timeouts
                agent_tasks = []
                
                # Claude agent
                try:
                    from src.agents.claude_agent import ClaudeCodeAgent
                    claude_agent = ClaudeCodeAgent()
                    self.agents["claude"] = claude_agent
                    self.logger.info("✅ Claude agent ready")
                except Exception as e:
                    self.logger.warning(f"⚠️ Claude agent failed: {e}")
                
                # OpenAI agent
                try:
                    from src.agents.openai_agent import OpenAIAgent
                    openai_agent = OpenAIAgent()
                    self.agents["openai"] = openai_agent
                    self.logger.info("✅ OpenAI agent ready")
                except Exception as e:
                    self.logger.warning(f"⚠️ OpenAI agent failed: {e}")
                
                # Gemini agent
                try:
                    from src.agents.gemini_agent import GeminiAgent
                    gemini_agent = GeminiAgent()
                    self.agents["gemini"] = gemini_agent
                    self.logger.info("✅ Gemini agent ready")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gemini agent failed: {e}")
                
                # Check if we have at least one working agent
                if not self.agents:
                    raise RuntimeError("No agents initialized successfully")
                
                # Initialize debate protocol with short timeouts
                self.debate_protocol = FixedDebateProtocol(
                    max_rounds=2,
                    agent_timeout=45.0,  # 15 seconds per agent
                    round_timeout=120.0,  # 45 seconds per round
                    total_timeout=90.0   # 90 seconds total
                )
                
                self.initialized = True
                self.logger.info(f"🎯 Initialization complete with {len(self.agents)} agents")
                return True
                
        except asyncio.TimeoutError:
            self.logger.error("❌ Agent initialization timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Agent initialization failed: {e}")
            return False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request with timeout protection."""
        method = request.get("method", "unknown")
        request_id = request.get("id", "unknown")
        
        try:
            # Apply overall request timeout
            async with timeout_protection(60.0, f"request_{method}"):
                
                if method == "initialize":
                    return await self._handle_initialize(request)
                elif method == "tools/call":
                    return await self._handle_tool_call(request)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                    
        except asyncio.TimeoutError:
            self.logger.error(f"Request {request_id} ({method}) timed out")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Request timed out",
                    "data": {"method": method, "timeout": "60s"}
                }
            }
        except Exception as e:
            self.logger.error(f"Request {request_id} ({method}) failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def _handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": [
                        {
                            "name": "quick_solve",
                            "description": "Quickly solve a task with the best available agent",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "task": {"type": "string", "description": "Task to solve"}
                                },
                                "required": ["task"]
                            }
                        },
                        {
                            "name": "collaborate",
                            "description": "Collaborate with multiple agents on a complex task",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "task": {"type": "string", "description": "Task for collaboration"},
                                    "use_debate": {"type": "boolean", "default": False, "description": "Use debate mode"}
                                },
                                "required": ["task"]
                            }
                        },
                        {
                            "name": "system_status",
                            "description": "Get system status and agent health",
                            "inputSchema": {"type": "object", "properties": {}}
                        }
                    ]
                },
                "serverInfo": {
                    "name": "AngelaMCP-Simple",
                    "version": __version__
                }
            }
        }
    
    async def _handle_tool_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        request_id = request.get("id")
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        # Initialize if needed
        if not self.initialized:
            init_success = await self.initialize_agents()
            if not init_success:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": "❌ Failed to initialize agents. Please check configuration."
                        }]
                    }
                }
        
        # Handle different tools
        if tool_name == "quick_solve":
            result = await self._quick_solve(arguments.get("task", ""))
        elif tool_name == "collaborate":
            result = await self._collaborate(
                arguments.get("task", ""),
                arguments.get("use_debate", False)
            )
        elif tool_name == "system_status":
            result = await self._system_status()
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{
                    "type": "text", 
                    "text": json.dumps(result, indent=2)
                }]
            }
        }
    
    async def _quick_solve(self, task: str) -> Dict[str, Any]:
        """Solve task with single best agent."""
        if not task:
            return {"error": "Task is required"}
        
        if not self.agents:
            return {"error": "No agents available"}
        
        # Try agents in order of preference
        agent_order = ["claude", "openai", "gemini"]
        
        for agent_name in agent_order:
            if agent_name in self.agents:
                try:
                    agent = self.agents[agent_name]
                    
                    # Create simple context
                    from src.models.task_context import TaskContext
                    context = TaskContext()
                    
                    # Call agent with timeout
                    async with timeout_protection(20.0, f"quick_solve_{agent_name}"):
                        response = await agent.generate(task, context)
                    
                    if response and hasattr(response, 'content'):
                        return {
                            "success": True,
                            "solution": response.content,
                            "agent": agent_name,
                            "duration": "< 20s"
                        }
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"Agent {agent_name} timed out on quick solve")
                    continue
                except Exception as e:
                    self.logger.error(f"Agent {agent_name} failed: {e}")
                    continue
        
        return {"error": "All agents failed to solve the task"}
    
    async def _collaborate(self, task: str, use_debate: bool = False) -> Dict[str, Any]:
        """Collaborate with multiple agents."""
        if not task:
            return {"error": "Task is required"}
        
        available_agents = [agent for agent in self.agents.values() if agent]
        
        if not available_agents:
            return {"error": "No agents available"}
        
        if len(available_agents) == 1 or not use_debate:
            # Single agent mode
            return await self._quick_solve(task)
        
        # Multi-agent debate mode
        if not self.debate_protocol:
            return {"error": "Debate protocol not initialized"}
        
        try:
            from src.models.task_context import TaskContext
            context = TaskContext()
            
            self.logger.info(f"🎭 Starting collaboration on: {task[:50]}...")
            
            # Conduct debate with timeout protection
            debate_result = await self.debate_protocol.conduct_debate(
                topic=task,
                agents=available_agents,
                context=context
            )
            
            if debate_result.success:
                return {
                    "success": True,
                    "solution": debate_result.final_consensus,
                    "summary": debate_result.summary,
                    "participants": debate_result.participating_agents,
                    "rounds": debate_result.rounds_completed,
                    "duration": f"{debate_result.total_duration:.1f}s",
                    "consensus_score": debate_result.consensus_score
                }
            else:
                return {
                    "success": False,
                    "error": debate_result.error_message or "Debate failed",
                    "partial_result": debate_result.final_consensus,
                    "summary": debate_result.summary
                }
                
        except Exception as e:
            self.logger.error(f"Collaboration failed: {e}")
            return {"error": f"Collaboration failed: {str(e)}"}
    
    async def _system_status(self) -> Dict[str, Any]:
        """Get system status."""
        task_manager = get_task_manager()
        task_stats = task_manager.get_stats()
        
        return {
            "system": "AngelaMCP Simple",
            "version": __version__,
            "initialized": self.initialized,
            "agents": {
                name: {
                    "available": True,
                    "type": type(agent).__name__
                }
                for name, agent in self.agents.items()
            },
            "task_manager": task_stats,
            "uptime": time.time(),  # Simple uptime
            "status": "healthy" if self.initialized else "initializing"
        }
    
    async def run(self):
        """Run the MCP server."""
        self.logger.info("🚀 Starting SimpleMCPServer...")
        
        try:
            while not self.shutdown_requested:
                try:
                    # Read input with timeout
                    line = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                        timeout=1.0
                    )
                    
                    if not line:  # EOF
                        break
                    
                    # Parse and handle request
                    try:
                        request = json.loads(line.strip())
                        response = await self.handle_request(request)
                        print(json.dumps(response), flush=True)
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON: {e}")
                        continue
                        
                except asyncio.TimeoutError:
                    # Normal timeout for checking shutdown
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing input: {e}")
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("🧹 Cleaning up resources...")
        
        try:
            # Cleanup async resources
            await cleanup_async_resources()
            
            # Clear agents
            self.agents.clear()
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


async def main():
    """Main entry point."""
    server = SimpleMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user", file=sys.stderr)
