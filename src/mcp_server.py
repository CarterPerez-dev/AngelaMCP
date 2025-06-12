#!/usr/bin/env python3
"""
MCP Server for AngelaMCP - Proper Model Context Protocol implementation.

This is the core MCP server that Claude Code connects to for multi-agent collaboration.
I'm implementing the official MCP protocol instead of subprocess calls.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

# MCP Protocol imports
from mcp.server import MCPServer, RequestContext
from mcp.server.models import (
    InitializeResult,
    ServerCapabilities,
    Tool,
    TextContent,
    CallToolResult,
)
from mcp.types import JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

# AngelaMCP imports
from config.settings import settings
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent
from src.orchestrator.manager import TaskOrchestrator
from src.persistence.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger("mcp_server")


@dataclass
class CollaborationResult:
    """Result of a collaboration session."""
    success: bool
    final_solution: str
    agent_responses: List[Dict[str, Any]]
    consensus_score: float
    debate_summary: Optional[str] = None
    execution_time: float = 0.0
    cost_breakdown: Optional[Dict[str, float]] = None


class CollaborationRequest(BaseModel):
    """Request for agent collaboration."""
    task_description: str = Field(..., description="The task to collaborate on")
    agents: List[str] = Field(default=["claude", "openai", "gemini"], description="Agents to include")
    strategy: str = Field(default="debate", description="Collaboration strategy: debate, parallel, consensus")
    max_rounds: int = Field(default=3, description="Maximum debate rounds")
    require_consensus: bool = Field(default=True, description="Whether consensus is required")


class DebateRequest(BaseModel):
    """Request for structured debate."""
    topic: str = Field(..., description="Topic to debate")
    agents: List[str] = Field(default=["claude", "openai", "gemini"], description="Participating agents")
    max_rounds: int = Field(default=3, description="Maximum debate rounds")
    timeout_seconds: int = Field(default=300, description="Timeout per round")


class AngelaMCPServer:
    """MCP Server for AngelaMCP multi-agent collaboration."""
    
    def __init__(self):
        self.server = MCPServer("angelamcp")
        self.orchestrator: Optional[TaskOrchestrator] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.agents: Dict[str, Any] = {}
        self.setup_tools()
        
    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            logger.info("Initializing AngelaMCP MCP Server...")
            
            # Initialize database
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize agents
            self.agents = {
                "claude": ClaudeCodeAgent(),
                "openai": OpenAIAgent(), 
                "gemini": GeminiAgent()
            }
            
            # Initialize orchestrator
            self.orchestrator = TaskOrchestrator(
                claude_agent=self.agents["claude"],
                openai_agent=self.agents["openai"],
                gemini_agent=self.agents["gemini"],
                db_manager=self.db_manager
            )
            
            logger.info("AngelaMCP MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}", exc_info=True)
            raise
    
    def setup_tools(self) -> None:
        """Setup MCP tools that Claude Code can call."""
        
        @self.server.call_tool()
        async def collaborate(arguments: Dict[str, Any]) -> List[TextContent]:
            """Orchestrate collaboration between multiple AI agents."""
            try:
                request = CollaborationRequest(**arguments)
                
                if not self.orchestrator:
                    return [TextContent(
                        type="text",
                        text="❌ AngelaMCP orchestrator not initialized"
                    )]
                
                logger.info(f"Starting collaboration on: {request.task_description}")
                
                # Execute collaboration
                result = await self.orchestrator.collaborate_on_task(
                    task_description=request.task_description,
                    agents=request.agents,
                    strategy=request.strategy,
                    max_rounds=request.max_rounds,
                    require_consensus=request.require_consensus
                )
                
                # Format response
                response_text = f"""
🤝 **Multi-Agent Collaboration Complete**

**Task:** {request.task_description}
**Strategy:** {request.strategy}
**Agents:** {', '.join(request.agents)}
**Success:** {'✅' if result.success else '❌'}
**Consensus Score:** {result.consensus_score:.2f}

**Final Solution:**
{result.final_solution}

**Agent Responses:**
"""
                
                for i, response in enumerate(result.agent_responses, 1):
                    agent_name = response.get('agent', 'Unknown')
                    content = response.get('content', 'No response')
                    response_text += f"\n**{i}. {agent_name}:**\n{content}\n"
                
                if result.debate_summary:
                    response_text += f"\n**Debate Summary:**\n{result.debate_summary}"
                
                return [TextContent(type="text", text=response_text)]
                
            except Exception as e:
                error_msg = f"❌ Collaboration failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [TextContent(type="text", text=error_msg)]
        
        @self.server.call_tool()
        async def debate(arguments: Dict[str, Any]) -> List[TextContent]:
            """Start a structured debate between AI agents."""
            try:
                request = DebateRequest(**arguments)
                
                if not self.orchestrator:
                    return [TextContent(
                        type="text",
                        text="❌ AngelaMCP orchestrator not initialized"
                    )]
                
                logger.info(f"Starting debate on: {request.topic}")
                
                # Execute debate
                result = await self.orchestrator.start_debate(
                    topic=request.topic,
                    agents=request.agents,
                    max_rounds=request.max_rounds,
                    timeout_seconds=request.timeout_seconds
                )
                
                # Format debate results
                response_text = f"""
🗣️ **Structured Debate Complete**

**Topic:** {request.topic}
**Participants:** {', '.join(request.agents)}
**Rounds:** {result.get('rounds_completed', 0)}/{request.max_rounds}

**Debate Results:**
"""
                
                # Add debate rounds
                for round_num, round_data in enumerate(result.get('rounds', []), 1):
                    response_text += f"\n**Round {round_num}:**\n"
                    for agent_response in round_data.get('responses', []):
                        agent = agent_response.get('agent', 'Unknown')
                        position = agent_response.get('content', 'No position')
                        response_text += f"- **{agent}:** {position}\n"
                
                # Add final consensus
                if 'consensus' in result:
                    consensus = result['consensus']
                    response_text += f"\n**Final Consensus:**\n{consensus['summary']}\n"
                    response_text += f"**Confidence Score:** {consensus.get('score', 0):.2f}\n"
                
                return [TextContent(type="text", text=response_text)]
                
            except Exception as e:
                error_msg = f"❌ Debate failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [TextContent(type="text", text=error_msg)]
        
        @self.server.call_tool()
        async def analyze_task_complexity(arguments: Dict[str, Any]) -> List[TextContent]:
            """Analyze task complexity and recommend collaboration strategy."""
            try:
                task_description = arguments.get("task_description", "")
                
                if not task_description:
                    return [TextContent(
                        type="text",
                        text="❌ No task description provided"
                    )]
                
                if not self.orchestrator:
                    return [TextContent(
                        type="text",
                        text="❌ AngelaMCP orchestrator not initialized"
                    )]
                
                # Analyze complexity
                analysis = await self.orchestrator.analyze_task_complexity(task_description)
                
                response_text = f"""
📊 **Task Complexity Analysis**

**Task:** {task_description}

**Complexity Score:** {analysis.get('complexity_score', 0):.2f}/10
**Estimated Time:** {analysis.get('estimated_time', 'Unknown')}
**Recommended Strategy:** {analysis.get('recommended_strategy', 'single_agent')}

**Analysis Details:**
- **Technical Complexity:** {analysis.get('technical_complexity', 'Unknown')}
- **Collaboration Benefit:** {analysis.get('collaboration_benefit', 'Unknown')}
- **Recommended Agents:** {', '.join(analysis.get('recommended_agents', []))}

**Reasoning:**
{analysis.get('reasoning', 'No reasoning provided')}
"""
                
                return [TextContent(type="text", text=response_text)]
                
            except Exception as e:
                error_msg = f"❌ Task analysis failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [TextContent(type="text", text=error_msg)]
        
        @self.server.call_tool()
        async def get_agent_status(arguments: Dict[str, Any]) -> List[TextContent]:
            """Get status of all AI agents."""
            try:
                status_text = "🤖 **Agent Status Report**\n\n"
                
                for agent_name, agent in self.agents.items():
                    try:
                        # Check agent health
                        health = await agent.health_check() if hasattr(agent, 'health_check') else {"status": "unknown"}
                        status = "✅ Online" if health.get("status") == "healthy" else "❌ Offline"
                        
                        status_text += f"**{agent_name.title()} Agent:** {status}\n"
                        if "model" in health:
                            status_text += f"  - Model: {health['model']}\n"
                        if "last_response_time" in health:
                            status_text += f"  - Response Time: {health['last_response_time']:.2f}s\n"
                        
                    except Exception as e:
                        status_text += f"**{agent_name.title()} Agent:** ❌ Error - {str(e)}\n"
                
                # Add orchestrator status
                if self.orchestrator:
                    status_text += f"\n**Orchestrator:** ✅ Ready\n"
                    status_text += f"**Database:** {'✅ Connected' if self.db_manager else '❌ Disconnected'}\n"
                else:
                    status_text += f"\n**Orchestrator:** ❌ Not initialized\n"
                
                return [TextContent(type="text", text=status_text)]
                
            except Exception as e:
                error_msg = f"❌ Status check failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [TextContent(type="text", text=error_msg)]

    async def run(self) -> None:
        """Run the MCP server."""
        try:
            await self.initialize()
            
            # Server capabilities
            capabilities = ServerCapabilities(
                tools={
                    "collaborate": Tool(
                        name="collaborate",
                        description="Orchestrate collaboration between multiple AI agents on a task",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Description of the task to collaborate on"
                                },
                                "agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of agents to include (claude, openai, gemini)",
                                    "default": ["claude", "openai", "gemini"]
                                },
                                "strategy": {
                                    "type": "string",
                                    "enum": ["debate", "parallel", "consensus"],
                                    "description": "Collaboration strategy to use",
                                    "default": "debate"
                                },
                                "max_rounds": {
                                    "type": "integer",
                                    "description": "Maximum number of debate rounds",
                                    "default": 3
                                },
                                "require_consensus": {
                                    "type": "boolean",
                                    "description": "Whether consensus is required",
                                    "default": True
                                }
                            },
                            "required": ["task_description"]
                        }
                    ),
                    "debate": Tool(
                        name="debate",
                        description="Start a structured debate between AI agents on a topic",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "Topic to debate"
                                },
                                "agents": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Agents to include in debate",
                                    "default": ["claude", "openai", "gemini"]
                                },
                                "max_rounds": {
                                    "type": "integer",
                                    "description": "Maximum debate rounds",
                                    "default": 3
                                },
                                "timeout_seconds": {
                                    "type": "integer",
                                    "description": "Timeout per round in seconds",
                                    "default": 300
                                }
                            },
                            "required": ["topic"]
                        }
                    ),
                    "analyze_task_complexity": Tool(
                        name="analyze_task_complexity",
                        description="Analyze task complexity and recommend collaboration strategy",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "task_description": {
                                    "type": "string",
                                    "description": "Task to analyze"
                                }
                            },
                            "required": ["task_description"]
                        }
                    ),
                    "get_agent_status": Tool(
                        name="get_agent_status",
                        description="Get current status of all AI agents",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        }
                    )
                }
            )
            
            # Initialize server
            init_result = InitializeResult(
                protocolVersion="2024-11-05",
                capabilities=capabilities,
                serverInfo={
                    "name": "AngelaMCP",
                    "version": "1.0.0"
                }
            )
            
            logger.info("AngelaMCP MCP Server ready - waiting for connections...")
            
            # Run the MCP server
            await self.server.run(
                transport="stdio",
                init_result=init_result
            )
            
        except Exception as e:
            logger.error(f"MCP server run failed: {e}", exc_info=True)
            raise


async def main():
    """Main entry point for MCP server."""
    try:
        server = AngelaMCPServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("MCP server shutdown requested")
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
