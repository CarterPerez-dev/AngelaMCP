#!/usr/bin/env python3
"""
MCP Server for AngelaMCP - Complete Model Context Protocol implementation.

This is the core MCP server that Claude Code connects to for multi-agent collaboration.
I'm implementing the official MCP protocol with full orchestrator integration.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

# MCP Protocol imports
from mcp.server import Server
from mcp import stdio_server
from mcp.types import TextContent, CallToolResult

# AngelaMCP imports
from config.settings import settings
from src.agents import ClaudeCodeAgent, OpenAIAgent, GeminiAgent, TaskContext, TaskType
from src.orchestrator import TaskOrchestrator, CollaborationStrategy
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
    strategy: str = Field(default="debate", description="Collaboration strategy: debate, parallel, consensus, auto")
    max_rounds: int = Field(default=3, description="Maximum debate rounds")
    require_consensus: bool = Field(default=True, description="Whether consensus is required")


class DebateRequest(BaseModel):
    """Request for structured debate."""
    topic: str = Field(..., description="Topic to debate")
    agents: List[str] = Field(default=["claude", "openai", "gemini"], description="Participating agents")
    max_rounds: int = Field(default=3, description="Maximum debate rounds")
    timeout_seconds: int = Field(default=300, description="Timeout per round")


class TaskAnalysisRequest(BaseModel):
    """Request for task complexity analysis."""
    task_description: str = Field(..., description="Task to analyze")


class AngelaMCPServer:
    """MCP Server for AngelaMCP multi-agent collaboration."""
    
    def __init__(self):
        self.server = Server("angelamcp")
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
        
        # Register list_tools handler
        @self.server.list_tools()
        async def list_available_tools():
            """List all available tools."""
            return [
                {
                    "name": "collaborate",
                    "description": "Orchestrate collaboration between multiple AI agents on a task",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "The task to collaborate on"
                            },
                            "agents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of agents to include (claude, openai, gemini)",
                                "default": ["claude", "openai", "gemini"]
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["debate", "parallel", "consensus", "auto"],
                                "description": "Collaboration strategy",
                                "default": "auto"
                            }
                        },
                        "required": ["task_description"]
                    }
                },
                {
                    "name": "debate",
                    "description": "Start a structured debate between AI agents on a topic",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Topic to debate"
                            },
                            "agents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Participating agents",
                                "default": ["claude", "openai", "gemini"]
                            }
                        },
                        "required": ["topic"]
                    }
                },
                {
                    "name": "analyze_task_complexity",
                    "description": "Analyze task complexity and recommend collaboration strategy",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Task to analyze"
                            }
                        },
                        "required": ["task_description"]
                    }
                },
                {
                    "name": "get_agent_status",
                    "description": "Get current status of all AI agents",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False
                    }
                }
            ]
        
        @self.server.call_tool()
        async def collaborate(arguments: Dict[str, Any]) -> CallToolResult:
            """Orchestrate collaboration between multiple AI agents."""
            try:
                # Parse request
                request = CollaborationRequest(**arguments)
                logger.info(f"Collaboration request: {request.task_description[:100]}...")
                
                # Create context
                context = TaskContext(
                    task_type=TaskType.GENERAL,
                    session_id=f"mcp_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}"
                )
                
                # Map strategy string to enum
                strategy_map = {
                    "debate": CollaborationStrategy.DEBATE,
                    "parallel": CollaborationStrategy.PARALLEL,
                    "consensus": CollaborationStrategy.CONSENSUS,
                    "single_agent": CollaborationStrategy.SINGLE_AGENT,
                    "auto": CollaborationStrategy.AUTO
                }
                
                strategy = strategy_map.get(request.strategy, CollaborationStrategy.AUTO)
                
                # Execute collaboration
                result = await self.orchestrator.execute_task(
                    request.task_description,
                    context,
                    strategy
                )
                
                # Format response
                response_text = f"""**AngelaMCP Collaboration Result**

**Strategy Used:** {result.strategy_used.value if result.strategy_used else 'unknown'}
**Success:** {result.success}
**Execution Time:** {result.execution_time:.2f}s
**Consensus Score:** {result.consensus_score:.2f}

**Final Solution:**
{result.final_solution}
"""
                
                if result.cost_breakdown:
                    cost_text = "**Cost Breakdown:**\n"
                    for agent, cost in result.cost_breakdown.items():
                        cost_text += f"- {agent}: ${cost:.4f}\n"
                    response_text += f"\n{cost_text}"
                
                if result.debate_summary:
                    response_text += f"\n**Debate Summary:**\n{result.debate_summary}"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=response_text)]
                )
                
            except Exception as e:
                logger.error(f"Collaboration failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"❌ Collaboration failed: {str(e)}")],
                    isError=True
                )
        
        @self.server.call_tool()
        async def debate(arguments: Dict[str, Any]) -> CallToolResult:
            """Start a structured debate between AI agents."""
            try:
                # Parse request
                request = DebateRequest(**arguments)
                logger.info(f"Debate request: {request.topic[:100]}...")
                
                # Create context
                context = TaskContext(
                    task_type=TaskType.DEBATE,
                    session_id=f"mcp_debate_{id(request)}"
                )
                
                # Start debate
                result = await self.orchestrator.start_debate(request.topic, context)
                
                # Format response
                if result.success:
                    response_text = f"""**AngelaMCP Debate Result**

**Topic:** {request.topic}
**Rounds Completed:** {result.rounds_completed}
**Consensus Score:** {result.consensus_score:.2f}

**Final Consensus:**
{result.final_consensus or 'No consensus reached'}

**Debate Summary:**
{result.summary or 'No summary available'}
"""
                    
                    if result.participant_votes:
                        vote_text = "**Final Votes:**\n"
                        for agent, vote_info in result.participant_votes.items():
                            vote_text += f"- {agent}: {vote_info}\n"
                        response_text += f"\n{vote_text}"
                else:
                    response_text = f"❌ Debate failed: {result.error_message or 'Unknown error'}"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=response_text)]
                )
                
            except Exception as e:
                logger.error(f"Debate failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"❌ Debate failed: {str(e)}")],
                    isError=True
                )
        
        @self.server.call_tool()
        async def analyze_task_complexity(arguments: Dict[str, Any]) -> CallToolResult:
            """Analyze task complexity and recommend collaboration strategy."""
            try:
                # Parse request
                request = TaskAnalysisRequest(**arguments)
                logger.info(f"Task analysis request: {request.task_description[:100]}...")
                
                # Create context for analysis
                context = TaskContext(task_type=TaskType.ANALYSIS)
                
                # Use orchestrator's strategy selection logic
                strategy = await self.orchestrator._select_strategy(
                    request.task_description, 
                    context
                )
                
                # Provide detailed analysis
                task_lower = request.task_description.lower()
                
                # Analyze complexity indicators
                complexity_indicators = []
                if any(word in task_lower for word in ["complex", "architecture", "system", "enterprise"]):
                    complexity_indicators.append("High complexity keywords detected")
                
                if len(request.task_description.split()) > 50:
                    complexity_indicators.append("Detailed/lengthy task description")
                
                if any(word in task_lower for word in ["compare", "analyze", "evaluate", "debate"]):
                    complexity_indicators.append("Requires multiple perspectives")
                
                # Agent capability analysis
                agent_recommendations = []
                if any(word in task_lower for word in ["code", "implement", "build", "create"]):
                    agent_recommendations.append("Claude Code - Primary implementation")
                    
                if any(word in task_lower for word in ["review", "optimize", "improve", "fix"]):
                    agent_recommendations.append("OpenAI - Code review and optimization")
                    
                if any(word in task_lower for word in ["research", "best practices", "documentation"]):
                    agent_recommendations.append("Gemini - Research and documentation")
                
                response_text = f"""**AngelaMCP Task Analysis**

**Task:** {request.task_description}

**Recommended Strategy:** {strategy.value}

**Complexity Indicators:**
{chr(10).join(f"- {indicator}" for indicator in complexity_indicators) if complexity_indicators else "- Standard complexity task"}

**Agent Recommendations:**
{chr(10).join(f"- {rec}" for rec in agent_recommendations) if agent_recommendations else "- All agents recommended"}

**Strategy Explanation:**
"""
                
                # Add strategy explanation
                if strategy == CollaborationStrategy.SINGLE_AGENT:
                    response_text += "This task appears straightforward and can be handled efficiently by a single agent (usually Claude Code)."
                elif strategy == CollaborationStrategy.DEBATE:
                    response_text += "This task benefits from multiple perspectives and structured debate between agents to reach the best solution."
                elif strategy == CollaborationStrategy.CONSENSUS:
                    response_text += "This task requires agreement from multiple agents to ensure quality and completeness."
                elif strategy == CollaborationStrategy.PARALLEL:
                    response_text += "This task can be broken down and executed in parallel by multiple agents for faster completion."
                
                return CallToolResult(
                    content=[TextContent(type="text", text=response_text)]
                )
                
            except Exception as e:
                logger.error(f"Task analysis failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"❌ Task analysis failed: {str(e)}")],
                    isError=True
                )
        
        @self.server.call_tool()
        async def get_agent_status(arguments: Dict[str, Any]) -> CallToolResult:
            """Get current status of all AI agents."""
            try:
                logger.info("Agent status request")
                
                # Get health check from orchestrator
                health_status = await self.orchestrator.health_check()
                
                response_text = "**AngelaMCP Agent Status**\n\n"
                
                for component, status in health_status.items():
                    if isinstance(status, dict):
                        if status.get("status") == "healthy":
                            emoji = "✅"
                        else:
                            emoji = "❌"
                        response_text += f"{emoji} **{component.title()}**: {status.get('status', 'unknown')}\n"
                    else:
                        if "error" in str(status).lower():
                            emoji = "❌"
                        else:
                            emoji = "✅"
                        response_text += f"{emoji} **{component.title()}**: {status}\n"
                
                # Add configuration info
                response_text += f"\n**Configuration:**\n"
                response_text += f"- OpenAI Model: {settings.openai_model}\n"
                response_text += f"- Gemini Model: {settings.gemini_model}\n"
                response_text += f"- Claude Vote Weight: {settings.claude_vote_weight}\n"
                response_text += f"- Debate Max Rounds: {settings.debate_max_rounds}\n"
                response_text += f"- Cost Tracking: {'Enabled' if settings.enable_cost_tracking else 'Disabled'}\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=response_text)]
                )
                
            except Exception as e:
                logger.error(f"Agent status check failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"❌ Agent status check failed: {str(e)}")],
                    isError=True
                )
    
    async def run(self) -> None:
        """Run the MCP server."""
        try:
            logger.info("Starting AngelaMCP MCP Server...")
            
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream, 
                    write_stream,
                    self.server.create_initialization_options()
                )
                
        except Exception as e:
            logger.error(f"MCP server run failed: {e}", exc_info=True)
            raise


async def main():
    """Main entry point for MCP server."""
    try:
        server = AngelaMCPServer()
        await server.initialize()
        await server.run()
    except KeyboardInterrupt:
        logger.info("MCP server shutdown requested")
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
