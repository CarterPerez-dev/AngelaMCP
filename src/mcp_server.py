# src/mcp_server.py
#!/usr/bin/env python3
"""
MCP Server for AngelaMCP - Fixed for MCP SDK compatibility.

This version handles the initialization parameter correctly.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, Any, List, Optional, Union, Sequence
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MCP SDK imports
from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource
)
from mcp import stdio_server

# AngelaMCP imports
from config.settings import settings
from src.agents import ClaudeCodeAgent, OpenAIAgent, GeminiAgent, TaskContext, TaskType
from src.orchestrator import TaskOrchestrator, CollaborationStrategy
from src.persistence.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger("mcp_server")

# Create the MCP server instance
server = Server("angelamcp")

# Global instances
orchestrator: Optional[TaskOrchestrator] = None
db_manager: Optional[DatabaseManager] = None
agents: Dict[str, Any] = {}


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """
    List available tools for the MCP client.
    """
    return [
        Tool(
            name="collaborate",
            description="Orchestrate collaboration between multiple AI agents on a task",
            inputSchema={
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
        ),
        Tool(
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
                        "description": "Participating agents",
                        "default": ["claude", "openai", "gemini"]
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
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
        Tool(
            name="get_agent_status",
            description="Get current status of all AI agents",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: Any
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """
    Handle tool execution requests from the MCP client.
    """
    global orchestrator
    
    # Ensure orchestrator is initialized
    if orchestrator is None:
        return [TextContent(type="text", text="❌ Server not fully initialized. Please try again in a moment.")]
    
    logger.info(f"Tool called: {name} with args: {arguments}")
    
    try:
        if name == "collaborate":
            # Create context
            context = TaskContext(
                task_type=TaskType.GENERAL,
                session_id=f"mcp_collab_{id(arguments)}"
            )
            
            # Map strategy
            strategy_map = {
                "debate": CollaborationStrategy.DEBATE,
                "parallel": CollaborationStrategy.PARALLEL,
                "consensus": CollaborationStrategy.CONSENSUS,
                "auto": CollaborationStrategy.AUTO
            }
            strategy = strategy_map.get(
                arguments.get("strategy", "auto"), 
                CollaborationStrategy.AUTO
            )
            
            # Execute collaboration
            result = await orchestrator.execute_task(
                arguments["task_description"],
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
{result.final_solution}"""
            
            if result.cost_breakdown:
                response_text += "\n\n**Cost Breakdown:**"
                for agent, cost in result.cost_breakdown.items():
                    response_text += f"\n- {agent}: ${cost:.4f}"
            
            if result.debate_summary:
                response_text += f"\n\n**Debate Summary:**\n{result.debate_summary}"
            
            return [TextContent(type="text", text=response_text)]
            
        elif name == "debate":
            # Create context for debate
            context = TaskContext(
                task_type=TaskType.DEBATE,
                session_id=f"mcp_debate_{id(arguments)}"
            )
            
            # Start debate
            result = await orchestrator.start_debate(
                arguments["topic"], 
                context
            )
            
            # Format response
            if result.success:
                response_text = f"""**AngelaMCP Debate Result**

**Topic:** {arguments['topic']}
**Rounds Completed:** {result.rounds_completed}
**Consensus Score:** {result.consensus_score:.2f}

**Final Consensus:**
{result.final_consensus or 'No consensus reached'}

**Debate Summary:**
{result.summary or 'No summary available'}"""
                
                if result.participant_votes:
                    response_text += "\n\n**Final Votes:**"
                    for agent, vote_info in result.participant_votes.items():
                        response_text += f"\n- {agent}: {vote_info}"
            else:
                response_text = f"❌ Debate failed: {result.error_message or 'Unknown error'}"
            
            return [TextContent(type="text", text=response_text)]
            
        elif name == "analyze_task_complexity":
            # Create context for analysis
            context = TaskContext(task_type=TaskType.ANALYSIS)
            
            # Use orchestrator's strategy selection logic
            strategy = await orchestrator._select_strategy(
                arguments["task_description"], 
                context
            )
            
            # Analyze complexity
            task_lower = arguments["task_description"].lower()
            
            complexity_indicators = []
            if any(word in task_lower for word in ["complex", "architecture", "system", "enterprise"]):
                complexity_indicators.append("High complexity keywords detected")
            if len(arguments["task_description"].split()) > 50:
                complexity_indicators.append("Detailed/lengthy task description")
            if any(word in task_lower for word in ["compare", "analyze", "evaluate", "debate"]):
                complexity_indicators.append("Requires multiple perspectives")
            
            agent_recommendations = []
            if any(word in task_lower for word in ["code", "implement", "build", "create"]):
                agent_recommendations.append("Claude Code - Primary implementation")
            if any(word in task_lower for word in ["review", "optimize", "improve", "fix"]):
                agent_recommendations.append("OpenAI - Code review and optimization")
            if any(word in task_lower for word in ["research", "best practices", "documentation"]):
                agent_recommendations.append("Gemini - Research and documentation")
            
            response_text = f"""**AngelaMCP Task Analysis**

**Task:** {arguments['task_description']}

**Recommended Strategy:** {strategy.value}

**Complexity Indicators:**
{chr(10).join(f"- {ind}" for ind in complexity_indicators) if complexity_indicators else "- Standard complexity task"}

**Agent Recommendations:**
{chr(10).join(f"- {rec}" for rec in agent_recommendations) if agent_recommendations else "- All agents recommended"}

**Strategy Explanation:**
"""
            
            if strategy == CollaborationStrategy.SINGLE_AGENT:
                response_text += "This task appears straightforward and can be handled efficiently by a single agent."
            elif strategy == CollaborationStrategy.DEBATE:
                response_text += "This task benefits from multiple perspectives and structured debate between agents."
            elif strategy == CollaborationStrategy.CONSENSUS:
                response_text += "This task requires agreement from multiple agents to ensure quality and completeness."
            elif strategy == CollaborationStrategy.PARALLEL:
                response_text += "This task can be broken down and executed in parallel by multiple agents."
            
            return [TextContent(type="text", text=response_text)]
            
        elif name == "get_agent_status":
            # Get health check from orchestrator
            health_status = await orchestrator.health_check()
            
            response_text = "**AngelaMCP Agent Status**\n\n"
            
            for component, status in health_status.items():
                if isinstance(status, dict):
                    emoji = "✅" if status.get("status") == "healthy" else "❌"
                    response_text += f"{emoji} **{component.title()}**: {status.get('status', 'unknown')}\n"
                else:
                    emoji = "❌" if "error" in str(status).lower() else "✅"
                    response_text += f"{emoji} **{component.title()}**: {status}\n"
            
            response_text += f"\n**Configuration:**\n"
            response_text += f"- OpenAI Model: {settings.openai_model}\n"
            response_text += f"- Gemini Model: {settings.gemini_model}\n"
            response_text += f"- Claude Vote Weight: {settings.claude_vote_weight}\n"
            response_text += f"- Debate Max Rounds: {settings.debate_max_rounds}\n"
            response_text += f"- Cost Tracking: {'Enabled' if settings.enable_cost_tracking else 'Disabled'}\n"
            
            return [TextContent(type="text", text=response_text)]
            
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}", exc_info=True)
        return [TextContent(type="text", text=f"❌ Tool {name} failed: {str(e)}")]


async def initialize_angelamcp() -> None:
    """Initialize all AngelaMCP components."""
    global orchestrator, db_manager, agents
    
    try:
        logger.info("Initializing AngelaMCP MCP Server...")
        
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Initialize agents
        agents = {
            "claude": ClaudeCodeAgent(),
            "openai": OpenAIAgent(), 
            "gemini": GeminiAgent()
        }
        
        # Initialize orchestrator
        orchestrator = TaskOrchestrator(
            claude_agent=agents["claude"],
            openai_agent=agents["openai"],
            gemini_agent=agents["gemini"],
            db_manager=db_manager
        )
        
        logger.info("AngelaMCP MCP Server initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP server: {e}", exc_info=True)
        raise


async def main():
    """Main entry point for MCP server."""
    try:
        # Initialize AngelaMCP components
        await initialize_angelamcp()
        
        # Run the MCP server using the standard pattern
        logger.info("Starting AngelaMCP MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            # The initialization_options parameter might be a dict or None
            # Let's try passing None or an empty dict
            initialization_options = {}
            
            # Run with the initialization options
            await server.run(read_stream, write_stream, initialization_options)
            
    except KeyboardInterrupt:
        logger.info("MCP server shutdown requested")
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
