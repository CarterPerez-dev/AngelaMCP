#!/usr/bin/env python3
"""
AngelaMCP MCP Server - JSON-RPC Implementation
Fixed logging and argument handling issues for Claude Code integration.
"""

import json
import sys
import os
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

# Ensure unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Server version
__version__ = "1.0.0"

class AngelaMCPServer:
    """JSON-RPC MCP Server for AngelaMCP."""
    
    def __init__(self):
        self.orchestrator: Optional[Any] = None
        self.agents: Dict[str, Any] = {}
        self.db_manager: Optional[Any] = None
        self.initialized = False
        
    async def initialize_components(self):
        """Initialize AngelaMCP components."""
        if self.initialized:
            return True
            
        try:
            # Import here to avoid circular imports
            from src.orchestrator.manager import TaskOrchestrator
            from src.persistence.database import DatabaseManager
            from src.agents.claude_agent import ClaudeCodeAgent
            from src.agents.openai_agent import OpenAIAgent  
            from src.agents.gemini_agent import GeminiAgent
            
            # Initialize database
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize agents with error handling
            self.agents = {}
            
            try:
                self.agents["claude"] = ClaudeCodeAgent()
            except Exception as e:
                print(f"Claude agent initialization failed: {e}", file=sys.stderr)
                
            try:
                self.agents["openai"] = OpenAIAgent()
            except Exception as e:
                print(f"OpenAI agent initialization failed: {e}", file=sys.stderr)
                
            try:
                self.agents["gemini"] = GeminiAgent()
            except Exception as e:
                print(f"Gemini agent initialization failed: {e}", file=sys.stderr)
            
            # Need at least one agent
            if not self.agents:
                raise RuntimeError("No agents initialized successfully")
            
            # Initialize orchestrator
            self.orchestrator = TaskOrchestrator(
                claude_agent=self.agents.get("claude"),
                openai_agent=self.agents.get("openai"),
                gemini_agent=self.agents.get("gemini"),
                db_manager=self.db_manager
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Component initialization failed: {e}", file=sys.stderr)
            return False

    def send_response(self, response: Dict[str, Any]):
        """Send a JSON-RPC response"""
        print(json.dumps(response), flush=True)

    def handle_initialize(self, request_id: Any) -> Dict[str, Any]:
        """Handle initialization"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "angelamcp",
                    "version": __version__
                }
            }
        }

    def handle_tools_list(self, request_id: Any) -> Dict[str, Any]:
        """List available tools"""
        tools = [
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
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools
            }
        }

    async def handle_tool_call(self, request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            
            # Ensure components are initialized
            if not self.initialized:
                await self.initialize_components()
            
            if tool_name == "collaborate":
                result = await self._handle_collaborate(arguments)
            elif tool_name == "debate":
                result = await self._handle_debate(arguments)
            elif tool_name == "analyze_task_complexity":
                result = await self._handle_analyze_task_complexity(arguments)
            elif tool_name == "get_agent_status":
                result = await self._handle_get_agent_status(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Tool {tool_name} failed: {str(e)}"
                }
            }

    async def _handle_collaborate(self, arguments: Dict[str, Any]) -> str:
        """Handle collaboration requests."""
        try:
            task_description = arguments.get("task_description", "")
            strategy = arguments.get("strategy", "auto")
            
            if not task_description:
                return "❌ Task description is required for collaboration"
            
            if not self.orchestrator:
                return "❌ AngelaMCP orchestrator not initialized"
            
            # Import here to avoid circular imports
            from src.agents.base import TaskContext, TaskType
            from src.orchestrator.manager import CollaborationStrategy
            
            # Create context
            context = TaskContext(
                task_type=TaskType.GENERAL,
                session_id=f"mcp_collab_{id(arguments)}"
            )
            
            # Map strategy string to enum
            strategy_map = {
                "debate": CollaborationStrategy.DEBATE,
                "parallel": CollaborationStrategy.PARALLEL,
                "consensus": CollaborationStrategy.CONSENSUS,
                "single_agent": CollaborationStrategy.SINGLE_AGENT,
                "auto": CollaborationStrategy.AUTO
            }
            
            collab_strategy = strategy_map.get(strategy, CollaborationStrategy.AUTO)
            
            # Execute collaboration
            result = await self.orchestrator.execute_task(
                task_description,
                context,
                collab_strategy
            )
            
            # Format response
            strategy_name = result.strategy_used.value if result.strategy_used else 'unknown'
            response_text = f"""**AngelaMCP Collaboration Result**

**Strategy Used:** {strategy_name}
**Success:** {result.success}
**Execution Time:** {result.execution_time:.2f}s
**Consensus Score:** {result.consensus_score:.2f}

**Final Solution:**
{result.final_solution}
"""
            
            if result.cost_breakdown:
                cost_text = "\n**Cost Breakdown:**\n"
                for agent, cost in result.cost_breakdown.items():
                    cost_text += f"- {agent}: ${cost:.4f}\n"
                response_text += cost_text
            
            if result.debate_summary:
                response_text += f"\n**Debate Summary:**\n{result.debate_summary}"
            
            return response_text
            
        except Exception as e:
            return f"❌ Collaboration failed: {str(e)}"

    async def _handle_debate(self, arguments: Dict[str, Any]) -> str:
        """Handle debate requests."""
        try:
            topic = arguments.get("topic", "")
            
            if not topic:
                return "❌ Topic is required for debate"
            
            if not self.orchestrator:
                return "❌ AngelaMCP orchestrator not initialized"
            
            # Import here to avoid circular imports
            from src.agents.base import TaskContext, TaskType
            
            # Create context
            context = TaskContext(
                task_type=TaskType.DEBATE,
                session_id=f"mcp_debate_{id(arguments)}"
            )
            
            # Start debate
            result = await self.orchestrator.start_debate(topic, context)
            
            # Format response
            if result.success:
                response_text = f"""**AngelaMCP Debate Result**

**Topic:** {topic}
**Rounds Completed:** {result.rounds_completed}
**Consensus Score:** {result.consensus_score:.2f}

**Final Consensus:**
{result.final_consensus or 'No consensus reached'}

**Debate Summary:**
{result.summary or 'No summary available'}
"""
                
                if result.participant_votes:
                    vote_text = "\n**Final Votes:**\n"
                    for agent, vote_info in result.participant_votes.items():
                        vote_text += f"- {agent}: {vote_info}\n"
                    response_text += vote_text
            else:
                response_text = f"❌ Debate failed: {result.error_message or 'Unknown error'}"
            
            return response_text
            
        except Exception as e:
            return f"❌ Debate failed: {str(e)}"

    async def _handle_analyze_task_complexity(self, arguments: Dict[str, Any]) -> str:
        """Handle task complexity analysis requests."""
        try:
            task_description = arguments.get("task_description", "")
            
            if not task_description:
                return "❌ Task description is required for analysis"
            
            if not self.orchestrator:
                # Provide basic analysis without orchestrator
                return f"""**AngelaMCP Task Analysis**

**Task:** {task_description}

**Basic Analysis:** This appears to be a {len(task_description.split())} word task.
**Recommended Strategy:** auto
**Note:** Full analysis requires orchestrator initialization.
"""
            
            # Import here to avoid circular imports
            from src.agents.base import TaskContext, TaskType
            from src.orchestrator.manager import CollaborationStrategy
            
            # Create context for analysis
            context = TaskContext(task_type=TaskType.ANALYSIS)
            
            # Use orchestrator's strategy selection logic
            strategy = await self.orchestrator._select_strategy(
                task_description, 
                context
            )
            
            # Provide detailed analysis
            task_lower = task_description.lower()
            
            # Analyze complexity indicators
            complexity_indicators = []
            if any(word in task_lower for word in ["complex", "architecture", "system", "enterprise"]):
                complexity_indicators.append("High complexity keywords detected")
            
            if len(task_description.split()) > 50:
                complexity_indicators.append("Detailed/lengthy task description")
            
            if any(word in task_lower for word in ["compare", "analyze", "evaluate", "debate"]):
                complexity_indicators.append("Requires multiple perspectives")
            
            indicators_text = "\n".join(f"- {indicator}" for indicator in complexity_indicators) if complexity_indicators else "- Standard complexity task"
            
            response_text = f"""**AngelaMCP Task Analysis**

**Task:** {task_description}

**Recommended Strategy:** {strategy.value}

**Complexity Indicators:**
{indicators_text}

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
            
            return response_text
            
        except Exception as e:
            return f"❌ Task analysis failed: {str(e)}"

    async def _handle_get_agent_status(self, arguments: Dict[str, Any]) -> str:
        """Handle agent status requests."""
        try:
            
            # Import settings here to avoid circular imports
            try:
                from config.settings import settings
            except Exception:
                settings = None
            
            # Basic status
            response_text = f"""**AngelaMCP Agent Status**

✅ **MCP Server**: Running (JSON-RPC v{__version__})
✅ **Database**: {'Connected' if self.db_manager else 'Not initialized'}  
✅ **Orchestrator**: {'Ready' if self.orchestrator else 'Not initialized'}

**Available Agents:**
"""
            
            if self.agents:
                for agent_name in self.agents.keys():
                    response_text += f"✅ **{agent_name.title()}**: Available\n"
            else:
                response_text += "⚠️ Agents not yet initialized\n"
            
            if settings:
                response_text += f"""
**Configuration:**
- OpenAI Model: {getattr(settings, 'openai_model', 'not configured')}
- Gemini Model: {getattr(settings, 'gemini_model', 'not configured')}
- Claude Vote Weight: {getattr(settings, 'claude_vote_weight', 'not configured')}
- Debate Max Rounds: {getattr(settings, 'debate_max_rounds', 'not configured')}
- Cost Tracking: {'Enabled' if getattr(settings, 'enable_cost_tracking', False) else 'Disabled'}
"""
            else:
                response_text += "\n**Configuration:** Not loaded"
            
            return response_text
            
        except Exception as e:
            return f"❌ Agent status check failed: {str(e)}"

    async def run(self):
        """Main server loop"""
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                method = request.get("method")
                request_id = request.get("id")
                params = request.get("params", {})
                
                if method == "initialize":
                    response = self.handle_initialize(request_id)
                elif method == "tools/list":
                    response = self.handle_tools_list(request_id)
                elif method == "tools/call":
                    response = await self.handle_tool_call(request_id, params)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }
                
                self.send_response(response)
                
            except json.JSONDecodeError as e:
                continue
            except EOFError:
                break
            except Exception as e:
                if 'request_id' in locals():
                    self.send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        }
                    })

async def main():
    """Main entry point for JSON-RPC MCP server."""
    try:
        server = AngelaMCPServer()
        await server.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
