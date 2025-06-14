"""
Unified Task Orchestrator for AngelaMCP multi-agent collaboration.

This is the core brain that coordinates between Claude Code, OpenAI, and Gemini agents.
I'm implementing a production-grade orchestration system with debate, voting, and consensus.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from src.agents import BaseAgent, AgentType, AgentResponse, TaskContext, TaskType
from src.agents import ClaudeCodeAgent
from src.agents import OpenAIAgent
from src.agents import GeminiAgent
from src.orchestrator.debate import DebateProtocol, DebateResult
from src.orchestrator.voting import VotingSystem, VotingResult
from src.orchestrator.task_queue import AsyncTaskQueue
from src.persistence import DatabaseManager
from src.persistence import Conversation, TaskExecution
from src.utils import get_logger
from src.utils import OrchestrationError
from config import settings

logger = get_logger("orchestrator.manager")


class CollaborationStrategy(str, Enum):
    """Strategy for agent collaboration."""
    SINGLE_AGENT = "single_agent"
    PARALLEL = "parallel"
    DEBATE = "debate"
    CONSENSUS = "consensus"
    AUTO = "auto"  # Let orchestrator decide


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class CollaborationResult:
    """Result of a collaboration session."""
    success: bool
    final_solution: str
    agent_responses: List[Dict[str, Any]] = field(default_factory=list)
    consensus_score: float = 0.0
    debate_summary: Optional[str] = None
    execution_time: float = 0.0
    cost_breakdown: Optional[Dict[str, float]] = None
    strategy_used: Optional[CollaborationStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class TaskOrchestrator:
    """
    Unified orchestrator for multi-agent collaboration.

    Coordinates Claude Code, OpenAI, and Gemini agents for complex tasks.
    """
    
    def __init__(
        self,
        claude_agent: Optional[ClaudeCodeAgent],
        openai_agent: Optional[OpenAIAgent],
        gemini_agent: Optional[GeminiAgent],
        db_manager: DatabaseManager
    ):
        self.claude_agent = claude_agent
        self.openai_agent = openai_agent
        self.gemini_agent = gemini_agent
        self.db_manager = db_manager
        
        # Initialize sub-systems
        self.debate_protocol = DebateProtocol()
        self.voting_system = VotingSystem()
        self.task_queue = AsyncTaskQueue()
        
        self.logger = get_logger("orchestrator")
        
        # Agent mapping - only include available agents
        self.agents = {}
        if claude_agent:
            self.agents[AgentType.CLAUDE] = claude_agent
        if openai_agent:
            self.agents[AgentType.OPENAI] = openai_agent
        if gemini_agent:
            self.agents[AgentType.GEMINI] = gemini_agent
    
    async def execute_task(
        self, 
        task_description: str, 
        context: TaskContext,
        strategy: CollaborationStrategy = CollaborationStrategy.AUTO
    ) -> CollaborationResult:
        """Execute a task with automatic or specified strategy."""
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Executing task {task_id[:8]}: {task_description[:100]}...")
            
            # Analyze task complexity if strategy is AUTO
            if strategy == CollaborationStrategy.AUTO:
                strategy = await self._select_strategy(task_description, context)
                self.logger.info(f"Auto-selected strategy: {strategy.value}")
            
            # Execute based on strategy
            if strategy == CollaborationStrategy.SINGLE_AGENT:
                result = await self._single_agent_execution(task_description, context)
            elif strategy == CollaborationStrategy.PARALLEL:
                result = await self._parallel_execution(task_description, context)
            elif strategy == CollaborationStrategy.DEBATE:
                result = await self._debate_execution(task_description, context)
            elif strategy == CollaborationStrategy.CONSENSUS:
                result = await self._consensus_execution(task_description, context)
            else:
                raise OrchestrationError(f"Unknown strategy: {strategy}")
            
            # Add metadata
            result.strategy_used = strategy
            result.execution_time = time.time() - start_time
            
            # Save to database
            await self._save_task_execution(task_id, task_description, result, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return CollaborationResult(
                success=False,
                final_solution="",
                execution_time=time.time() - start_time,
                strategy_used=strategy,
                error_message=str(e)
            )
    
    async def collaborate(
        self,
        task_description: str,
        strategy: CollaborationStrategy = CollaborationStrategy.DEBATE,
        context: Optional[TaskContext] = None
    ) -> CollaborationResult:
        """Collaborate on a task using specified strategy."""
        if context is None:
            context = TaskContext()
        
        return await self.execute_task(task_description, context, strategy)
    
    async def start_debate(
        self,
        topic: str,
        context: Optional[TaskContext] = None
    ) -> DebateResult:
        """Start a structured debate on a topic."""
        if context is None:
            context = TaskContext(task_type=TaskType.DEBATE)
        
        # Use only available agents
        agents = [agent for agent in [self.claude_agent, self.openai_agent, self.gemini_agent] if agent is not None]
        
        if len(agents) < 2:
            # Return failed debate result
            return DebateResult(
                debate_id=str(uuid.uuid4()),
                topic=topic,
                success=False,
                error_message="Need at least 2 agents for debate",
                participating_agents=[agent.name for agent in agents] if agents else []
            )
        
        return await self.debate_protocol.conduct_debate(topic, agents, context)
    
    async def _select_strategy(
        self, 
        task_description: str, 
        context: TaskContext
    ) -> CollaborationStrategy:
        """Automatically select the best strategy for a task."""
        
        # Simple heuristics for strategy selection
        task_lower = task_description.lower()
        
        # Keywords that suggest debate/collaboration
        collaboration_keywords = [
            "compare", "debate", "discuss", "analyze", "evaluate",
            "pros and cons", "best approach", "recommend", "decide"
        ]
        
        # Keywords that suggest single agent
        simple_keywords = [
            "create", "write", "implement", "build", "generate",
            "fix", "update", "modify", "add"
        ]
        
        if any(keyword in task_lower for keyword in collaboration_keywords):
            return CollaborationStrategy.DEBATE
        elif any(keyword in task_lower for keyword in simple_keywords):
            return CollaborationStrategy.SINGLE_AGENT
        else:
            # Default to consensus for complex tasks
            return CollaborationStrategy.CONSENSUS
    
    async def _single_agent_execution(
        self, 
        task_description: str, 
        context: TaskContext
    ) -> CollaborationResult:
        """Execute task with the best single agent (usually Claude)."""
        
        try:
            # Use Claude as primary agent, fallback to others if needed
            agent = self.claude_agent or self.openai_agent or self.gemini_agent
            agent_name = "claude" if self.claude_agent else ("openai" if self.openai_agent else "gemini")
            
            if not agent:
                raise OrchestrationError("No agents available for execution")
            
            response = await agent.generate(task_description, context)
            
            return CollaborationResult(
                success=response.success if hasattr(response, 'success') else True,
                final_solution=response.content,
                agent_responses=[{
                    "agent": agent_name,
                    "response": response.content,
                    "metadata": getattr(response, 'metadata', {})
                }],
                cost_breakdown=self._calculate_costs([response])
            )
            
        except Exception as e:
            self.logger.error(f"Single agent execution failed: {e}")
            return CollaborationResult(
                success=False,
                final_solution="",
                error_message=str(e)
            )
    
    async def _parallel_execution(
        self, 
        task_description: str, 
        context: TaskContext
    ) -> CollaborationResult:
        """Execute task with all agents in parallel."""
        
        try:
            # Run available agents in parallel
            tasks = []
            agent_names = []
            
            if self.claude_agent:
                tasks.append(self.claude_agent.generate(task_description, context))
                agent_names.append("claude")
            if self.openai_agent:
                tasks.append(self.openai_agent.generate(task_description, context))
                agent_names.append("openai")
            if self.gemini_agent:
                tasks.append(self.gemini_agent.generate(task_description, context))
                agent_names.append("gemini")
            
            if not tasks:
                raise OrchestrationError("No agents available for parallel execution")
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful responses
            successful_responses = []
            
            for i, response in enumerate(responses):
                if not isinstance(response, Exception):
                    successful_responses.append({
                        "agent": agent_names[i],
                        "response": response.content,
                        "metadata": getattr(response, 'metadata', {})
                    })
            
            if not successful_responses:
                raise OrchestrationError("All agents failed")
            
            # Use Claude's response as primary, others as supporting
            claude_response = next(
                (r for r in successful_responses if r["agent"] == "claude"), 
                successful_responses[0]
            )
            
            return CollaborationResult(
                success=True,
                final_solution=claude_response["response"],
                agent_responses=successful_responses,
                cost_breakdown=self._calculate_costs(responses)
            )
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return CollaborationResult(
                success=False,
                final_solution="",
                error_message=str(e)
            )
    
    async def _debate_execution(
        self, 
        task_description: str, 
        context: TaskContext
    ) -> CollaborationResult:
        """Execute task through structured debate."""
        
        try:
            # Use only available agents for debate
            agents = [agent for agent in [self.claude_agent, self.openai_agent, self.gemini_agent] if agent is not None]
            
            if len(agents) < 2:
                raise OrchestrationError("Need at least 2 agents for debate")
            
            debate_result = await self.debate_protocol.conduct_debate(
                task_description, agents, context
            )
            
            if debate_result.success and debate_result.rounds:
                # Get final proposals from last round
                final_round = debate_result.rounds[-1]
                
                if final_round.proposals:
                    # Vote on proposals
                    voting_result = await self.voting_system.conduct_voting(
                        final_round.proposals, agents, context
                    )
                    
                    if voting_result.success and voting_result.winning_proposal:
                        return CollaborationResult(
                            success=True,
                            final_solution=voting_result.winning_proposal.content,
                            consensus_score=voting_result.consensus_score,
                            debate_summary=debate_result.summary,
                            agent_responses=[{
                                "agent": p.agent_name,
                                "response": p.content,
                                "score": getattr(p, 'score', 0)
                            } for p in final_round.proposals]
                        )
                
                # Fallback to consensus from debate
                return CollaborationResult(
                    success=bool(debate_result.final_consensus),
                    final_solution=debate_result.final_consensus or "No consensus reached",
                    consensus_score=debate_result.consensus_score,
                    debate_summary=debate_result.summary
                )
            
            raise OrchestrationError("Debate failed to produce results")
            
        except Exception as e:
            self.logger.error(f"Debate execution failed: {e}")
            return CollaborationResult(
                success=False,
                final_solution="",
                error_message=str(e)
            )
    
    async def _consensus_execution(
        self, 
        task_description: str, 
        context: TaskContext
    ) -> CollaborationResult:
        """Execute task requiring consensus from all agents."""
        
        try:
            # Get initial responses from available agents
            agents = [agent for agent in [self.claude_agent, self.openai_agent, self.gemini_agent] if agent is not None]
            
            if not agents:
                raise OrchestrationError("No agents available for consensus")
            
            responses = []
            for agent in agents:
                response = await agent.generate(task_description, context)
                responses.append(response)
            
            # Check for natural consensus
            consensus_score = self._calculate_consensus(responses)
            
            if consensus_score >= 0.8:  # High consensus
                # Use best response (prefer Claude)
                best_response = responses[0]  # Claude
                
                return CollaborationResult(
                    success=True,
                    final_solution=best_response.content,
                    consensus_score=consensus_score,
                    agent_responses=[{
                        "agent": f"agent_{i}",
                        "response": r.content
                    } for i, r in enumerate(responses)]
                )
            else:
                # Need debate to reach consensus
                return await self._debate_execution(task_description, context)
            
        except Exception as e:
            self.logger.error(f"Consensus execution failed: {e}")
            return CollaborationResult(
                success=False,
                final_solution="",
                error_message=str(e)
            )
    
    def _calculate_consensus(self, responses: List[AgentResponse]) -> float:
        """Calculate consensus score between responses."""
        # Simple consensus calculation - would need more sophisticated NLP
        if len(responses) < 2:
            return 1.0
        
        # For now, return moderate consensus to trigger debate
        return 0.6
    
    def _calculate_costs(self, responses: List[Any]) -> Dict[str, float]:
        """Calculate cost breakdown for responses."""
        costs = {}
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                continue
                
            agent_name = ["claude", "openai", "gemini"][i] if i < 3 else f"agent_{i}"
            
            # Calculate based on token usage if available
            if hasattr(response, 'token_usage') and response.token_usage:
                input_tokens = response.token_usage.get('input_tokens', 0)
                output_tokens = response.token_usage.get('output_tokens', 0)
                
                if agent_name == "openai":
                    costs[agent_name] = (
                        input_tokens * settings.openai_input_cost / 1000 +
                        output_tokens * settings.openai_output_cost / 1000
                    )
                elif agent_name == "gemini":
                    costs[agent_name] = (
                        input_tokens * settings.gemini_input_cost / 1000 +
                        output_tokens * settings.gemini_output_cost / 1000
                    )
                else:
                    costs[agent_name] = 0.0  # Claude Code is free
            else:
                costs[agent_name] = 0.0
        
        return costs
    
    async def _save_task_execution(
        self,
        task_id: str,
        task_description: str,
        result: CollaborationResult,
        context: TaskContext
    ):
        """Save task execution to database."""
        try:
            async with self.db_manager.get_session() as session:
                execution = TaskExecution(
                    id=task_id,
                    task_description=task_description,
                    strategy=result.strategy_used.value if result.strategy_used else "unknown",
                    success=result.success,
                    final_solution=result.final_solution,
                    execution_time_ms=result.execution_time * 1000,
                    consensus_score=result.consensus_score,
                    agent_responses=result.agent_responses,
                    cost_breakdown=result.cost_breakdown,
                    metadata_json=result.metadata,
                    conversation_id=context.conversation_id
                )
                
                session.add(execution)
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save task execution: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        status = {}
        
        try:
            # Check database
            await self.db_manager.health_check()
            status["database"] = "healthy"
        except Exception as e:
            status["database"] = f"error: {e}"
        
        # Check agents
        for agent_type, agent in self.agents.items():
            try:
                health = await agent.health_check()
                status[f"agent_{agent_type.value}"] = health
            except Exception as e:
                status[f"agent_{agent_type.value}"] = f"error: {e}"
        
        return status
