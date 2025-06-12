"""
Task orchestrator for AngelaMCP multi-agent collaboration.

This module implements the core orchestration engine that manages task distribution,
agent coordination, and workflow execution across Claude Code, OpenAI, and Gemini agents.
I'm building a production-grade system that can handle complex multi-step tasks with
intelligent agent selection and fallback mechanisms.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from pydantic import BaseModel, Field

from src.agents.base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType, AgentRole,
    agent_registry
)
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent
from src.persistence.database import DatabaseManager
from src.persistence.models import Conversation, Message, TaskExecution
from src.utils.logger import get_logger, log_context, AsyncPerformanceLogger
from config.settings import settings

logger = get_logger("orchestration.orchestrator")


class TaskPriority(str, Enum):
    """Task priority levels for orchestration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrchestrationStrategy(str, Enum):
    """Strategies for agent orchestration."""
    SINGLE_AGENT = "single_agent"          # Use best agent for task
    PARALLEL = "parallel"                  # Run multiple agents in parallel
    SEQUENTIAL = "sequential"              # Run agents in sequence
    DEBATE = "debate"                      # Enable debate between agents
    CONSENSUS = "consensus"                # Require consensus from multiple agents


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    success: bool
    content: str
    agent_responses: List[AgentResponse] = field(default_factory=list)
    execution_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    strategy_used: Optional[OrchestrationStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class OrchestrationTask(BaseModel):
    """Task definition for orchestration."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str = Field(description="Task description or prompt")
    task_type: TaskType = Field(description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    
    # Strategy configuration
    strategy: OrchestrationStrategy = Field(default=OrchestrationStrategy.SINGLE_AGENT)
    preferred_agents: List[AgentType] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    
    # Collaboration settings
    enable_debate: bool = Field(default=False)
    max_debate_rounds: int = Field(default=3)
    require_consensus: bool = Field(default=False)
    consensus_threshold: float = Field(default=0.7)
    
    # Resource constraints
    max_tokens: Optional[int] = Field(None)
    max_cost_usd: Optional[float] = Field(None)
    timeout_seconds: Optional[int] = Field(None)
    
    # Context
    conversation_id: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
    context_data: Dict[str, Any] = Field(default_factory=dict)


class TaskOrchestrator:
    """
    Core task orchestrator for managing multi-agent collaboration.
    
    I'm implementing intelligent task routing, agent selection, and coordination
    to maximize the effectiveness of multi-agent problem solving.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = get_logger("orchestration.task_orchestrator")
        
        # Task tracking
        self._active_tasks: Dict[str, OrchestrationTask] = {}
        self._task_results: Dict[str, TaskResult] = {}
        
        # Agent selection weights (higher = preferred)
        self._agent_weights = {
            AgentType.CLAUDE_CODE: {
                TaskType.CODE_GENERATION: 1.0,
                TaskType.CODE_REVIEW: 0.8,
                TaskType.DEBUGGING: 1.0,
                TaskType.TESTING: 0.9,
                TaskType.DOCUMENTATION: 0.7,
                TaskType.ANALYSIS: 0.6,
                TaskType.RESEARCH: 0.4,
                TaskType.COLLABORATION: 0.8,
                TaskType.CUSTOM: 0.7
            },
            AgentType.OPENAI: {
                TaskType.CODE_GENERATION: 0.7,
                TaskType.CODE_REVIEW: 1.0,
                TaskType.DEBUGGING: 0.8,
                TaskType.TESTING: 0.7,
                TaskType.DOCUMENTATION: 0.8,
                TaskType.ANALYSIS: 1.0,
                TaskType.RESEARCH: 0.8,
                TaskType.COLLABORATION: 0.9,
                TaskType.CUSTOM: 0.8
            },
            AgentType.GEMINI: {
                TaskType.CODE_GENERATION: 0.6,
                TaskType.CODE_REVIEW: 0.7,
                TaskType.DEBUGGING: 0.6,
                TaskType.TESTING: 0.6,
                TaskType.DOCUMENTATION: 0.9,
                TaskType.ANALYSIS: 0.8,
                TaskType.RESEARCH: 1.0,
                TaskType.COLLABORATION: 0.7,
                TaskType.CUSTOM: 0.9
            }
        }
        
        # Performance tracking
        self._total_tasks = 0
        self._successful_tasks = 0
        self._total_cost = 0.0
        self._start_time = time.time()
        
        self.logger.info("Task orchestrator initialized")
    
    def _select_best_agent(self, task: OrchestrationTask) -> Optional[BaseAgent]:
        """Select the best agent for a given task."""
        available_agents = agent_registry.get_all_agents()
        
        if not available_agents:
            self.logger.error("No agents available for task execution")
            return None
        
        # Filter by preferred agents if specified
        if task.preferred_agents:
            available_agents = [
                agent for agent in available_agents
                if agent.agent_type in task.preferred_agents
            ]
        
        # Filter by required capabilities
        if task.required_capabilities:
            available_agents = [
                agent for agent in available_agents
                if all(agent.supports_capability(cap) for cap in task.required_capabilities)
            ]
        
        if not available_agents:
            self.logger.warning("No agents match task requirements")
            return None
        
        # Score agents based on task type and capabilities
        agent_scores = []
        for agent in available_agents:
            base_score = self._agent_weights.get(agent.agent_type, {}).get(task.task_type, 0.5)
            
            # Bonus for required capabilities
            capability_bonus = 0.1 * len([
                cap for cap in task.required_capabilities
                if agent.supports_capability(cap)
            ])
            
            # Performance-based adjustment
            metrics = agent.performance_metrics
            if metrics["total_requests"] > 0:
                success_rate = (metrics["total_requests"] - 
                              metrics.get("failed_requests", 0)) / metrics["total_requests"]
                performance_bonus = (success_rate - 0.8) * 0.2  # Bonus for >80% success rate
            else:
                performance_bonus = 0
            
            total_score = base_score + capability_bonus + performance_bonus
            agent_scores.append((agent, total_score))
            
            self.logger.debug(
                f"Agent {agent.name} score: {total_score:.3f} "
                f"(base: {base_score}, capability: {capability_bonus}, "
                f"performance: {performance_bonus})"
            )
        
        # Sort by score and return best agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        best_agent = agent_scores[0][0]
        
        self.logger.info(
            f"Selected agent {best_agent.name} for task {task.task_id} "
            f"(score: {agent_scores[0][1]:.3f})"
        )
        
        return best_agent
    
    def _select_multiple_agents(self, task: OrchestrationTask, count: int) -> List[BaseAgent]:
        """Select multiple agents for parallel or collaborative execution."""
        available_agents = agent_registry.get_all_agents()
        
        if len(available_agents) < count:
            self.logger.warning(f"Only {len(available_agents)} agents available, requested {count}")
            return available_agents
        
        # Use similar scoring logic as single agent selection
        agent_scores = []
        for agent in available_agents:
            base_score = self._agent_weights.get(agent.agent_type, {}).get(task.task_type, 0.5)
            
            # Add diversity bonus to encourage different agent types
            type_penalty = sum(1 for scored_agent, _ in agent_scores 
                             if scored_agent.agent_type == agent.agent_type) * 0.1
            
            total_score = base_score - type_penalty
            agent_scores.append((agent, total_score))
        
        # Sort and take top agents
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agents = [agent for agent, _ in agent_scores[:count]]
        
        self.logger.info(
            f"Selected {len(selected_agents)} agents for collaborative task: "
            f"{[agent.name for agent in selected_agents]}"
        )
        
        return selected_agents
    
    async def _execute_single_agent(self, task: OrchestrationTask, agent: BaseAgent) -> TaskResult:
        """Execute task with a single agent."""
        start_time = time.time()
        
        try:
            # Create task context
            context = TaskContext(
                task_id=task.task_id,
                task_type=task.task_type,
                conversation_id=task.conversation_id,
                session_id=task.session_id,
                agent_role=AgentRole.PRIMARY,
                max_tokens=task.max_tokens,
                timeout_seconds=task.timeout_seconds,
                context_data=task.context_data
            )
            
            # Execute task
            async with AsyncPerformanceLogger(
                self.logger, f"single_agent_execution", 
                task_id=task.task_id, agent=agent.name
            ):
                response = await agent.generate(task.description, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Create result
            result = TaskResult(
                task_id=task.task_id,
                success=response.success,
                content=response.content,
                agent_responses=[response],
                execution_time_ms=execution_time,
                total_cost_usd=response.cost_usd or 0.0,
                total_tokens=response.tokens_used or 0,
                strategy_used=OrchestrationStrategy.SINGLE_AGENT,
                metadata={
                    "agent_used": agent.name,
                    "agent_type": agent.agent_type.value
                },
                error_message=response.error_message
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Single agent execution failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                content="",
                execution_time_ms=execution_time,
                strategy_used=OrchestrationStrategy.SINGLE_AGENT,
                error_message=str(e),
                metadata={"agent_used": agent.name}
            )
    
    async def _execute_parallel_agents(self, task: OrchestrationTask, agents: List[BaseAgent]) -> TaskResult:
        """Execute task with multiple agents in parallel."""
        start_time = time.time()
        
        try:
            # Create tasks for each agent
            agent_tasks = []
            for i, agent in enumerate(agents):
                context = TaskContext(
                    task_id=f"{task.task_id}_parallel_{i}",
                    task_type=task.task_type,
                    conversation_id=task.conversation_id,
                    session_id=task.session_id,
                    agent_role=AgentRole.PRIMARY,
                    max_tokens=task.max_tokens,
                    timeout_seconds=task.timeout_seconds,
                    context_data=task.context_data
                )
                
                agent_tasks.append(agent.generate(task.description, context))
            
            # Execute all tasks in parallel
            async with AsyncPerformanceLogger(
                self.logger, f"parallel_agent_execution",
                task_id=task.task_id, agent_count=len(agents)
            ):
                responses = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Process responses
            valid_responses = []
            total_cost = 0.0
            total_tokens = 0
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    self.logger.error(f"Agent {agents[i].name} failed: {response}")
                    continue
                
                valid_responses.append(response)
                if response.cost_usd:
                    total_cost += response.cost_usd
                if response.tokens_used:
                    total_tokens += response.tokens_used
            
            if not valid_responses:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    content="All parallel agents failed",
                    execution_time_ms=execution_time,
                    strategy_used=OrchestrationStrategy.PARALLEL,
                    error_message="All agents failed"
                )
            
            # Combine responses (take the best one or combine them)
            best_response = max(valid_responses, key=lambda r: 1 if r.success else 0)
            
            # Create combined content
            if len(valid_responses) > 1:
                combined_content = f"Combined results from {len(valid_responses)} agents:\n\n"
                for i, response in enumerate(valid_responses):
                    combined_content += f"=== Agent {agents[i].name} ===\n{response.content}\n\n"
            else:
                combined_content = best_response.content
            
            result = TaskResult(
                task_id=task.task_id,
                success=any(r.success for r in valid_responses),
                content=combined_content,
                agent_responses=valid_responses,
                execution_time_ms=execution_time,
                total_cost_usd=total_cost,
                total_tokens=total_tokens,
                strategy_used=OrchestrationStrategy.PARALLEL,
                metadata={
                    "agents_used": [agent.name for agent in agents],
                    "successful_responses": len([r for r in valid_responses if r.success]),
                    "total_responses": len(valid_responses)
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Parallel agent execution failed: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                content="",
                execution_time_ms=execution_time,
                strategy_used=OrchestrationStrategy.PARALLEL,
                error_message=str(e)
            )
    
    async def execute_task(self, task: OrchestrationTask) -> TaskResult:
        """Execute a task using the appropriate orchestration strategy."""
        self._active_tasks[task.task_id] = task
        self._total_tasks += 1
        
        with log_context(task.task_id, task.session_id):
            self.logger.info(
                f"Executing task {task.task_id} with strategy {task.strategy.value}",
                extra={
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "strategy": task.strategy.value
                }
            )
            
            try:
                if task.strategy == OrchestrationStrategy.SINGLE_AGENT:
                    agent = self._select_best_agent(task)
                    if not agent:
                        raise ValueError("No suitable agent found for task")
                    result = await self._execute_single_agent(task, agent)
                
                elif task.strategy == OrchestrationStrategy.PARALLEL:
                    agents = self._select_multiple_agents(task, 2)
                    if not agents:
                        raise ValueError("No agents available for parallel execution")
                    result = await self._execute_parallel_agents(task, agents)
                
                elif task.strategy == OrchestrationStrategy.DEBATE:
                    # Import here to avoid circular imports
                    from .debate import DebateProtocol
                    debate = DebateProtocol(self, task)
                    result = await debate.execute_debate()
                
                else:
                    # Fallback to single agent
                    self.logger.warning(f"Strategy {task.strategy.value} not implemented, using single agent")
                    agent = self._select_best_agent(task)
                    if not agent:
                        raise ValueError("No suitable agent found for task")
                    result = await self._execute_single_agent(task, agent)
                
                # Store result and update metrics
                self._task_results[task.task_id] = result
                if result.success:
                    self._successful_tasks += 1
                self._total_cost += result.total_cost_usd
                
                # Persist to database
                await self._persist_task_execution(task, result)
                
                self.logger.info(
                    f"Task {task.task_id} completed - Success: {result.success}, "
                    f"Cost: ${result.total_cost_usd:.4f}, Time: {result.execution_time_ms:.1f}ms"
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
                result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    content="",
                    error_message=str(e)
                )
                self._task_results[task.task_id] = result
                return result
            
            finally:
                # Clean up active task
                self._active_tasks.pop(task.task_id, None)
    
    async def _persist_task_execution(self, task: OrchestrationTask, result: TaskResult) -> None:
        """Persist task execution results to database."""
        try:
            async with self.db.get_session() as session:
                task_execution = TaskExecution(
                    id=str(uuid.uuid4()),
                    conversation_id=task.conversation_id,
                    task_type=task.task_type.value,
                    task_description=task.description[:1000],
                    strategy_used=task.strategy.value,
                    participants=[resp.agent_type.value for resp in result.agent_responses],
                    success=result.success,
                    final_solution=result.content[:5000],  
                    consensus_score=result.metadata.get("consensus_score"),
                    execution_time_ms=result.execution_time_ms,
                    total_tokens=result.total_tokens,
                    total_cost_usd=result.total_cost_usd,
                    error_info={"message": result.error_message} if result.error_message else None,
                    metadata_json=result.metadata
                )
                
                session.add(task_execution)
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to persist task execution: {e}")
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a completed task."""
        return self._task_results.get(task_id)
    
    def get_active_tasks(self) -> List[OrchestrationTask]:
        """Get list of currently active tasks."""
        return list(self._active_tasks.values())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        uptime = time.time() - self._start_time
        success_rate = self._successful_tasks / max(self._total_tasks, 1)
        
        return {
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "success_rate": success_rate,
            "total_cost_usd": self._total_cost,
            "average_cost_per_task": self._total_cost / max(self._total_tasks, 1),
            "uptime_seconds": uptime,
            "tasks_per_minute": (self._total_tasks / max(uptime / 60, 1)),
            "active_tasks_count": len(self._active_tasks)
        }


class OrchestrationEngine:
    """
    High-level orchestration engine that manages the overall workflow.
    
    I'm providing a simplified interface for complex multi-agent operations
    while handling all the orchestration complexity internally.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.orchestrator = TaskOrchestrator(db_manager)
        self.logger = get_logger("orchestration.engine")
        
    async def process_request(
        self,
        prompt: str,
        task_type: TaskType = TaskType.CUSTOM,
        strategy: OrchestrationStrategy = OrchestrationStrategy.SINGLE_AGENT,
        **kwargs
    ) -> TaskResult:
        """Process a user request with intelligent orchestration."""
        
        task = OrchestrationTask(
            description=prompt,
            task_type=task_type,
            strategy=strategy,
            **kwargs
        )
        
        return await self.orchestrator.execute_task(task)
    
    async def analyze_and_route(self, prompt: str, **kwargs) -> TaskResult:
        """Automatically analyze prompt and route to best strategy."""
        
        # Simple heuristics for task type detection
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["code", "function", "class", "implement", "write"]):
            task_type = TaskType.CODE_GENERATION
            strategy = OrchestrationStrategy.SINGLE_AGENT
        elif any(word in prompt_lower for word in ["review", "check", "analyze", "critique"]):
            task_type = TaskType.CODE_REVIEW
            strategy = OrchestrationStrategy.PARALLEL
        elif any(word in prompt_lower for word in ["debug", "error", "fix", "problem"]):
            task_type = TaskType.DEBUGGING
            strategy = OrchestrationStrategy.SINGLE_AGENT
        elif any(word in prompt_lower for word in ["research", "investigate", "study", "learn"]):
            task_type = TaskType.RESEARCH
            strategy = OrchestrationStrategy.PARALLEL
        elif any(word in prompt_lower for word in ["compare", "debate", "discuss", "opinion"]):
            task_type = TaskType.ANALYSIS
            strategy = OrchestrationStrategy.DEBATE
        else:
            task_type = TaskType.CUSTOM
            strategy = OrchestrationStrategy.SINGLE_AGENT
        
        self.logger.info(f"Auto-routed task: {task_type.value} with {strategy.value}")
        
        return await self.process_request(prompt, task_type, strategy, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall orchestration engine status."""
        return {
            "orchestrator_metrics": self.orchestrator.get_performance_metrics(),
            "available_agents": len(agent_registry.get_all_agents()),
            "agent_health": {
                agent.name: "healthy" if agent else "unknown"
                for agent in agent_registry.get_all_agents()
            }
        }
