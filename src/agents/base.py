"""
Base agent interface for AngelaMCP.

This defines the common interface that all AI agents must implement.
I'm using ABC to ensure consistent implementation across Claude, OpenAI, and Gemini.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

from src.utils import get_logger, AgentLogger


class AgentType(str, Enum):
    """Types of AI agents in the system."""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


class TaskType(str, Enum):
    """Types of tasks agents can handle."""
    GENERAL = "general"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXECUTION = "code_execution"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    DOCUMENTATION = "documentation"
    CREATIVE = "creative"
    DEBATE = "debate"
    CONSENSUS = "consensus"
    CRITIQUE = "critique"


class AgentRole(str, Enum):
    """Roles agents can play in collaboration."""
    PRIMARY = "primary"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    PROPOSER = "proposer"
    DEBATER = "debater"
    SYNTHESIZER = "synthesizer"
    MODERATOR = "moderator"


@dataclass
class TaskContext:
    """Context information for task execution."""
    task_type: TaskType = TaskType.GENERAL
    agent_role: Optional[AgentRole] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def model_copy(self) -> 'TaskContext':
        """Create a copy of the context."""
        return TaskContext(
            task_type=self.task_type,
            agent_role=self.agent_role,
            conversation_id=self.conversation_id,
            session_id=self.session_id,
            user_preferences=self.user_preferences.copy(),
            constraints=self.constraints.copy(),
            metadata=self.metadata.copy()
        )


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class AgentResponse:
    """Response from an AI agent."""
    agent_type: AgentType
    agent_name: str
    content: str
    success: bool = True
    confidence: float = 0.8
    execution_time_ms: float = 0.0
    token_usage: Optional[TokenUsage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def cost_usd(self) -> float:
        """Calculate cost in USD based on token usage."""
        if not self.token_usage:
            return 0.0
        
        # Import here to avoid circular imports
        from config.settings import settings
        
        if self.agent_type == AgentType.OPENAI:
            input_cost = self.token_usage.input_tokens * settings.openai_input_cost / 1000
            output_cost = self.token_usage.output_tokens * settings.openai_output_cost / 1000
            return input_cost + output_cost
        elif self.agent_type == AgentType.GEMINI:
            input_cost = self.token_usage.input_tokens * settings.gemini_input_cost / 1000
            output_cost = self.token_usage.output_tokens * settings.gemini_output_cost / 1000
            return input_cost + output_cost
        else:
            return 0.0  # Claude Code is free


@dataclass
class AgentCapabilities:
    """Capabilities supported by an agent."""
    can_execute_code: bool = False
    can_read_files: bool = False
    can_write_files: bool = False
    can_browse_web: bool = False
    can_use_tools: bool = False
    supported_languages: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    max_context_length: int = 4096
    supports_streaming: bool = False
    supports_function_calling: bool = False


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    All agents (Claude, OpenAI, Gemini) must implement this interface.
    """
    
    def __init__(
        self, 
        agent_type: AgentType, 
        name: str, 
        capabilities: AgentCapabilities = None
    ):
        self.agent_type = agent_type
        self.name = name
        self.capabilities = capabilities or AgentCapabilities()
        
        # Logging
        self.logger = get_logger(f"agents.{name}")
        self.agent_logger = AgentLogger(name)
        
        # Performance tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_execution_time = 0.0
        
        # Rate limiting
        self._request_times: List[float] = []
        self._rate_limit_window = 60.0  # 1 minute
    
    @abstractmethod
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt/task
            context: Task context with additional information
            
        Returns:
            AgentResponse with the generated content
        """
        pass
    
    async def critique(self, content: str, context: TaskContext) -> AgentResponse:
        """
        Critique the given content.
        
        Default implementation creates a critique prompt and calls generate.
        Override for agent-specific critique behavior.
        """
        critique_prompt = f"""Please provide a constructive critique of the following content:

{content}

Consider:
1. Strengths and positive aspects
2. Areas for improvement
3. Specific suggestions for enhancement
4. Any concerns or potential issues

Provide a balanced, professional review."""

        critique_context = context.model_copy()
        critique_context.task_type = TaskType.CRITIQUE
        critique_context.agent_role = AgentRole.CRITIC
        
        return await self.generate(critique_prompt, critique_context)
    
    async def propose_solution(self, task_description: str, constraints: List[str], context: TaskContext) -> AgentResponse:
        """
        Propose a solution for the given task.
        
        Default implementation creates a proposal prompt and calls generate.
        Override for agent-specific proposal behavior.
        """
        constraints_text = "\n".join(f"- {constraint}" for constraint in constraints) if constraints else "None specified"
        
        proposal_prompt = f"""Please propose a solution for the following task:

**Task:** {task_description}

**Constraints:**
{constraints_text}

Provide:
1. **Proposed Solution:** Clear, actionable approach
2. **Implementation Steps:** How to execute this solution
3. **Benefits:** Why this approach is effective
4. **Considerations:** Any limitations or alternatives to consider

Focus on practicality and effectiveness."""

        proposal_context = context.model_copy()
        proposal_context.task_type = TaskType.GENERAL
        proposal_context.agent_role = AgentRole.PROPOSER
        
        return await self.generate(proposal_prompt, proposal_context)
    
    async def analyze_task(self, task_description: str, context: TaskContext) -> AgentResponse:
        """
        Analyze a task and provide insights.
        
        Default implementation creates an analysis prompt and calls generate.
        Override for agent-specific analysis behavior.
        """
        analysis_prompt = f"""Please analyze the following task:

{task_description}

Provide:
1. **Task Complexity:** Assessment of difficulty level
2. **Required Skills:** What capabilities are needed
3. **Approach Recommendations:** Best strategies to tackle this
4. **Potential Challenges:** What difficulties might arise
5. **Success Criteria:** How to measure completion

Be thorough and insightful in your analysis."""

        analysis_context = context.model_copy()
        analysis_context.task_type = TaskType.ANALYSIS
        analysis_context.agent_role = AgentRole.SPECIALIST
        
        return await self.generate(analysis_prompt, analysis_context)
    
    def supports_capability(self, capability: str) -> bool:
        """Check if agent supports a specific capability."""
        capability_map = {
            "code_execution": self.capabilities.can_execute_code,
            "file_reading": self.capabilities.can_read_files,
            "file_writing": self.capabilities.can_write_files,
            "web_browsing": self.capabilities.can_browse_web,
            "tool_usage": self.capabilities.can_use_tools,
            "streaming": self.capabilities.supports_streaming,
            "function_calling": self.capabilities.supports_function_calling
        }
        
        return capability_map.get(capability.lower(), False)
    
    def supports_language(self, language: str) -> bool:
        """Check if agent supports a programming language."""
        return language.lower() in [lang.lower() for lang in self.capabilities.supported_languages]
    
    def supports_format(self, format_type: str) -> bool:
        """Check if agent supports a content format."""
        return format_type.lower() in [fmt.lower() for fmt in self.capabilities.supported_formats]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent."""
        try:
            # Simple test generation
            test_context = TaskContext(task_type=TaskType.GENERAL)
            start_time = time.time()
            
            response = await self.generate("Hello, how are you?", test_context)
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy" if response.success else "error",
                "response_time_ms": response_time,
                "last_check": time.time(),
                "total_requests": self._total_requests,
                "success_rate": self._successful_requests / max(self._total_requests, 1),
                "average_tokens": self._total_tokens / max(self._total_requests, 1),
                "total_cost_usd": self._total_cost
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_check": time.time()
            }
    
    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        self.logger.info(f"Shutting down agent {self.name}")
        # Override in subclasses for specific cleanup
    
    def _track_request(self, response: AgentResponse) -> None:
        """Track request metrics."""
        self._total_requests += 1
        
        if response.success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1
        
        self._total_execution_time += response.execution_time_ms
        
        if response.token_usage:
            self._total_tokens += response.token_usage.total_tokens
        
        self._total_cost += response.cost_usd
        
        # Track request timing for rate limiting
        current_time = time.time()
        self._request_times.append(current_time)
        
        # Clean old request times (outside window)
        cutoff_time = current_time - self._rate_limit_window
        self._request_times = [t for t in self._request_times if t > cutoff_time]
    
    def _check_rate_limit(self, max_requests_per_minute: int) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        cutoff_time = current_time - self._rate_limit_window
        
        # Count recent requests
        recent_requests = len([t for t in self._request_times if t > cutoff_time])
        
        return recent_requests < max_requests_per_minute
    
    async def _wait_for_rate_limit(self, max_requests_per_minute: int) -> None:
        """Wait if rate limit is exceeded."""
        while not self._check_rate_limit(max_requests_per_minute):
            wait_time = 60.0 / max_requests_per_minute
            self.logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": self._successful_requests / max(self._total_requests, 1),
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "total_execution_time_ms": self._total_execution_time,
            "average_execution_time_ms": self._total_execution_time / max(self._total_requests, 1),
            "average_tokens_per_request": self._total_tokens / max(self._total_requests, 1),
            "current_rate": len(self._request_times)  # Requests in last minute
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, type={self.agent_type.value})>"


# Performance monitoring decorator for agent methods
def track_performance(func):
    """Decorator to track performance of agent methods."""
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(self, *args, **kwargs)
            
            # Track successful request
            if hasattr(result, 'execution_time_ms'):
                result.execution_time_ms = (time.time() - start_time) * 1000
            
            if hasattr(self, '_track_request'):
                self._track_request(result)
            
            return result
            
        except Exception as e:
            # Track failed request
            execution_time = (time.time() - start_time) * 1000
            
            if hasattr(self, '_track_request'):
                error_response = AgentResponse(
                    agent_type=self.agent_type,
                    agent_name=self.name,
                    content="",
                    success=False,
                    execution_time_ms=execution_time,
                    error=str(e)
                )
                self._track_request(error_response)
            
            raise
    
    return wrapper


# Global agent registry for easy access
class AgentRegistry:
    """
    Registry for managing all active agents.
    
    I'm implementing a centralized registry to track all active agents
    and provide easy access for the orchestration system.
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[AgentType, List[BaseAgent]] = {
            agent_type: [] for agent_type in AgentType
        }
        self.logger = get_logger("agents.registry")
    
    def register(self, agent: BaseAgent) -> None:
        """Register a new agent."""
        if agent.name in self._agents:
            raise ValueError(f"Agent with name '{agent.name}' already registered")
        
        self._agents[agent.name] = agent
        self._agent_types[agent.agent_type].append(agent)
        
        self.logger.info(f"Registered agent: {agent}")
    
    def unregister(self, agent_name: str) -> None:
        """Unregister an agent."""
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        agent = self._agents[agent_name]
        del self._agents[agent_name]
        self._agent_types[agent.agent_type].remove(agent)
        
        self.logger.info(f"Unregistered agent: {agent_name}")
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agents.get(agent_name)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type."""
        return self._agent_types[agent_type].copy()
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        return list(self._agents.values())
    
    def get_available_agents(self, capability: Optional[str] = None) -> List[BaseAgent]:
        """Get agents that support a specific capability."""
        if capability is None:
            return self.get_all_agents()
        
        return [
            agent for agent in self._agents.values()
            if agent.supports_capability(capability)
        ]
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all registered agents."""
        results = {}
        
        for agent_name, agent in self._agents.items():
            try:
                results[agent_name] = await agent.health_check()
            except Exception as e:
                results[agent_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered agents."""
        self.logger.info("Shutting down all registered agents...")
        
        for agent in self._agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {agent.name}: {e}")
        
        self._agents.clear()
        for agent_list in self._agent_types.values():
            agent_list.clear()


# Global registry instance
agent_registry = AgentRegistry()
