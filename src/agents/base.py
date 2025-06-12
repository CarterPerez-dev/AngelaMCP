"""
Base agent implementation for AngelaMCP.

This module defines the abstract base class and common interfaces for all AI agents
in the multi-agent collaboration platform. I'm implementing a standardized interface
that ensures consistent behavior across Claude Code, OpenAI, and Gemini agents.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field, ConfigDict

from src.utils.logger import get_logger, log_agent_interaction, AsyncPerformanceLogger

logger = get_logger("agents.base")


class AgentType(str, Enum):
    """Enumeration of supported agent types."""
    CLAUDE_CODE = "claude_code"
    OPENAI = "openai"
    GEMINI = "gemini"


class TaskType(str, Enum):
    """Types of tasks that agents can execute."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    COLLABORATION = "collaboration"
    CUSTOM = "custom"


class AgentRole(str, Enum):
    """Roles agents can play in collaborative tasks."""
    PRIMARY = "primary"      # Main agent executing the task
    REVIEWER = "reviewer"    # Agent providing review/critique
    RESEARCHER = "researcher" # Agent gathering information
    SPECIALIST = "specialist" # Agent with specific expertise


class AgentCapability(BaseModel):
    """Represents a capability that an agent supports."""
    name: str = Field(description="Name of the capability")
    description: str = Field(description="Description of what this capability does")
    supported_formats: List[str] = Field(default_factory=list, description="Supported input/output formats")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")
    cost_per_request: Optional[float] = Field(None, description="Estimated cost per request in USD")


class AgentResponse(BaseModel):
    """Standardized response from agent operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool = Field(description="Whether the operation succeeded")
    content: str = Field(description="The main response content")
    
    # Metadata
    agent_type: str = Field(description="Type of agent that generated this response")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    
    # Performance metrics
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens consumed")
    cost_usd: Optional[float] = Field(None, description="Cost in USD")
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    confidence_score: Optional[float] = Field(None, description="Confidence in the response (0-1)")
    
    # Tool usage tracking
    tools_used: List[str] = Field(default_factory=list, description="Tools/functions used")
    function_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Function calls made")


class TaskContext(BaseModel):
    """Context information for task execution."""
    task_id: str = Field(description="Unique task identifier")
    task_type: TaskType = Field(description="Type of task being executed")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    
    # Agent assignment
    primary_agent: Optional[str] = Field(None, description="Primary agent for this task")
    participating_agents: List[str] = Field(default_factory=list, description="All participating agents")
    agent_role: AgentRole = Field(default=AgentRole.PRIMARY, description="Role of current agent")
    
    # Collaboration settings
    requires_collaboration: bool = Field(default=False, description="Whether task requires multi-agent collaboration")
    enable_debate: bool = Field(default=False, description="Whether to enable debate mode")
    max_debate_rounds: int = Field(default=3, description="Maximum debate rounds")
    
    # Constraints and preferences
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to use")
    temperature: Optional[float] = Field(None, description="Temperature setting for generation")
    timeout_seconds: Optional[int] = Field(None, description="Maximum execution time")
    
    # Additional context
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    previous_attempts: List[Dict[str, Any]] = Field(default_factory=list, description="Previous attempt results")


class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, agent_type: Optional[str] = None, 
                 error_code: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.agent_type = agent_type
        self.error_code = error_code
        self.metadata = metadata or {}


class AgentRateLimitError(AgentError):
    """Exception raised when agent hits rate limits."""
    pass


class AgentTimeoutError(AgentError):
    """Exception raised when agent operations timeout."""
    pass


class AgentAuthenticationError(AgentError):
    """Exception raised when agent authentication fails."""
    pass


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in AngelaMCP.
    
    I'm defining a comprehensive interface that ensures all agents provide
    consistent functionality for generation, collaboration, and monitoring.
    """
    
    def __init__(self, agent_type: AgentType, name: str, settings: Any):
        self.agent_type = agent_type
        self.name = name
        self.settings = settings
        self.logger = get_logger(f"agents.{name}")
        
        # Performance tracking
        self._total_requests = 0
        self._total_cost = 0.0
        self._total_tokens = 0
        self._start_time = time.time()
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_count_window = []
        
        # Capabilities (to be defined by subclasses)
        self._capabilities: List[AgentCapability] = []
        
        # Configuration
        self.max_retries = getattr(settings, f"{agent_type.value}_max_retries", 3)
        self.retry_delay = getattr(settings, f"{agent_type.value}_retry_delay", 1.0)
        self.rate_limit = getattr(settings, f"{agent_type.value}_rate_limit", 60)
        self.timeout = getattr(settings, f"{agent_type.value}_timeout", 180)
        
        self.logger.info(f"Initialized {agent_type.value} agent: {name}")
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities."""
        return self._capabilities.copy()
    
    @property
    def performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        uptime = time.time() - self._start_time
        return {
            "total_requests": self._total_requests,
            "total_cost_usd": self._total_cost,
            "total_tokens": self._total_tokens,
            "uptime_seconds": uptime,
            "average_cost_per_request": self._total_cost / max(self._total_requests, 1),
            "requests_per_minute": (self._total_requests / max(uptime / 60, 1)),
        }
    
    def _check_rate_limit(self) -> None:
        """Check if we're within rate limits."""
        current_time = time.time()
        
        # Clean old requests from window (last 60 seconds)
        self._request_count_window = [
            req_time for req_time in self._request_count_window
            if current_time - req_time < 60
        ]
        
        if len(self._request_count_window) >= self.rate_limit:
            raise AgentRateLimitError(
                f"Rate limit exceeded: {len(self._request_count_window)} requests in last minute",
                agent_type=self.agent_type.value,
                error_code="RATE_LIMIT_EXCEEDED"
            )
        
        self._request_count_window.append(current_time)
        self._last_request_time = current_time
    
    def _update_metrics(self, response: AgentResponse) -> None:
        """Update performance metrics based on response."""
        self._total_requests += 1
        
        if response.cost_usd:
            self._total_cost += response.cost_usd
        
        if response.tokens_used:
            self._total_tokens += response.tokens_used
    
    @abstractmethod
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt or question
            context: Task context and configuration
            
        Returns:
            AgentResponse containing the generated content and metadata
        """
        pass
    
    @abstractmethod
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """
        Provide critique or review of content.
        
        Args:
            content: The content to critique
            original_task: The original task context
            context: Task context and configuration
            
        Returns:
            AgentResponse containing the critique and suggestions
        """
        pass
    
    @abstractmethod
    async def propose_solution(self, task_description: str, constraints: List[str], 
                             context: TaskContext) -> AgentResponse:
        """
        Propose a solution for the given task.
        
        Args:
            task_description: Description of the task to solve
            constraints: List of constraints or requirements
            context: Task context and configuration
            
        Returns:
            AgentResponse containing the proposed solution
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the agent.
        
        Returns:
            Dictionary containing health status and metrics
        """
        try:
            # Test basic functionality with a simple request
            test_context = TaskContext(
                task_id=str(uuid.uuid4()),
                task_type=TaskType.CUSTOM
            )
            
            async with AsyncPerformanceLogger(self.logger, "health_check"):
                response = await self.generate("Test connection", test_context)
            
            return {
                "status": "healthy",
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "response_time_ms": response.execution_time_ms,
                "capabilities_count": len(self._capabilities),
                "performance_metrics": self.performance_metrics,
                "last_request_time": self._last_request_time
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_type": self.agent_type.value,
                "agent_name": self.name,
                "error": str(e),
                "performance_metrics": self.performance_metrics
            }
    
    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute an operation with automatic retry logic.
        
        Args:
            operation: The async operation to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check rate limits before each attempt
                self._check_rate_limit()
                
                # Execute the operation
                return await operation(*args, **kwargs)
                
            except AgentRateLimitError:
                # Don't retry rate limit errors
                raise
            except AgentAuthenticationError:
                # Don't retry auth errors
                raise
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        # If we get here, all retries failed
        raise last_exception
    
    def supports_capability(self, capability_name: str) -> bool:
        """Check if agent supports a specific capability."""
        return any(cap.name == capability_name for cap in self._capabilities)
    
    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get details about a specific capability."""
        for cap in self._capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the agent.
        
        Subclasses should override this to perform cleanup operations.
        """
        self.logger.info(f"Shutting down agent: {self.name}")
        
        # Log final metrics
        metrics = self.performance_metrics
        self.logger.info(
            f"Agent {self.name} final metrics: "
            f"{metrics['total_requests']} requests, "
            f"${metrics['total_cost_usd']:.4f} cost, "
            f"{metrics['total_tokens']} tokens"
        )
    
    def __str__(self) -> str:
        return f"{self.agent_type.value}:{self.name}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self.agent_type.value}, name={self.name})>"


class AgentRegistry:
    """
    Registry for managing agent instances.
    
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
        self.logger.info("Shutting down all agents")
        
        for agent in self._agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down agent {agent.name}: {e}")
        
        self._agents.clear()
        for agent_list in self._agent_types.values():
            agent_list.clear()


# Global agent registry instance
agent_registry = AgentRegistry()
