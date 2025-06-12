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

from src.utils.logger import get_logger


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
    ANALYSIS = "analysis"
    RESEARCH = "research"
    DOCUMENTATION = "documentation"
    CREATIVE = "creative"
    DEBATE = "debate"


class AgentRole(str, Enum):
    """Roles agents can play in collaboration."""
    PRIMARY = "primary"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    SPECIALIST = "specialist"
    CRITIC = "critic"
    PROPOSER = "proposer"
    DEBATER = "debater"


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
class AgentResponse:
    """Response from an AI agent."""
    agent_type: AgentType
    content: str
    confidence: float = 0.8
    execution_time_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    All agents (Claude, OpenAI, Gemini) must implement this interface.
    """
    
    def __init__(
        self, 
        agent_type: AgentType, 
        name: str, 
        capabilities: List[str] = None
    ):
        self.agent_type = agent_type
        self.name = name
        self.capabilities = capabilities or []
        self.logger = get_logger(f"agents.{agent_type.value}")
        self._last_response_time: Optional[float] = None
        self._total_requests = 0
        self._failed_requests = 0
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.agent_type.value}>"
    
    @abstractmethod
    async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The input prompt/question
            context: Task context and metadata
            
        Returns:
            AgentResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
        """
        Critique another agent's response or solution.
        
        Args:
            content: The content to critique
            original_task: The original task description
            context: Task context
            
        Returns:
            AgentResponse with critique and suggestions
        """
        pass
    
    @abstractmethod
    async def propose_solution(self, task_description: str, constraints: List[str], context: TaskContext) -> AgentResponse:
        """
        Propose a solution for a given task.
        
        Args:
            task_description: Description of the task
            constraints: Any constraints or requirements
            context: Task context
            
        Returns:
            AgentResponse with proposed solution
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check agent health and availability.
        
        Returns:
            Dict with health status information
        """
        try:
            start_time = time.time()
            
            # Basic test - should be overridden by implementations
            test_context = TaskContext(task_type=TaskType.GENERAL)
            response = await self.generate("Hello", test_context)
            
            response_time = time.time() - start_time
            self._last_response_time = response_time
            
            return {
                "status": "healthy",
                "agent_type": self.agent_type.value,
                "name": self.name,
                "response_time": response_time,
                "total_requests": self._total_requests,
                "failed_requests": self._failed_requests,
                "success_rate": ((self._total_requests - self._failed_requests) / max(1, self._total_requests)) * 100,
                "capabilities": self.capabilities,
                "last_check": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "agent_type": self.agent_type.value,
                "name": self.name,
                "error": str(e),
                "last_check": time.time()
            }
    
    def supports_capability(self, capability: str) -> bool:
        """Check if agent supports a specific capability."""
        return capability in self.capabilities
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": ((self._total_requests - self._failed_requests) / max(1, self._total_requests)) * 100,
            "last_response_time": self._last_response_time,
            "capabilities": self.capabilities
        }
    
    async def _track_request(self, func, *args, **kwargs):
        """Track request statistics."""
        self._total_requests += 1
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            self._failed_requests += 1
            raise e


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
        for agent in self._agents.values():
            try:
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {agent.name}: {e}")


# Global registry instance
agent_registry = AgentRegistry()
