"""Tests for the base agent functionality."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock

from src.agents.base import (
    BaseAgent, AgentType, AgentResponse, TaskContext, TaskType, AgentRole,
    AgentCapability, AgentError, AgentRateLimitError, AgentTimeoutError,
    AgentAuthenticationError, AgentRegistry, agent_registry
)


class TestBaseAgent:
    """Test the BaseAgent abstract class functionality."""
    
    class MockAgent(BaseAgent):
        """Mock implementation of BaseAgent for testing."""
        
        def __init__(self, agent_type, name, settings):
            super().__init__(agent_type, name, settings)
            self._capabilities = [
                AgentCapability(
                    name="test_capability",
                    description="Test capability",
                    supported_formats=["text"],
                    cost_per_request=0.001
                )
            ]
        
        async def generate(self, prompt: str, context: TaskContext) -> AgentResponse:
            await asyncio.sleep(0.01)  # Simulate processing time
            return AgentResponse(
                success=True,
                content=f"Response to: {prompt}",
                agent_type=self.agent_type.value,
                execution_time_ms=10.0,
                tokens_used=25,
                cost_usd=0.001
            )
        
        async def critique(self, content: str, original_task: str, context: TaskContext) -> AgentResponse:
            return AgentResponse(
                success=True,
                content=f"Critique of: {content}",
                agent_type=self.agent_type.value,
                execution_time_ms=15.0
            )
        
        async def propose_solution(self, task_description: str, constraints: list, context: TaskContext) -> AgentResponse:
            return AgentResponse(
                success=True,
                content=f"Solution for: {task_description}",
                agent_type=self.agent_type.value,
                execution_time_ms=20.0
            )
    
    @pytest.fixture
    def mock_agent(self, mock_settings):
        """Create a mock agent instance."""
        return self.MockAgent(AgentType.CLAUDE_CODE, "test_agent", mock_settings)
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.agent_type == AgentType.CLAUDE_CODE
        assert mock_agent.name == "test_agent"
        assert mock_agent._total_requests == 0
        assert mock_agent._total_cost == 0.0
        assert mock_agent._total_tokens == 0
        assert len(mock_agent.capabilities) == 1
    
    def test_agent_capabilities(self, mock_agent):
        """Test agent capabilities management."""
        capabilities = mock_agent.capabilities
        assert len(capabilities) == 1
        assert capabilities[0].name == "test_capability"
        
        assert mock_agent.supports_capability("test_capability")
        assert not mock_agent.supports_capability("nonexistent_capability")
        
        capability = mock_agent.get_capability("test_capability")
        assert capability is not None
        assert capability.cost_per_request == 0.001
    
    def test_performance_metrics(self, mock_agent):
        """Test performance metrics tracking."""
        metrics = mock_agent.performance_metrics
        
        assert "total_requests" in metrics
        assert "total_cost_usd" in metrics
        assert "total_tokens" in metrics
        assert "uptime_seconds" in metrics
        assert "average_cost_per_request" in metrics
        assert "requests_per_minute" in metrics
    
    async def test_generate_method(self, mock_agent, mock_task_context):
        """Test the generate method."""
        response = await mock_agent.generate("Test prompt", mock_task_context)
        
        assert response.success
        assert response.content == "Response to: Test prompt"
        assert response.agent_type == "claude_code"
        assert response.execution_time_ms == 10.0
    
    async def test_critique_method(self, mock_agent, mock_task_context):
        """Test the critique method."""
        response = await mock_agent.critique("Test content", "Original task", mock_task_context)
        
        assert response.success
        assert response.content == "Critique of: Test content"
        assert response.execution_time_ms == 15.0
    
    async def test_propose_solution_method(self, mock_agent, mock_task_context):
        """Test the propose_solution method."""
        response = await mock_agent.propose_solution("Test task", ["constraint1"], mock_task_context)
        
        assert response.success
        assert response.content == "Solution for: Test task"
        assert response.execution_time_ms == 20.0
    
    def test_rate_limiting(self, mock_agent):
        """Test rate limiting functionality."""
        # Set a low rate limit for testing
        mock_agent.rate_limit = 2
        
        # First two requests should pass
        mock_agent._check_rate_limit()
        mock_agent._check_rate_limit()
        
        # Third request should raise rate limit error
        with pytest.raises(AgentRateLimitError):
            mock_agent._check_rate_limit()
    
    def test_metrics_update(self, mock_agent, mock_agent_response):
        """Test metrics update functionality."""
        initial_requests = mock_agent._total_requests
        initial_cost = mock_agent._total_cost
        initial_tokens = mock_agent._total_tokens
        
        mock_agent._update_metrics(mock_agent_response)
        
        assert mock_agent._total_requests == initial_requests + 1
        assert mock_agent._total_cost == initial_cost + mock_agent_response.cost_usd
        assert mock_agent._total_tokens == initial_tokens + mock_agent_response.tokens_used
    
    async def test_retry_logic(self, mock_agent):
        """Test retry logic with exponential backoff."""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        # Should succeed after 2 retries
        result = await mock_agent.execute_with_retry(failing_operation)
        assert result == "success"
        assert call_count == 3
    
    async def test_retry_limit_exceeded(self, mock_agent):
        """Test retry limit exceeded."""
        async def always_failing_operation():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            await mock_agent.execute_with_retry(always_failing_operation)
    
    async def test_no_retry_for_auth_errors(self, mock_agent):
        """Test that authentication errors are not retried."""
        call_count = 0
        
        async def auth_failing_operation():
            nonlocal call_count
            call_count += 1
            raise AgentAuthenticationError("Auth failed")
        
        with pytest.raises(AgentAuthenticationError):
            await mock_agent.execute_with_retry(auth_failing_operation)
        
        # Should only be called once (no retries)
        assert call_count == 1
    
    async def test_health_check(self, mock_agent):
        """Test health check functionality."""
        health_info = await mock_agent.health_check()
        
        assert health_info["status"] == "healthy"
        assert health_info["agent_type"] == "claude_code"
        assert health_info["agent_name"] == "test_agent"
        assert "response_time_ms" in health_info
        assert "performance_metrics" in health_info
    
    async def test_shutdown(self, mock_agent):
        """Test agent shutdown."""
        await mock_agent.shutdown()
        # Test that shutdown completes without error


class TestAgentRegistry:
    """Test the AgentRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = AgentRegistry()
        assert len(registry._agents) == 0
        assert len(registry._agent_types) == len(AgentType)
    
    def test_agent_registration(self, mock_claude_agent):
        """Test agent registration."""
        registry = AgentRegistry()
        
        registry.register(mock_claude_agent)
        
        assert mock_claude_agent.name in registry._agents
        assert mock_claude_agent in registry._agent_types[AgentType.CLAUDE_CODE]
    
    def test_duplicate_registration(self, mock_claude_agent):
        """Test that duplicate registration raises error."""
        registry = AgentRegistry()
        
        registry.register(mock_claude_agent)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(mock_claude_agent)
    
    def test_agent_retrieval(self, mock_claude_agent):
        """Test agent retrieval methods."""
        registry = AgentRegistry()
        registry.register(mock_claude_agent)
        
        # Test get_agent
        agent = registry.get_agent(mock_claude_agent.name)
        assert agent == mock_claude_agent
        
        # Test get_agents_by_type
        claude_agents = registry.get_agents_by_type(AgentType.CLAUDE_CODE)
        assert mock_claude_agent in claude_agents
        
        # Test get_all_agents
        all_agents = registry.get_all_agents()
        assert mock_claude_agent in all_agents
    
    def test_agent_unregistration(self, mock_claude_agent):
        """Test agent unregistration."""
        registry = AgentRegistry()
        registry.register(mock_claude_agent)
        
        registry.unregister(mock_claude_agent.name)
        
        assert mock_claude_agent.name not in registry._agents
        assert mock_claude_agent not in registry._agent_types[AgentType.CLAUDE_CODE]
    
    def test_unregister_nonexistent_agent(self):
        """Test unregistering nonexistent agent raises error."""
        registry = AgentRegistry()
        
        with pytest.raises(ValueError, match="not found"):
            registry.unregister("nonexistent_agent")
    
    def test_get_available_agents(self, mock_claude_agent):
        """Test getting available agents with capability filtering."""
        registry = AgentRegistry()
        registry.register(mock_claude_agent)
        
        # Test without capability filter
        agents = registry.get_available_agents()
        assert mock_claude_agent in agents
        
        # Test with existing capability
        agents = registry.get_available_agents("test_capability")
        assert mock_claude_agent in agents
        
        # Test with nonexistent capability
        agents = registry.get_available_agents("nonexistent_capability")
        assert len(agents) == 0
    
    async def test_health_check_all(self, mock_claude_agent):
        """Test health check for all agents."""
        registry = AgentRegistry()
        registry.register(mock_claude_agent)
        
        results = await registry.health_check_all()
        
        assert mock_claude_agent.name in results
        assert results[mock_claude_agent.name]["status"] == "healthy"
    
    async def test_shutdown_all(self, mock_claude_agent):
        """Test shutdown all agents."""
        registry = AgentRegistry()
        registry.register(mock_claude_agent)
        
        await registry.shutdown_all()
        
        # Registry should be empty after shutdown
        assert len(registry._agents) == 0
        assert all(len(agent_list) == 0 for agent_list in registry._agent_types.values())


class TestAgentExceptions:
    """Test agent exception classes."""
    
    def test_agent_error(self):
        """Test AgentError base exception."""
        error = AgentError("Test error", "claude_code", "TEST_ERROR", {"key": "value"})
        
        assert str(error) == "Test error"
        assert error.agent_type == "claude_code"
        assert error.error_code == "TEST_ERROR"
        assert error.metadata == {"key": "value"}
    
    def test_agent_rate_limit_error(self):
        """Test AgentRateLimitError."""
        error = AgentRateLimitError("Rate limit exceeded")
        assert isinstance(error, AgentError)
    
    def test_agent_timeout_error(self):
        """Test AgentTimeoutError."""
        error = AgentTimeoutError("Operation timed out")
        assert isinstance(error, AgentError)
    
    def test_agent_authentication_error(self):
        """Test AgentAuthenticationError."""
        error = AgentAuthenticationError("Authentication failed")
        assert isinstance(error, AgentError)


class TestAgentResponse:
    """Test AgentResponse data class."""
    
    def test_agent_response_creation(self):
        """Test AgentResponse creation and properties."""
        response = AgentResponse(
            success=True,
            content="Test content",
            agent_type="claude_code",
            execution_time_ms=100.0,
            tokens_used=50,
            cost_usd=0.001,
            metadata={"test": "data"}
        )
        
        assert response.success
        assert response.content == "Test content"
        assert response.agent_type == "claude_code"
        assert response.execution_time_ms == 100.0
        assert response.tokens_used == 50
        assert response.cost_usd == 0.001
        assert response.metadata == {"test": "data"}
        assert response.timestamp > 0  # Should have a timestamp
    
    def test_agent_response_defaults(self):
        """Test AgentResponse with default values."""
        response = AgentResponse(
            success=False,
            content="Error content",
            agent_type="test_agent"
        )
        
        assert not response.success
        assert response.content == "Error content"
        assert response.agent_type == "test_agent"
        assert response.execution_time_ms is None
        assert response.tokens_used is None
        assert response.cost_usd is None
        assert response.metadata == {}


class TestTaskContext:
    """Test TaskContext data class."""
    
    def test_task_context_creation(self, mock_task_context):
        """Test TaskContext creation and properties."""
        assert mock_task_context.task_type == TaskType.CUSTOM
        assert mock_task_context.agent_role == AgentRole.PRIMARY
        assert mock_task_context.max_tokens == 1000
        assert mock_task_context.timeout_seconds == 30
        assert mock_task_context.context_data == {}
    
    def test_task_context_defaults(self):
        """Test TaskContext with default values."""
        import uuid
        
        context = TaskContext(
            task_id=str(uuid.uuid4()),
            task_type=TaskType.CODE_GENERATION
        )
        
        assert context.task_type == TaskType.CODE_GENERATION
        assert context.agent_role == AgentRole.PRIMARY
        assert context.requires_collaboration is False
        assert context.enable_debate is False
        assert context.max_debate_rounds == 3
        assert context.context_data == {}
        assert context.previous_attempts == []