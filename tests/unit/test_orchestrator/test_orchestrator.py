"""Tests for the orchestration system."""

import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, Mock, patch

from src.orchestration.orchestrator import (
    TaskOrchestrator, OrchestrationEngine, OrchestrationTask, TaskResult,
    OrchestrationStrategy, TaskPriority, TaskStatus, TaskType
)
from src.agents.base import AgentType, TaskContext, agent_registry


class TestOrchestrationTask:
    """Test the OrchestrationTask data class."""
    
    def test_task_creation(self):
        """Test task creation with defaults."""
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION
        )
        
        assert task.description == "Test task"
        assert task.task_type == TaskType.CODE_GENERATION
        assert task.priority == TaskPriority.MEDIUM
        assert task.strategy == OrchestrationStrategy.SINGLE_AGENT
        assert task.preferred_agents == []
        assert task.required_capabilities == []
        assert task.constraints == []
        assert not task.enable_debate
        assert task.max_debate_rounds == 3
        assert not task.require_consensus
        assert task.consensus_threshold == 0.7
    
    def test_task_with_custom_values(self):
        """Test task creation with custom values."""
        task = OrchestrationTask(
            description="Complex task",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            strategy=OrchestrationStrategy.DEBATE,
            preferred_agents=[AgentType.CLAUDE_CODE, AgentType.OPENAI],
            required_capabilities=["analysis", "reasoning"],
            constraints=["time_limit_5min"],
            enable_debate=True,
            max_debate_rounds=5,
            require_consensus=True,
            consensus_threshold=0.8,
            max_tokens=2000,
            max_cost_usd=0.10,
            timeout_seconds=300
        )
        
        assert task.description == "Complex task"
        assert task.task_type == TaskType.ANALYSIS
        assert task.priority == TaskPriority.HIGH
        assert task.strategy == OrchestrationStrategy.DEBATE
        assert AgentType.CLAUDE_CODE in task.preferred_agents
        assert "analysis" in task.required_capabilities
        assert "time_limit_5min" in task.constraints
        assert task.enable_debate
        assert task.max_debate_rounds == 5
        assert task.require_consensus
        assert task.consensus_threshold == 0.8
        assert task.max_tokens == 2000
        assert task.max_cost_usd == 0.10
        assert task.timeout_seconds == 300


class TestTaskResult:
    """Test the TaskResult data class."""
    
    def test_result_creation(self):
        """Test task result creation."""
        task_id = str(uuid.uuid4())
        
        result = TaskResult(
            task_id=task_id,
            success=True,
            content="Task completed successfully",
            execution_time_ms=1500.0,
            total_cost_usd=0.05,
            total_tokens=200,
            strategy_used=OrchestrationStrategy.SINGLE_AGENT,
            metadata={"agent_used": "claude_code"}
        )
        
        assert result.task_id == task_id
        assert result.success
        assert result.content == "Task completed successfully"
        assert result.execution_time_ms == 1500.0
        assert result.total_cost_usd == 0.05
        assert result.total_tokens == 200
        assert result.strategy_used == OrchestrationStrategy.SINGLE_AGENT
        assert result.metadata["agent_used"] == "claude_code"
        assert result.agent_responses == []
        assert result.error_message is None


class TestTaskOrchestrator:
    """Test the TaskOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self, mock_database):
        """Create a task orchestrator instance."""
        return TaskOrchestrator(mock_database)
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator._total_tasks == 0
        assert orchestrator._successful_tasks == 0
        assert orchestrator._total_cost == 0.0
        assert len(orchestrator._active_tasks) == 0
        assert len(orchestrator._task_results) == 0
    
    def test_agent_weights(self, orchestrator):
        """Test agent weight configuration."""
        weights = orchestrator._agent_weights
        
        # Check that all agent types have weights
        assert AgentType.CLAUDE_CODE in weights
        assert AgentType.OPENAI in weights
        assert AgentType.GEMINI in weights
        
        # Check that all task types have weights for each agent
        for agent_type in AgentType:
            for task_type in TaskType:
                assert task_type in weights[agent_type]
                assert isinstance(weights[agent_type][task_type], (int, float))
    
    def test_select_best_agent_no_agents(self, orchestrator):
        """Test agent selection when no agents are available."""
        # Clear agent registry
        agent_registry._agents.clear()
        for agent_list in agent_registry._agent_types.values():
            agent_list.clear()
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION
        )
        
        result = orchestrator._select_best_agent(task)
        assert result is None
    
    def test_select_best_agent_with_preferences(self, orchestrator, mock_claude_agent):
        """Test agent selection with preferred agents."""
        # Register mock agent
        agent_registry.register(mock_claude_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION,
            preferred_agents=[AgentType.CLAUDE_CODE]
        )
        
        result = orchestrator._select_best_agent(task)
        assert result == mock_claude_agent
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
    
    def test_select_best_agent_with_capabilities(self, orchestrator, mock_claude_agent):
        """Test agent selection with required capabilities."""
        # Register mock agent
        agent_registry.register(mock_claude_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION,
            required_capabilities=["test_capability"]
        )
        
        result = orchestrator._select_best_agent(task)
        assert result == mock_claude_agent
        
        # Test with non-existent capability
        task.required_capabilities = ["nonexistent_capability"]
        result = orchestrator._select_best_agent(task)
        assert result is None
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
    
    def test_select_multiple_agents(self, orchestrator, mock_claude_agent, mock_openai_agent):
        """Test selection of multiple agents."""
        # Register mock agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.ANALYSIS
        )
        
        result = orchestrator._select_multiple_agents(task, 2)
        assert len(result) == 2
        assert mock_claude_agent in result
        assert mock_openai_agent in result
        
        # Test requesting more agents than available
        result = orchestrator._select_multiple_agents(task, 5)
        assert len(result) == 2  # Should return all available
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    async def test_execute_single_agent(self, orchestrator, mock_claude_agent):
        """Test single agent execution."""
        # Register mock agent
        agent_registry.register(mock_claude_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION
        )
        
        result = await orchestrator._execute_single_agent(task, mock_claude_agent)
        
        assert result.success
        assert result.task_id == task.task_id
        assert result.strategy_used == OrchestrationStrategy.SINGLE_AGENT
        assert len(result.agent_responses) == 1
        assert result.agent_responses[0].success
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
    
    async def test_execute_parallel_agents(self, orchestrator, mock_claude_agent, mock_openai_agent):
        """Test parallel agent execution."""
        # Register mock agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.ANALYSIS
        )
        
        agents = [mock_claude_agent, mock_openai_agent]
        result = await orchestrator._execute_parallel_agents(task, agents)
        
        assert result.success
        assert result.task_id == task.task_id
        assert result.strategy_used == OrchestrationStrategy.PARALLEL
        assert len(result.agent_responses) == 2
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    async def test_execute_task_single_agent(self, orchestrator, mock_claude_agent):
        """Test full task execution with single agent strategy."""
        # Register mock agent
        agent_registry.register(mock_claude_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION,
            strategy=OrchestrationStrategy.SINGLE_AGENT
        )
        
        result = await orchestrator.execute_task(task)
        
        assert result.success
        assert result.task_id == task.task_id
        assert result.strategy_used == OrchestrationStrategy.SINGLE_AGENT
        
        # Check metrics were updated
        assert orchestrator._total_tasks == 1
        assert orchestrator._successful_tasks == 1
        
        # Check task was stored
        assert task.task_id in orchestrator._task_results
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
    
    async def test_execute_task_parallel(self, orchestrator, mock_claude_agent, mock_openai_agent):
        """Test full task execution with parallel strategy."""
        # Register mock agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.ANALYSIS,
            strategy=OrchestrationStrategy.PARALLEL
        )
        
        result = await orchestrator.execute_task(task)
        
        assert result.success
        assert result.strategy_used == OrchestrationStrategy.PARALLEL
        assert len(result.agent_responses) == 2
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    async def test_execute_task_no_agents(self, orchestrator):
        """Test task execution when no agents are available."""
        # Clear agent registry
        agent_registry._agents.clear()
        for agent_list in agent_registry._agent_types.values():
            agent_list.clear()
        
        task = OrchestrationTask(
            description="Test task",
            task_type=TaskType.CODE_GENERATION
        )
        
        result = await orchestrator.execute_task(task)
        
        assert not result.success
        assert "No suitable agent found" in result.error_message
    
    def test_get_task_result(self, orchestrator):
        """Test task result retrieval."""
        task_id = str(uuid.uuid4())
        result = TaskResult(
            task_id=task_id,
            success=True,
            content="Test result"
        )
        
        orchestrator._task_results[task_id] = result
        
        retrieved_result = orchestrator.get_task_result(task_id)
        assert retrieved_result == result
        
        # Test non-existent task
        nonexistent_result = orchestrator.get_task_result("nonexistent")
        assert nonexistent_result is None
    
    def test_get_active_tasks(self, orchestrator):
        """Test active tasks retrieval."""
        task = OrchestrationTask(
            description="Active task",
            task_type=TaskType.CUSTOM
        )
        
        orchestrator._active_tasks[task.task_id] = task
        
        active_tasks = orchestrator.get_active_tasks()
        assert len(active_tasks) == 1
        assert active_tasks[0] == task
    
    def test_get_performance_metrics(self, orchestrator):
        """Test performance metrics retrieval."""
        # Simulate some activity
        orchestrator._total_tasks = 10
        orchestrator._successful_tasks = 8
        orchestrator._total_cost = 0.50
        
        metrics = orchestrator.get_performance_metrics()
        
        assert metrics["total_tasks"] == 10
        assert metrics["successful_tasks"] == 8
        assert metrics["success_rate"] == 0.8
        assert metrics["total_cost_usd"] == 0.50
        assert metrics["average_cost_per_task"] == 0.05
        assert "uptime_seconds" in metrics
        assert "tasks_per_minute" in metrics
        assert "active_tasks_count" in metrics


class TestOrchestrationEngine:
    """Test the OrchestrationEngine class."""
    
    @pytest.fixture
    def engine(self, mock_database):
        """Create an orchestration engine instance."""
        return OrchestrationEngine(mock_database)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert hasattr(engine, 'orchestrator')
        assert isinstance(engine.orchestrator, TaskOrchestrator)
    
    async def test_process_request(self, engine, mock_claude_agent):
        """Test processing a simple request."""
        # Register mock agent
        agent_registry.register(mock_claude_agent)
        
        result = await engine.process_request(
            "Test prompt",
            task_type=TaskType.CUSTOM,
            strategy=OrchestrationStrategy.SINGLE_AGENT
        )
        
        assert result.success
        assert result.strategy_used == OrchestrationStrategy.SINGLE_AGENT
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
    
    async def test_analyze_and_route_code_generation(self, engine, mock_claude_agent):
        """Test automatic routing for code generation."""
        # Register mock agent
        agent_registry.register(mock_claude_agent)
        
        result = await engine.analyze_and_route("Write a function to sort a list")
        
        assert result.success
        # Should detect code generation and use single agent
        assert result.strategy_used == OrchestrationStrategy.SINGLE_AGENT
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
    
    async def test_analyze_and_route_review(self, engine, mock_claude_agent, mock_openai_agent):
        """Test automatic routing for code review."""
        # Register mock agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        result = await engine.analyze_and_route("Please review this code for issues")
        
        assert result.success
        # Should detect review and use parallel agents
        assert result.strategy_used == OrchestrationStrategy.PARALLEL
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    async def test_analyze_and_route_research(self, engine, mock_claude_agent, mock_openai_agent):
        """Test automatic routing for research tasks."""
        # Register mock agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        result = await engine.analyze_and_route("Research the latest trends in AI")
        
        assert result.success
        # Should detect research and use parallel agents
        assert result.strategy_used == OrchestrationStrategy.PARALLEL
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    async def test_analyze_and_route_debate(self, engine, mock_claude_agent, mock_openai_agent):
        """Test automatic routing for debate tasks."""
        # Register mock agents  
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        with patch('src.orchestration.debate.DebateProtocol') as mock_debate:
            # Mock debate protocol
            mock_debate_instance = mock_debate.return_value
            mock_debate_instance.execute_debate = AsyncMock(return_value=TaskResult(
                task_id="test",
                success=True,
                content="Debate result",
                strategy_used=OrchestrationStrategy.DEBATE
            ))
            
            result = await engine.analyze_and_route("Compare and debate the merits of different approaches")
            
            assert result.success
            # Should detect debate and use debate strategy
            assert result.strategy_used == OrchestrationStrategy.DEBATE
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    def test_get_status(self, engine):
        """Test engine status retrieval."""
        status = engine.get_status()
        
        assert "orchestrator_metrics" in status
        assert "available_agents" in status
        assert "agent_health" in status
        
        assert isinstance(status["available_agents"], int)
        assert isinstance(status["orchestrator_metrics"], dict)
        assert isinstance(status["agent_health"], dict)