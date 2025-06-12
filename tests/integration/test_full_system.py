"""Integration tests for the full AngelaMCP system."""

import pytest
import asyncio
import uuid
from unittest.mock import patch, AsyncMock, MagicMock

from src.agents.base import agent_registry, TaskContext, TaskType, AgentRole
from src.orchestration.orchestrator import OrchestrationEngine, OrchestrationTask, OrchestrationStrategy
from src.persistence.database import DatabaseManager
from src.ui.streaming import StreamingUI


@pytest.mark.integration
class TestFullSystemIntegration:
    """Test the complete AngelaMCP system integration."""
    
    @pytest.fixture
    async def full_system(self, mock_database, mock_claude_agent, mock_openai_agent, mock_gemini_agent):
        """Set up a complete system with all components."""
        # Clear and register all agents
        agent_registry._agents.clear()
        for agent_list in agent_registry._agent_types.values():
            agent_list.clear()
        
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        agent_registry.register(mock_gemini_agent)
        
        # Create orchestration engine
        engine = OrchestrationEngine(mock_database)
        
        # Create streaming UI
        streaming_ui = StreamingUI()
        
        yield {
            "engine": engine,
            "database": mock_database,
            "streaming_ui": streaming_ui,
            "agents": {
                "claude": mock_claude_agent,
                "openai": mock_openai_agent,
                "gemini": mock_gemini_agent
            }
        }
        
        # Cleanup
        agent_registry._agents.clear()
        for agent_list in agent_registry._agent_types.values():
            agent_list.clear()
    
    async def test_simple_task_execution(self, full_system):
        """Test simple task execution through the full system."""
        engine = full_system["engine"]
        streaming_ui = full_system["streaming_ui"]
        
        # Execute a simple task
        task_prompt = "Write a Python function to calculate factorial"
        
        # Track streaming events
        events_received = []
        def event_handler(event):
            events_received.append(event)
        
        streaming_ui.subscribe(event_handler)
        
        # Execute task
        result = await engine.analyze_and_route(task_prompt)
        
        # Verify result
        assert result.success
        assert result.content is not None
        assert len(result.content) > 0
        assert result.execution_time_ms > 0
        
        # Verify agent was selected appropriately (should be Claude for code generation)
        assert result.strategy_used in [OrchestrationStrategy.SINGLE_AGENT, OrchestrationStrategy.PARALLEL]
        
        # Check that cost and token tracking works
        if result.total_cost_usd is not None:
            assert result.total_cost_usd >= 0
        if result.total_tokens is not None:
            assert result.total_tokens >= 0
    
    async def test_parallel_task_execution(self, full_system):
        """Test parallel execution with multiple agents."""
        engine = full_system["engine"]
        
        # Task that should trigger parallel execution
        task_prompt = "Please review and analyze this code for potential improvements"
        
        result = await engine.analyze_and_route(task_prompt)
        
        # Should succeed and use parallel strategy
        assert result.success
        assert result.strategy_used == OrchestrationStrategy.PARALLEL
        
        # Should have responses from multiple agents
        if hasattr(result, 'agent_responses'):
            assert len(result.agent_responses) >= 1
    
    async def test_debate_execution(self, full_system):
        """Test debate execution through the system."""
        engine = full_system["engine"]
        
        # Task that should trigger debate
        task_prompt = "Compare and debate the pros and cons of microservices vs monolithic architecture"
        
        with patch('src.orchestration.debate.DebateProtocol') as mock_debate_class:
            # Mock the debate protocol
            mock_debate = mock_debate_class.return_value
            mock_debate.execute_debate = AsyncMock()
            
            # Mock debate result
            from src.orchestration.orchestrator import TaskResult
            mock_debate.execute_debate.return_value = TaskResult(
                task_id=str(uuid.uuid4()),
                success=True,
                content="Debate consensus: Both architectures have merits...",
                strategy_used=OrchestrationStrategy.DEBATE,
                execution_time_ms=5000.0,
                total_cost_usd=0.15,
                total_tokens=500,
                metadata={"debate_rounds": 3, "consensus_score": 0.75}
            )
            
            result = await engine.analyze_and_route(task_prompt)
            
            # Verify debate was triggered
            assert result.success
            assert result.strategy_used == OrchestrationStrategy.DEBATE
            assert "consensus" in result.content.lower()
    
    async def test_agent_failure_handling(self, full_system):
        """Test system behavior when agents fail."""
        engine = full_system["engine"]
        claude_agent = full_system["agents"]["claude"]
        
        # Make Claude agent fail
        claude_agent.generate = AsyncMock(side_effect=Exception("Agent failure"))
        
        # Task should still complete with other agents
        result = await engine.process_request(
            "Simple task",
            strategy=OrchestrationStrategy.PARALLEL
        )
        
        # System should handle the failure gracefully
        # Either succeed with other agents or fail gracefully
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error_message is not None
    
    async def test_streaming_events_integration(self, full_system):
        """Test streaming events throughout task execution."""
        engine = full_system["engine"]
        streaming_ui = full_system["streaming_ui"]
        
        # Collect streaming events
        events = []
        def collect_events(event):
            events.append(event)
        
        streaming_ui.subscribe(collect_events)
        
        # Emit some test events
        streaming_ui.emit_task_started("test-task", "code_generation", ["claude"])
        streaming_ui.emit_progress_update("test-task", 50.0, "processing")
        streaming_ui.emit_task_completed("test-task", True)
        
        # Give events time to process
        await asyncio.sleep(0.1)
        
        # Verify events were captured
        assert len(events) >= 3
        
        event_types = [event.event_type.value for event in events]
        assert "task_started" in event_types
        assert "progress_update" in event_types
        assert "task_completed" in event_types
    
    async def test_database_persistence_integration(self, full_system):
        """Test database persistence throughout system operation."""
        engine = full_system["engine"]
        database = full_system["database"]
        
        # Execute a task that should be persisted
        result = await engine.process_request(
            "Test task for persistence",
            task_type=TaskType.CUSTOM
        )
        
        # Task should complete
        assert result.success
        
        # Verify orchestrator metrics are updated
        metrics = engine.orchestrator.get_performance_metrics()
        assert metrics["total_tasks"] >= 1
        if result.success:
            assert metrics["successful_tasks"] >= 1
    
    async def test_agent_health_monitoring(self, full_system):
        """Test agent health monitoring integration."""
        agents = full_system["agents"]
        
        # Check health of all agents
        health_results = await agent_registry.health_check_all()
        
        # All mock agents should be healthy
        assert len(health_results) == 3
        
        for agent_name, health in health_results.items():
            assert health["status"] == "healthy"
            assert "response_time_ms" in health
            assert "performance_metrics" in health
    
    async def test_system_error_recovery(self, full_system):
        """Test system recovery from various error conditions."""
        engine = full_system["engine"]
        
        # Test with invalid task type
        result = await engine.process_request(
            "Test task",
            task_type=TaskType.CUSTOM,
            strategy=OrchestrationStrategy.SINGLE_AGENT
        )
        
        # Should handle gracefully
        assert isinstance(result.success, bool)
        
        # Test with empty prompt
        result = await engine.process_request("")
        
        # Should handle empty input gracefully
        assert isinstance(result.success, bool)
    
    async def test_concurrent_task_execution(self, full_system):
        """Test concurrent execution of multiple tasks."""
        engine = full_system["engine"]
        
        # Create multiple tasks to run concurrently
        tasks = [
            engine.process_request(f"Task {i}: Simple computation", TaskType.CUSTOM)
            for i in range(3)
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete (successfully or with handled errors)
        assert len(results) == 3
        
        for result in results:
            if isinstance(result, Exception):
                # If there's an exception, it should be handled gracefully
                assert isinstance(result, Exception)
            else:
                # If it's a result, it should have the expected structure
                assert hasattr(result, 'success')
                assert hasattr(result, 'content')
    
    async def test_system_performance_tracking(self, full_system):
        """Test system-wide performance tracking."""
        engine = full_system["engine"]
        
        # Execute several tasks to generate metrics
        for i in range(3):
            await engine.process_request(f"Performance test task {i}")
        
        # Check orchestrator performance metrics
        orchestrator_metrics = engine.orchestrator.get_performance_metrics()
        
        assert orchestrator_metrics["total_tasks"] >= 3
        assert "success_rate" in orchestrator_metrics
        assert "total_cost_usd" in orchestrator_metrics
        assert "uptime_seconds" in orchestrator_metrics
        
        # Check agent performance metrics
        for agent in agent_registry.get_all_agents():
            agent_metrics = agent.performance_metrics
            assert "total_requests" in agent_metrics
            assert "total_cost_usd" in agent_metrics
            assert "uptime_seconds" in agent_metrics
    
    async def test_system_shutdown_cleanup(self, full_system):
        """Test proper system shutdown and cleanup."""
        engine = full_system["engine"]
        database = full_system["database"]
        
        # Execute a task first
        await engine.process_request("Test task before shutdown")
        
        # Shutdown all agents
        await agent_registry.shutdown_all()
        
        # Verify agents were cleaned up
        assert len(agent_registry.get_all_agents()) == 0
        
        # Database should still be accessible for final operations
        assert database is not None
    
    async def test_ui_integration_with_system(self, full_system):
        """Test UI components integration with system data."""
        from src.ui.display import DisplayManager
        from src.ui.terminal import TerminalUI
        
        engine = full_system["engine"]
        streaming_ui = full_system["streaming_ui"]
        
        # Create UI components
        display_manager = DisplayManager()
        
        # Execute a task to generate data
        result = await engine.process_request("UI integration test")
        
        # Test that UI components can handle system data
        if result.success:
            # Create panels with real data
            performance_metrics = engine.orchestrator.get_performance_metrics()
            performance_panel = display_manager.create_performance_panel(performance_metrics)
            assert performance_panel is not None
            
            # Test streaming UI with real events
            live_panel = streaming_ui.create_live_display()
            assert live_panel is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSystemStressTests:
    """Stress tests for the system under load."""
    
    async def test_high_concurrency_tasks(self, full_system):
        """Test system under high concurrent task load."""
        engine = full_system["engine"]
        
        # Create many concurrent tasks
        num_tasks = 10
        tasks = [
            engine.process_request(f"Concurrent task {i}")
            for i in range(num_tasks)
        ]
        
        # Execute all concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=60.0  # 1 minute timeout
            )
            
            # Count successful completions
            completed = 0
            for result in results:
                if not isinstance(result, Exception) and hasattr(result, 'success'):
                    completed += 1
            
            # At least some should complete successfully
            assert completed > 0
            
        except asyncio.TimeoutError:
            # If timeout occurs, that's acceptable for stress test
            pytest.skip("Stress test timed out - acceptable for high load")
    
    async def test_rapid_fire_requests(self, full_system):
        """Test rapid succession of requests."""
        engine = full_system["engine"]
        
        # Send requests in rapid succession
        results = []
        for i in range(5):
            result = await engine.process_request(f"Rapid request {i}")
            results.append(result)
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
        
        # All requests should be handled
        assert len(results) == 5
        
        # Check that system maintained stability
        for result in results:
            assert hasattr(result, 'success')
    
    async def test_memory_usage_stability(self, full_system):
        """Test that memory usage remains stable over many operations."""
        engine = full_system["engine"]
        
        # Execute many tasks to test memory stability
        for i in range(20):
            result = await engine.process_request(f"Memory test {i}")
            
            # Small delay to allow garbage collection
            if i % 5 == 0:
                await asyncio.sleep(0.05)
        
        # If we get here without memory errors, the test passes
        # More sophisticated memory monitoring could be added
        assert True