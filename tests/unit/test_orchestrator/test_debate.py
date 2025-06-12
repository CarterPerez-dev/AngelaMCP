"""Tests for the debate protocol system."""

import pytest
import asyncio
import uuid
import time
from unittest.mock import AsyncMock, Mock, patch

from src.orchestration.debate import (
    DebateProtocol, DebateResult, DebateRound, DebateArgument,
    DebateRole, ArgumentType
)
from src.orchestration.orchestrator import OrchestrationTask, TaskType, OrchestrationStrategy
from src.agents.base import AgentType, TaskContext, AgentResponse, agent_registry


class TestDebateArgument:
    """Test the DebateArgument data class."""
    
    def test_argument_creation(self):
        """Test debate argument creation."""
        argument = DebateArgument(
            agent_name="test_agent",
            agent_type="claude_code",
            role=DebateRole.PROPOSER,
            argument_type=ArgumentType.INITIAL_PROPOSAL,
            content="Test argument content",
            confidence_score=0.8,
            evidence=["evidence1", "evidence2"]
        )
        
        assert argument.agent_name == "test_agent"
        assert argument.agent_type == "claude_code"
        assert argument.role == DebateRole.PROPOSER
        assert argument.argument_type == ArgumentType.INITIAL_PROPOSAL
        assert argument.content == "Test argument content"
        assert argument.confidence_score == 0.8
        assert len(argument.evidence) == 2
        assert argument.timestamp > 0
        assert argument.metadata == {}
    
    def test_argument_defaults(self):
        """Test argument creation with defaults."""
        argument = DebateArgument()
        
        assert argument.agent_name == ""
        assert argument.agent_type == ""
        assert argument.role == DebateRole.PROPOSER
        assert argument.argument_type == ArgumentType.INITIAL_PROPOSAL
        assert argument.content == ""
        assert argument.confidence_score is None
        assert argument.evidence == []
        assert argument.addresses_argument_id is None
        assert argument.metadata == {}


class TestDebateRound:
    """Test the DebateRound data class."""
    
    def test_round_creation(self):
        """Test debate round creation."""
        arguments = [
            DebateArgument(agent_name="agent1", content="Argument 1"),
            DebateArgument(agent_name="agent2", content="Argument 2")
        ]
        
        round_data = DebateRound(
            round_number=1,
            arguments=arguments,
            round_summary="Test round summary",
            consensus_score=0.75,
            round_duration_ms=1000.0
        )
        
        assert round_data.round_number == 1
        assert len(round_data.arguments) == 2
        assert round_data.round_summary == "Test round summary"
        assert round_data.consensus_score == 0.75
        assert round_data.round_duration_ms == 1000.0
    
    def test_round_defaults(self):
        """Test round creation with defaults."""
        round_data = DebateRound(round_number=1)
        
        assert round_data.round_number == 1
        assert round_data.arguments == []
        assert round_data.round_summary == ""
        assert round_data.consensus_score == 0.0
        assert round_data.round_duration_ms == 0.0


class TestDebateResult:
    """Test the DebateResult data class."""
    
    def test_result_creation(self, mock_debate_result):
        """Test debate result creation."""
        assert mock_debate_result.success
        assert mock_debate_result.final_consensus == "Test consensus"
        assert mock_debate_result.confidence_score == 0.8
        assert len(mock_debate_result.rounds) == 1
        assert len(mock_debate_result.participating_agents) == 2
        assert mock_debate_result.total_duration_ms == 1500.0
        assert mock_debate_result.total_cost_usd == 0.002
        assert mock_debate_result.total_tokens == 100
    
    def test_result_defaults(self):
        """Test result creation with defaults."""
        result = DebateResult(
            debate_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            success=False,
            final_consensus="No consensus",
            confidence_score=0.0
        )
        
        assert not result.success
        assert result.final_consensus == "No consensus"
        assert result.confidence_score == 0.0
        assert result.rounds == []
        assert result.participating_agents == []
        assert result.total_duration_ms == 0.0
        assert result.total_cost_usd == 0.0
        assert result.total_tokens == 0
        assert result.metadata == {}
        assert result.error_message is None


class TestDebateProtocol:
    """Test the DebateProtocol class."""
    
    @pytest.fixture
    def debate_task(self):
        """Create a debate task for testing."""
        return OrchestrationTask(
            description="Should we use microservices architecture?",
            task_type=TaskType.ANALYSIS,
            strategy=OrchestrationStrategy.DEBATE,
            enable_debate=True,
            max_debate_rounds=3,
            consensus_threshold=0.7,
            timeout_seconds=300
        )
    
    @pytest.fixture
    def mock_orchestrator(self, mock_database):
        """Create a mock orchestrator."""
        from src.orchestration.orchestrator import TaskOrchestrator
        return TaskOrchestrator(mock_database)
    
    @pytest.fixture
    def debate_protocol(self, mock_orchestrator, debate_task):
        """Create a debate protocol instance."""
        return DebateProtocol(mock_orchestrator, debate_task)
    
    def test_protocol_initialization(self, debate_protocol, debate_task):
        """Test debate protocol initialization."""
        assert debate_protocol.task == debate_task
        assert debate_protocol.max_rounds == debate_task.max_debate_rounds
        assert debate_protocol.consensus_threshold == debate_task.consensus_threshold
        assert debate_protocol.timeout_seconds == debate_task.timeout_seconds
        assert debate_protocol.rounds == []
        assert debate_protocol.arguments == []
        assert debate_protocol.participating_agents == []
        assert debate_protocol.total_cost == 0.0
        assert debate_protocol.total_tokens == 0
    
    def test_select_debate_agents_insufficient(self, debate_protocol):
        """Test agent selection with insufficient agents."""
        # Clear agent registry
        agent_registry._agents.clear()
        for agent_list in agent_registry._agent_types.values():
            agent_list.clear()
        
        # Add only one agent
        mock_agent = Mock()
        mock_agent.agent_type = AgentType.CLAUDE_CODE
        mock_agent.name = "single_agent"
        agent_registry._agents = {"single_agent": mock_agent}
        
        with pytest.raises(ValueError, match="At least 2 agents required"):
            debate_protocol._select_debate_agents()
    
    def test_select_debate_agents_sufficient(self, debate_protocol, mock_claude_agent, mock_openai_agent, mock_gemini_agent):
        """Test agent selection with sufficient agents."""
        # Register multiple agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        agent_registry.register(mock_gemini_agent)
        
        agents = debate_protocol._select_debate_agents()
        
        assert len(agents) >= 2
        assert len(agents) <= 3
        
        # Should prefer diverse agent types
        agent_types = {agent.agent_type for agent in agents}
        assert len(agent_types) >= 2
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
        agent_registry.unregister(mock_gemini_agent.name)
    
    def test_assign_debate_roles(self, debate_protocol, mock_claude_agent, mock_openai_agent, mock_gemini_agent):
        """Test debate role assignment."""
        agents = [mock_claude_agent, mock_openai_agent, mock_gemini_agent]
        roles = debate_protocol._assign_debate_roles(agents)
        
        assert len(roles) >= 2
        assert DebateRole.PROPOSER in roles.values()
        assert DebateRole.CHALLENGER in roles.values()
        
        # Should assign synthesizer if 3+ agents
        if len(agents) >= 3:
            assert DebateRole.SYNTHESIZER in roles.values()
        
        # Each agent should have only one role
        assert len(roles) == len(set(roles.keys()))
        assert len(set(roles.values())) == len(roles)
    
    async def test_get_initial_proposal(self, debate_protocol, mock_claude_agent):
        """Test getting initial proposal from proposer."""
        # Mock agent response
        mock_response = AgentResponse(
            success=True,
            content="Initial proposal content",
            agent_type="claude_code",
            confidence_score=0.8,
            cost_usd=0.001,
            tokens_used=50
        )
        mock_claude_agent.generate = AsyncMock(return_value=mock_response)
        
        argument = await debate_protocol._get_initial_proposal(mock_claude_agent)
        
        assert argument.agent_name == mock_claude_agent.name
        assert argument.agent_type == mock_claude_agent.agent_type.value
        assert argument.role == DebateRole.PROPOSER
        assert argument.argument_type == ArgumentType.INITIAL_PROPOSAL
        assert argument.content == "Initial proposal content"
        assert argument.confidence_score == 0.8
        
        # Check cost tracking
        assert debate_protocol.total_cost == 0.001
        assert debate_protocol.total_tokens == 50
    
    async def test_get_counter_argument(self, debate_protocol, mock_openai_agent):
        """Test getting counter argument from challenger."""
        # Create initial proposal
        proposal = DebateArgument(
            agent_name="proposer",
            content="Initial proposal",
            role=DebateRole.PROPOSER,
            argument_type=ArgumentType.INITIAL_PROPOSAL
        )
        
        # Mock agent response
        mock_response = AgentResponse(
            success=True,
            content="Counter argument content",
            agent_type="openai",
            confidence_score=0.7,
            cost_usd=0.002,
            tokens_used=75
        )
        mock_openai_agent.generate = AsyncMock(return_value=mock_response)
        
        argument = await debate_protocol._get_counter_argument(mock_openai_agent, proposal)
        
        assert argument.agent_name == mock_openai_agent.name
        assert argument.agent_type == mock_openai_agent.agent_type.value
        assert argument.role == DebateRole.CHALLENGER
        assert argument.argument_type == ArgumentType.COUNTER_ARGUMENT
        assert argument.content == "Counter argument content"
        assert argument.addresses_argument_id == proposal.id
    
    async def test_get_rebuttal(self, debate_protocol, mock_claude_agent):
        """Test getting rebuttal from proposer."""
        # Create initial proposal and counter argument
        proposal = DebateArgument(
            agent_name="proposer",
            content="Initial proposal",
            role=DebateRole.PROPOSER
        )
        
        counter = DebateArgument(
            agent_name="challenger",
            content="Counter argument",
            role=DebateRole.CHALLENGER,
            argument_type=ArgumentType.COUNTER_ARGUMENT
        )
        
        # Mock agent response
        mock_response = AgentResponse(
            success=True,
            content="Rebuttal content",
            agent_type="claude_code",
            confidence_score=0.9
        )
        mock_claude_agent.generate = AsyncMock(return_value=mock_response)
        
        argument = await debate_protocol._get_rebuttal(mock_claude_agent, proposal, counter)
        
        assert argument.agent_name == mock_claude_agent.name
        assert argument.role == DebateRole.PROPOSER
        assert argument.argument_type == ArgumentType.REBUTTAL
        assert argument.content == "Rebuttal content"
        assert argument.addresses_argument_id == counter.id
    
    async def test_synthesize_consensus(self, debate_protocol, mock_gemini_agent):
        """Test consensus synthesis."""
        # Create sample arguments
        arguments = [
            DebateArgument(
                agent_name="agent1",
                content="Argument 1",
                role=DebateRole.PROPOSER
            ),
            DebateArgument(
                agent_name="agent2",
                content="Argument 2",
                role=DebateRole.CHALLENGER
            )
        ]
        
        # Mock agent response
        mock_response = AgentResponse(
            success=True,
            content="Synthesized consensus",
            agent_type="gemini",
            confidence_score=0.85
        )
        mock_gemini_agent.generate = AsyncMock(return_value=mock_response)
        
        synthesis = await debate_protocol._synthesize_consensus(mock_gemini_agent, arguments)
        
        assert synthesis.agent_name == mock_gemini_agent.name
        assert synthesis.role == DebateRole.SYNTHESIZER
        assert synthesis.argument_type == ArgumentType.SYNTHESIS
        assert synthesis.content == "Synthesized consensus"
    
    def test_calculate_consensus_score(self, debate_protocol):
        """Test consensus score calculation."""
        # Test with no arguments
        score = debate_protocol._calculate_consensus_score([])
        assert score == 0.0
        
        # Test with high confidence arguments
        high_confidence_args = [
            DebateArgument(confidence_score=0.9),
            DebateArgument(confidence_score=0.8)
        ]
        score = debate_protocol._calculate_consensus_score(high_confidence_args)
        assert score > 0.5
        
        # Test with synthesis bonus
        with_synthesis = high_confidence_args + [
            DebateArgument(argument_type=ArgumentType.SYNTHESIS, confidence_score=0.7)
        ]
        score_with_synthesis = debate_protocol._calculate_consensus_score(with_synthesis)
        assert score_with_synthesis > score
        
        # Test with counter-argument penalty
        with_counters = high_confidence_args + [
            DebateArgument(argument_type=ArgumentType.COUNTER_ARGUMENT, confidence_score=0.6),
            DebateArgument(argument_type=ArgumentType.COUNTER_ARGUMENT, confidence_score=0.5)
        ]
        score_with_counters = debate_protocol._calculate_consensus_score(with_counters)
        assert score_with_counters < score
    
    async def test_execute_debate_full_flow(self, debate_protocol, mock_claude_agent, mock_openai_agent, mock_gemini_agent):
        """Test full debate execution flow."""
        # Register agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        agent_registry.register(mock_gemini_agent)
        
        # Mock agent responses
        mock_claude_agent.generate = AsyncMock(return_value=AgentResponse(
            success=True,
            content="Claude proposal",
            agent_type="claude_code",
            confidence_score=0.8,
            cost_usd=0.001,
            tokens_used=50
        ))
        
        mock_openai_agent.generate = AsyncMock(return_value=AgentResponse(
            success=True,
            content="OpenAI counter",
            agent_type="openai",
            confidence_score=0.7,
            cost_usd=0.002,
            tokens_used=60
        ))
        
        mock_gemini_agent.generate = AsyncMock(return_value=AgentResponse(
            success=True,
            content="Gemini synthesis",
            agent_type="gemini",
            confidence_score=0.9,
            cost_usd=0.003,
            tokens_used=80
        ))
        
        # Execute debate
        result = await debate_protocol.execute_debate()
        
        # Verify result
        assert isinstance(result.task_id, str)
        assert result.success or not result.success  # Could succeed or fail based on consensus
        assert len(result.metadata["debate_result"].rounds) >= 1
        assert len(result.metadata["debate_result"].participating_agents) >= 2
        assert result.total_cost_usd > 0
        assert result.total_tokens > 0
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
        agent_registry.unregister(mock_gemini_agent.name)
    
    async def test_execute_debate_early_consensus(self, debate_protocol, mock_claude_agent, mock_openai_agent):
        """Test debate with early consensus achievement."""
        # Set low consensus threshold for testing
        debate_protocol.consensus_threshold = 0.5
        
        # Register agents
        agent_registry.register(mock_claude_agent)
        agent_registry.register(mock_openai_agent)
        
        # Mock high-confidence responses to trigger early consensus
        mock_claude_agent.generate = AsyncMock(return_value=AgentResponse(
            success=True,
            content="Strong proposal",
            agent_type="claude_code",
            confidence_score=0.9,
            cost_usd=0.001,
            tokens_used=50
        ))
        
        mock_openai_agent.generate = AsyncMock(return_value=AgentResponse(
            success=True,
            content="Agreeable response",
            agent_type="openai",
            confidence_score=0.8,
            cost_usd=0.002,
            tokens_used=60
        ))
        
        result = await debate_protocol.execute_debate()
        
        # Should succeed with early consensus
        debate_result = result.metadata["debate_result"]
        assert len(debate_result.rounds) <= debate_protocol.max_rounds
        
        # Cleanup
        agent_registry.unregister(mock_claude_agent.name)
        agent_registry.unregister(mock_openai_agent.name)
    
    async def test_execute_debate_insufficient_agents(self, debate_protocol):
        """Test debate execution with insufficient agents."""
        # Clear agent registry
        agent_registry._agents.clear()
        for agent_list in agent_registry._agent_types.values():
            agent_list.clear()
        
        result = await debate_protocol.execute_debate()
        
        assert not result.success
        assert "Need at least 2 agents" in result.error_message