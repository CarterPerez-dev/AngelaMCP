"""Pytest configuration and fixtures."""

import pytest
import asyncio
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
import os
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set test environment
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///test_angelamcp.db"
os.environ["LOG_LEVEL"] = "DEBUG"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def mock_settings(temp_dir):
    """Create mock settings for testing."""
    from config.settings import Settings
    from pydantic import SecretStr
    
    return Settings(
        # Core settings
        app_name="AngelaMCP-Test",
        app_version="1.0.0-test",
        debug=True,
        log_level="DEBUG",
        log_file=str(temp_dir / "test.log"),
        
        # Database
        database_url="sqlite:///test_angelamcp.db",
        database_pool_size=5,
        database_max_overflow=10,
        database_echo=False,
        
        # Agent paths and keys
        claude_code_path=str(temp_dir / "mock_claude"),
        openai_api_key=SecretStr("test-openai-key"),
        google_api_key=SecretStr("test-google-key"),
        
        # Claude Code settings
        claude_session_dir=str(temp_dir / "claude_sessions"),
        claude_session_persist=True,
        claude_code_max_turns=5,
        claude_code_output_format="json",
        
        # OpenAI settings
        openai_model="o3-mini",
        openai_max_tokens=4000,
        openai_temperature=0.7,
        
        # Gemini settings
        gemini_model="gemini-2.5-pro",
        gemini_max_output_tokens=4000,
        gemini_temperature=0.7,
        
        # Agent retry settings
        claude_code_max_retries=3,
        openai_max_retries=3,
        gemini_max_retries=3,
    )

@pytest.fixture
async def mock_database():
    """Create a mock database for testing."""
    from src.persistence.database import DatabaseManager
    
    # Use in-memory SQLite for tests
    db_manager = DatabaseManager("sqlite:///:memory:")
    await db_manager.initialize()
    
    yield db_manager
    
    await db_manager.close()

@pytest.fixture
def mock_console():
    """Create a mock Rich console for UI testing."""
    from rich.console import Console
    from io import StringIO
    
    string_io = StringIO()
    console = Console(file=string_io, force_terminal=True, width=120)
    return console

@pytest.fixture
def mock_agent_response():
    """Create a mock agent response."""
    from src.agents.base import AgentResponse
    
    return AgentResponse(
        success=True,
        content="Test response content",
        agent_type="test_agent",
        execution_time_ms=100.0,
        tokens_used=50,
        cost_usd=0.001,
        metadata={"test": "data"}
    )

@pytest.fixture
def mock_task_context():
    """Create a mock task context."""
    from src.agents.base import TaskContext, TaskType, AgentRole
    
    return TaskContext(
        task_id=str(uuid.uuid4()),
        task_type=TaskType.CUSTOM,
        agent_role=AgentRole.PRIMARY,
        max_tokens=1000,
        timeout_seconds=30
    )

@pytest.fixture
def mock_claude_agent(mock_settings):
    """Create a mock Claude Code agent."""
    from src.agents.claude_agent import ClaudeCodeAgent
    
    # Create mock claude executable
    mock_claude_path = Path(mock_settings.claude_code_path)
    mock_claude_path.parent.mkdir(parents=True, exist_ok=True)
    mock_claude_path.write_text("#!/bin/bash\necho 'Mock Claude response'")
    mock_claude_path.chmod(0o755)
    
    with patch('src.agents.claude_agent.ClaudeCodeAgent._verify_claude_installation'):
        agent = ClaudeCodeAgent("test_claude", str(mock_claude_path))
        agent._execute_command = AsyncMock(return_value={
            "type": "result",
            "subtype": "success", 
            "result": "Mock Claude response",
            "cost_usd": 0.001,
            "num_turns": 1
        })
        yield agent

@pytest.fixture
def mock_openai_agent(mock_settings):
    """Create a mock OpenAI agent."""
    from src.agents.openai_agent import OpenAIAgent
    from unittest.mock import MagicMock
    
    agent = OpenAIAgent("test_openai")
    
    # Mock the client
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Mock OpenAI response"
    mock_completion.choices[0].finish_reason = "stop"
    mock_completion.usage.completion_tokens = 25
    mock_completion.model = "o3-mini"
    mock_completion.id = "test-id"
    mock_completion.system_fingerprint = "test-fp"
    
    agent.client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    yield agent

@pytest.fixture
def mock_gemini_agent(mock_settings):
    """Create a mock Gemini agent."""
    from src.agents.gemini_agent import GeminiAgent
    from unittest.mock import MagicMock
    
    agent = GeminiAgent("test_gemini")
    
    # Mock the client
    mock_response = MagicMock()
    mock_response.text = "Mock Gemini response"
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Mock Gemini response"
    mock_response.candidates[0].finish_reason = "STOP"
    mock_response.candidates[0].safety_ratings = []
    
    agent.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    yield agent

@pytest.fixture
async def mock_orchestrator(mock_database, mock_claude_agent, mock_openai_agent, mock_gemini_agent):
    """Create a mock orchestrator with agents."""
    from src.orchestration.orchestrator import TaskOrchestrator, OrchestrationEngine
    from src.agents.base import agent_registry
    
    # Clear and register mock agents
    agent_registry._agents.clear()
    for agent_list in agent_registry._agent_types.values():
        agent_list.clear()
    
    agent_registry.register(mock_claude_agent)
    agent_registry.register(mock_openai_agent) 
    agent_registry.register(mock_gemini_agent)
    
    orchestrator = TaskOrchestrator(mock_database)
    engine = OrchestrationEngine(mock_database)
    
    yield engine
    
    # Cleanup
    agent_registry._agents.clear()
    for agent_list in agent_registry._agent_types.values():
        agent_list.clear()

@pytest.fixture
def mock_debate_result():
    """Create a mock debate result."""
    from src.orchestration.debate import DebateResult, DebateRound, DebateArgument, DebateRole, ArgumentType
    
    arguments = [
        DebateArgument(
            agent_name="test_claude",
            agent_type="claude_code",
            role=DebateRole.PROPOSER,
            argument_type=ArgumentType.INITIAL_PROPOSAL,
            content="Initial proposal content",
            confidence_score=0.8
        ),
        DebateArgument(
            agent_name="test_openai",
            agent_type="openai",
            role=DebateRole.CHALLENGER,
            argument_type=ArgumentType.COUNTER_ARGUMENT,
            content="Counter argument content",
            confidence_score=0.7
        )
    ]
    
    rounds = [
        DebateRound(
            round_number=1,
            arguments=arguments,
            round_summary="Test round",
            consensus_score=0.75,
            round_duration_ms=1000.0
        )
    ]
    
    return DebateResult(
        debate_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        success=True,
        final_consensus="Test consensus",
        confidence_score=0.8,
        rounds=rounds,
        participating_agents=["test_claude", "test_openai"],
        total_duration_ms=1500.0,
        total_cost_usd=0.002,
        total_tokens=100
    )

@pytest.fixture
def mock_vote_result():
    """Create a mock vote result."""
    from src.orchestration.voting import VoteResult, Vote, VoteType, VotingMethod
    
    votes = [
        Vote(
            agent_name="test_claude",
            agent_type="claude_code",
            vote_type=VoteType.APPROVE,
            confidence=0.9,
            reasoning="Good proposal",
            weight=1.5
        ),
        Vote(
            agent_name="test_openai", 
            agent_type="openai",
            vote_type=VoteType.REJECT,
            confidence=0.7,
            reasoning="Has issues",
            weight=1.0
        )
    ]
    
    return VoteResult(
        vote_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        success=False,
        final_decision=VoteType.REJECT,
        confidence_score=0.7,
        votes=votes,
        total_weight=2.5,
        approve_weight=1.5,
        reject_weight=1.0,
        abstain_weight=0.0,
        has_veto=False,
        voting_method=VotingMethod.CLAUDE_VETO,
        duration_ms=500.0
    )

@pytest.fixture
def mock_stream_events():
    """Create mock stream events for UI testing."""
    from src.ui.streaming import StreamEvent, StreamEventType
    import time
    
    return [
        StreamEvent(
            event_type=StreamEventType.TASK_STARTED,
            timestamp=time.time(),
            source="orchestrator",
            data={
                "task_id": str(uuid.uuid4()),
                "task_type": "test",
                "agents": ["test_claude"]
            }
        ),
        StreamEvent(
            event_type=StreamEventType.AGENT_RESPONSE,
            timestamp=time.time(),
            source="test_claude",
            data={
                "agent_name": "test_claude",
                "success": True,
                "execution_time_ms": 100.0
            }
        )
    ]

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
