"""Tests for the UI display components."""

import pytest
from unittest.mock import Mock, MagicMock
from rich.console import Console
from io import StringIO

from src.ui.display import DisplayManager, DisplayTheme
from src.agents.base import BaseAgent, AgentType, AgentCapability
from src.orchestration.debate import DebateResult, DebateRound, DebateArgument, DebateRole, ArgumentType
from src.orchestration.voting import VoteResult, Vote, VoteType, VotingMethod


class TestDisplayTheme:
    """Test the DisplayTheme configuration."""
    
    def test_default_theme(self):
        """Test default theme creation."""
        theme = DisplayTheme()
        
        assert theme.primary == "bright_blue"
        assert theme.secondary == "bright_cyan"
        assert theme.success == "bright_green"
        assert theme.warning == "bright_yellow"
        assert theme.error == "bright_red"
        assert theme.info == "bright_white"
        assert theme.muted == "dim white"
        assert theme.claude_color == "bright_magenta"
        assert theme.openai_color == "bright_green"
        assert theme.gemini_color == "bright_blue"
    
    def test_custom_theme(self):
        """Test custom theme creation."""
        theme = DisplayTheme(
            primary="red",
            claude_color="yellow",
            openai_color="blue"
        )
        
        assert theme.primary == "red"
        assert theme.claude_color == "yellow"
        assert theme.openai_color == "blue"
        # Should keep defaults for non-specified values
        assert theme.secondary == "bright_cyan"


class TestDisplayManager:
    """Test the DisplayManager class."""
    
    @pytest.fixture
    def display_manager(self, mock_console):
        """Create a display manager instance."""
        return DisplayManager(mock_console)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        agent = Mock(spec=BaseAgent)
        agent.name = "test_agent"
        agent.agent_type = AgentType.CLAUDE_CODE
        agent.capabilities = [
            AgentCapability(
                name="test_capability",
                description="Test capability description",
                supported_formats=["text", "json"],
                cost_per_request=0.001
            )
        ]
        agent.performance_metrics = {
            "total_requests": 10,
            "total_cost_usd": 0.05,
            "uptime_seconds": 3600
        }
        return agent
    
    def test_manager_initialization(self, display_manager):
        """Test display manager initialization."""
        assert isinstance(display_manager.console, Console)
        assert isinstance(display_manager.theme, DisplayTheme)
    
    def test_get_agent_color(self, display_manager):
        """Test agent color mapping."""
        assert display_manager.get_agent_color("claude_code") == "bright_magenta"
        assert display_manager.get_agent_color("openai") == "bright_green"
        assert display_manager.get_agent_color("gemini") == "bright_blue"
        assert display_manager.get_agent_color("unknown") == "bright_white"
    
    def test_format_timestamp(self, display_manager):
        """Test timestamp formatting."""
        timestamp = 1642680000.0  # Known timestamp
        formatted = display_manager.format_timestamp(timestamp)
        assert ":" in formatted  # Should contain time separator
        assert len(formatted) == 8  # HH:MM:SS format
    
    def test_format_duration(self, display_manager):
        """Test duration formatting."""
        # Test milliseconds
        assert display_manager.format_duration(500) == "500ms"
        
        # Test seconds
        assert display_manager.format_duration(1500) == "1.5s"
        
        # Test minutes
        assert display_manager.format_duration(90000) == "1.5m"
    
    def test_create_agent_panel_basic(self, display_manager, mock_agent):
        """Test basic agent panel creation."""
        panel = display_manager.create_agent_panel(mock_agent, detailed=False)
        
        assert panel.title == "Agent: test_agent"
        # Panel should be created without errors
        assert panel is not None
    
    def test_create_agent_panel_detailed(self, display_manager, mock_agent):
        """Test detailed agent panel creation."""
        panel = display_manager.create_agent_panel(mock_agent, detailed=True)
        
        assert panel.title == "Agent: test_agent"
        # Should include capabilities in detailed view
        assert panel is not None
    
    def test_create_debate_panel(self, display_manager, mock_debate_result):
        """Test debate panel creation."""
        panel = display_manager.create_debate_panel(mock_debate_result, detailed=True)
        
        assert panel.title == "Debate Results"
        assert panel is not None
    
    def test_create_debate_panel_basic(self, display_manager, mock_debate_result):
        """Test basic debate panel creation."""
        panel = display_manager.create_debate_panel(mock_debate_result, detailed=False)
        
        assert panel.title == "Debate Results"
        assert panel is not None
    
    def test_create_voting_panel(self, display_manager, mock_vote_result):
        """Test voting panel creation."""
        panel = display_manager.create_voting_panel(mock_vote_result, detailed=True)
        
        assert panel.title == "Voting Results"
        assert panel is not None
    
    def test_create_voting_panel_basic(self, display_manager, mock_vote_result):
        """Test basic voting panel creation."""
        panel = display_manager.create_voting_panel(mock_vote_result, detailed=False)
        
        assert panel.title == "Voting Results"
        assert panel is not None
    
    def test_create_voting_panel_with_veto(self, display_manager):
        """Test voting panel with veto."""
        vote_result = VoteResult(
            vote_id="test-vote",
            task_id="test-task",
            success=False,
            final_decision=VoteType.VETO,
            confidence_score=0.9,
            votes=[
                Vote(
                    agent_name="claude",
                    agent_type="claude_code",
                    vote_type=VoteType.VETO,
                    confidence=0.9,
                    reasoning="Critical security flaw detected",
                    weight=1.5
                )
            ],
            total_weight=1.5,
            approve_weight=0.0,
            reject_weight=0.0,
            abstain_weight=0.0,
            has_veto=True,
            veto_reason="Critical security flaw detected",
            voting_method=VotingMethod.CLAUDE_VETO,
            duration_ms=1000.0
        )
        
        panel = display_manager.create_voting_panel(vote_result, detailed=True)
        
        assert panel.title == "Voting Results"
        assert panel is not None
    
    def test_create_task_progress_panel(self, display_manager):
        """Test task progress panel creation."""
        progress_info = {
            "type": "code_generation",
            "strategy": "single_agent",
            "status": "in_progress",
            "progress": 75,
            "agents": ["claude_code", "openai"]
        }
        
        panel = display_manager.create_task_progress_panel("test-task-123", progress_info)
        
        assert panel.title == "Task Progress"
        assert panel is not None
    
    def test_create_performance_panel(self, display_manager):
        """Test performance panel creation."""
        metrics = {
            "total_tasks": 100,
            "success_rate": 0.85,
            "average_cost_per_task": 0.025,
            "tasks_per_minute": 2.5,
            "agent_performance": {
                "claude_code": {
                    "requests": 50,
                    "success_rate": 0.9,
                    "avg_cost": 0.02
                },
                "openai": {
                    "requests": 30,
                    "success_rate": 0.8,
                    "avg_cost": 0.03
                }
            }
        }
        
        panel = display_manager.create_performance_panel(metrics)
        
        assert panel.title == "Performance Dashboard"
        assert panel is not None
    
    def test_create_performance_panel_minimal(self, display_manager):
        """Test performance panel with minimal metrics."""
        metrics = {
            "total_tasks": 5,
            "success_rate": 1.0,
            "average_cost_per_task": 0.01,
            "tasks_per_minute": 0.5
        }
        
        panel = display_manager.create_performance_panel(metrics)
        
        assert panel.title == "Performance Dashboard"
        assert panel is not None
    
    def test_create_error_panel(self, display_manager):
        """Test error panel creation."""
        panel = display_manager.create_error_panel("Test error message")
        
        assert panel.title == "⚠️ Error"
        assert panel is not None
    
    def test_create_error_panel_with_details(self, display_manager):
        """Test error panel with details."""
        details = {
            "error_code": "AGENT_ERROR",
            "agent_type": "claude_code",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        panel = display_manager.create_error_panel("Test error message", details)
        
        assert panel.title == "⚠️ Error"
        assert panel is not None
    
    def test_panel_rendering_no_exceptions(self, display_manager, mock_agent, mock_debate_result, mock_vote_result):
        """Test that all panels can be rendered without exceptions."""
        console = Console(file=StringIO(), force_terminal=False)
        display_manager.console = console
        
        # Test agent panel
        agent_panel = display_manager.create_agent_panel(mock_agent, detailed=True)
        console.print(agent_panel)
        
        # Test debate panel
        debate_panel = display_manager.create_debate_panel(mock_debate_result, detailed=True)
        console.print(debate_panel)
        
        # Test voting panel
        voting_panel = display_manager.create_voting_panel(mock_vote_result, detailed=True)
        console.print(voting_panel)
        
        # Test progress panel
        progress_panel = display_manager.create_task_progress_panel("test", {"type": "test"})
        console.print(progress_panel)
        
        # Test performance panel
        performance_panel = display_manager.create_performance_panel({"total_tasks": 1})
        console.print(performance_panel)
        
        # Test error panel
        error_panel = display_manager.create_error_panel("Test error")
        console.print(error_panel)
        
        # If we get here without exceptions, all panels rendered successfully
        assert True


class TestPanelContent:
    """Test the content and structure of generated panels."""
    
    @pytest.fixture
    def string_console(self):
        """Create a console that outputs to string for content testing."""
        return Console(file=StringIO(), force_terminal=False, width=120)
    
    def test_agent_panel_content(self, mock_console):
        """Test agent panel contains expected content."""
        display_manager = DisplayManager(mock_console)
        
        # Create mock agent
        mock_agent = Mock()
        mock_agent.name = "test_claude"
        mock_agent.agent_type = AgentType.CLAUDE_CODE
        mock_agent.capabilities = []
        mock_agent.performance_metrics = {
            "total_requests": 42,
            "total_cost_usd": 1.25,
            "uptime_seconds": 7200
        }
        
        panel = display_manager.create_agent_panel(mock_agent)
        
        # Panel should have correct title and style
        assert panel.title == "Agent: test_claude"
    
    def test_debate_panel_safety_blocks(self, mock_console):
        """Test debate panel with safety-blocked content."""
        display_manager = DisplayManager(mock_console)
        
        # Create debate result with safety issues
        from src.orchestration.debate import DebateResult
        import uuid
        
        debate_result = DebateResult(
            debate_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            success=False,
            final_consensus="Content blocked by safety filters",
            confidence_score=0.0,
            rounds=[],
            participating_agents=["claude", "openai"],
            total_duration_ms=500.0,
            total_cost_usd=0.001,
            total_tokens=10
        )
        
        panel = display_manager.create_debate_panel(debate_result)
        
        assert panel.title == "Debate Results"
        assert panel is not None
    
    def test_voting_panel_unanimous_decision(self, mock_console):
        """Test voting panel with unanimous decision."""
        display_manager = DisplayManager(mock_console)
        
        # Create unanimous vote result
        votes = [
            Vote(
                agent_name="claude",
                agent_type="claude_code",
                vote_type=VoteType.APPROVE,
                confidence=0.9,
                reasoning="Excellent proposal",
                weight=1.5
            ),
            Vote(
                agent_name="openai",
                agent_type="openai",
                vote_type=VoteType.APPROVE,
                confidence=0.8,
                reasoning="Well thought out",
                weight=1.0
            )
        ]
        
        vote_result = VoteResult(
            vote_id="unanimous-test",
            task_id="test-task",
            success=True,
            final_decision=VoteType.APPROVE,
            confidence_score=0.85,
            votes=votes,
            total_weight=2.5,
            approve_weight=2.5,
            reject_weight=0.0,
            abstain_weight=0.0,
            has_veto=False,
            voting_method=VotingMethod.WEIGHTED_MAJORITY,
            duration_ms=800.0
        )
        
        panel = display_manager.create_voting_panel(vote_result)
        
        assert panel.title == "Voting Results"
        assert panel is not None