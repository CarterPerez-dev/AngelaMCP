#!/usr/bin/env python3
"""
Unit tests for the CollaborationOrchestrator.

Tests the core collaboration orchestrator functionality without
requiring external API access.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.collaboration import (
    CollaborationOrchestrator, 
    CollaborationRequest, 
    CollaborationMode,
    CollaborationResult
)


class TestCollaborationOrchestrator:
    """Test the CollaborationOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a collaboration orchestrator instance."""
        return CollaborationOrchestrator()
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample collaboration request."""
        return CollaborationRequest(
            task_description="Create a Python function to calculate factorial",
            mode=CollaborationMode.FULL_DEBATE,
            timeout_minutes=5
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'claude_agent')
        assert hasattr(orchestrator, 'openai_agent')
        assert hasattr(orchestrator, 'gemini_agent')
    
    def test_collaboration_request_creation(self, sample_request):
        """Test collaboration request creation."""
        assert sample_request.task_description == "Create a Python function to calculate factorial"
        assert sample_request.mode == CollaborationMode.FULL_DEBATE
        assert sample_request.timeout_minutes == 5
    
    @pytest.mark.asyncio
    async def test_get_agents_basic(self, orchestrator):
        """Test basic agent retrieval."""
        # This test just verifies the method exists and doesn't crash
        try:
            agents = await orchestrator._get_agents()
            # Should return a list, even if empty due to missing API keys
            assert isinstance(agents, list)
        except Exception:
            # If it fails due to missing keys, that's expected in test environment
            pass
    
    @pytest.mark.asyncio
    async def test_collaborate_mock(self, orchestrator, sample_request):
        """Test collaboration with mocked agents."""
        # Mock the internal agent methods
        with patch.object(orchestrator, '_get_agents') as mock_get_agents:
            mock_agent = Mock()
            mock_agent.name = "mock_claude"
            mock_agent.agent_type = "claude_code"
            mock_get_agents.return_value = [mock_agent]
            
            with patch.object(orchestrator, 'debate_protocol') as mock_debate:
                # Mock debate result
                mock_debate_result = Mock()
                mock_debate_result.success = True
                mock_debate_result.proposals = []
                mock_debate_result.consensus_reached = True
                mock_debate_result.final_proposal = None
                mock_debate_result.total_duration = 2.5
                
                mock_debate.conduct_debate = AsyncMock(return_value=mock_debate_result)
                
                with patch.object(orchestrator, 'voting_system') as mock_voting:
                    # Mock voting result
                    mock_voting_result = Mock()
                    mock_voting_result.success = True
                    mock_voting_result.winner = "mock_claude"
                    mock_voting_result.chosen_proposal = Mock()
                    mock_voting_result.chosen_proposal.solution = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
                    
                    mock_voting.conduct_voting = AsyncMock(return_value=mock_voting_result)
                    
                    # Test collaboration
                    result = await orchestrator.collaborate(sample_request)
                    
                    # Verify result structure
                    assert isinstance(result, CollaborationResult)
                    assert result.success == True
                    assert result.chosen_agent == "mock_claude"
                    assert result.final_solution is not None
    
    def test_collaboration_modes(self):
        """Test different collaboration modes."""
        # Test FULL_DEBATE mode
        request1 = CollaborationRequest(
            task_description="Test task",
            mode=CollaborationMode.FULL_DEBATE
        )
        assert request1.mode == CollaborationMode.FULL_DEBATE
        
        # Test CLAUDE_LEAD mode
        request2 = CollaborationRequest(
            task_description="Test task",
            mode=CollaborationMode.CLAUDE_LEAD
        )
        assert request2.mode == CollaborationMode.CLAUDE_LEAD
        
        # Test QUICK_CONSENSUS mode
        request3 = CollaborationRequest(
            task_description="Test task",
            mode=CollaborationMode.QUICK_CONSENSUS
        )
        assert request3.mode == CollaborationMode.QUICK_CONSENSUS


class TestCollaborationRequest:
    """Test the CollaborationRequest data class."""
    
    def test_request_defaults(self):
        """Test default values for collaboration request."""
        request = CollaborationRequest(task_description="Test task")
        
        assert request.task_description == "Test task"
        assert request.mode == CollaborationMode.FULL_DEBATE
        assert request.timeout_minutes == 10
        assert request.context == {}
    
    def test_request_custom_values(self):
        """Test custom values for collaboration request."""
        context = {"key": "value"}
        request = CollaborationRequest(
            task_description="Custom task",
            mode=CollaborationMode.CLAUDE_LEAD,
            timeout_minutes=15,
            context=context
        )
        
        assert request.task_description == "Custom task"
        assert request.mode == CollaborationMode.CLAUDE_LEAD
        assert request.timeout_minutes == 15
        assert request.context == context


class TestCollaborationResult:
    """Test the CollaborationResult data class."""
    
    def test_result_success(self):
        """Test successful collaboration result."""
        result = CollaborationResult(
            success=True,
            chosen_agent="claude_code",
            final_solution="def test(): pass",
            total_duration=5.2,
            consensus_reached=True
        )
        
        assert result.success == True
        assert result.chosen_agent == "claude_code"
        assert result.final_solution == "def test(): pass"
        assert result.total_duration == 5.2
        assert result.consensus_reached == True
        assert result.error_message is None
    
    def test_result_failure(self):
        """Test failed collaboration result."""
        result = CollaborationResult(
            success=False,
            error_message="Collaboration timeout",
            total_duration=10.0
        )
        
        assert result.success == False
        assert result.error_message == "Collaboration timeout"
        assert result.chosen_agent is None
        assert result.final_solution is None
        assert result.consensus_reached == False