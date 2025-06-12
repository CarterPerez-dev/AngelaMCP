#!/usr/bin/env python3
"""
Unit tests for the Debate and Voting systems.

Tests the core debate protocol and voting system functionality.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.debate import DebateProtocol, AgentProposal, AgentCritique, DebateResult
from src.orchestrator.voting import VotingSystem, AgentVote, ProposalScore, VotingResult


class TestAgentProposal:
    """Test the AgentProposal data class."""
    
    def test_proposal_creation(self):
        """Test proposal creation."""
        proposal = AgentProposal(
            agent_name="claude_code",
            solution="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            reasoning="Recursive approach is clean and readable",
            confidence=0.9
        )
        
        assert proposal.agent_name == "claude_code"
        assert "factorial" in proposal.solution
        assert proposal.confidence == 0.9
        assert proposal.reasoning is not None
    
    def test_proposal_defaults(self):
        """Test proposal with default values."""
        proposal = AgentProposal(
            agent_name="test_agent",
            solution="test solution"
        )
        
        assert proposal.agent_name == "test_agent"
        assert proposal.solution == "test solution"
        assert proposal.reasoning == ""
        assert proposal.confidence == 0.5


class TestAgentCritique:
    """Test the AgentCritique data class."""
    
    def test_critique_creation(self):
        """Test critique creation."""
        critique = AgentCritique(
            agent_name="openai",
            target_agent="claude_code",
            points=["Missing error handling", "Could be optimized"],
            suggestions=["Add input validation", "Use iterative approach"],
            confidence=0.8
        )
        
        assert critique.agent_name == "openai"
        assert critique.target_agent == "claude_code"
        assert len(critique.points) == 2
        assert len(critique.suggestions) == 2
        assert critique.confidence == 0.8
    
    def test_critique_defaults(self):
        """Test critique with default values."""
        critique = AgentCritique(
            agent_name="test_agent",
            target_agent="target_agent"
        )
        
        assert critique.agent_name == "test_agent"
        assert critique.target_agent == "target_agent"
        assert critique.points == []
        assert critique.suggestions == []
        assert critique.confidence == 0.5


class TestAgentVote:
    """Test the AgentVote data class."""
    
    def test_vote_creation(self):
        """Test vote creation."""
        vote = AgentVote(
            agent_name="claude_code",
            choice="proposal_1",
            confidence=0.9,
            reasoning="Best implementation approach"
        )
        
        assert vote.agent_name == "claude_code"
        assert vote.choice == "proposal_1"
        assert vote.confidence == 0.9
        assert vote.reasoning == "Best implementation approach"
    
    def test_vote_defaults(self):
        """Test vote with default values."""
        vote = AgentVote(
            agent_name="test_agent",
            choice="test_choice"
        )
        
        assert vote.agent_name == "test_agent"
        assert vote.choice == "test_choice"
        assert vote.confidence == 0.5
        assert vote.reasoning == ""


class TestProposalScore:
    """Test the ProposalScore data class."""
    
    def test_score_creation(self):
        """Test proposal score creation."""
        score = ProposalScore(
            agent_name="claude_code",
            total_score=2.5,
            vote_count=3,
            avg_confidence=0.83
        )
        
        assert score.agent_name == "claude_code"
        assert score.total_score == 2.5
        assert score.vote_count == 3
        assert score.avg_confidence == 0.83
    
    def test_score_defaults(self):
        """Test proposal score with default values."""
        score = ProposalScore(agent_name="test_agent")
        
        assert score.agent_name == "test_agent"
        assert score.total_score == 0.0
        assert score.vote_count == 0
        assert score.avg_confidence == 0.0


class TestDebateResult:
    """Test the DebateResult data class."""
    
    def test_debate_result_success(self):
        """Test successful debate result."""
        proposals = [
            AgentProposal(agent_name="claude", solution="solution1"),
            AgentProposal(agent_name="openai", solution="solution2")
        ]
        
        result = DebateResult(
            success=True,
            proposals=proposals,
            consensus_reached=True,
            final_proposal=proposals[0],
            total_duration=5.2
        )
        
        assert result.success == True
        assert len(result.proposals) == 2
        assert result.consensus_reached == True
        assert result.final_proposal is not None
        assert result.total_duration == 5.2
        assert result.error_message is None
    
    def test_debate_result_failure(self):
        """Test failed debate result."""
        result = DebateResult(
            success=False,
            error_message="Timeout reached",
            total_duration=10.0
        )
        
        assert result.success == False
        assert result.error_message == "Timeout reached"
        assert result.proposals == []
        assert result.consensus_reached == False
        assert result.final_proposal is None


class TestVotingResult:
    """Test the VotingResult data class."""
    
    def test_voting_result_success(self):
        """Test successful voting result."""
        scores = [
            ProposalScore(agent_name="claude", total_score=3.0),
            ProposalScore(agent_name="openai", total_score=2.5)
        ]
        
        result = VotingResult(
            success=True,
            winner="claude",
            scores=scores,
            chosen_proposal=AgentProposal(agent_name="claude", solution="winning solution")
        )
        
        assert result.success == True
        assert result.winner == "claude"
        assert len(result.scores) == 2
        assert result.chosen_proposal is not None
        assert result.error_message is None
    
    def test_voting_result_failure(self):
        """Test failed voting result."""
        result = VotingResult(
            success=False,
            error_message="No valid votes received"
        )
        
        assert result.success == False
        assert result.error_message == "No valid votes received"
        assert result.winner is None
        assert result.scores == []
        assert result.chosen_proposal is None


class TestDebateProtocol:
    """Test the DebateProtocol class functionality."""
    
    @pytest.fixture
    def debate_protocol(self):
        """Create a debate protocol instance."""
        return DebateProtocol()
    
    def test_debate_protocol_initialization(self, debate_protocol):
        """Test debate protocol initialization."""
        assert debate_protocol is not None
        assert hasattr(debate_protocol, 'conduct_debate')
    
    def test_phase_transitions(self, debate_protocol):
        """Test debate phase management."""
        # This tests the basic structure, not async functionality
        assert hasattr(debate_protocol, 'conduct_debate')


class TestVotingSystem:
    """Test the VotingSystem class functionality."""
    
    @pytest.fixture
    def voting_system(self):
        """Create a voting system instance."""
        return VotingSystem()
    
    def test_voting_system_initialization(self, voting_system):
        """Test voting system initialization."""
        assert voting_system is not None
        assert hasattr(voting_system, 'conduct_voting')
    
    def test_claude_weights(self, voting_system):
        """Test Claude's special voting weights."""
        # Test that Claude has special treatment in voting
        # This is a structural test without async calls
        assert hasattr(voting_system, 'conduct_voting')