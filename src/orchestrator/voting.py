"""
Voting System for AngelaMCP multi-agent collaboration.

This implements weighted voting with Claude veto power, consensus detection,
and sophisticated vote aggregation. I'm creating a production-grade voting
system that ensures quality decision-making.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.agents import BaseAgent, AgentResponse, TaskContext, TaskType, AgentRole
from src.orchestrator.debate import AgentProposal
from src.utils import get_logger, VotingLogger, log_context, AsyncPerformanceLogger
from src.utils import OrchestrationError
from config import settings


class VoteType(str, Enum):
    """Types of votes agents can cast."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    VETO = "veto"  # Special vote type for Claude


class VotingMethod(str, Enum):
    """Different voting methods available."""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    UNANIMOUS = "unanimous"
    CLAUDE_VETO = "claude_veto"  # Default: weighted with Claude veto power


@dataclass
class AgentVote:
    """A single vote from an agent."""
    agent_name: str
    agent_type: str
    vote_type: VoteType
    confidence: float
    reasoning: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProposalScore:
    """Scoring for a single proposal."""
    proposal: AgentProposal
    votes: List[AgentVote] = field(default_factory=list)
    total_weight: float = 0.0
    weighted_score: float = 0.0
    approval_count: int = 0
    rejection_count: int = 0
    abstain_count: int = 0
    claude_vetoed: bool = False
    consensus_score: float = 0.0


@dataclass
class VotingResult:
    """Complete result of a voting session."""
    voting_id: str
    success: bool
    winner: Optional[str] = None
    winning_proposal: Optional[AgentProposal] = None
    proposal_scores: List[ProposalScore] = field(default_factory=list)
    voting_method: VotingMethod = VotingMethod.CLAUDE_VETO
    total_duration: float = 0.0
    consensus_reached: bool = False
    claude_used_veto: bool = False
    voting_summary: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VotingSystem:
    """
    Weighted voting system with Claude veto power.
    
    Manages vote collection, aggregation, and consensus determination.
    """
    
    def __init__(self):
        self.logger = get_logger("orchestrator.voting")
        self.voting_logger = VotingLogger()
        
        # Voting configuration from settings
        self.claude_vote_weight = settings.claude_vote_weight
        self.openai_vote_weight = settings.openai_vote_weight
        self.gemini_vote_weight = settings.gemini_vote_weight
        self.enable_claude_veto = settings.claude_veto_enabled
        self.voting_timeout = settings.voting_timeout
        
        # Agent weight mapping
        self.agent_weights = {
            "claude": self.claude_vote_weight,
            "openai": self.openai_vote_weight,
            "gemini": self.gemini_vote_weight
        }
    
    async def conduct_voting(
        self,
        proposals: List[AgentProposal],
        agents: List[BaseAgent],
        context: TaskContext,
        voting_method: VotingMethod = VotingMethod.CLAUDE_VETO,
        require_consensus: bool = False
    ) -> VotingResult:
        """
        Conduct complete voting session on proposals.
        
        Args:
            proposals: List of proposals to vote on
            agents: List of participating agents
            context: Task context
            voting_method: Method for vote aggregation
            require_consensus: Whether consensus is required
            
        Returns:
            VotingResult with complete voting information
        """
        
        voting_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            with log_context(request_id=voting_id):
                self.voting_logger.log_voting_start(voting_id, len(proposals))
                
                if not proposals:
                    raise OrchestrationError("No proposals to vote on")
                
                if not agents:
                    raise OrchestrationError("No agents available for voting")
                
                # Collect votes from all agents for all proposals
                proposal_scores = await self._collect_votes(proposals, agents, context, voting_id)
                
                # Determine winner
                winner_result = self._determine_winner(proposal_scores, voting_id)
                
                total_duration = time.time() - start_time
                
                # Create voting summary
                summary = self._create_voting_summary(proposal_scores, winner_result)
                
                result = VotingResult(
                    voting_id=voting_id,
                    success=winner_result is not None,
                    winner=winner_result.proposal.agent_name if winner_result else None,
                    winning_proposal=winner_result.proposal if winner_result else None,
                    proposal_scores=proposal_scores,
                    voting_method=voting_method,
                    total_duration=total_duration,
                    consensus_reached=self._check_consensus(proposal_scores),
                    claude_used_veto=any(score.claude_vetoed for score in proposal_scores),
                    voting_summary=summary,
                    metadata={
                        "total_proposals": len(proposals),
                        "claude_vote_weight": self.claude_vote_weight,
                        "veto_enabled": self.enable_claude_veto
                    }
                )
                
                self.voting_logger.log_voting_end(
                    voting_id, 
                    result.winner, 
                    result.consensus_reached
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Voting {voting_id[:8]} failed: {e}")
            return VotingResult(
                voting_id=voting_id,
                success=False,
                voting_method=voting_method,
                total_duration=time.time() - start_time,
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
    
    async def _collect_votes(
        self,
        proposals: List[AgentProposal],
        agents: List[BaseAgent],
        context: TaskContext,
        voting_id: str
    ) -> List[ProposalScore]:
        """Collect votes from all agents for all proposals."""
        
        # Initialize proposal scores
        proposal_scores = [
            ProposalScore(proposal=proposal)
            for proposal in proposals
        ]
        
        # Collect votes for each proposal from each agent
        async def vote_on_proposal(voter_agent: BaseAgent, proposal: AgentProposal) -> List[AgentVote]:
            """Get a single agent's vote on a single proposal."""
            
            # Don't vote on own proposal (automatic approval)
            if voter_agent.name == proposal.agent_name:
                return [AgentVote(
                    agent_name=voter_agent.name,
                    agent_type=voter_agent.agent_type.value,
                    vote_type=VoteType.APPROVE,
                    confidence=1.0,
                    reasoning="Author's proposal - automatic approval",
                    weight=self.agent_weights.get(voter_agent.name.lower(), 1.0)
                )]
            
            try:
                self.logger.info(f"[{voting_id[:8]}] 🗳️ {voter_agent.name} voting on {proposal.agent_name}'s proposal")
                
                # Create voting prompt
                voting_prompt = f"""You are evaluating a proposal in a collaborative decision-making process.

**Proposal by {proposal.agent_name}:**
{proposal.content}

**Your Task:** Vote on this proposal with detailed reasoning.

**Voting Options:**
- APPROVE: This proposal is good and should be implemented
- REJECT: This proposal has significant issues and should not be implemented
- ABSTAIN: You cannot make a clear decision or lack expertise in this area
{f"- VETO: (Claude only) This proposal has critical flaws that make it dangerous" if voter_agent.name.lower() == "claude" and self.enable_claude_veto else ""}

Please provide:
1. **Vote:** Your decision (APPROVE/REJECT/ABSTAIN{"/VETO" if voter_agent.name.lower() == "claude" and self.enable_claude_veto else ""})
2. **Confidence:** How confident you are (0.0-1.0)
3. **Reasoning:** Detailed explanation of your decision
4. **Key Factors:** What influenced your vote most

Be objective and constructive in your evaluation."""

                voting_context = context.model_copy()
                voting_context.task_type = TaskType.ANALYSIS
                voting_context.agent_role = AgentRole.REVIEWER
                
                async with AsyncPerformanceLogger(
                    self.logger, f"vote_{voter_agent.name}_{proposal.agent_name}", 
                    voting_id=voting_id
                ):
                    response = await asyncio.wait_for(
                        voter_agent.generate(voting_prompt, voting_context),
                        timeout=self.voting_timeout
                    )
                
                if response and response.content:
                    # Parse vote from response
                    vote = self._parse_vote_response(
                        response.content, 
                        voter_agent, 
                        proposal
                    )
                    
                    self.voting_logger.log_vote_cast(
                        voting_id, 
                        voter_agent.name, 
                        vote.vote_type.value, 
                        vote.confidence
                    )
                    
                    return [vote]
                else:
                    self.logger.warning(f"[{voting_id[:8]}] ❌ {voter_agent.name} vote failed")
                    return []
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[{voting_id[:8]}] ⏱️ {voter_agent.name} vote timed out")
                return []
            except Exception as e:
                self.logger.error(f"[{voting_id[:8]}] 💥 Vote error for {voter_agent.name}: {e}")
                return []
        
        # Collect all votes concurrently
        vote_tasks = []
        for proposal in proposals:
            for agent in agents:
                vote_tasks.append(vote_on_proposal(agent, proposal))
        
        vote_results = await asyncio.gather(*vote_tasks, return_exceptions=True)
        
        # Organize votes by proposal
        vote_index = 0
        for i, proposal in enumerate(proposals):
            for agent in agents:
                if vote_index < len(vote_results):
                    votes = vote_results[vote_index]
                    if isinstance(votes, list) and votes:
                        proposal_scores[i].votes.extend(votes)
                vote_index += 1
        
        # Calculate scores for each proposal
        for score in proposal_scores:
            self._calculate_proposal_score(score)
        
        return proposal_scores
    
    def _parse_vote_response(
        self, 
        response_content: str, 
        voter_agent: BaseAgent, 
        proposal: AgentProposal
    ) -> AgentVote:
        """Parse vote response from agent."""
        
        response_lower = response_content.lower()
        
        # Determine vote type
        vote_type = VoteType.ABSTAIN  # Default
        
        if "veto" in response_lower and voter_agent.name.lower() == "claude" and self.enable_claude_veto:
            vote_type = VoteType.VETO
        elif "approve" in response_lower or "support" in response_lower or "yes" in response_lower:
            vote_type = VoteType.APPROVE
        elif "reject" in response_lower or "oppose" in response_lower or "no" in response_lower:
            vote_type = VoteType.REJECT
        elif "abstain" in response_lower:
            vote_type = VoteType.ABSTAIN
        
        # Extract confidence (look for numbers between 0 and 1)
        confidence = 0.8  # Default
        import re
        
        # Look for confidence patterns
        confidence_patterns = [
            r"confidence[:\s]*([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*confidence",
            r"([0-9]*\.?[0-9]+)/10",
            r"([0-9]*\.?[0-9]+)%"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    val = float(match.group(1))
                    if val <= 1.0:
                        confidence = val
                    elif val <= 10.0:
                        confidence = val / 10.0
                    elif val <= 100.0:
                        confidence = val / 100.0
                    break
                except ValueError:
                    pass
        
        # Get weight for this agent
        weight = self.agent_weights.get(voter_agent.name.lower(), 1.0)
        
        return AgentVote(
            agent_name=voter_agent.name,
            agent_type=voter_agent.agent_type.value,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=response_content,
            weight=weight
        )
    
    def _calculate_proposal_score(self, proposal_score: ProposalScore) -> None:
        """Calculate weighted score for a proposal."""
        
        total_weight = 0.0
        weighted_score = 0.0
        approval_count = 0
        rejection_count = 0
        abstain_count = 0
        claude_vetoed = False
        
        for vote in proposal_score.votes:
            total_weight += vote.weight
            
            if vote.vote_type == VoteType.APPROVE:
                weighted_score += vote.weight * vote.confidence
                approval_count += 1
            elif vote.vote_type == VoteType.REJECT:
                weighted_score -= vote.weight * vote.confidence
                rejection_count += 1
            elif vote.vote_type == VoteType.ABSTAIN:
                abstain_count += 1
            elif vote.vote_type == VoteType.VETO:
                claude_vetoed = True
                weighted_score = -float('inf')  # Veto overrides everything
        
        # Normalize score by total weight
        if total_weight > 0 and not claude_vetoed:
            weighted_score = weighted_score / total_weight
        
        # Calculate consensus score (agreement level)
        if proposal_score.votes:
            dominant_vote_count = max(approval_count, rejection_count, abstain_count)
            consensus_score = dominant_vote_count / len(proposal_score.votes)
        else:
            consensus_score = 0.0
        
        # Update proposal score
        proposal_score.total_weight = total_weight
        proposal_score.weighted_score = weighted_score
        proposal_score.approval_count = approval_count
        proposal_score.rejection_count = rejection_count
        proposal_score.abstain_count = abstain_count
        proposal_score.claude_vetoed = claude_vetoed
        proposal_score.consensus_score = consensus_score
    
    def _determine_winner(
        self, 
        proposal_scores: List[ProposalScore], 
        voting_id: str
    ) -> Optional[ProposalScore]:
        """Determine the winning proposal."""
        
        if not proposal_scores:
            return None
        
        # Filter out vetoed proposals
        valid_proposals = [
            score for score in proposal_scores 
            if not score.claude_vetoed
        ]
        
        if not valid_proposals:
            self.logger.info(f"[{voting_id[:8]}] 🚫 All proposals vetoed by Claude")
            return None
        
        # Find proposal with highest weighted score
        winner = max(valid_proposals, key=lambda x: x.weighted_score)
        
        # Require positive score for approval
        if winner.weighted_score <= 0:
            self.logger.info(f"[{voting_id[:8]}] 🚫 No proposal received positive approval")
            return None
        
        self.logger.info(
            f"[{voting_id[:8]}] 🏆 Winner: {winner.proposal.agent_name} "
            f"(score: {winner.weighted_score:.2f})"
        )
        
        return winner
    
    def _check_consensus(self, proposal_scores: List[ProposalScore]) -> bool:
        """Check if consensus was reached."""
        
        if not proposal_scores:
            return False
        
        # Find highest scoring proposal
        best_score = max(proposal_scores, key=lambda x: x.weighted_score)
        
        # Check if consensus score is high enough
        consensus_threshold = 0.7  # 70% agreement
        
        return (
            best_score.weighted_score > 0 and 
            best_score.consensus_score >= consensus_threshold and
            not best_score.claude_vetoed
        )
    
    def _create_voting_summary(
        self, 
        proposal_scores: List[ProposalScore], 
        winner: Optional[ProposalScore]
    ) -> str:
        """Create human-readable voting summary."""
        
        if not proposal_scores:
            return "No proposals to vote on."
        
        summary_lines = ["**🗳️ Voting Results:**"]
        
        # Sort proposals by score for display
        sorted_scores = sorted(proposal_scores, key=lambda x: x.weighted_score, reverse=True)
        
        for i, score in enumerate(sorted_scores):
            status = "🏆 WINNER" if winner and score.proposal.agent_name == winner.proposal.agent_name else ""
            if score.claude_vetoed:
                status = "🚫 VETOED"
            
            summary_lines.append(
                f"{i+1}. **{score.proposal.agent_name}** "
                f"(Score: {score.weighted_score:.2f}, "
                f"Votes: ✅{score.approval_count} ❌{score.rejection_count} ➖{score.abstain_count}) "
                f"{status}"
            )
        
        if winner:
            summary_lines.append(f"\n**Final Decision:** {winner.proposal.agent_name}'s solution will be implemented")
        else:
            summary_lines.append(f"\n**Final Decision:** No solution reached consensus")
        
        return "\n".join(summary_lines)
    
    async def quick_approval_vote(
        self,
        proposal: AgentProposal,
        agents: List[BaseAgent],
        context: TaskContext
    ) -> bool:
        """Quick yes/no approval vote on a single proposal."""
        
        voting_result = await self.conduct_voting(
            [proposal], agents, context, 
            voting_method=VotingMethod.WEIGHTED_MAJORITY
        )
        
        return voting_result.success and voting_result.winner is not None
    
    def get_agent_weight(self, agent_name: str) -> float:
        """Get voting weight for an agent."""
        return self.agent_weights.get(agent_name.lower(), 1.0)
    
    def set_agent_weight(self, agent_name: str, weight: float) -> None:
        """Set voting weight for an agent."""
        self.agent_weights[agent_name.lower()] = weight
        self.logger.info(f"Updated {agent_name} vote weight to {weight}")


class VotingError(Exception):
    """Exception raised during voting operations."""
    pass
