"""
Voting System for AngelaMCP.

This module implements a weighted voting system where AI agents vote on proposals
from debates. Claude Code has senior developer voting weight and veto power since
it's the agent with actual file system access and execution capabilities.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.agents.base import BaseAgent, AgentResponse, TaskContext, AgentType
from src.orchestrator.debate import DebateResult, AgentProposal
from src.utils.logger import get_logger, AsyncPerformanceLogger

logger = get_logger("orchestrator.voting")


class VoteChoice(str, Enum):
    """Possible vote choices."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class AgentVote:
    """A vote from an agent."""
    agent_name: str
    agent_type: str
    proposal_target: str  # The agent whose proposal is being voted on
    choice: VoteChoice
    confidence: float = 0.5  # 0.0 to 1.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposalScore:
    """Scoring for a single proposal."""
    proposal: AgentProposal
    total_score: float = 0.0
    weighted_score: float = 0.0
    votes: List[AgentVote] = field(default_factory=list)
    approval_count: int = 0
    rejection_count: int = 0
    abstain_count: int = 0
    claude_approved: bool = False
    claude_vetoed: bool = False


@dataclass
class VotingResult:
    """Result of the voting process."""
    voting_id: str
    success: bool
    winner: Optional[str] = None
    winning_proposal: Optional[AgentProposal] = None
    proposal_scores: List[ProposalScore] = field(default_factory=list)
    total_duration: float = 0.0
    consensus_reached: bool = False
    claude_used_veto: bool = False
    voting_summary: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VotingSystem:
    """
    Manages weighted voting between AI agents with Claude Code as senior developer.
    
    Voting weights:
    - Claude Code: 2.0 (senior developer with file system access)
    - OpenAI: 1.0 (reviewer)
    - Gemini: 1.0 (researcher)
    
    Special rules:
    - Claude Code can veto any proposal (overrides all other votes)
    - Requires majority approval to win
    - In case of tie, Claude Code's preference wins
    """
    
    def __init__(self, claude_vote_weight: float = 2.0, enable_claude_veto: bool = True, voting_timeout: int = 120):
        """
        Initialize the voting system.
        
        Args:
            claude_vote_weight: Weight multiplier for Claude Code votes
            enable_claude_veto: Whether Claude Code can veto proposals
            voting_timeout: Timeout for voting phase in seconds
        """
        self.claude_vote_weight = claude_vote_weight
        self.enable_claude_veto = enable_claude_veto
        self.voting_timeout = voting_timeout
        self.logger = get_logger("voting")
        
        # Agent vote weights
        self.vote_weights = {
            AgentType.CLAUDE_CODE.value: claude_vote_weight,
            AgentType.OPENAI.value: 1.0,
            AgentType.GEMINI.value: 1.0
        }
    
    async def conduct_voting(
        self,
        debate_result: DebateResult,
        agents: List[BaseAgent],
        context: TaskContext
    ) -> VotingResult:
        """
        Conduct voting on proposals from a debate.
        
        Args:
            debate_result: Result from the debate phase
            agents: List of participating agents
            context: Task context for voting
            
        Returns:
            VotingResult with winner and vote breakdown
        """
        voting_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"🗳️ Starting voting {voting_id[:8]} on {len(debate_result.rounds[0].proposals)} proposals")
        
        try:
            async with AsyncPerformanceLogger(self.logger, "voting_full", task_id=voting_id):
                # Get proposals from debate result
                if not debate_result.rounds or not debate_result.rounds[0].proposals:
                    return VotingResult(
                        voting_id=voting_id,
                        success=False,
                        total_duration=time.time() - start_time,
                        error_message="No proposals to vote on"
                    )
                
                proposals = debate_result.rounds[0].proposals
                
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
                
                self.logger.info(f"🏆 Voting {voting_id[:8]} completed: Winner is {result.winner or 'None'}")
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Voting {voting_id[:8]} failed: {e}")
            return VotingResult(
                voting_id=voting_id,
                success=False,
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
            
            # Don't vote on your own proposal
            if voter_agent.name == proposal.agent_name:
                return []
            
            try:
                self.logger.info(f"[{voting_id[:8]}] 🗳️ {voter_agent.name} voting on {proposal.agent_name}'s proposal...")
                
                # Create voting prompt
                voting_prompt = f"""You are participating in a collaborative AI voting process. Please evaluate the following proposal and vote on it.

**Task Context:** {context.task_type.value if hasattr(context, 'task_type') else 'General Task'}

**Proposal from {proposal.agent_name}:**
{proposal.content}

Please provide your vote and reasoning in this exact format:

**VOTE:** [APPROVE/REJECT/ABSTAIN]
**CONFIDENCE:** [0.0-1.0]
**REASONING:** [Your detailed reasoning for this vote]

Voting guidelines:
- APPROVE: This proposal is technically sound, practical, and well-implemented
- REJECT: This proposal has significant issues that make it unsuitable
- ABSTAIN: You cannot adequately evaluate this proposal or it's outside your expertise

Consider:
- Technical correctness and feasibility
- Code quality and best practices (if applicable)
- Completeness and thoroughness
- Potential risks or issues
- Overall value and effectiveness

Be objective and constructive in your evaluation."""

                # Get vote response
                async with AsyncPerformanceLogger(
                    self.logger, f"vote_{voter_agent.name}_on_{proposal.agent_name}", task_id=voting_id
                ):
                    response = await asyncio.wait_for(
                        voter_agent.generate(voting_prompt, context),
                        timeout=self.voting_timeout
                    )
                
                if response.success:
                    # Parse vote from response
                    vote = self._parse_vote_response(
                        voter_agent, proposal.agent_name, response.content
                    )
                    if vote:
                        self.logger.info(f"[{voting_id[:8]}] ✅ {voter_agent.name} voted {vote.choice.value} on {proposal.agent_name}")
                        return [vote]
                    else:
                        self.logger.warning(f"[{voting_id[:8]}] ❓ Could not parse vote from {voter_agent.name}")
                else:
                    self.logger.warning(f"[{voting_id[:8]}] ❌ {voter_agent.name} voting failed: {response.error_message}")
                
                return []
                
            except asyncio.TimeoutError:
                self.logger.warning(f"[{voting_id[:8]}] ⏱️ {voter_agent.name} vote timed out")
                return []
            except Exception as e:
                self.logger.error(f"[{voting_id[:8]}] 💥 Voting error for {voter_agent.name}: {e}")
                return []
        
        # Collect all votes concurrently
        voting_tasks = []
        for voter_agent in agents:
            for proposal in proposals:
                voting_tasks.append(vote_on_proposal(voter_agent, proposal))
        
        # Execute all voting tasks
        vote_results = await asyncio.gather(*voting_tasks, return_exceptions=True)
        
        # Organize votes by proposal
        for i, vote_list in enumerate(vote_results):
            if isinstance(vote_list, list) and vote_list:
                vote = vote_list[0]
                
                # Find the corresponding proposal score
                for score in proposal_scores:
                    if score.proposal.agent_name == vote.proposal_target:
                        score.votes.append(vote)
                        
                        # Update vote counts
                        if vote.choice == VoteChoice.APPROVE:
                            score.approval_count += 1
                        elif vote.choice == VoteChoice.REJECT:
                            score.rejection_count += 1
                        else:
                            score.abstain_count += 1
                        
                        # Check for Claude Code special handling
                        if vote.agent_type == AgentType.CLAUDE_CODE.value:
                            if vote.choice == VoteChoice.APPROVE:
                                score.claude_approved = True
                            elif vote.choice == VoteChoice.REJECT and self.enable_claude_veto:
                                score.claude_vetoed = True
                                self.logger.info(f"[{voting_id[:8]}] 🚫 Claude Code VETOED {score.proposal.agent_name}'s proposal")
                        
                        break
        
        # Calculate weighted scores
        for score in proposal_scores:
            total_weighted_score = 0.0
            total_raw_score = 0.0
            
            for vote in score.votes:
                vote_value = 0.0
                if vote.choice == VoteChoice.APPROVE:
                    vote_value = 1.0 * vote.confidence
                elif vote.choice == VoteChoice.REJECT:
                    vote_value = -1.0 * vote.confidence
                # ABSTAIN = 0.0
                
                weight = self.vote_weights.get(vote.agent_type, 1.0)
                total_weighted_score += vote_value * weight
                total_raw_score += vote_value
            
            score.total_score = total_raw_score
            score.weighted_score = total_weighted_score
        
        self.logger.info(f"[{voting_id[:8]}] 📊 Collected votes: {sum(len(s.votes) for s in proposal_scores)} total votes")
        return proposal_scores
    
    def _parse_vote_response(self, voter_agent: BaseAgent, proposal_target: str, response_content: str) -> Optional[AgentVote]:
        """Parse vote information from agent response."""
        try:
            lines = response_content.strip().split('\n')
            
            vote_choice = None
            confidence = 0.5
            reasoning = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("**VOTE:**"):
                    vote_text = line.replace("**VOTE:**", "").strip().upper()
                    if "APPROVE" in vote_text:
                        vote_choice = VoteChoice.APPROVE
                    elif "REJECT" in vote_text:
                        vote_choice = VoteChoice.REJECT
                    elif "ABSTAIN" in vote_text:
                        vote_choice = VoteChoice.ABSTAIN
                
                elif line.startswith("**CONFIDENCE:**"):
                    conf_text = line.replace("**CONFIDENCE:**", "").strip()
                    try:
                        confidence = float(conf_text)
                        confidence = max(0.0, min(1.0, confidence))  
                    except ValueError:
                        confidence = 0.5
                
                elif line.startswith("**REASONING:**"):
                    reasoning = line.replace("**REASONING:**", "").strip()
            
            if not reasoning:
                reasoning_start = response_content.find("**REASONING:**")
                if reasoning_start != -1:
                    reasoning = response_content[reasoning_start + len("**REASONING:") :].strip()
            
            if vote_choice:
                return AgentVote(
                    agent_name=voter_agent.name,
                    agent_type=voter_agent.agent_type.value,
                    proposal_target=proposal_target,
                    choice=vote_choice,
                    confidence=confidence,
                    reasoning=reasoning or "No reasoning provided"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing vote response: {e}")
            return None
    
    def _determine_winner(self, proposal_scores: List[ProposalScore], voting_id: str) -> Optional[ProposalScore]:
        """Determine the winning proposal based on votes and weights."""
        
        if not proposal_scores:
            return None
        
        # First, eliminate any Claude-vetoed proposals
        valid_proposals = [
            score for score in proposal_scores
            if not score.claude_vetoed
        ]
        
        if not valid_proposals:
            self.logger.info(f"[{voting_id[:8]}] ❌ All proposals were vetoed by Claude Code")
            return None
        
        # Sort by weighted score (highest first)
        valid_proposals.sort(key=lambda x: x.weighted_score, reverse=True)
        
        # Check if we have a clear winner
        winner = valid_proposals[0]
        
        # Must have positive weighted score to win
        if winner.weighted_score <= 0:
            self.logger.info(f"[{voting_id[:8]}] ❌ No proposal received positive approval")
            return None
        
        self.logger.info(f"[{voting_id[:8]}] 🏆 Winner: {winner.proposal.agent_name} (score: {winner.weighted_score:.2f})")
        return winner
    
    def _check_consensus(self, proposal_scores: List[ProposalScore]) -> bool:
        """Check if there's strong consensus in the voting."""
        if not proposal_scores:
            return False
        
        # Sort by weighted score
        sorted_scores = sorted(proposal_scores, key=lambda x: x.weighted_score, reverse=True)
        
        if len(sorted_scores) < 2:
            return True
        
        # Check if winner has significantly higher score than second place
        winner_score = sorted_scores[0].weighted_score
        second_score = sorted_scores[1].weighted_score
        
        # Consider consensus if winner has at least 2x the score of second place
        return winner_score > 0 and (second_score <= 0 or winner_score >= 2 * second_score)
    
    def _create_voting_summary(self, proposal_scores: List[ProposalScore], winner: Optional[ProposalScore]) -> str:
        """Create a human-readable voting summary."""
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


class VotingError(Exception):
    """Exception raised during voting operations."""
    pass
