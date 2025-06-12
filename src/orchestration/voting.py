"""
Voting system for AngelaMCP multi-agent consensus building.

This module implements weighted voting mechanisms that allow agents to evaluate
solutions, proposals, and reach consensus through democratic processes with
special provisions for Claude Code's veto power as specified in the roadmap.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, AgentResponse, TaskContext, AgentRole, AgentType, agent_registry
from src.utils.logger import get_logger, log_context, AsyncPerformanceLogger

logger = get_logger("orchestration.voting")


class VoteType(str, Enum):
    """Types of votes that can be cast."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    VETO = "veto"  # Special vote type for Claude Code


class VotingMethod(str, Enum):
    """Different voting methods available."""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    UNANIMOUS = "unanimous"
    CLAUDE_VETO = "claude_veto"  # Claude Code has veto power


@dataclass
class Vote:
    """Represents a single vote from an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    agent_type: str = ""
    vote_type: VoteType = VoteType.ABSTAIN
    confidence: float = 0.5
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoteResult:
    """Result of a voting session."""
    vote_id: str
    task_id: str
    success: bool
    final_decision: VoteType
    confidence_score: float
    votes: List[Vote] = field(default_factory=list)
    total_weight: float = 0.0
    approve_weight: float = 0.0
    reject_weight: float = 0.0
    abstain_weight: float = 0.0
    has_veto: bool = False
    veto_reason: str = ""
    voting_method: VotingMethod = VotingMethod.SIMPLE_MAJORITY
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VotingSystem:
    """
    Weighted voting system for multi-agent consensus building.
    
    I'm implementing a sophisticated voting mechanism that:
    1. Allows weighted votes based on agent expertise
    2. Supports different voting methods
    3. Provides Claude Code with veto power as specified
    4. Gathers detailed reasoning for each vote
    """
    
    def __init__(self):
        self.logger = get_logger("orchestration.voting_system")
        
        # Default agent weights for different voting contexts
        self._default_weights = {
            AgentType.CLAUDE_CODE: 1.5,  # Higher weight + veto power
            AgentType.OPENAI: 1.0,       # Standard weight
            AgentType.GEMINI: 1.0        # Standard weight
        }
        
        # Task-specific weight adjustments
        self._task_weight_modifiers = {
            "code_generation": {
                AgentType.CLAUDE_CODE: 1.3,
                AgentType.OPENAI: 0.9,
                AgentType.GEMINI: 0.8
            },
            "code_review": {
                AgentType.CLAUDE_CODE: 1.1,
                AgentType.OPENAI: 1.2,
                AgentType.GEMINI: 0.9
            },
            "research": {
                AgentType.CLAUDE_CODE: 0.8,
                AgentType.OPENAI: 1.0,
                AgentType.GEMINI: 1.3
            },
            "analysis": {
                AgentType.CLAUDE_CODE: 0.9,
                AgentType.OPENAI: 1.2,
                AgentType.GEMINI: 1.1
            }
        }
    
    def _calculate_agent_weight(self, agent: BaseAgent, context: str = "general") -> float:
        """Calculate voting weight for an agent based on context."""
        base_weight = self._default_weights.get(agent.agent_type, 1.0)
        
        # Apply task-specific modifiers
        modifiers = self._task_weight_modifiers.get(context, {})
        modifier = modifiers.get(agent.agent_type, 1.0)
        
        # Performance-based adjustment
        metrics = agent.performance_metrics
        if metrics["total_requests"] > 10:  # Only adjust if we have enough data
            success_rate = (metrics["total_requests"] - 
                          metrics.get("failed_requests", 0)) / metrics["total_requests"]
            performance_modifier = 0.8 + (success_rate * 0.4)  # Range: 0.8 to 1.2
        else:
            performance_modifier = 1.0
        
        final_weight = base_weight * modifier * performance_modifier
        
        self.logger.debug(
            f"Agent {agent.name} weight: {final_weight:.2f} "
            f"(base: {base_weight}, modifier: {modifier}, performance: {performance_modifier})"
        )
        
        return final_weight
    
    async def _collect_vote(self, agent: BaseAgent, proposal: str, context: TaskContext, 
                          weight: float) -> Vote:
        """Collect a vote from a single agent."""
        
        # Special prompt for Claude Code with veto power
        if agent.agent_type == AgentType.CLAUDE_CODE:
            vote_prompt = f"""You are participating in a voting session to evaluate this proposal:

PROPOSAL:
{proposal}

ORIGINAL TASK:
{context.context_data.get('original_task', 'Not specified')}

As Claude Code, you have SPECIAL VETO POWER in this voting system. You can:

1. **APPROVE** - You support this proposal
2. **REJECT** - You oppose this proposal  
3. **VETO** - You strongly oppose and want to prevent this proposal (overrides all other votes)
4. **ABSTAIN** - You have no strong opinion

Please provide your vote with detailed reasoning:

**VOTE**: [APPROVE/REJECT/VETO/ABSTAIN]

**CONFIDENCE**: [0.0 to 1.0]

**REASONING**: 
- Explain your evaluation of the proposal
- Identify strengths and weaknesses
- Consider implementation feasibility
- Assess potential risks or benefits
- If voting VETO, explain why this proposal should be blocked

**EVIDENCE**:
- List specific points that support your vote
- Reference concrete examples or standards
- Cite relevant best practices or concerns

Your vote carries extra weight ({weight:.2f}x) and your VETO can override majority approval."""

        else:
            vote_prompt = f"""You are participating in a voting session to evaluate this proposal:

PROPOSAL:
{proposal}

ORIGINAL TASK:
{context.context_data.get('original_task', 'Not specified')}

Please cast your vote with detailed reasoning:

**VOTE**: [APPROVE/REJECT/ABSTAIN]

**CONFIDENCE**: [0.0 to 1.0] - How confident are you in this assessment?

**REASONING**: 
- Evaluate the proposal's merit and feasibility
- Consider strengths, weaknesses, and trade-offs
- Assess alignment with the original task requirements
- Identify potential risks or implementation challenges

**EVIDENCE**:
- List specific points that support your vote
- Reference concrete examples or standards
- Provide technical or practical justifications

Your vote weight is {weight:.2f} in this decision process."""
        
        try:
            async with AsyncPerformanceLogger(
                self.logger, f"collect_vote", 
                agent=agent.name, task_id=context.task_id
            ):
                response = await agent.generate(vote_prompt, context)
            
            # Parse the vote response
            vote = self._parse_vote_response(agent, response, weight)
            
            self.logger.info(
                f"Collected vote from {agent.name}: {vote.vote_type.value} "
                f"(confidence: {vote.confidence:.2f}, weight: {vote.weight:.2f})"
            )
            
            return vote
            
        except Exception as e:
            self.logger.error(f"Failed to collect vote from {agent.name}: {e}")
            
            # Return abstain vote as fallback
            return Vote(
                agent_name=agent.name,
                agent_type=agent.agent_type.value,
                vote_type=VoteType.ABSTAIN,
                confidence=0.0,
                reasoning=f"Failed to collect vote: {e}",
                weight=weight,
                metadata={"error": str(e)}
            )
    
    def _parse_vote_response(self, agent: BaseAgent, response: AgentResponse, weight: float) -> Vote:
        """Parse agent response into a structured vote."""
        content = response.content.lower()
        
        # Extract vote type
        vote_type = VoteType.ABSTAIN  # Default
        if "veto" in content and agent.agent_type == AgentType.CLAUDE_CODE:
            vote_type = VoteType.VETO
        elif "approve" in content or "support" in content:
            vote_type = VoteType.APPROVE
        elif "reject" in content or "oppose" in content:
            vote_type = VoteType.REJECT
        elif "abstain" in content:
            vote_type = VoteType.ABSTAIN
        
        # Extract confidence (look for patterns like "confidence: 0.8" or "0.8/1.0")
        confidence = 0.5  # Default
        import re
        confidence_patterns = [
            r"confidence[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]*\.?[0-9]+)\s*/\s*1\.?0?",
            r"([0-9]*\.?[0-9]+)\s*confidence"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    confidence = float(match.group(1))
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    break
                except ValueError:
                    continue
        
        # Extract reasoning and evidence sections
        reasoning = ""
        evidence = []
        
        # Look for reasoning section
        reasoning_match = re.search(r"reasoning[:\s]+(.*?)(?:evidence|$)", content, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # Look for evidence section
        evidence_match = re.search(r"evidence[:\s]+(.*?)$", content, re.DOTALL | re.IGNORECASE)
        if evidence_match:
            evidence_text = evidence_match.group(1).strip()
            # Split by bullet points or dashes
            evidence = [
                item.strip() 
                for item in re.split(r'[-•*]\s*', evidence_text)
                if item.strip()
            ]
        
        # If we couldn't parse structured sections, use the full content as reasoning
        if not reasoning:
            reasoning = response.content
        
        return Vote(
            agent_name=agent.name,
            agent_type=agent.agent_type.value,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            weight=weight,
            metadata={
                "response_success": response.success,
                "response_tokens": response.tokens_used,
                "response_cost": response.cost_usd
            }
        )
    
    async def conduct_vote(self, proposal: str, agents: List[BaseAgent], 
                         method: VotingMethod = VotingMethod.CLAUDE_VETO,
                         context: Optional[TaskContext] = None,
                         task_context: str = "general") -> VoteResult:
        """Conduct a voting session among the specified agents."""
        
        vote_id = str(uuid.uuid4())
        start_time = time.time()
        
        if context is None:
            context = TaskContext(
                task_id=vote_id,
                task_type="voting"
            )
        
        # Add original proposal to context
        context.context_data["original_task"] = context.context_data.get("original_task", proposal)
        
        with log_context(vote_id, context.session_id):
            self.logger.info(
                f"Starting vote {vote_id} with {len(agents)} agents using {method.value}"
            )
            
            try:
                # Calculate weights for all agents
                agent_weights = {
                    agent: self._calculate_agent_weight(agent, task_context) 
                    for agent in agents
                }
                
                # Collect votes from all agents in parallel
                vote_tasks = [
                    self._collect_vote(agent, proposal, context, agent_weights[agent])
                    for agent in agents
                ]
                
                async with AsyncPerformanceLogger(
                    self.logger, "collect_all_votes", 
                    vote_id=vote_id, agent_count=len(agents)
                ):
                    votes = await asyncio.gather(*vote_tasks, return_exceptions=True)
                
                # Filter out exceptions and create valid votes list
                valid_votes = []
                for i, vote in enumerate(votes):
                    if isinstance(vote, Exception):
                        self.logger.error(f"Vote collection failed for {agents[i].name}: {vote}")
                        # Create abstain vote as fallback
                        fallback_vote = Vote(
                            agent_name=agents[i].name,
                            agent_type=agents[i].agent_type.value,
                            vote_type=VoteType.ABSTAIN,
                            confidence=0.0,
                            reasoning=f"Vote collection failed: {vote}",
                            weight=agent_weights[agents[i]],
                            metadata={"error": str(vote)}
                        )
                        valid_votes.append(fallback_vote)
                    else:
                        valid_votes.append(vote)
                
                # Calculate vote result
                result = self._calculate_vote_result(
                    vote_id, context.task_id, valid_votes, method, start_time
                )
                
                self.logger.info(
                    f"Vote {vote_id} completed - Decision: {result.final_decision.value} "
                    f"(confidence: {result.confidence_score:.2f}, "
                    f"approve: {result.approve_weight:.1f}, "
                    f"reject: {result.reject_weight:.1f}, "
                    f"veto: {result.has_veto})"
                )
                
                return result
                
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                self.logger.error(f"Vote conduct failed: {e}")
                
                return VoteResult(
                    vote_id=vote_id,
                    task_id=context.task_id,
                    success=False,
                    final_decision=VoteType.REJECT,
                    confidence_score=0.0,
                    voting_method=method,
                    duration_ms=duration,
                    metadata={"error": str(e)}
                )
    
    def _calculate_vote_result(self, vote_id: str, task_id: str, votes: List[Vote], 
                             method: VotingMethod, start_time: float) -> VoteResult:
        """Calculate the final result of a vote."""
        
        # Calculate weight totals
        total_weight = sum(vote.weight for vote in votes)
        approve_weight = sum(vote.weight for vote in votes if vote.vote_type == VoteType.APPROVE)
        reject_weight = sum(vote.weight for vote in votes if vote.vote_type == VoteType.REJECT)
        abstain_weight = sum(vote.weight for vote in votes if vote.vote_type == VoteType.ABSTAIN)
        
        # Check for veto
        veto_votes = [vote for vote in votes if vote.vote_type == VoteType.VETO]
        has_veto = len(veto_votes) > 0
        veto_reason = ""
        
        if has_veto:
            veto_reason = "; ".join([vote.reasoning for vote in veto_votes])
        
        # Determine final decision based on method
        final_decision = VoteType.REJECT  # Default
        confidence_score = 0.0
        
        if method == VotingMethod.CLAUDE_VETO:
            if has_veto:
                final_decision = VoteType.VETO
                confidence_score = max([vote.confidence for vote in veto_votes])
            elif approve_weight > reject_weight:
                final_decision = VoteType.APPROVE
                # Weight confidence by vote weights
                approve_votes = [vote for vote in votes if vote.vote_type == VoteType.APPROVE]
                if approve_votes:
                    weighted_confidence = sum(vote.confidence * vote.weight for vote in approve_votes)
                    confidence_score = weighted_confidence / approve_weight
            else:
                final_decision = VoteType.REJECT
                reject_votes = [vote for vote in votes if vote.vote_type == VoteType.REJECT]
                if reject_votes:
                    weighted_confidence = sum(vote.confidence * vote.weight for vote in reject_votes)
                    confidence_score = weighted_confidence / max(reject_weight, 0.1)
        
        elif method == VotingMethod.SIMPLE_MAJORITY:
            approve_count = len([vote for vote in votes if vote.vote_type == VoteType.APPROVE])
            reject_count = len([vote for vote in votes if vote.vote_type == VoteType.REJECT])
            
            if approve_count > reject_count:
                final_decision = VoteType.APPROVE
            else:
                final_decision = VoteType.REJECT
            
            # Simple average confidence
            relevant_votes = [vote for vote in votes if vote.vote_type == final_decision]
            if relevant_votes:
                confidence_score = sum(vote.confidence for vote in relevant_votes) / len(relevant_votes)
        
        elif method == VotingMethod.WEIGHTED_MAJORITY:
            if approve_weight > reject_weight:
                final_decision = VoteType.APPROVE
                confidence_score = approve_weight / max(total_weight - abstain_weight, 0.1)
            else:
                final_decision = VoteType.REJECT
                confidence_score = reject_weight / max(total_weight - abstain_weight, 0.1)
        
        elif method == VotingMethod.UNANIMOUS:
            # All non-abstaining votes must agree
            non_abstain_votes = [vote for vote in votes if vote.vote_type != VoteType.ABSTAIN]
            if non_abstain_votes:
                first_vote_type = non_abstain_votes[0].vote_type
                if all(vote.vote_type == first_vote_type for vote in non_abstain_votes):
                    final_decision = first_vote_type
                    confidence_score = sum(vote.confidence for vote in non_abstain_votes) / len(non_abstain_votes)
                else:
                    final_decision = VoteType.REJECT  # No consensus
                    confidence_score = 0.0
        
        duration = (time.time() - start_time) * 1000
        
        return VoteResult(
            vote_id=vote_id,
            task_id=task_id,
            success=final_decision in [VoteType.APPROVE],
            final_decision=final_decision,
            confidence_score=confidence_score,
            votes=votes,
            total_weight=total_weight,
            approve_weight=approve_weight,
            reject_weight=reject_weight,
            abstain_weight=abstain_weight,
            has_veto=has_veto,
            veto_reason=veto_reason,
            voting_method=method,
            duration_ms=duration,
            metadata={
                "total_votes": len(votes),
                "vote_breakdown": {
                    "approve": len([v for v in votes if v.vote_type == VoteType.APPROVE]),
                    "reject": len([v for v in votes if v.vote_type == VoteType.REJECT]),
                    "abstain": len([v for v in votes if v.vote_type == VoteType.ABSTAIN]),
                    "veto": len([v for v in votes if v.vote_type == VoteType.VETO])
                }
            }
        )


class ConsensusBuilder:
    """
    High-level consensus building system that combines voting with iteration.
    
    I'm implementing a system that can refine proposals based on feedback
    and seek consensus through multiple rounds of voting and revision.
    """
    
    def __init__(self, voting_system: VotingSystem):
        self.voting_system = voting_system
        self.logger = get_logger("orchestration.consensus_builder")
    
    async def build_consensus(self, initial_proposal: str, agents: List[BaseAgent],
                            max_iterations: int = 3,
                            consensus_threshold: float = 0.8,
                            context: Optional[TaskContext] = None) -> VoteResult:
        """Build consensus through iterative voting and proposal refinement."""
        
        current_proposal = initial_proposal
        iteration = 0
        
        self.logger.info(f"Starting consensus building with {len(agents)} agents")
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"Consensus iteration {iteration}")
            
            # Conduct vote on current proposal
            vote_result = await self.voting_system.conduct_vote(
                current_proposal, agents, VotingMethod.CLAUDE_VETO, context
            )
            
            # Check if we achieved consensus
            if vote_result.success and vote_result.confidence_score >= consensus_threshold:
                self.logger.info(f"Consensus achieved in iteration {iteration}")
                return vote_result
            
            # If vetoed, we need major changes
            if vote_result.has_veto:
                self.logger.info("Proposal vetoed, seeking major revision")
                break
            
            # If we have more iterations, refine the proposal based on feedback
            if iteration < max_iterations:
                self.logger.info("Refining proposal based on feedback")
                current_proposal = await self._refine_proposal(
                    current_proposal, vote_result, agents, context
                )
        
        # Return the final vote result
        self.logger.info(
            f"Consensus building completed after {iteration} iterations. "
            f"Final decision: {vote_result.final_decision.value}"
        )
        
        return vote_result
    
    async def _refine_proposal(self, original_proposal: str, vote_result: VoteResult,
                             agents: List[BaseAgent], context: Optional[TaskContext]) -> str:
        """Refine proposal based on voting feedback."""
        
        # Collect feedback from agents who rejected or had concerns
        feedback_items = []
        for vote in vote_result.votes:
            if vote.vote_type in [VoteType.REJECT, VoteType.VETO]:
                feedback_items.append(f"**{vote.agent_name}**: {vote.reasoning}")
        
        if not feedback_items:
            return original_proposal  # No specific feedback to incorporate
        
        feedback_text = "\n".join(feedback_items)
        
        # Use the best-performing agent to refine the proposal
        best_agent = max(agents, key=lambda a: a.performance_metrics.get("success_rate", 0.5))
        
        refinement_prompt = f"""Please refine this proposal based on the feedback received:

ORIGINAL PROPOSAL:
{original_proposal}

FEEDBACK FROM VOTING:
{feedback_text}

Please create an improved version that addresses the concerns raised while maintaining the core intent. The refined proposal should:

1. Address the specific issues mentioned in the feedback
2. Incorporate valid suggestions and improvements
3. Resolve conflicts and contradictions
4. Maintain feasibility and practicality
5. Keep the core objectives intact

Provide only the refined proposal without additional commentary."""

        try:
            if context:
                context.task_id = f"{context.task_id}_refinement"
            else:
                context = TaskContext(task_id="proposal_refinement")
            
            response = await best_agent.generate(refinement_prompt, context)
            
            if response.success:
                self.logger.info(f"Proposal refined by {best_agent.name}")
                return response.content
            else:
                self.logger.warning("Proposal refinement failed, using original")
                return original_proposal
                
        except Exception as e:
            self.logger.error(f"Error refining proposal: {e}")
            return original_proposal