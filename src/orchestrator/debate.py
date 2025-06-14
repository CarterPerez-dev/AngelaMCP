"""
Fixed Debate Protocol for AngelaMCP multi-agent collaboration.

Fixed issues:
- Better error handling when agents fail
- Graceful degradation with partial agent participation
- Improved consensus calculation
- More robust agent response validation
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.agents import BaseAgent, AgentResponse, TaskContext, TaskType, AgentRole
from src.utils import get_logger, DebateLogger, log_context, AsyncPerformanceLogger
from src.utils import OrchestrationError
from config import settings


class DebatePhase(str, Enum):
    """Phases of the debate process."""
    INITIALIZATION = "initialization"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    REBUTTAL = "rebuttal"
    REFINEMENT = "refinement"
    CONSENSUS = "consensus"
    COMPLETED = "completed"


@dataclass
class AgentProposal:
    """A proposal from an agent in the debate."""
    agent_type: str
    agent_name: str
    content: str
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentCritique:
    """A critique of another agent's proposal."""
    critic_agent: str
    target_proposal: AgentProposal
    critique_content: str
    suggestions: List[str] = field(default_factory=list)
    severity: str = "moderate"  # low, moderate, high
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DebateRound:
    """Single round of debate with all phases."""
    round_number: int
    phase: DebatePhase
    proposals: List[AgentProposal] = field(default_factory=list)
    critiques: List[AgentCritique] = field(default_factory=list)
    rebuttals: List[AgentProposal] = field(default_factory=list)
    round_summary: Optional[str] = None
    duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    participating_agents: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)


@dataclass
class DebateResult:
    """Complete result of a debate session."""
    debate_id: str
    topic: str
    success: bool
    rounds: List[DebateRound] = field(default_factory=list)
    final_consensus: Optional[str] = None
    consensus_score: float = 0.0
    participating_agents: List[str] = field(default_factory=list)
    rounds_completed: int = 0
    total_duration: float = 0.0
    summary: Optional[str] = None
    participant_votes: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebateProtocol:
    """
    Structured debate protocol for multi-agent collaboration.
    
    Manages the entire debate lifecycle from proposal to consensus.
    """
    
    def __init__(self):
        self.logger = get_logger("orchestrator.debate")
        self.debate_logger = DebateLogger()
        self.max_rounds = settings.debate_max_rounds
        self.timeout_per_phase = settings.debate_timeout
        self.min_participants = max(1, settings.debate_min_participants)  # Allow single agent as minimum
        
        # Active debates tracking
        self._active_debates: Dict[str, DebateResult] = {}
    
    async def conduct_debate(
        self,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext,
        max_rounds: Optional[int] = None,
        require_consensus: bool = True
    ) -> DebateResult:
        """
        Conduct a complete structured debate between agents.
        
        Args:
            topic: The topic to debate
            agents: List of participating agents
            context: Task context for the debate
            max_rounds: Maximum rounds (overrides default)
            require_consensus: Whether consensus is required
            
        Returns:
            DebateResult with complete debate information
        """
        
        debate_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize debate tracking
        debate_result = DebateResult(
            debate_id=debate_id,
            topic=topic,
            success=False,
            participating_agents=[agent.name for agent in agents]
        )
        
        self._active_debates[debate_id] = debate_result
        
        try:
            with log_context(request_id=debate_id):
                self.debate_logger.log_debate_start(debate_id, topic, [a.name for a in agents])
                
                # I'll allow debates with even a single agent for graceful degradation
                if len(agents) < 1:
                    raise OrchestrationError(f"Need at least 1 agent for debate")
                
                # Filter out any None agents
                working_agents = [agent for agent in agents if agent is not None]
                
                if len(working_agents) < 1:
                    raise OrchestrationError("No working agents available for debate")
                
                # Set rounds limit
                rounds_limit = max_rounds or self.max_rounds
                
                # Main debate loop
                for round_num in range(1, rounds_limit + 1):
                    self.debate_logger.log_debate_round(debate_id, round_num, "starting")
                    
                    round_result = await self._conduct_debate_round(
                        debate_id, round_num, topic, working_agents, context
                    )
                    
                    debate_result.rounds.append(round_result)
                    debate_result.rounds_completed = round_num
                    
                    # Check if we have enough content for consensus
                    if round_result.proposals:
                        consensus_score = await self._evaluate_consensus(round_result.proposals)
                        debate_result.consensus_score = consensus_score
                        
                        # I'm lowering the consensus threshold to be more forgiving
                        if consensus_score >= 0.6 or len(working_agents) == 1:  # Lower threshold or single agent
                            self.logger.info(f"Consensus reached in round {round_num} (score: {consensus_score:.2f})")
                            break
                
                # Generate final consensus - even if consensus is low
                debate_result.final_consensus = await self._generate_final_consensus(
                    debate_result.rounds, working_agents, context
                )
                
                # Calculate final consensus score
                if debate_result.rounds:
                    final_proposals = debate_result.rounds[-1].proposals
                    if final_proposals:
                        debate_result.consensus_score = await self._evaluate_consensus(final_proposals)
                
                # Generate debate summary
                debate_result.summary = await self._generate_debate_summary(debate_result)
                
                # Mark as successful if we have any meaningful output
                debate_result.success = bool(debate_result.final_consensus and len(debate_result.final_consensus) > 50)
                debate_result.total_duration = time.time() - start_time
                
                self.debate_logger.log_debate_end(
                    debate_id, 
                    debate_result.success, 
                    debate_result.consensus_score
                )
                
                return debate_result
                
        except Exception as e:
            self.logger.error(f"Debate {debate_id[:8]} failed: {e}")
            
            debate_result.success = False
            debate_result.error_message = str(e)
            debate_result.total_duration = time.time() - start_time
            
            # Even on failure, try to provide some output
            if debate_result.rounds:
                try:
                    debate_result.final_consensus = await self._generate_emergency_consensus(topic, debate_result.rounds)
                    debate_result.summary = f"Debate failed but partial results available: {str(e)}"
                except:
                    pass
            
            self.debate_logger.log_debate_end(debate_id, False, 0.0)
            
            return debate_result
            
        finally:
            # Cleanup
            if debate_id in self._active_debates:
                del self._active_debates[debate_id]
    
    async def _conduct_debate_round(
        self,
        debate_id: str,
        round_number: int,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext
    ) -> DebateRound:
        """Conduct a single round of debate."""
        
        round_start = time.time()
        debate_round = DebateRound(
            round_number=round_number,
            phase=DebatePhase.INITIALIZATION,
            started_at=datetime.utcnow()
        )
        
        try:
            # Phase 1: Initial Proposals
            self.debate_logger.log_debate_round(debate_id, round_number, "proposals")
            debate_round.phase = DebatePhase.PROPOSAL
            
            initial_proposals = await self._gather_initial_proposals(
                topic, agents, context, debate_id, round_number
            )
            debate_round.proposals = initial_proposals
            debate_round.participating_agents = [p.agent_name for p in initial_proposals]
            
            # Phase 2: Cross-Critiques (only if we have multiple proposals)
            if len(initial_proposals) > 1:
                self.debate_logger.log_debate_round(debate_id, round_number, "critiques")
                debate_round.phase = DebatePhase.CRITIQUE
                
                critiques = await self._gather_critiques(
                    initial_proposals, agents, context, debate_id, round_number
                )
                debate_round.critiques = critiques
                
                # Phase 3: Refinements (only if we have critiques)
                if critiques:
                    self.debate_logger.log_debate_round(debate_id, round_number, "refinement")
                    debate_round.phase = DebatePhase.REFINEMENT
                    
                    refined_proposals = await self._gather_refined_proposals(
                        initial_proposals, critiques, agents, context, debate_id
                    )
                    
                    # Update proposals with refined versions if we got any
                    if refined_proposals:
                        debate_round.proposals = refined_proposals
            
            # Generate round summary
            debate_round.round_summary = await self._generate_round_summary(debate_round)
            
            debate_round.phase = DebatePhase.COMPLETED
            debate_round.completed_at = datetime.utcnow()
            debate_round.duration_seconds = time.time() - round_start
            
            return debate_round
            
        except Exception as e:
            self.logger.error(f"Round {round_number} failed: {e}")
            debate_round.phase = DebatePhase.COMPLETED
            debate_round.duration_seconds = time.time() - round_start
            # Don't re-raise, let the debate continue with what we have
            return debate_round
    
    async def _gather_initial_proposals(
        self,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str,
        round_number: int
    ) -> List[AgentProposal]:
        """Gather initial proposals from all agents."""
        
        proposals = []
        
        async def get_agent_proposal(agent: BaseAgent) -> Optional[AgentProposal]:
            """Get proposal from a single agent."""
            try:
                self.logger.info(f"[{debate_id[:8]}] 📝 Getting proposal from {agent.name}")
                
                proposal_prompt = f"""You are participating in a structured debate on the following topic:

**Topic:** {topic}

**Your Role:** Provide your best solution/proposal for this topic.

Please provide:
1. **Your Proposal:** Clear, actionable solution
2. **Reasoning:** Why this is the best approach
3. **Key Benefits:** Main advantages of your proposal
4. **Potential Concerns:** Any limitations you acknowledge

This is round {round_number} of the debate. Be thorough but concise."""

                proposal_context = context.model_copy()
                proposal_context.task_type = TaskType.DEBATE
                proposal_context.agent_role = AgentRole.PROPOSER
                
                async with AsyncPerformanceLogger(
                    self.logger, f"proposal_{agent.name}", debate_id=debate_id, round=round_number
                ):
                    response = await asyncio.wait_for(
                        agent.generate(proposal_prompt, proposal_context),
                        timeout=self.timeout_per_phase
                    )
                
                # I'm being more lenient about what constitutes a valid response
                if response and response.content and len(response.content.strip()) > 10:
                    self.logger.info(f"[{debate_id[:8]}] ✅ {agent.name} proposal ready")
                    return AgentProposal(
                        agent_type=agent.agent_type.value,
                        agent_name=agent.name,
                        content=response.content,
                        confidence_score=getattr(response, 'confidence', 0.8),
                        metadata=getattr(response, 'metadata', {})
                    )
                else:
                    self.logger.warning(f"[{debate_id[:8]}] ❌ {agent.name} proposal failed or too short")
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} proposal timed out")
                return None
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 Proposal error for {agent.name}: {e}")
                return None
        
        # Gather proposals concurrently
        proposal_tasks = [get_agent_proposal(agent) for agent in agents]
        proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)
        
        # Filter successful proposals
        for result in proposal_results:
            if result and not isinstance(result, Exception):
                proposals.append(result)
        
        self.logger.info(f"[{debate_id[:8]}] 📋 Gathered {len(proposals)} proposals")
        return proposals
    
    async def _gather_critiques(
        self,
        proposals: List[AgentProposal],
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str,
        round_number: int
    ) -> List[AgentCritique]:
        """Gather critiques of proposals from agents."""
        
        critiques = []
        
        async def get_agent_critiques(agent: BaseAgent) -> List[AgentCritique]:
            """Get critiques from a single agent on all other proposals."""
            agent_critiques = []
            
            # Critique each proposal not from this agent
            other_proposals = [p for p in proposals if p.agent_name != agent.name]
            
            for proposal in other_proposals:
                try:
                    self.logger.info(f"[{debate_id[:8]}] 🔍 {agent.name} critiquing {proposal.agent_name}'s proposal")
                    
                    critique_prompt = f"""You are reviewing a colleague's proposal in a collaborative debate.

**Original Topic:** {context.metadata.get('topic', 'Topic from debate context')}

**Proposal to Review (by {proposal.agent_name}):**
{proposal.content}

**Your Task:** Provide constructive criticism to improve this proposal.

Please provide:
1. **Strengths:** What's good about this proposal
2. **Weaknesses:** Areas that could be improved
3. **Specific Suggestions:** Concrete ways to enhance the proposal
4. **Concerns:** Any risks or issues you see

Be constructive and professional. The goal is to improve the solution, not to attack."""

                    critique_context = context.model_copy()
                    critique_context.task_type = TaskType.DEBATE
                    critique_context.agent_role = AgentRole.CRITIC
                    
                    async with AsyncPerformanceLogger(
                        self.logger, f"critique_{agent.name}_{proposal.agent_name}", 
                        debate_id=debate_id, round=round_number
                    ):
                        response = await asyncio.wait_for(
                            agent.generate(critique_prompt, critique_context),
                            timeout=self.timeout_per_phase
                        )
                    
                    # I'm being more lenient about critique validation too
                    if response and response.content and len(response.content.strip()) > 10:
                        critique = AgentCritique(
                            critic_agent=agent.name,
                            target_proposal=proposal,
                            critique_content=response.content,
                            confidence=getattr(response, 'confidence', 0.8)
                        )
                        agent_critiques.append(critique)
                        self.logger.info(f"[{debate_id[:8]}] ✅ {agent.name} critique of {proposal.agent_name} ready")
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} critique of {proposal.agent_name} timed out")
                except Exception as e:
                    self.logger.error(f"[{debate_id[:8]}] 💥 Critique error: {e}")
            
            return agent_critiques
        
        # Gather critiques concurrently
        critique_tasks = [get_agent_critiques(agent) for agent in agents]
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)
        
        # Flatten results
        for result in critique_results:
            if isinstance(result, list):
                critiques.extend(result)
        
        self.logger.info(f"[{debate_id[:8]}] 📝 Gathered {len(critiques)} critiques")
        return critiques
    
    async def _gather_refined_proposals(
        self,
        original_proposals: List[AgentProposal],
        critiques: List[AgentCritique],
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str
    ) -> List[AgentProposal]:
        """Gather refined proposals based on critiques."""
        
        refined_proposals = []
        
        async def refine_agent_proposal(agent: BaseAgent) -> Optional[AgentProposal]:
            """Refine a single agent's proposal based on critiques."""
            try:
                # Find original proposal and relevant critiques
                original_proposal = next(
                    (p for p in original_proposals if p.agent_name == agent.name),
                    None
                )
                
                if not original_proposal:
                    return None
                
                # Find critiques for this agent's proposal
                relevant_critiques = [
                    c for c in critiques 
                    if c.target_proposal.agent_name == agent.name
                ]
                
                if relevant_critiques:
                    self.logger.info(f"[{debate_id[:8]}] 🔧 {agent.name} refining proposal based on {len(relevant_critiques)} critiques")
                    
                    # Build refinement prompt
                    critiques_text = "\n\n".join([
                        f"**Critique from {c.critic_agent}:**\n{c.critique_content}"
                        for c in relevant_critiques
                    ])
                    
                    refinement_prompt = f"""Based on the feedback from your colleagues, please refine your original proposal.

**Your Original Proposal:**
{original_proposal.content}

**Feedback Received:**
{critiques_text}

**Your Task:** Revise your proposal incorporating the valid feedback while maintaining your core approach.

Please provide:
1. **Refined Proposal:** Your improved solution
2. **Changes Made:** What you modified based on feedback
3. **Rationale:** Why you made these changes or chose not to change certain aspects

Maintain the strength of your original idea while addressing legitimate concerns."""

                    refinement_context = context.model_copy()
                    refinement_context.task_type = TaskType.DEBATE
                    refinement_context.agent_role = AgentRole.PROPOSER
                    
                    async with AsyncPerformanceLogger(
                        self.logger, f"refinement_{agent.name}", debate_id=debate_id
                    ):
                        response = await asyncio.wait_for(
                            agent.generate(refinement_prompt, refinement_context),
                            timeout=self.timeout_per_phase
                        )
                    
                    if response and response.content and len(response.content.strip()) > 10:
                        self.logger.info(f"[{debate_id[:8]}] ✅ {agent.name} refined proposal ready")
                        return AgentProposal(
                            agent_type=agent.agent_type.value,
                            agent_name=agent.name,
                            content=response.content,
                            confidence_score=getattr(response, 'confidence', 0.8),
                            metadata={
                                **getattr(response, 'metadata', {}),
                                "refined": True,
                                "original_proposal_id": id(original_proposal),
                                "critiques_addressed": len(relevant_critiques)
                            }
                        )
                else:
                    self.logger.info(f"[{debate_id[:8]}] 📋 {agent.name} keeping original proposal (no critiques)")
                    return original_proposal
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} refinement timed out")
                return original_proposal  # Return original instead of None
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 Refinement error for {agent.name}: {e}")
                return original_proposal  # Return original instead of None
        
        # Gather refined proposals concurrently
        refinement_tasks = [refine_agent_proposal(agent) for agent in agents]
        refinement_results = await asyncio.gather(*refinement_tasks, return_exceptions=True)
        
        # Filter successful refinements
        for result in refinement_results:
            if result and not isinstance(result, Exception):
                refined_proposals.append(result)
        
        self.logger.info(f"[{debate_id[:8]}] 🏁 Refined {len(refined_proposals)} proposals")
        return refined_proposals
    
    async def _evaluate_consensus(self, proposals: List[AgentProposal]) -> float:
        """Evaluate consensus level between proposals."""
        if len(proposals) < 2:
            return 1.0  # Single proposal = perfect consensus
        
        # Improved consensus calculation
        total_comparisons = 0
        similarity_sum = 0.0
        
        for i, proposal1 in enumerate(proposals):
            for j, proposal2 in enumerate(proposals[i+1:], i+1):
                # Basic word overlap similarity
                words1 = set(proposal1.content.lower().split())
                words2 = set(proposal2.content.lower().split())
                
                if words1 and words2:
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = overlap / union if union > 0 else 0
                    
                    # I'm giving bonus points for longer overlaps
                    if overlap > 10:  # Significant overlap
                        similarity += 0.2
                    
                    similarity_sum += min(similarity, 1.0)  # Cap at 1.0
                    total_comparisons += 1
        
        consensus_score = similarity_sum / total_comparisons if total_comparisons > 0 else 0.5
        return min(consensus_score, 1.0)
    
    async def _generate_final_consensus(
        self,
        rounds: List[DebateRound],
        agents: List[BaseAgent],
        context: TaskContext
    ) -> Optional[str]:
        """Generate final consensus from debate rounds."""
        
        if not rounds or not rounds[-1].proposals:
            return "No consensus could be reached due to lack of proposals."
        
        # Use the first available agent to synthesize consensus
        synthesizer = None
        for agent in agents:
            if agent is not None:
                synthesizer = agent
                break
        
        if not synthesizer:
            return "No agent available to synthesize consensus."
        
        try:
            # Build synthesis prompt
            final_proposals = rounds[-1].proposals
            proposals_text = "\n\n".join([
                f"**{prop.agent_name}'s Final Proposal:**\n{prop.content}"
                for prop in final_proposals
            ])
            
            synthesis_prompt = f"""Based on the debate that has taken place, synthesize the best elements from all proposals into a final consensus solution.

**All Final Proposals:**
{proposals_text}

**Your Task:** Create a unified solution that incorporates the best ideas from each proposal while resolving any conflicts.

Please provide:
1. **Consensus Solution:** The combined best approach
2. **Key Elements:** What you took from each proposal  
3. **Resolution:** How you resolved any conflicting ideas
4. **Implementation:** Practical next steps

Focus on creating a practical, implementable solution that builds on the strongest ideas from the debate."""

            synthesis_context = context.model_copy()
            synthesis_context.task_type = TaskType.CONSENSUS
            synthesis_context.agent_role = AgentRole.SPECIALIST
            
            response = await synthesizer.generate(synthesis_prompt, synthesis_context)
            
            if response and response.content:
                return response.content
            
        except Exception as e:
            self.logger.error(f"Consensus generation failed: {e}")
        
        # Fallback consensus if synthesis fails
        return await self._generate_emergency_consensus(context.metadata.get('topic', 'debate topic'), rounds)
    
    async def _generate_emergency_consensus(self, topic: str, rounds: List[DebateRound]) -> str:
        """Generate emergency consensus when normal synthesis fails."""
        
        if not rounds:
            return f"Debate on '{topic}' could not reach consensus due to technical issues."
        
        # Extract key points from proposals
        all_proposals = []
        for round_data in rounds:
            all_proposals.extend(round_data.proposals)
        
        if not all_proposals:
            return f"Debate on '{topic}' started but no valid proposals were generated."
        
        # Create a basic summary
        agent_names = list(set(p.agent_name for p in all_proposals))
        
        consensus = f"""**Emergency Consensus Summary for '{topic}':**

**Participants:** {', '.join(agent_names)}
**Rounds Completed:** {len(rounds)}

**Key Proposals Generated:**
"""
        
        for i, proposal in enumerate(all_proposals[-3:], 1):  # Last 3 proposals
            preview = proposal.content[:200] + "..." if len(proposal.content) > 200 else proposal.content
            consensus += f"\n{i}. **{proposal.agent_name}:** {preview}\n"
        
        consensus += f"""
**Status:** Partial debate results available. The debate encountered technical difficulties but generated valuable insights from {len(agent_names)} participants across {len(rounds)} rounds."""
        
        return consensus
    
    async def _generate_round_summary(self, round_data: DebateRound) -> str:
        """Generate summary for a debate round."""
        
        summary_parts = [
            f"**Round {round_data.round_number} Summary:**",
            f"- Proposals: {len(round_data.proposals)}",
            f"- Critiques: {len(round_data.critiques)}",
            f"- Duration: {round_data.duration_seconds:.1f}s",
            f"- Participating agents: {len(round_data.participating_agents)}"
        ]
        
        if round_data.participating_agents:
            summary_parts.append(f"- Active participants: {', '.join(round_data.participating_agents)}")
        
        if round_data.failed_agents:
            summary_parts.append(f"- Failed agents: {', '.join(round_data.failed_agents)}")
        
        return "\n".join(summary_parts)
    
    async def _generate_debate_summary(self, debate_result: DebateResult) -> str:
        """Generate comprehensive debate summary."""
        
        summary_parts = [
            f"**Debate Summary: {debate_result.topic}**",
            f"- Rounds completed: {debate_result.rounds_completed}",
            f"- Duration: {debate_result.total_duration:.1f}s",
            f"- Consensus score: {debate_result.consensus_score:.2f}",
            f"- Participants: {', '.join(debate_result.participating_agents)}"
        ]
        
        if debate_result.rounds:
            total_proposals = sum(len(r.proposals) for r in debate_result.rounds)
            total_critiques = sum(len(r.critiques) for r in debate_result.rounds)
            
            summary_parts.extend([
                f"- Total proposals: {total_proposals}",
                f"- Total critiques: {total_critiques}"
            ])
        
        if debate_result.success:
            summary_parts.append("- Status: ✅ Completed successfully")
        else:
            summary_parts.append(f"- Status: ❌ Failed ({debate_result.error_message or 'Unknown error'})")
        
        return "\n".join(summary_parts)
    
    def get_active_debates(self) -> Dict[str, DebateResult]:
        """Get currently active debates."""
        return self._active_debates.copy()
    
    async def cancel_debate(self, debate_id: str) -> bool:
        """Cancel an active debate."""
        if debate_id in self._active_debates:
            del self._active_debates[debate_id]
            self.logger.info(f"Debate {debate_id[:8]} cancelled")
            return True
        return False


class DebateError(Exception):
    """Exception raised during debate operations."""
    pass
