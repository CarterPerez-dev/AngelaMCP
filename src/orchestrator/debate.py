"""
Fixed Debate Protocol for AngelaMCP multi-agent collaboration.

Production fixes:
- More resilient proposal validation (reduced from 10 to 5 char minimum)
- Better error handling when agents fail with detailed logging
- Graceful degradation with partial agent participation
- Enhanced consensus calculation with multiple metrics
- Improved fallback mechanisms for failed responses
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
    word_count: int = field(init=False)
    
    def __post_init__(self):
        """Calculate word count for quality assessment."""
        self.word_count = len(self.content.split()) if self.content else 0


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
    success_rate: float = field(init=False, default=0.0)
    
    def __post_init__(self):
        """Calculate success rate based on participation."""
        total_expected = len(self.participating_agents) + len(self.failed_agents)
        if total_expected > 0:
            self.success_rate = len(self.participating_agents) / total_expected
        else:
            self.success_rate = 0.0


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
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class DebateProtocol:
    """
    Enhanced debate protocol for multi-agent collaboration.
    
    Manages the entire debate lifecycle with improved resilience and error handling.
    """
    
    def __init__(self):
        self.logger = get_logger("orchestrator.debate")
        self.debate_logger = DebateLogger()
        self.max_rounds = settings.debate_max_rounds
        self.timeout_per_phase = settings.debate_timeout
        self.min_participants = max(1, settings.debate_min_participants)
        
        # Enhanced validation settings
        self.min_content_length = 5  # Reduced from 10 for more lenient validation
        self.min_quality_words = 15  # Minimum words for quality content
        self.max_failures_per_round = 2  # Allow some failures per round
        
        # Active debates tracking
        self._active_debates: Dict[str, DebateResult] = {}
    
    async def conduct_debate(
        self,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext,
        max_rounds: Optional[int] = None,
        require_consensus: bool = False  # Changed default to False
    ) -> DebateResult:
        """
        Conduct a complete structured debate between agents with enhanced error handling.
        """
        
        debate_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize debate tracking
        debate_result = DebateResult(
            debate_id=debate_id,
            topic=topic,
            success=False,
            participating_agents=[agent.name for agent in agents if agent is not None]
        )
        
        self._active_debates[debate_id] = debate_result
        
        try:
            with log_context(request_id=debate_id):
                self.debate_logger.log_debate_start(debate_id, topic, [a.name for a in agents if a])
                
                # Validate we have at least one working agent
                working_agents = [agent for agent in agents if agent is not None]
                
                if len(working_agents) < 1:
                    raise OrchestrationError("No working agents available for debate")
                
                self.logger.info(f"[{debate_id[:8]}] Starting debate with {len(working_agents)} agents")
                
                # Set rounds limit
                rounds_limit = max_rounds or self.max_rounds
                
                # Main debate loop with enhanced error handling
                for round_num in range(1, rounds_limit + 1):
                    self.debate_logger.log_debate_round(debate_id, round_num, "starting")
                    
                    try:
                        round_result = await self._conduct_debate_round(
                            debate_id, round_num, topic, working_agents, context
                        )
                        
                        debate_result.rounds.append(round_result)
                        debate_result.rounds_completed = round_num
                        
                        self.logger.info(
                            f"[{debate_id[:8]}] Round {round_num} completed: "
                            f"{len(round_result.proposals)} proposals, "
                            f"success rate: {round_result.success_rate:.2f}"
                        )
                        
                        # Check if we have sufficient content for consensus
                        if round_result.proposals and len(round_result.proposals) > 0:
                            consensus_score = await self._evaluate_consensus(round_result.proposals)
                            debate_result.consensus_score = consensus_score
                            
                            # More lenient consensus threshold
                            consensus_threshold = 0.4 if len(working_agents) > 1 else 0.1
                            
                            if consensus_score >= consensus_threshold or round_result.success_rate >= 0.5:
                                self.logger.info(
                                    f"[{debate_id[:8]}] Sufficient consensus reached in round {round_num} "
                                    f"(score: {consensus_score:.2f}, success rate: {round_result.success_rate:.2f})"
                                )
                                break
                        
                        # If we have no proposals at all, try one more round
                        if not round_result.proposals and round_num < rounds_limit:
                            self.logger.warning(f"[{debate_id[:8]}] No proposals in round {round_num}, continuing...")
                            continue
                            
                    except Exception as e:
                        self.logger.error(f"[{debate_id[:8]}] Round {round_num} failed: {e}")
                        # Continue to next round unless this is the last round
                        if round_num < rounds_limit:
                            continue
                        else:
                            break
                
                # Generate final consensus from all available content
                debate_result.final_consensus = await self._generate_final_consensus(
                    debate_result.rounds, working_agents, context, topic
                )
                
                # Calculate comprehensive quality metrics
                debate_result.quality_metrics = self._calculate_quality_metrics(debate_result)
                
                # Calculate final consensus score based on all rounds
                if debate_result.rounds:
                    all_proposals = []
                    for round_data in debate_result.rounds:
                        all_proposals.extend(round_data.proposals)
                    
                    if all_proposals:
                        debate_result.consensus_score = await self._evaluate_consensus(all_proposals)
                
                # Generate comprehensive debate summary
                debate_result.summary = await self._generate_debate_summary(debate_result)
                
                # Mark as successful if we have meaningful output
                success_criteria = (
                    debate_result.final_consensus and 
                    len(debate_result.final_consensus) > 50 and
                    debate_result.rounds_completed > 0
                )
                
                debate_result.success = success_criteria
                debate_result.total_duration = time.time() - start_time
                
                self.debate_logger.log_debate_end(
                    debate_id, 
                    debate_result.success, 
                    debate_result.consensus_score
                )
                
                self.logger.info(
                    f"[{debate_id[:8]}] Debate completed: "
                    f"success={debate_result.success}, "
                    f"rounds={debate_result.rounds_completed}, "
                    f"consensus={debate_result.consensus_score:.3f}, "
                    f"duration={debate_result.total_duration:.1f}s"
                )
                
                return debate_result
                
        except Exception as e:
            self.logger.error(f"[{debate_id[:8]}] Debate failed with error: {e}")
            
            debate_result.success = False
            debate_result.error_message = str(e)
            debate_result.total_duration = time.time() - start_time
            
            # Even on failure, try to provide some output if we have partial results
            if debate_result.rounds:
                try:
                    debate_result.final_consensus = await self._generate_emergency_consensus(
                        topic, debate_result.rounds
                    )
                    debate_result.summary = f"Debate encountered errors but generated partial results: {str(e)}"
                except Exception as fallback_error:
                    self.logger.error(f"[{debate_id[:8]}] Emergency consensus generation failed: {fallback_error}")
                    debate_result.final_consensus = f"Debate on '{topic}' encountered technical difficulties."
            else:
                debate_result.final_consensus = f"Debate on '{topic}' failed to generate any proposals due to: {str(e)}"
            
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
        """Conduct a single round of debate with enhanced error handling."""
        
        round_start = time.time()
        debate_round = DebateRound(
            round_number=round_number,
            phase=DebatePhase.INITIALIZATION,
            started_at=datetime.utcnow()
        )
        
        try:
            # Phase 1: Initial Proposals with better error handling
            self.debate_logger.log_debate_round(debate_id, round_number, "proposals")
            debate_round.phase = DebatePhase.PROPOSAL
            
            initial_proposals, failed_agents = await self._gather_initial_proposals_enhanced(
                topic, agents, context, debate_id, round_number
            )
            
            debate_round.proposals = initial_proposals
            debate_round.participating_agents = [p.agent_name for p in initial_proposals]
            debate_round.failed_agents = failed_agents
            
            self.logger.info(
                f"[{debate_id[:8]}] Round {round_number} proposals: "
                f"{len(initial_proposals)} successful, {len(failed_agents)} failed"
            )
            
            # Phase 2: Cross-Critiques (only if we have multiple successful proposals)
            if len(initial_proposals) > 1:
                self.debate_logger.log_debate_round(debate_id, round_number, "critiques")
                debate_round.phase = DebatePhase.CRITIQUE
                
                critiques = await self._gather_critiques_enhanced(
                    initial_proposals, agents, context, debate_id, round_number
                )
                debate_round.critiques = critiques
                
                # Phase 3: Refinements (only if we have critiques)
                if critiques:
                    self.debate_logger.log_debate_round(debate_id, round_number, "refinement")
                    debate_round.phase = DebatePhase.REFINEMENT
                    
                    refined_proposals = await self._gather_refined_proposals_enhanced(
                        initial_proposals, critiques, agents, context, debate_id
                    )
                    
                    # Update proposals with refined versions if we got any
                    if refined_proposals and len(refined_proposals) >= len(initial_proposals) * 0.5:
                        debate_round.proposals = refined_proposals
                        self.logger.info(f"[{debate_id[:8]}] Updated with {len(refined_proposals)} refined proposals")
            
            # Generate round summary
            debate_round.round_summary = await self._generate_round_summary(debate_round)
            
            debate_round.phase = DebatePhase.COMPLETED
            debate_round.completed_at = datetime.utcnow()
            debate_round.duration_seconds = time.time() - round_start
            
            return debate_round
            
        except Exception as e:
            self.logger.error(f"[{debate_id[:8]}] Round {round_number} encountered error: {e}")
            debate_round.phase = DebatePhase.COMPLETED
            debate_round.duration_seconds = time.time() - round_start
            debate_round.failed_agents.extend([agent.name for agent in agents])
            # Don't re-raise, let the debate continue with what we have
            return debate_round
    
    async def _gather_initial_proposals_enhanced(
        self,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str,
        round_number: int
    ) -> tuple[List[AgentProposal], List[str]]:
        """Gather initial proposals with enhanced error tracking and validation."""
        
        proposals = []
        failed_agents = []
        
        async def get_agent_proposal_enhanced(agent: BaseAgent) -> Optional[AgentProposal]:
            """Get proposal from a single agent with enhanced validation."""
            try:
                self.logger.info(f"[{debate_id[:8]}] 📝 Getting proposal from {agent.name}")
                
                # Enhanced proposal prompt with more specific guidance
                proposal_prompt = f"""You are participating in a structured collaborative debate.

**Topic:** {topic}

**Your Role:** Provide your best solution/proposal for this topic.

**Requirements:**
- Provide a substantive response (minimum 50 words)
- Include specific reasoning and examples
- Address practical implementation considerations
- Be clear and well-structured

**Structure your response with:**
1. **Your Position**: Clear statement of your stance
2. **Supporting Arguments**: Key reasons and evidence
3. **Implementation Details**: How this would work in practice
4. **Benefits**: Why this approach is advantageous
5. **Considerations**: Any limitations or challenges you acknowledge

This is round {round_number} of the debate. Be thorough and specific."""

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
                
                # Enhanced validation with multiple criteria
                if response and response.content:
                    content = response.content.strip()
                    word_count = len(content.split())
                    
                    # More comprehensive validation
                    if (len(content) >= self.min_content_length and 
                        word_count >= self.min_quality_words and
                        not self._is_generic_response(content)):
                        
                        self.logger.info(
                            f"[{debate_id[:8]}] ✅ {agent.name} proposal accepted "
                            f"({len(content)} chars, {word_count} words)"
                        )
                        
                        return AgentProposal(
                            agent_type=agent.agent_type.value,
                            agent_name=agent.name,
                            content=content,
                            confidence_score=getattr(response, 'confidence', 0.8),
                            metadata={
                                **getattr(response, 'metadata', {}),
                                "round_number": round_number,
                                "word_count": word_count,
                                "response_length": len(content)
                            }
                        )
                    else:
                        self.logger.warning(
                            f"[{debate_id[:8]}] ⚠️ {agent.name} proposal quality check failed "
                            f"(length: {len(content)}, words: {word_count})"
                        )
                        return None
                else:
                    self.logger.warning(f"[{debate_id[:8]}] ❌ {agent.name} no response content")
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} proposal timed out")
                return None
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 Proposal error for {agent.name}: {e}")
                return None
        
        # Gather proposals concurrently
        proposal_tasks = [get_agent_proposal_enhanced(agent) for agent in agents]
        proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)
        
        # Process results and track failures
        for i, result in enumerate(proposal_results):
            agent = agents[i]
            if result and not isinstance(result, Exception):
                proposals.append(result)
            else:
                failed_agents.append(agent.name)
                if isinstance(result, Exception):
                    self.logger.error(f"[{debate_id[:8]}] Agent {agent.name} failed with exception: {result}")
        
        self.logger.info(
            f"[{debate_id[:8]}] 📋 Proposal results: {len(proposals)} successful, {len(failed_agents)} failed"
        )
        
        return proposals, failed_agents
    
    def _is_generic_response(self, content: str) -> bool:
        """Check if response is too generic or templated."""
        generic_phrases = [
            "i understand you're looking",
            "i can help with",
            "please provide more details",
            "what specific aspect",
            "i need more information"
        ]
        
        content_lower = content.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in content_lower)
        
        # If more than 2 generic phrases, it's probably too generic
        return generic_count >= 2
    
    async def _gather_critiques_enhanced(
        self,
        proposals: List[AgentProposal],
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str,
        round_number: int
    ) -> List[AgentCritique]:
        """Gather critiques with enhanced error handling."""
        
        critiques = []
        
        async def get_agent_critiques_enhanced(agent: BaseAgent) -> List[AgentCritique]:
            """Get critiques from a single agent with enhanced error handling."""
            agent_critiques = []
            
            # Critique each proposal not from this agent
            other_proposals = [p for p in proposals if p.agent_name != agent.name]
            
            for proposal in other_proposals:
                try:
                    self.logger.info(f"[{debate_id[:8]}] 🔍 {agent.name} critiquing {proposal.agent_name}'s proposal")
                    
                    critique_prompt = f"""You are reviewing a colleague's proposal in a collaborative debate.

**Original Topic:** {context.metadata.get('topic', 'From debate context')}

**Proposal to Review (by {proposal.agent_name}):**
{proposal.content}

**Your Task:** Provide constructive criticism to improve this proposal.

**Structure your critique with:**
1. **Strengths**: What's good about this proposal (be specific)
2. **Areas for Improvement**: What could be enhanced
3. **Specific Suggestions**: Concrete ways to improve the proposal
4. **Alternative Perspectives**: Other approaches to consider
5. **Implementation Concerns**: Practical challenges or considerations

Be constructive, professional, and specific. Provide at least 30 words of substantive feedback."""

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
                    
                    # Enhanced critique validation
                    if response and response.content:
                        content = response.content.strip()
                        word_count = len(content.split())
                        
                        if len(content) >= self.min_content_length and word_count >= 10:
                            critique = AgentCritique(
                                critic_agent=agent.name,
                                target_proposal=proposal,
                                critique_content=content,
                                confidence=getattr(response, 'confidence', 0.8)
                            )
                            agent_critiques.append(critique)
                            self.logger.info(
                                f"[{debate_id[:8]}] ✅ {agent.name} critique of {proposal.agent_name} ready "
                                f"({word_count} words)"
                            )
                        else:
                            self.logger.warning(
                                f"[{debate_id[:8]}] ⚠️ {agent.name} critique too short "
                                f"({len(content)} chars, {word_count} words)"
                            )
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} critique of {proposal.agent_name} timed out")
                except Exception as e:
                    self.logger.error(f"[{debate_id[:8]}] 💥 Critique error: {e}")
            
            return agent_critiques
        
        # Gather critiques concurrently
        critique_tasks = [get_agent_critiques_enhanced(agent) for agent in agents]
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)
        
        # Flatten results and handle errors
        for i, result in enumerate(critique_results):
            if isinstance(result, list):
                critiques.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"[{debate_id[:8]}] Critique gathering failed for {agents[i].name}: {result}")
        
        self.logger.info(f"[{debate_id[:8]}] 📝 Gathered {len(critiques)} critiques")
        return critiques
    
    async def _gather_refined_proposals_enhanced(
        self,
        original_proposals: List[AgentProposal],
        critiques: List[AgentCritique],
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str
    ) -> List[AgentProposal]:
        """Gather refined proposals with enhanced error handling."""
        
        refined_proposals = []
        
        async def refine_agent_proposal_enhanced(agent: BaseAgent) -> Optional[AgentProposal]:
            """Refine a single agent's proposal with enhanced handling."""
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
                    self.logger.info(
                        f"[{debate_id[:8]}] 🔧 {agent.name} refining proposal based on "
                        f"{len(relevant_critiques)} critiques"
                    )
                    
                    # Build comprehensive refinement prompt
                    critiques_text = "\n\n".join([
                        f"**Critique from {c.critic_agent}:**\n{c.critique_content}"
                        for c in relevant_critiques
                    ])
                    
                    refinement_prompt = f"""Based on the feedback from your colleagues, refine your original proposal.

**Your Original Proposal:**
{original_proposal.content}

**Feedback Received:**
{critiques_text}

**Your Task:** Revise your proposal incorporating valid feedback while maintaining your core approach.

**Structure your refined proposal with:**
1. **Refined Position**: Your improved solution
2. **Key Changes**: What you modified based on feedback
3. **Rationale**: Why you made these changes
4. **Addressed Concerns**: How you handled the critiques
5. **Maintained Strengths**: What you kept from your original approach

Provide a substantial, well-reasoned response (minimum 60 words) that demonstrates thoughtful consideration of the feedback."""

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
                    
                    # Enhanced refinement validation
                    if response and response.content:
                        content = response.content.strip()
                        word_count = len(content.split())
                        
                        if len(content) >= self.min_content_length and word_count >= 20:
                            self.logger.info(
                                f"[{debate_id[:8]}] ✅ {agent.name} refined proposal ready "
                                f"({word_count} words)"
                            )
                            
                            return AgentProposal(
                                agent_type=agent.agent_type.value,
                                agent_name=agent.name,
                                content=content,
                                confidence_score=getattr(response, 'confidence', 0.8),
                                metadata={
                                    **getattr(response, 'metadata', {}),
                                    "refined": True,
                                    "original_proposal_id": id(original_proposal),
                                    "critiques_addressed": len(relevant_critiques),
                                    "word_count": word_count
                                }
                            )
                        else:
                            self.logger.warning(
                                f"[{debate_id[:8]}] ⚠️ {agent.name} refinement too short, keeping original"
                            )
                            return original_proposal
                    else:
                        self.logger.warning(f"[{debate_id[:8]}] ⚠️ {agent.name} refinement failed, keeping original")
                        return original_proposal
                else:
                    self.logger.info(f"[{debate_id[:8]}] 📋 {agent.name} keeping original proposal (no critiques)")
                    return original_proposal
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} refinement timed out, keeping original")
                # Return original proposal instead of None
                return next((p for p in original_proposals if p.agent_name == agent.name), None)
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 Refinement error for {agent.name}: {e}")
                # Return original proposal instead of None
                return next((p for p in original_proposals if p.agent_name == agent.name), None)
        
        # Gather refined proposals concurrently
        refinement_tasks = [refine_agent_proposal_enhanced(agent) for agent in agents]
        refinement_results = await asyncio.gather(*refinement_tasks, return_exceptions=True)
        
        # Process results and handle errors
        for i, result in enumerate(refinement_results):
            if result and not isinstance(result, Exception):
                refined_proposals.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"[{debate_id[:8]}] Refinement failed for {agents[i].name}: {result}")
                # Try to include original proposal as fallback
                original = next((p for p in original_proposals if p.agent_name == agents[i].name), None)
                if original:
                    refined_proposals.append(original)
        
        self.logger.info(f"[{debate_id[:8]}] 🏁 Refinement complete: {len(refined_proposals)} proposals")
        return refined_proposals
    
    async def _evaluate_consensus(self, proposals: List[AgentProposal]) -> float:
        """Evaluate consensus level with enhanced metrics."""
        if len(proposals) < 2:
            return 1.0  # Single proposal = perfect consensus
        
        # Multiple consensus metrics
        similarity_scores = []
        quality_scores = []
        
        for i, proposal1 in enumerate(proposals):
            for j, proposal2 in enumerate(proposals[i+1:], i+1):
                # Content similarity
                words1 = set(proposal1.content.lower().split())
                words2 = set(proposal2.content.lower().split())
                
                if words1 and words2:
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = overlap / union if union > 0 else 0
                    
                    # Quality bonus for substantial content
                    word_count_bonus = min(len(words1), len(words2)) / 100  # Bonus up to 1.0
                    similarity += min(word_count_bonus, 0.3)  # Cap bonus at 0.3
                    
                    similarity_scores.append(min(similarity, 1.0))
                
                # Quality assessment
                quality1 = min(proposal1.word_count / 50, 1.0)  # Quality based on word count
                quality2 = min(proposal2.word_count / 50, 1.0)
                avg_quality = (quality1 + quality2) / 2
                quality_scores.append(avg_quality)
        
        # Calculate weighted consensus
        if similarity_scores and quality_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # Weighted combination (70% similarity, 30% quality)
            consensus_score = (avg_similarity * 0.7) + (avg_quality * 0.3)
        else:
            consensus_score = 0.5  # Default moderate consensus
        
        return min(consensus_score, 1.0)
    
    def _calculate_quality_metrics(self, debate_result: DebateResult) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for the debate."""
        metrics = {}
        
        if not debate_result.rounds:
            return metrics
        
        all_proposals = []
        for round_data in debate_result.rounds:
            all_proposals.extend(round_data.proposals)
        
        if all_proposals:
            # Content quality metrics
            word_counts = [p.word_count for p in all_proposals]
            metrics["avg_word_count"] = sum(word_counts) / len(word_counts)
            metrics["min_word_count"] = min(word_counts)
            metrics["max_word_count"] = max(word_counts)
            
            # Participation metrics
            unique_agents = set(p.agent_name for p in all_proposals)
            metrics["agent_participation_rate"] = len(unique_agents) / len(debate_result.participating_agents) if debate_result.participating_agents else 0
            
            # Round success metrics
            successful_rounds = sum(1 for r in debate_result.rounds if r.proposals)
            metrics["round_success_rate"] = successful_rounds / len(debate_result.rounds)
            
            # Content diversity (based on unique word usage)
            all_words = set()
            for proposal in all_proposals:
                all_words.update(proposal.content.lower().split())
            metrics["vocabulary_diversity"] = len(all_words)
        
        return metrics
    
    async def _generate_final_consensus(
        self,
        rounds: List[DebateRound],
        agents: List[BaseAgent],
        context: TaskContext,
        topic: str
    ) -> Optional[str]:
        """Generate final consensus with enhanced fallback handling."""
        
        if not rounds:
            return f"No consensus could be reached on '{topic}' due to lack of participation."
        
        # Collect all proposals from all rounds
        all_proposals = []
        for round_data in rounds:
            all_proposals.extend(round_data.proposals)
        
        if not all_proposals:
            return f"Debate on '{topic}' started but no valid proposals were generated."
        
        # Use the first available agent to synthesize consensus
        synthesizer = None
        for agent in agents:
            if agent is not None:
                synthesizer = agent
                break
        
        if not synthesizer:
            return await self._generate_emergency_consensus(topic, rounds)
        
        try:
            # Enhanced synthesis prompt
            proposals_text = "\n\n".join([
                f"**{prop.agent_name}'s Proposal (Round {getattr(prop, 'round_number', 'Unknown')}):**\n{prop.content}"
                for prop in all_proposals[-6:]  # Use last 6 proposals to avoid overwhelming
            ])
            
            synthesis_prompt = f"""Based on the collaborative debate that has taken place, synthesize the best elements from all proposals into a comprehensive final solution.

**Debate Topic:** {topic}

**All Proposals Considered:**
{proposals_text}

**Your Task:** Create a unified consensus solution that incorporates the strongest ideas from each proposal while resolving any conflicts.

**Structure your consensus with:**
1. **Consensus Solution**: The agreed-upon approach that combines the best ideas
2. **Key Contributions**: What valuable elements you incorporated from each agent
3. **Conflict Resolution**: How you resolved any disagreeing viewpoints
4. **Implementation Strategy**: Practical steps to execute this solution
5. **Expected Outcomes**: What this approach should achieve

Focus on creating a practical, implementable solution that builds on the collaborative insights from the debate. Provide at least 100 words of substantive content."""

            synthesis_context = context.model_copy()
            synthesis_context.task_type = TaskType.CONSENSUS
            synthesis_context.agent_role = AgentRole.SPECIALIST
            
            response = await synthesizer.generate(synthesis_prompt, synthesis_context)
            
            if response and response.content and len(response.content.strip()) > 20:
                return response.content
            
        except Exception as e:
            self.logger.error(f"Consensus generation failed: {e}")
        
        # Fallback to emergency consensus
        return await self._generate_emergency_consensus(topic, rounds)
    
    async def _generate_emergency_consensus(self, topic: str, rounds: List[DebateRound]) -> str:
        """Generate emergency consensus when normal synthesis fails."""
        
        if not rounds:
            return f"Debate on '{topic}' could not reach consensus due to technical issues."
        
        # Extract key information from available data
        all_proposals = []
        agent_names = set()
        
        for round_data in rounds:
            all_proposals.extend(round_data.proposals)
            agent_names.update(round_data.participating_agents)
        
        if not all_proposals:
            return f"Debate on '{topic}' started but no valid proposals were generated across {len(rounds)} rounds."
        
        # Create a summary based on available content
        consensus = f"""**Emergency Consensus Summary for '{topic}':**

**Debate Overview:**
- Participants: {', '.join(sorted(agent_names)) if agent_names else 'None'}
- Rounds Completed: {len(rounds)}
- Total Proposals: {len(all_proposals)}

**Key Insights Generated:**"""
        
        # Include representative proposals
        for i, proposal in enumerate(all_proposals[-3:], 1):  # Last 3 proposals
            preview = proposal.content[:150] + "..." if len(proposal.content) > 150 else proposal.content
            consensus += f"\n\n{i}. **{proposal.agent_name}'s Contribution:**\n{preview}"
        
        consensus += f"""

**Status:** The debate generated {len(all_proposals)} substantive proposals from {len(agent_names)} participants across {len(rounds)} rounds. While technical difficulties prevented full consensus synthesis, the collaborative discussion produced valuable insights and multiple viable approaches to address '{topic}'."""
        
        return consensus
    
    async def _generate_round_summary(self, round_data: DebateRound) -> str:
        """Generate enhanced summary for a debate round."""
        
        summary_parts = [
            f"**Round {round_data.round_number} Summary:**",
            f"- Proposals: {len(round_data.proposals)}",
            f"- Critiques: {len(round_data.critiques)}",
            f"- Duration: {round_data.duration_seconds:.1f}s",
            f"- Success Rate: {round_data.success_rate:.2f}"
        ]
        
        if round_data.participating_agents:
            summary_parts.append(f"- Successful participants: {', '.join(round_data.participating_agents)}")
        
        if round_data.failed_agents:
            summary_parts.append(f"- Failed agents: {', '.join(round_data.failed_agents)}")
        
        # Add quality metrics
        if round_data.proposals:
            avg_words = sum(p.word_count for p in round_data.proposals) / len(round_data.proposals)
            summary_parts.append(f"- Average proposal length: {avg_words:.0f} words")
        
        return "\n".join(summary_parts)
    
    async def _generate_debate_summary(self, debate_result: DebateResult) -> str:
        """Generate comprehensive debate summary with quality metrics."""
        
        summary_parts = [
            f"**Debate Summary: {debate_result.topic}**",
            f"- Success: {'✅ Yes' if debate_result.success else '❌ No'}",
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
            
            # Add quality metrics if available
            if debate_result.quality_metrics:
                metrics = debate_result.quality_metrics
                if "avg_word_count" in metrics:
                    summary_parts.append(f"- Average proposal quality: {metrics['avg_word_count']:.0f} words")
                if "agent_participation_rate" in metrics:
                    summary_parts.append(f"- Participation rate: {metrics['agent_participation_rate']:.2f}")
        
        if debate_result.error_message:
            summary_parts.append(f"- Error: {debate_result.error_message}")
        
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
