"""
Debate Protocol for AngelaMCP.

This module implements structured debates between AI agents where they can
propose solutions, critique each other's work, and reach consensus through
voting. I'm implementing a simple but effective debate flow focusing on
the core collaborative experience.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from src.agents.base import BaseAgent, AgentResponse, TaskContext, TaskType, AgentType
from src.utils.logger import get_logger, AsyncPerformanceLogger

logger = get_logger("orchestrator.debate")


class DebatePhase(str, Enum):
    """Phases of the debate process."""
    INITIALIZATION = "initialization"
    PROPOSALS = "proposals"
    CRITIQUE = "critique"
    REBUTTAL = "rebuttal"
    FINAL_PROPOSALS = "final_proposals"
    VOTING = "voting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentProposal:
    """A proposal from an agent."""
    agent_type: str
    agent_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCritique:
    """A critique of another agent's proposal."""
    critic_agent: str
    target_proposal: str  # agent_type of the proposal being critiqued
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "moderate"  # low, moderate, high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """Information about a single debate round."""
    round_number: int
    phase: DebatePhase
    started_at: datetime
    completed_at: Optional[datetime] = None
    proposals: List[AgentProposal] = field(default_factory=list)
    critiques: List[AgentCritique] = field(default_factory=list)
    phase_duration: Optional[float] = None


@dataclass
class DebateResult:
    """Final result of a debate."""
    debate_id: str
    topic: str
    success: bool
    winner: Optional[str] = None
    winning_proposal: Optional[AgentProposal] = None
    total_duration: float = 0.0
    rounds: List[DebateRound] = field(default_factory=list)
    participating_agents: List[str] = field(default_factory=list)
    consensus_reached: bool = False
    vote_breakdown: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebateProtocol:
    """
    Manages structured debates between AI agents.
    
    I'm implementing a collaborative debate system where agents propose solutions,
    critique each other's work, provide rebuttals, and reach consensus through voting.
    The focus is on creating engaging, productive multi-agent interactions.
    """
    
    def __init__(self, timeout_per_phase: int = 120, max_rounds: int = 3):
        """
        Initialize the debate protocol.
        
        Args:
            timeout_per_phase: Maximum time per debate phase in seconds
            max_rounds: Maximum number of debate rounds
        """
        self.timeout_per_phase = timeout_per_phase
        self.max_rounds = max_rounds
        self.logger = get_logger("debate")
        
        # Track active debates
        self._active_debates: Dict[str, Dict[str, Any]] = {}
    
    async def conduct_debate(
        self,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext,
        require_all_agents: bool = False
    ) -> DebateResult:
        """
        Conduct a structured debate between agents.
        
        Args:
            topic: The topic/task for agents to debate
            agents: List of participating agents
            context: Task context for the debate
            require_all_agents: Whether all agents must participate successfully
            
        Returns:
            DebateResult with complete debate transcript and outcome
        """
        debate_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Initialize debate tracking
        debate_info = {
            "debate_id": debate_id,
            "topic": topic,
            "agents": {agent.name: agent for agent in agents},
            "start_time": start_time,
            "current_phase": DebatePhase.INITIALIZATION
        }
        self._active_debates[debate_id] = debate_info
        
        self.logger.info(f"🎪 Starting debate {debate_id[:8]} on: {topic[:60]}...")
        
        try:
            async with AsyncPerformanceLogger(self.logger, "debate_full", task_id=debate_id):
                # Phase 1: Initial Proposals
                self.logger.info(f"[{debate_id[:8]}] 💡 Phase 1: Getting proposals from all agents")
                proposals = await self._phase_initial_proposals(topic, agents, context, debate_id)
                
                if not proposals and require_all_agents:
                    raise DebateError("Failed to get initial proposals from all agents")
                
                if not proposals:
                    self.logger.warning(f"[{debate_id[:8]}] No proposals received - ending debate")
                    return DebateResult(
                        debate_id=debate_id,
                        topic=topic,
                        success=False,
                        total_duration=time.time() - start_time,
                        participating_agents=[agent.name for agent in agents],
                        error_message="No proposals received from any agent"
                    )
                
                # Phase 2: Critique Round
                self.logger.info(f"[{debate_id[:8]}] 🔍 Phase 2: Agents critiquing each other's work")
                critiques = await self._phase_critique_round(proposals, agents, context, debate_id)
                
                # Phase 3: Final Proposals (agents refine based on feedback)
                self.logger.info(f"[{debate_id[:8]}] ✨ Phase 3: Refined proposals based on feedback")
                final_proposals = await self._phase_final_proposals(
                    topic, proposals, critiques, agents, context, debate_id
                )
                
                # Calculate total duration
                total_duration = time.time() - start_time
                
                # Create debate result with all the data needed for voting
                result = DebateResult(
                    debate_id=debate_id,
                    topic=topic,
                    success=len(final_proposals) > 0,
                    total_duration=total_duration,
                    participating_agents=[agent.name for agent in agents],
                    metadata={
                        "initial_proposals": len(proposals),
                        "critiques_generated": len(critiques),
                        "final_proposals": len(final_proposals),
                        "context": context.model_dump() if hasattr(context, 'model_dump') else {}
                    }
                )
                
                # Store proposals for voting
                result.rounds = [
                    DebateRound(
                        round_number=1,
                        phase=DebatePhase.PROPOSALS,
                        started_at=datetime.utcnow(),
                        proposals=[
                            AgentProposal(
                                agent_type=agent.agent_type.value,
                                agent_name=agent.name,
                                content=prop_content,
                                confidence_score=getattr(prop_response, 'confidence_score', None),
                                metadata=prop_response.metadata if hasattr(prop_response, 'metadata') else {}
                            )
                            for agent, prop_response, prop_content in final_proposals
                        ]
                    )
                ]
                
                self.logger.info(f"🎉 Debate {debate_id[:8]} completed in {total_duration:.1f}s with {len(final_proposals)} proposals")
                return result
                
        except Exception as e:
            total_duration = time.time() - start_time
            self.logger.error(f"❌ Debate {debate_id[:8]} failed: {e}")
            
            return DebateResult(
                debate_id=debate_id,
                topic=topic,
                success=False,
                total_duration=total_duration,
                participating_agents=[agent.name for agent in agents],
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
        
        finally:
            # Cleanup
            if debate_id in self._active_debates:
                del self._active_debates[debate_id]
    
    async def _phase_initial_proposals(
        self,
        topic: str,
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str
    ) -> List[tuple]:
        """Phase 1: Each agent proposes their solution."""
        
        proposals = []
        
        # Get proposals from all agents in parallel
        async def get_agent_proposal(agent: BaseAgent) -> Optional[tuple]:
            try:
                self.logger.info(f"[{debate_id[:8]}] 🤖 Getting proposal from {agent.name}...")
                
                async with AsyncPerformanceLogger(
                    self.logger, f"proposal_{agent.name}", task_id=debate_id
                ):
                    response = await asyncio.wait_for(
                        agent.propose_solution(topic, [], context),
                        timeout=self.timeout_per_phase
                    )
                    
                    if response.success:
                        self.logger.info(f"[{debate_id[:8]}] ✅ {agent.name} proposal received ({len(response.content)} chars)")
                        return (agent, response, response.content)
                    else:
                        self.logger.warning(f"[{debate_id[:8]}] ❌ {agent.name} proposal failed: {response.error_message}")
                        return None
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} proposal timed out after {self.timeout_per_phase}s")
                return None
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 {agent.name} proposal error: {e}")
                return None
        
        # Execute all proposals concurrently
        proposal_tasks = [get_agent_proposal(agent) for agent in agents]
        proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)
        
        # Filter out failed proposals
        for result in proposal_results:
            if result and not isinstance(result, Exception):
                proposals.append(result)
        
        self.logger.info(f"[{debate_id[:8]}] 📊 Collected {len(proposals)}/{len(agents)} proposals")
        return proposals
    
    async def _phase_critique_round(
        self,
        proposals: List[tuple],
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str
    ) -> List[AgentCritique]:
        """Phase 2: Each agent critiques other agents' proposals."""
        
        critiques = []
        
        # Create critique tasks for each agent to review each other agent's proposals
        async def generate_critique(critic_agent: BaseAgent, target_proposal: tuple) -> Optional[AgentCritique]:
            target_agent, target_response, target_content = target_proposal
            
            # Don't critique your own proposal
            if critic_agent.name == target_agent.name:
                return None
            
            try:
                self.logger.info(f"[{debate_id[:8]}] 🔍 {critic_agent.name} reviewing {target_agent.name}'s work...")
                
                async with AsyncPerformanceLogger(
                    self.logger, f"critique_{critic_agent.name}_vs_{target_agent.name}", task_id=debate_id
                ):
                    critique_response = await asyncio.wait_for(
                        critic_agent.critique(target_content, f"Solution from {target_agent.name}", context),
                        timeout=self.timeout_per_phase
                    )
                    
                    if critique_response.success:
                        self.logger.info(f"[{debate_id[:8]}] ✅ {critic_agent.name} critique of {target_agent.name} complete")
                        return AgentCritique(
                            critic_agent=critic_agent.name,
                            target_proposal=target_agent.name,
                            content=critique_response.content,
                            metadata=critique_response.metadata if hasattr(critique_response, 'metadata') else {}
                        )
                    else:
                        self.logger.warning(
                            f"[{debate_id[:8]}] ❌ {critic_agent.name} critique of {target_agent.name} failed"
                        )
                        return None
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {critic_agent.name} critique timed out")
                return None
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 Critique error: {e}")
                return None
        
        # Generate all critique tasks
        critique_tasks = []
        for critic_agent in agents:
            for target_proposal in proposals:
                critique_tasks.append(generate_critique(critic_agent, target_proposal))
        
        # Execute critiques concurrently
        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)
        
        # Filter successful critiques
        for result in critique_results:
            if result and not isinstance(result, Exception):
                critiques.append(result)
        
        self.logger.info(f"[{debate_id[:8]}] 📝 Generated {len(critiques)} critiques")
        return critiques
    
    async def _phase_final_proposals(
        self,
        topic: str,
        initial_proposals: List[tuple],
        critiques: List[AgentCritique],
        agents: List[BaseAgent],
        context: TaskContext,
        debate_id: str
    ) -> List[tuple]:
        """Phase 3: Agents refine their proposals based on critiques."""
        
        final_proposals = []
        
        async def generate_final_proposal(agent: BaseAgent) -> Optional[tuple]:
            try:
                # Find critiques of this agent's proposal
                agent_critiques = [
                    c for c in critiques 
                    if c.target_proposal == agent.name
                ]
                
                if agent_critiques:
                    self.logger.info(f"[{debate_id[:8]}] 🔄 {agent.name} refining based on {len(agent_critiques)} critiques...")
                    
                    # Create prompt incorporating feedback
                    critique_text = "\n\n".join([
                        f"**Feedback from {c.critic_agent}:**\n{c.content}"
                        for c in agent_critiques
                    ])
                    
                    refinement_prompt = f"""Based on the feedback from other AI agents, please refine your solution:

**Original Task:** {topic}

**Feedback Received:**
{critique_text}

Please provide an improved solution that:
- Addresses the valid concerns raised
- Improves any weak areas identified  
- Maintains your solution's core strengths
- Creates a more robust overall approach

Focus on creating the best possible solution incorporating this collaborative feedback."""
                    
                    # Get refined proposal
                    async with AsyncPerformanceLogger(
                        self.logger, f"final_proposal_{agent.name}", task_id=debate_id
                    ):
                        response = await asyncio.wait_for(
                            agent.generate(refinement_prompt, context),
                            timeout=self.timeout_per_phase
                        )
                else:
                    self.logger.info(f"[{debate_id[:8]}] 📋 {agent.name} keeping original proposal (no critiques)")
                    
                    # No critiques received, return original proposal
                    original_proposal = next(
                        (prop for a, resp, prop in initial_proposals if a.name == agent.name),
                        None
                    )
                    
                    if original_proposal:
                        response = next(resp for a, resp, prop in initial_proposals if a.name == agent.name)
                    else:
                        # Fallback - generate new proposal
                        response = await agent.propose_solution(topic, [], context)
                
                if response.success:
                    self.logger.info(f"[{debate_id[:8]}] ✅ {agent.name} final proposal ready")
                    return (agent, response, response.content)
                else:
                    self.logger.warning(f"[{debate_id[:8]}] ❌ {agent.name} final proposal failed")
                    return None
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"[{debate_id[:8]}] ⏱️ {agent.name} final proposal timed out")
                return None
            except Exception as e:
                self.logger.error(f"[{debate_id[:8]}] 💥 Final proposal error for {agent.name}: {e}")
                return None
        
        # Generate final proposals concurrently
        final_tasks = [generate_final_proposal(agent) for agent in agents]
        final_results = await asyncio.gather(*final_tasks, return_exceptions=True)
        
        # Filter successful proposals
        for result in final_results:
            if result and not isinstance(result, Exception):
                final_proposals.append(result)
        
        self.logger.info(f"[{debate_id[:8]}] 🏁 Final phase complete: {len(final_proposals)} refined proposals")
        return final_proposals


class DebateError(Exception):
    """Exception raised during debate operations."""
    pass