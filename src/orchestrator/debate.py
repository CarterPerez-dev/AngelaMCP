#!/usr/bin/env python3
"""
Fixed Debate Orchestrator that prevents hanging issues.
Uses proper async task management with cleanup and cancellation.
"""

import asyncio
import time
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import the fixed async task manager
from .manager import (
    get_task_manager, 
    managed_gather, 
    timeout_protection,
    AsyncTaskManager
)


class DebatePhase(Enum):
    """Phases of a debate round."""
    INITIALIZATION = "initialization"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    REFINEMENT = "refinement"
    VOTING = "voting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentProposal:
    """Proposal from an agent."""
    agent_type: str
    agent_name: str
    content: str
    confidence_score: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentCritique:
    """Critique of a proposal."""
    critic_agent: str
    target_proposal: AgentProposal
    critique_content: str
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)


@dataclass
class DebateRound:
    """Single round of debate."""
    round_number: int
    phase: DebatePhase = DebatePhase.INITIALIZATION
    proposals: List[AgentProposal] = field(default_factory=list)
    critiques: List[AgentCritique] = field(default_factory=list)
    participating_agents: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    round_summary: Optional[str] = None
    success_rate: float = 0.0


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
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FixedDebateProtocol:
    """
    Fixed debate protocol that doesn't hang.
    Uses proper async task management with timeouts and cancellation.
    """
    
    def __init__(self, 
                 max_rounds: int = 2,
                 agent_timeout: float = 15.0,
                 round_timeout: float = 45.0,
                 total_timeout: float = 120.0):
        self.logger = logging.getLogger("debate.fixed")
        self.max_rounds = max_rounds
        self.agent_timeout = agent_timeout
        self.round_timeout = round_timeout
        self.total_timeout = total_timeout
        
        # Quality thresholds
        self.min_content_length = 10
        self.min_words = 5
        
        # Task manager
        self.task_manager = get_task_manager()
    
    async def conduct_debate(
        self,
        topic: str,
        agents: List[Any],  # BaseAgent type
        context: Any,  # TaskContext type
        max_rounds: Optional[int] = None
    ) -> DebateResult:
        """
        Conduct a debate with proper async handling and no hanging.
        """
        debate_id = str(uuid.uuid4())
        start_time = time.time()
        max_rounds = max_rounds or self.max_rounds
        
        # Initialize result
        result = DebateResult(
            debate_id=debate_id,
            topic=topic,
            success=False,
            participating_agents=[agent.name for agent in agents if agent]
        )
        
        self.logger.info(f"🎭 Starting debate {debate_id[:8]}: {topic}")
        
        try:
            # Use timeout protection for the entire debate
            async with timeout_protection(self.total_timeout, f"debate_{debate_id[:8]}"):
                
                # Filter working agents
                working_agents = [agent for agent in agents if agent is not None]
                if not working_agents:
                    raise RuntimeError("No working agents available")
                
                self.logger.info(f"🤖 Debate with {len(working_agents)} agents: {[a.name for a in working_agents]}")
                
                # Conduct debate rounds
                for round_num in range(1, max_rounds + 1):
                    self.logger.info(f"⚡ Round {round_num} starting...")
                    
                    try:
                        round_result = await self._conduct_round_safe(
                            debate_id, round_num, topic, working_agents, context
                        )
                        
                        result.rounds.append(round_result)
                        result.rounds_completed = round_num
                        
                        # Check if we have enough for consensus
                        if round_result.proposals:
                            break  # We have something to work with
                            
                    except Exception as e:
                        self.logger.error(f"Round {round_num} failed: {e}")
                        # Continue to next round if possible
                        if round_num == max_rounds:
                            raise  # Last round, give up
                
                # Generate consensus if we have any rounds
                if result.rounds and any(r.proposals for r in result.rounds):
                    result.final_consensus = await self._generate_consensus_safe(result.rounds, topic)
                    result.consensus_score = self._calculate_consensus_score(result.rounds)
                    result.success = True
                else:
                    result.final_consensus = f"No successful proposals generated for: {topic}"
                    result.success = False
                
                # Generate summary
                result.summary = self._generate_debate_summary(result)
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Debate {debate_id[:8]} timed out after {self.total_timeout}s")
            result.error_message = f"Debate timed out after {self.total_timeout}s"
            result.final_consensus = f"Debate on '{topic}' was interrupted due to timeout."
            
        except Exception as e:
            self.logger.error(f"Debate {debate_id[:8]} failed: {e}")
            result.error_message = str(e)
            result.final_consensus = f"Debate on '{topic}' encountered an error: {str(e)}"
        
        finally:
            result.total_duration = time.time() - start_time
            self.logger.info(f"🏁 Debate {debate_id[:8]} completed in {result.total_duration:.1f}s")
        
        return result
    
    async def _conduct_round_safe(
        self,
        debate_id: str,
        round_num: int,
        topic: str,
        agents: List[Any],
        context: Any
    ) -> DebateRound:
        """Conduct a single debate round with proper error handling."""
        
        round_result = DebateRound(
            round_number=round_num,
            phase=DebatePhase.INITIALIZATION,
            started_at=datetime.utcnow()
        )
        
        try:
            async with timeout_protection(self.round_timeout, f"round_{round_num}"):
                
                # Phase 1: Get proposals
                round_result.phase = DebatePhase.PROPOSAL
                self.logger.info(f"📝 Getting proposals from {len(agents)} agents...")
                
                proposals = await self._get_proposals_safe(debate_id, round_num, topic, agents, context)
                round_result.proposals = proposals
                round_result.participating_agents = [p.agent_name for p in proposals]
                
                self.logger.info(f"✅ Got {len(proposals)} proposals")
                
                # Phase 2: Get critiques (if multiple proposals)
                if len(proposals) > 1:
                    round_result.phase = DebatePhase.CRITIQUE
                    self.logger.info(f"🔍 Getting critiques...")
                    
                    critiques = await self._get_critiques_safe(debate_id, round_num, proposals, agents, context)
                    round_result.critiques = critiques
                    
                    self.logger.info(f"💭 Got {len(critiques)} critiques")
                
                round_result.phase = DebatePhase.COMPLETED
                round_result.success_rate = len(proposals) / len(agents) if agents else 0.0
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Round {round_num} timed out")
            round_result.phase = DebatePhase.FAILED
            round_result.failed_agents = [agent.name for agent in agents]
            
        except Exception as e:
            self.logger.error(f"Round {round_num} failed: {e}")
            round_result.phase = DebatePhase.FAILED
            round_result.failed_agents = [agent.name for agent in agents]
        
        finally:
            round_result.completed_at = datetime.utcnow()
            if round_result.started_at:
                duration = round_result.completed_at - round_result.started_at
                round_result.duration_seconds = duration.total_seconds()
        
        return round_result
    
    async def _get_proposals_safe(
        self,
        debate_id: str,
        round_num: int,
        topic: str,
        agents: List[Any],
        context: Any
    ) -> List[AgentProposal]:
        """Get proposals from agents with proper error handling."""
        
        async def get_single_proposal(agent: Any) -> Optional[AgentProposal]:
            """Get proposal from a single agent."""
            try:
                # Create a focused prompt
                prompt = f"""Provide a solution for: {topic}

Requirements:
- Give a clear, practical solution
- Explain your reasoning
- Keep it concise but substantive (at least {self.min_words} words)

This is round {round_num} of a collaborative discussion."""
                
                # Call the agent with timeout
                async with timeout_protection(self.agent_timeout, f"proposal_{agent.name}"):
                    response = await agent.generate(prompt, context)
                
                # Validate response
                if response and hasattr(response, 'content') and response.content:
                    content = response.content.strip()
                    word_count = len(content.split())
                    
                    if len(content) >= self.min_content_length and word_count >= self.min_words:
                        self.logger.info(f"✅ {agent.name}: {word_count} words")
                        
                        return AgentProposal(
                            agent_type=getattr(agent, 'agent_type', 'unknown'),
                            agent_name=agent.name,
                            content=content,
                            confidence_score=getattr(response, 'confidence', 0.8),
                            metadata={
                                "round": round_num,
                                "word_count": word_count,
                                "length": len(content)
                            }
                        )
                    else:
                        self.logger.warning(f"⚠️ {agent.name}: response too short ({word_count} words)")
                else:
                    self.logger.warning(f"❌ {agent.name}: no valid response")
                
                return None
                
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱️ {agent.name}: proposal timed out")
                return None
            except Exception as e:
                self.logger.error(f"💥 {agent.name}: proposal failed - {e}")
                return None
        
        # Use managed gather for proper cleanup
        coroutines = [get_single_proposal(agent) for agent in agents]
        names = [f"proposal_{agent.name}" for agent in agents]
        
        try:
            results = await managed_gather(
                coroutines, 
                names, 
                timeout=self.agent_timeout,
                return_exceptions=True,
                cancel_on_first_error=False  # Don't cancel others if one fails
            )
            
            # Filter successful proposals
            proposals = [result for result in results if isinstance(result, AgentProposal)]
            
            return proposals
            
        except Exception as e:
            self.logger.error(f"Proposal gathering failed: {e}")
            return []
    
    async def _get_critiques_safe(
        self,
        debate_id: str,
        round_num: int,
        proposals: List[AgentProposal],
        agents: List[Any],
        context: Any
    ) -> List[AgentCritique]:
        """Get critiques with proper error handling."""
        
        async def get_single_critique(agent: Any, proposal: AgentProposal) -> Optional[AgentCritique]:
            """Get critique from one agent for one proposal."""
            # Don't critique your own proposal
            if agent.name == proposal.agent_name:
                return None
            
            try:
                prompt = f"""Review this proposal and provide constructive feedback:

**Proposal by {proposal.agent_name}:**
{proposal.content}

Provide:
1. What's good about this approach
2. What could be improved
3. Specific suggestions

Keep it brief but helpful (at least {self.min_words} words)."""
                
                async with timeout_protection(self.agent_timeout, f"critique_{agent.name}"):
                    response = await agent.generate(prompt, context)
                
                if response and hasattr(response, 'content') and response.content:
                    content = response.content.strip()
                    word_count = len(content.split())
                    
                    if len(content) >= self.min_content_length and word_count >= self.min_words:
                        return AgentCritique(
                            critic_agent=agent.name,
                            target_proposal=proposal,
                            critique_content=content,
                            confidence=getattr(response, 'confidence', 0.8)
                        )
                
                return None
                
            except Exception as e:
                self.logger.error(f"Critique failed {agent.name} -> {proposal.agent_name}: {e}")
                return None
        
        # Create critique tasks for all agent-proposal combinations
        coroutines = []
        names = []
        
        for agent in agents:
            for proposal in proposals:
                if agent.name != proposal.agent_name:  # Don't critique own proposal
                    coroutines.append(get_single_critique(agent, proposal))
                    names.append(f"critique_{agent.name}_{proposal.agent_name}")
        
        if not coroutines:
            return []
        
        try:
            results = await managed_gather(
                coroutines,
                names,
                timeout=self.agent_timeout,
                return_exceptions=True,
                cancel_on_first_error=False
            )
            
            # Filter successful critiques
            critiques = [result for result in results if isinstance(result, AgentCritique)]
            
            return critiques
            
        except Exception as e:
            self.logger.error(f"Critique gathering failed: {e}")
            return []
    
    async def _generate_consensus_safe(self, rounds: List[DebateRound], topic: str) -> str:
        """Generate consensus with timeout protection."""
        try:
            async with timeout_protection(15.0, "consensus_generation"):
                
                # Collect all proposals
                all_proposals = []
                for round_data in rounds:
                    all_proposals.extend(round_data.proposals)
                
                if not all_proposals:
                    return f"No proposals generated for: {topic}"
                
                # Simple consensus: use the best proposal or combine top proposals
                if len(all_proposals) == 1:
                    return f"**Solution for '{topic}':**\n\n{all_proposals[0].content}"
                else:
                    # Combine insights from multiple proposals
                    consensus = f"**Collaborative Solution for '{topic}':**\n\n"
                    
                    for i, proposal in enumerate(all_proposals[:3], 1):  # Max 3 proposals
                        preview = proposal.content[:200] + "..." if len(proposal.content) > 200 else proposal.content
                        consensus += f"{i}. **{proposal.agent_name}'s approach:**\n{preview}\n\n"
                    
                    consensus += "**Recommended approach:** Combine the strongest elements from each proposal above."
                    
                    return consensus
                    
        except asyncio.TimeoutError:
            self.logger.warning("Consensus generation timed out")
            return f"Consensus generation timed out for: {topic}"
        except Exception as e:
            self.logger.error(f"Consensus generation failed: {e}")
            return f"Unable to generate consensus for: {topic}"
    
    def _calculate_consensus_score(self, rounds: List[DebateRound]) -> float:
        """Calculate consensus score based on round results."""
        if not rounds:
            return 0.0
        
        total_score = 0.0
        total_rounds = 0
        
        for round_data in rounds:
            if round_data.proposals:
                round_score = round_data.success_rate
                total_score += round_score
                total_rounds += 1
        
        return total_score / total_rounds if total_rounds > 0 else 0.0
    
    def _generate_debate_summary(self, result: DebateResult) -> str:
        """Generate a summary of the debate."""
        if not result.rounds:
            return f"Debate on '{result.topic}' failed to complete any rounds."
        
        total_proposals = sum(len(r.proposals) for r in result.rounds)
        total_critiques = sum(len(r.critiques) for r in result.rounds)
        
        summary = f"""**Debate Summary: {result.topic}**

📊 **Results:**
- Status: {'✅ Success' if result.success else '❌ Failed'}
- Rounds: {result.rounds_completed}
- Duration: {result.total_duration:.1f}s
- Proposals: {total_proposals}
- Critiques: {total_critiques}
- Consensus Score: {result.consensus_score:.2f}

👥 **Participants:** {', '.join(result.participating_agents)}"""
        
        if result.error_message:
            summary += f"\n\n⚠️ **Issue:** {result.error_message}"
        
        return summary
