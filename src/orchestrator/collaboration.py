"""
Main Collaboration Orchestrator for AngelaMCP.

This module coordinates the complete multi-agent collaboration flow:
1. Initialize agents (Claude Code, OpenAI, Gemini)
2. Conduct structured debate
3. Run weighted voting
4. Return final collaborative result

This is the core of what makes AngelaMCP different from single-agent tools.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from src.agents.base import BaseAgent, TaskContext, TaskType, AgentType
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.gemini_agent import GeminiAgent
from src.orchestrator.debate import DebateProtocol, DebateResult
from src.orchestrator.voting import VotingSystem, VotingResult
from src.utils.logger import get_logger, AsyncPerformanceLogger
from config.settings import settings

logger = get_logger("orchestrator.collaboration")


class CollaborationMode(str, Enum):
    """Different modes of collaboration."""
    FULL_DEBATE = "full_debate"  # Complete debate + voting
    QUICK_VOTE = "quick_vote"    # Skip critique phase, go straight to voting
    CLAUDE_LEAD = "claude_lead"  # Claude Code proposes, others review
    PARALLEL_WORK = "parallel_work"  # All agents work independently, best result wins


@dataclass
class CollaborationRequest:
    """Request for multi-agent collaboration."""
    task_description: str
    mode: CollaborationMode = CollaborationMode.FULL_DEBATE
    require_consensus: bool = True
    timeout_minutes: int = 10
    context: Optional[Dict[str, Any]] = None


@dataclass
class CollaborationResult:
    """Complete result of multi-agent collaboration."""
    collaboration_id: str
    request: CollaborationRequest
    success: bool
    
    # Final outcome
    final_solution: Optional[str] = None
    chosen_agent: Optional[str] = None
    
    # Process details
    debate_result: Optional[DebateResult] = None
    voting_result: Optional[VotingResult] = None
    
    # Metrics
    total_duration: float = 0.0
    agents_participated: List[str] = field(default_factory=list)
    consensus_reached: bool = False
    
    # Execution summary
    summary: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CollaborationOrchestrator:
    """
    Main orchestrator for multi-agent collaboration.
    
    This is the heart of AngelaMCP - it coordinates Claude Code (with its MCP tools),
    OpenAI (reviewer), and Gemini (researcher) to work together on tasks.
    """
    
    def __init__(
        self,
        claude_agent: Optional[ClaudeCodeAgent] = None,
        openai_agent: Optional[OpenAIAgent] = None,
        gemini_agent: Optional[GeminiAgent] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the collaboration orchestrator.
        
        Args:
            claude_agent: Claude Code agent instance
            openai_agent: OpenAI agent instance  
            gemini_agent: Gemini agent instance
            status_callback: Optional callback for real-time status updates
        """
        self.logger = get_logger("collaboration")
        self.status_callback = status_callback
        
        # Initialize agents
        self.claude_agent = claude_agent or ClaudeCodeAgent()
        self.openai_agent = openai_agent or OpenAIAgent()
        self.gemini_agent = gemini_agent or GeminiAgent()
        
        # Initialize collaboration components
        self.debate_protocol = DebateProtocol(
            timeout_per_phase=settings.debate_timeout,
            max_rounds=settings.debate_max_rounds
        )
        
        self.voting_system = VotingSystem(
            claude_vote_weight=settings.claude_vote_weight,
            enable_claude_veto=settings.claude_veto_enabled,
            voting_timeout=settings.voting_timeout
        )
        
        # Track active collaborations
        self._active_collaborations: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("🎭 Collaboration orchestrator initialized with 3 agents")
    
    def _update_status(self, message: str) -> None:
        """Update status via callback if available."""
        if self.status_callback:
            self.status_callback(message)
        self.logger.info(message)
    
    async def collaborate(self, request: CollaborationRequest) -> CollaborationResult:
        """
        Execute a complete multi-agent collaboration.
        
        Args:
            request: The collaboration request
            
        Returns:
            CollaborationResult with final outcome and process details
        """
        collaboration_id = str(uuid.uuid4())
        start_time = time.time()
        
        self._update_status(f"🚀 Starting collaboration {collaboration_id[:8]} on: {request.task_description[:60]}...")
        
        # Track collaboration
        self._active_collaborations[collaboration_id] = {
            "id": collaboration_id,
            "request": request,
            "start_time": start_time,
            "status": "initializing"
        }
        
        try:
            async with AsyncPerformanceLogger(self.logger, "collaboration_full", task_id=collaboration_id):
                
                # Create task context
                context = TaskContext(
                    task_id=collaboration_id,
                    task_type=self._determine_task_type(request.task_description),
                    session_id=collaboration_id,
                    requires_collaboration=True,
                    enable_debate=True,
                    max_debate_rounds=settings.debate_max_rounds,
                    timeout_seconds=request.timeout_minutes * 60,
                    context_data=request.context or {}
                )
                
                # Get list of participating agents
                agents = [self.claude_agent, self.openai_agent, self.gemini_agent]
                participating_agents = [agent.name for agent in agents]
                
                # Execute collaboration based on mode
                if request.mode == CollaborationMode.FULL_DEBATE:
                    result = await self._full_debate_collaboration(
                        request, agents, context, collaboration_id
                    )
                elif request.mode == CollaborationMode.CLAUDE_LEAD:
                    result = await self._claude_lead_collaboration(
                        request, agents, context, collaboration_id
                    )
                else:
                    # Default to full debate for now
                    result = await self._full_debate_collaboration(
                        request, agents, context, collaboration_id
                    )
                
                total_duration = time.time() - start_time
                
                # Create final result
                final_result = CollaborationResult(
                    collaboration_id=collaboration_id,
                    request=request,
                    success=result["success"],
                    final_solution=result.get("solution"),
                    chosen_agent=result.get("chosen_agent"),
                    debate_result=result.get("debate_result"),
                    voting_result=result.get("voting_result"),
                    total_duration=total_duration,
                    agents_participated=participating_agents,
                    consensus_reached=result.get("consensus_reached", False),
                    summary=result.get("summary", ""),
                    error_message=result.get("error"),
                    metadata={
                        "mode": request.mode,
                        "agents_count": len(agents),
                        "timeout_minutes": request.timeout_minutes
                    }
                )
                
                self._update_status(f"✅ Collaboration completed: {final_result.chosen_agent or 'No winner'} in {total_duration:.1f}s")
                return final_result
                
        except Exception as e:
            total_duration = time.time() - start_time
            self.logger.error(f"❌ Collaboration {collaboration_id[:8]} failed: {e}")
            
            return CollaborationResult(
                collaboration_id=collaboration_id,
                request=request,
                success=False,
                total_duration=total_duration,
                agents_participated=[agent.name for agent in [self.claude_agent, self.openai_agent, self.gemini_agent]],
                error_message=str(e),
                metadata={"error_type": type(e).__name__}
            )
        
        finally:
            # Cleanup
            if collaboration_id in self._active_collaborations:
                del self._active_collaborations[collaboration_id]
    
    async def _full_debate_collaboration(
        self,
        request: CollaborationRequest,
        agents: List[BaseAgent],
        context: TaskContext,
        collaboration_id: str
    ) -> Dict[str, Any]:
        """Execute full debate + voting collaboration."""
        
        self._update_status(f"[{collaboration_id[:8]}] 🎪 Starting full debate mode")
        
        # Phase 1: Conduct Debate
        self._update_status(f"[{collaboration_id[:8]}] 💬 Phase 1: Agent debate in progress...")
        debate_result = await self.debate_protocol.conduct_debate(
            topic=request.task_description,
            agents=agents,
            context=context,
            require_all_agents=False  # Continue even if some agents fail
        )
        
        if not debate_result.success:
            return {
                "success": False,
                "error": f"Debate failed: {debate_result.error_message}",
                "debate_result": debate_result
            }
        
        # Phase 2: Conduct Voting
        self._update_status(f"[{collaboration_id[:8]}] 🗳️ Phase 2: Voting on proposals...")
        voting_result = await self.voting_system.conduct_voting(
            debate_result=debate_result,
            agents=agents,
            context=context
        )
        
        if not voting_result.success:
            return {
                "success": False,
                "error": f"Voting failed: {voting_result.error_message}",
                "debate_result": debate_result,
                "voting_result": voting_result
            }
        
        # Phase 3: Final Implementation (if we have a winner)
        final_solution = None
        if voting_result.winner and voting_result.winning_proposal:
            self._update_status(f"[{collaboration_id[:8]}] 🏆 Phase 3: Implementing {voting_result.winner}'s solution...")
            
            # If Claude Code won, we can potentially execute the solution
            if voting_result.winner == self.claude_agent.name:
                # Claude Code can execute its own solution
                final_solution = voting_result.winning_proposal.content
            else:
                # For other agents, we use their proposal as-is
                final_solution = voting_result.winning_proposal.content
        
        # Create summary
        summary = f"""
**🎭 Multi-Agent Collaboration Complete**

**Task:** {request.task_description[:100]}...

**Participants:** {', '.join([agent.name for agent in agents])}

**Debate Results:**
- Proposals generated: {len(debate_result.rounds[0].proposals) if debate_result.rounds else 0}
- Duration: {debate_result.total_duration:.1f}s

**Voting Results:**
{voting_result.voting_summary}

**Final Outcome:** {"✅ " + voting_result.winner if voting_result.winner else "❌ No consensus reached"}
""".strip()
        
        return {
            "success": True,
            "solution": final_solution,
            "chosen_agent": voting_result.winner,
            "debate_result": debate_result,
            "voting_result": voting_result,
            "consensus_reached": voting_result.consensus_reached,
            "summary": summary
        }
    
    async def _claude_lead_collaboration(
        self,
        request: CollaborationRequest,
        agents: List[BaseAgent],
        context: TaskContext,
        collaboration_id: str
    ) -> Dict[str, Any]:
        """Execute Claude-lead collaboration (Claude proposes, others review)."""
        
        self._update_status(f"[{collaboration_id[:8]}] 👑 Claude leading collaboration")
        
        # Phase 1: Claude Code creates initial solution
        self._update_status(f"[{collaboration_id[:8]}] 🔧 Claude Code implementing solution...")
        claude_response = await self.claude_agent.propose_solution(
            request.task_description, [], context
        )
        
        if not claude_response.success:
            return {
                "success": False,
                "error": f"Claude Code failed to create solution: {claude_response.error_message}"
            }
        
        # Phase 2: Other agents review and suggest improvements
        self._update_status(f"[{collaboration_id[:8]}] 👀 Other agents reviewing Claude's work...")
        
        reviews = []
        for agent in agents:
            if agent.name == self.claude_agent.name:
                continue  # Skip Claude reviewing itself
            
            try:
                review = await agent.critique(
                    claude_response.content,
                    f"Solution from Claude Code for: {request.task_description}",
                    context
                )
                if review.success:
                    reviews.append({
                        "agent": agent.name,
                        "review": review.content
                    })
            except Exception as e:
                self.logger.warning(f"Review from {agent.name} failed: {e}")
        
        # Phase 3: Claude Code optionally refines based on feedback
        final_solution = claude_response.content
        if reviews:
            self._update_status(f"[{collaboration_id[:8]}] 🔄 Claude Code refining based on feedback...")
            
            # Create refinement prompt
            review_text = "\n\n".join([
                f"**Review from {r['agent']}:**\n{r['review']}"
                for r in reviews
            ])
            
            refinement_prompt = f"""Based on feedback from other AI agents, please refine your solution:

**Original Task:** {request.task_description}

**Your Current Solution:**
{claude_response.content}

**Feedback Received:**
{review_text}

Please provide an improved solution that addresses valid concerns while maintaining the core strengths of your approach."""
            
            try:
                refined_response = await self.claude_agent.generate(refinement_prompt, context)
                if refined_response.success:
                    final_solution = refined_response.content
            except Exception as e:
                self.logger.warning(f"Claude Code refinement failed: {e}")
        
        summary = f"""
**👑 Claude Code Lead Collaboration**

**Task:** {request.task_description[:100]}...

**Process:**
- Claude Code created initial solution
- {len(reviews)} agents provided reviews
- {"Solution refined based on feedback" if reviews else "No refinements needed"}

**Final Solution:** Implemented by Claude Code
""".strip()
        
        return {
            "success": True,
            "solution": final_solution,
            "chosen_agent": self.claude_agent.name,
            "consensus_reached": True,  # Claude Code is authoritative
            "summary": summary
        }
    
    def _determine_task_type(self, task_description: str) -> TaskType:
        """Determine the task type from description."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["test", "testing", "unittest", "pytest"]):
            return TaskType.TESTING
        elif any(word in task_lower for word in ["debug", "error", "fix", "bug"]):
            return TaskType.DEBUGGING
        elif any(word in task_lower for word in ["review", "critique", "analyze"]):
            return TaskType.CODE_REVIEW
        elif any(word in task_lower for word in ["document", "docs", "readme", "comment"]):
            return TaskType.DOCUMENTATION
        elif any(word in task_lower for word in ["research", "investigate", "explore"]):
            return TaskType.RESEARCH
        elif any(word in task_lower for word in ["create", "build", "implement", "develop", "code"]):
            return TaskType.CODE_GENERATION
        else:
            return TaskType.CUSTOM
    
    async def quick_task(self, task_description: str, timeout_minutes: int = 5) -> CollaborationResult:
        """Execute a quick collaborative task with reasonable defaults."""
        request = CollaborationRequest(
            task_description=task_description,
            mode=CollaborationMode.FULL_DEBATE,
            timeout_minutes=timeout_minutes
        )
        return await self.collaborate(request)
    
    async def claude_lead_task(self, task_description: str, timeout_minutes: int = 3) -> CollaborationResult:
        """Execute a Claude-lead task where Claude implements and others review."""
        request = CollaborationRequest(
            task_description=task_description,
            mode=CollaborationMode.CLAUDE_LEAD,
            timeout_minutes=timeout_minutes
        )
        return await self.collaborate(request)
    
    def get_active_collaborations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active collaborations."""
        return {
            collab_id: {
                "task": info["request"].task_description[:100],
                "status": info["status"],
                "duration": time.time() - info["start_time"]
            }
            for collab_id, info in self._active_collaborations.items()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all agents and collaboration system."""
        agent_health = {}
        
        # Check each agent
        for agent_name, agent in [
            ("claude_code", self.claude_agent),
            ("openai", self.openai_agent),
            ("gemini", self.gemini_agent)
        ]:
            try:
                health = await agent.health_check()
                agent_health[agent_name] = health
            except Exception as e:
                agent_health[agent_name] = {"status": "error", "error": str(e)}
        
        # Overall system health
        healthy_agents = sum(1 for h in agent_health.values() if h.get("status") == "healthy")
        
        return {
            "overall_status": "healthy" if healthy_agents >= 2 else "degraded" if healthy_agents >= 1 else "unhealthy",
            "healthy_agents": healthy_agents,
            "total_agents": 3,
            "agent_health": agent_health,
            "collaboration_features": {
                "debate_protocol": "available",
                "voting_system": "available",
                "claude_veto": settings.claude_veto_enabled
            }
        }


class CollaborationError(Exception):
    """Exception raised during collaboration operations."""
    pass