"""
Debate protocol for AngelaMCP multi-agent collaboration.

This module implements structured debate protocols that allow agents to engage
in productive disagreement, present arguments, counter-arguments, and reach
consensus through iterative discussion. I'm building a system that leverages
the different strengths and perspectives of each agent type.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, AgentResponse, TaskContext, AgentRole, agent_registry
from src.utils.logger import get_logger, log_context, AsyncPerformanceLogger

logger = get_logger("orchestration.debate")


class DebateRole(str, Enum):
    """Roles agents can play in a debate."""
    PROPOSER = "proposer"        # Agent proposing initial solution
    CHALLENGER = "challenger"    # Agent challenging the proposal
    MODERATOR = "moderator"      # Agent moderating the debate
    SYNTHESIZER = "synthesizer"  # Agent creating final synthesis


class ArgumentType(str, Enum):
    """Types of arguments in a debate."""
    INITIAL_PROPOSAL = "initial_proposal"
    COUNTER_ARGUMENT = "counter_argument"
    REBUTTAL = "rebuttal"
    CLARIFICATION = "clarification"
    EVIDENCE = "evidence"
    SYNTHESIS = "synthesis"


@dataclass
class DebateArgument:
    """Represents a single argument in a debate."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    agent_type: str = ""
    role: DebateRole = DebateRole.PROPOSER
    argument_type: ArgumentType = ArgumentType.INITIAL_PROPOSAL
    content: str = ""
    confidence_score: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    addresses_argument_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """Represents a single round of debate."""
    round_number: int
    arguments: List[DebateArgument] = field(default_factory=list)
    round_summary: str = ""
    consensus_score: float = 0.0
    round_duration_ms: float = 0.0


@dataclass
class DebateResult:
    """Final result of a debate session."""
    debate_id: str
    task_id: str
    success: bool
    final_consensus: str
    confidence_score: float
    rounds: List[DebateRound] = field(default_factory=list)
    participating_agents: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class DebateProtocol:
    """
    Structured debate protocol for multi-agent collaboration.
    
    I'm implementing a sophisticated debate system that allows agents to:
    1. Present initial proposals
    2. Challenge and counter-argue
    3. Provide evidence and clarification
    4. Reach consensus through structured discussion
    """
    
    def __init__(self, orchestrator, task: 'OrchestrationTask'):
        self.orchestrator = orchestrator
        self.task = task
        self.logger = get_logger("orchestration.debate_protocol")
        
        # Debate configuration
        self.max_rounds = task.max_debate_rounds
        self.consensus_threshold = task.consensus_threshold
        self.timeout_seconds = task.timeout_seconds or 300  # 5 minutes default
        
        # Debate state
        self.debate_id = str(uuid.uuid4())
        self.rounds: List[DebateRound] = []
        self.arguments: List[DebateArgument] = []
        self.participating_agents: List[BaseAgent] = []
        
        # Performance tracking
        self.start_time = 0.0
        self.total_cost = 0.0
        self.total_tokens = 0
        
    def _select_debate_agents(self) -> List[BaseAgent]:
        """Select agents for debate based on diversity and capability."""
        available_agents = agent_registry.get_all_agents()
        
        if len(available_agents) < 2:
            raise ValueError("At least 2 agents required for debate")
        
        # Prefer diverse agent types for better debate
        selected_agents = []
        agent_types_used = set()
        
        # First, select one agent of each type if available
        for agent in available_agents:
            if agent.agent_type not in agent_types_used and len(selected_agents) < 3:
                selected_agents.append(agent)
                agent_types_used.add(agent.agent_type)
        
        # If we need more agents, add based on performance
        if len(selected_agents) < 3 and len(available_agents) > len(selected_agents):
            remaining_agents = [
                agent for agent in available_agents 
                if agent not in selected_agents
            ]
            # Sort by performance and add best remaining
            remaining_agents.sort(
                key=lambda a: a.performance_metrics.get("success_rate", 0.5),
                reverse=True
            )
            selected_agents.extend(remaining_agents[:3 - len(selected_agents)])
        
        self.logger.info(f"Selected {len(selected_agents)} agents for debate: {[a.name for a in selected_agents]}")
        return selected_agents
    
    def _assign_debate_roles(self, agents: List[BaseAgent]) -> Dict[BaseAgent, DebateRole]:
        """Assign roles to agents based on their strengths."""
        roles = {}
        
        if len(agents) >= 2:
            # Assign proposer to agent best suited for the task type
            weights = self.orchestrator._agent_weights
            best_proposer = max(
                agents,
                key=lambda a: weights.get(a.agent_type, {}).get(self.task.task_type, 0.5)
            )
            roles[best_proposer] = DebateRole.PROPOSER
            
            # Assign challenger to a different agent type if possible
            remaining_agents = [a for a in agents if a != best_proposer]
            challenger = next(
                (a for a in remaining_agents if a.agent_type != best_proposer.agent_type),
                remaining_agents[0] if remaining_agents else None
            )
            if challenger:
                roles[challenger] = DebateRole.CHALLENGER
            
            # Assign synthesizer if we have a third agent
            if len(agents) >= 3:
                synthesizer_candidates = [a for a in agents if a not in roles]
                if synthesizer_candidates:
                    roles[synthesizer_candidates[0]] = DebateRole.SYNTHESIZER
        
        return roles
    
    async def _get_initial_proposal(self, proposer: BaseAgent) -> DebateArgument:
        """Get initial proposal from the proposer agent."""
        context = TaskContext(
            task_id=f"{self.task.task_id}_proposal",
            task_type=self.task.task_type,
            conversation_id=self.task.conversation_id,
            session_id=self.task.session_id,
            agent_role=AgentRole.PRIMARY,
            max_tokens=self.task.max_tokens,
            timeout_seconds=self.task.timeout_seconds
        )
        
        proposal_prompt = f"""You are participating in a structured debate about this task:

{self.task.description}

As the PROPOSER, please provide your initial solution or approach. Your proposal should include:

1. **Core Solution**: Your main approach to solving this task
2. **Reasoning**: Why you believe this is the best approach
3. **Evidence**: Any supporting arguments or examples
4. **Potential Concerns**: What challenges or limitations you acknowledge
5. **Implementation Details**: Specific steps or considerations

Be thorough but concise. This proposal will be challenged by other agents, so make it robust and well-reasoned."""

        response = await proposer.generate(proposal_prompt, context)
        
        # Track cost and tokens
        if response.cost_usd:
            self.total_cost += response.cost_usd
        if response.tokens_used:
            self.total_tokens += response.tokens_used
        
        return DebateArgument(
            agent_name=proposer.name,
            agent_type=proposer.agent_type.value,
            role=DebateRole.PROPOSER,
            argument_type=ArgumentType.INITIAL_PROPOSAL,
            content=response.content,
            confidence_score=response.confidence_score,
            metadata={"response_metadata": response.metadata}
        )
    
    async def _get_counter_argument(self, challenger: BaseAgent, proposal: DebateArgument) -> DebateArgument:
        """Get counter-argument from challenger agent."""
        context = TaskContext(
            task_id=f"{self.task.task_id}_challenge",
            task_type=self.task.task_type,
            conversation_id=self.task.conversation_id,
            session_id=self.task.session_id,
            agent_role=AgentRole.REVIEWER,
            max_tokens=self.task.max_tokens,
            timeout_seconds=self.task.timeout_seconds
        )
        
        challenge_prompt = f"""You are participating in a structured debate. Here is the original task:

{self.task.description}

The PROPOSER has suggested this solution:

{proposal.content}

As the CHALLENGER, please provide a thorough critique and alternative approach. Your response should include:

1. **Critical Analysis**: What are the weaknesses or problems with the proposed solution?
2. **Alternative Approach**: What different solution or modifications do you recommend?
3. **Supporting Evidence**: Why is your approach better or more suitable?
4. **Risk Assessment**: What risks does the original proposal have that your approach avoids?
5. **Implementation Advantages**: How is your approach easier, faster, or more reliable?

Be constructive but rigorous in your critique. Present a compelling alternative that addresses the task better."""

        response = await challenger.generate(challenge_prompt, context)
        
        # Track cost and tokens
        if response.cost_usd:
            self.total_cost += response.cost_usd
        if response.tokens_used:
            self.total_tokens += response.tokens_used
        
        return DebateArgument(
            agent_name=challenger.name,
            agent_type=challenger.agent_type.value,
            role=DebateRole.CHALLENGER,
            argument_type=ArgumentType.COUNTER_ARGUMENT,
            content=response.content,
            confidence_score=response.confidence_score,
            addresses_argument_id=proposal.id,
            metadata={"response_metadata": response.metadata}
        )
    
    async def _get_rebuttal(self, proposer: BaseAgent, proposal: DebateArgument, 
                          counter_argument: DebateArgument) -> DebateArgument:
        """Get rebuttal from original proposer."""
        context = TaskContext(
            task_id=f"{self.task.task_id}_rebuttal",
            task_type=self.task.task_type,
            conversation_id=self.task.conversation_id,
            session_id=self.task.session_id,
            agent_role=AgentRole.PRIMARY,
            max_tokens=self.task.max_tokens,
            timeout_seconds=self.task.timeout_seconds
        )
        
        rebuttal_prompt = f"""You are continuing a structured debate. Here is the original task:

{self.task.description}

Your original proposal was:

{proposal.content}

The CHALLENGER responded with this critique and alternative:

{counter_argument.content}

As the original PROPOSER, please provide a rebuttal that:

1. **Defends Your Approach**: Address the specific criticisms raised
2. **Identifies Flaws**: Point out problems with the challenger's alternative
3. **Provides Evidence**: Offer concrete examples or reasoning for your position
4. **Refined Solution**: Improve your original proposal based on valid points
5. **Comparative Analysis**: Explain why your approach is still superior overall

Be open to valid criticism but defend your position where you believe it's stronger."""

        response = await proposer.generate(rebuttal_prompt, context)
        
        # Track cost and tokens
        if response.cost_usd:
            self.total_cost += response.cost_usd
        if response.tokens_used:
            self.total_tokens += response.tokens_used
        
        return DebateArgument(
            agent_name=proposer.name,
            agent_type=proposer.agent_type.value,
            role=DebateRole.PROPOSER,
            argument_type=ArgumentType.REBUTTAL,
            content=response.content,
            confidence_score=response.confidence_score,
            addresses_argument_id=counter_argument.id,
            metadata={"response_metadata": response.metadata}
        )
    
    async def _synthesize_consensus(self, synthesizer: BaseAgent, 
                                  arguments: List[DebateArgument]) -> DebateArgument:
        """Create synthesis from all arguments."""
        context = TaskContext(
            task_id=f"{self.task.task_id}_synthesis",
            task_type=self.task.task_type,
            conversation_id=self.task.conversation_id,
            session_id=self.task.session_id,
            agent_role=AgentRole.SPECIALIST,
            max_tokens=self.task.max_tokens,
            timeout_seconds=self.task.timeout_seconds
        )
        
        # Build comprehensive prompt with all arguments
        arguments_text = ""
        for i, arg in enumerate(arguments, 1):
            arguments_text += f"""
=== Argument {i} ({arg.role.value} - {arg.argument_type.value}) ===
Agent: {arg.agent_name}
Content: {arg.content}

"""
        
        synthesis_prompt = f"""You are synthesizing a debate about this task:

{self.task.description}

Here are all the arguments presented during the debate:

{arguments_text}

As the SYNTHESIZER, create a final consensus solution that:

1. **Integrated Solution**: Combines the best elements from all perspectives
2. **Addresses Concerns**: Resolves the main criticisms and concerns raised
3. **Balanced Approach**: Incorporates valid points from all participants
4. **Implementation Plan**: Provides a clear, actionable solution
5. **Risk Mitigation**: Addresses potential problems identified in the debate
6. **Consensus Reasoning**: Explains how this synthesis resolves the disagreements

Create a solution that all participants could reasonably accept while solving the original task effectively."""

        response = await synthesizer.generate(synthesis_prompt, context)
        
        # Track cost and tokens
        if response.cost_usd:
            self.total_cost += response.cost_usd
        if response.tokens_used:
            self.total_tokens += response.tokens_used
        
        return DebateArgument(
            agent_name=synthesizer.name,
            agent_type=synthesizer.agent_type.value,
            role=DebateRole.SYNTHESIZER,
            argument_type=ArgumentType.SYNTHESIS,
            content=response.content,
            confidence_score=response.confidence_score,
            metadata={"response_metadata": response.metadata}
        )
    
    def _calculate_consensus_score(self, arguments: List[DebateArgument]) -> float:
        """Calculate consensus score based on argument patterns."""
        if len(arguments) < 2:
            return 0.0
        
        # Simple heuristic: higher confidence scores indicate stronger consensus
        confidence_scores = [arg.confidence_score for arg in arguments if arg.confidence_score is not None]
        
        if not confidence_scores:
            return 0.5  # Default moderate consensus
        
        # Average confidence across all arguments
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Bonus for synthesis arguments (indicates resolution)
        synthesis_bonus = 0.2 if any(arg.argument_type == ArgumentType.SYNTHESIS for arg in arguments) else 0.0
        
        # Penalty for many counter-arguments (indicates ongoing disagreement)
        counter_penalty = len([arg for arg in arguments if arg.argument_type == ArgumentType.COUNTER_ARGUMENT]) * 0.1
        
        consensus_score = min(1.0, max(0.0, avg_confidence + synthesis_bonus - counter_penalty))
        
        return consensus_score
    
    async def execute_debate(self) -> 'TaskResult':
        """Execute the full debate protocol."""
        from .orchestrator import TaskResult, OrchestrationStrategy
        
        self.start_time = time.time()
        
        with log_context(self.debate_id, self.task.session_id):
            self.logger.info(f"Starting debate for task {self.task.task_id}")
            
            try:
                # Select and assign agents
                agents = self._select_debate_agents()
                roles = self._assign_debate_roles(agents)
                self.participating_agents = agents
                
                if len(roles) < 2:
                    raise ValueError("Need at least 2 agents with assigned roles for debate")
                
                proposer = next(agent for agent, role in roles.items() if role == DebateRole.PROPOSER)
                challenger = next(agent for agent, role in roles.items() if role == DebateRole.CHALLENGER)
                synthesizer = next((agent for agent, role in roles.items() if role == DebateRole.SYNTHESIZER), None)
                
                # Execute debate rounds
                for round_num in range(1, self.max_rounds + 1):
                    round_start = time.time()
                    round_arguments = []
                    
                    self.logger.info(f"Starting debate round {round_num}")
                    
                    if round_num == 1:
                        # Initial proposal
                        async with AsyncPerformanceLogger(self.logger, f"debate_proposal", round=round_num):
                            proposal = await self._get_initial_proposal(proposer)
                        round_arguments.append(proposal)
                        self.arguments.append(proposal)
                        
                        # Counter-argument
                        async with AsyncPerformanceLogger(self.logger, f"debate_challenge", round=round_num):
                            counter = await self._get_counter_argument(challenger, proposal)
                        round_arguments.append(counter)
                        self.arguments.append(counter)
                    else:
                        # Subsequent rounds: rebuttals and clarifications
                        last_argument = self.arguments[-1]
                        if last_argument.role == DebateRole.CHALLENGER:
                            # Proposer rebuts
                            async with AsyncPerformanceLogger(self.logger, f"debate_rebuttal", round=round_num):
                                rebuttal = await self._get_rebuttal(proposer, self.arguments[0], last_argument)
                            round_arguments.append(rebuttal)
                            self.arguments.append(rebuttal)
                        else:
                            # Challenger provides follow-up
                            async with AsyncPerformanceLogger(self.logger, f"debate_followup", round=round_num):
                                followup = await self._get_counter_argument(challenger, last_argument)
                            round_arguments.append(followup)
                            self.arguments.append(followup)
                    
                    # Calculate round consensus
                    consensus_score = self._calculate_consensus_score(round_arguments)
                    round_duration = (time.time() - round_start) * 1000
                    
                    # Create round summary
                    round_summary = f"Round {round_num}: {len(round_arguments)} arguments, consensus: {consensus_score:.2f}"
                    
                    debate_round = DebateRound(
                        round_number=round_num,
                        arguments=round_arguments,
                        round_summary=round_summary,
                        consensus_score=consensus_score,
                        round_duration_ms=round_duration
                    )
                    self.rounds.append(debate_round)
                    
                    self.logger.info(f"Completed round {round_num} - Consensus: {consensus_score:.2f}")
                    
                    # Check for early consensus
                    if consensus_score >= self.consensus_threshold:
                        self.logger.info(f"Early consensus reached at round {round_num}")
                        break
                
                # Final synthesis if we have a synthesizer
                final_content = ""
                final_confidence = 0.0
                
                if synthesizer and len(self.arguments) >= 2:
                    self.logger.info("Creating final synthesis")
                    async with AsyncPerformanceLogger(self.logger, "debate_synthesis"):
                        synthesis = await self._synthesize_consensus(synthesizer, self.arguments)
                    self.arguments.append(synthesis)
                    final_content = synthesis.content
                    final_confidence = synthesis.confidence_score or 0.0
                else:
                    # Use the last argument as final result
                    last_arg = self.arguments[-1] if self.arguments else None
                    if last_arg:
                        final_content = last_arg.content
                        final_confidence = last_arg.confidence_score or 0.0
                
                # Calculate final consensus
                final_consensus_score = self._calculate_consensus_score(self.arguments)
                total_duration = (time.time() - self.start_time) * 1000
                
                # Create debate result
                debate_result = DebateResult(
                    debate_id=self.debate_id,
                    task_id=self.task.task_id,
                    success=final_consensus_score >= self.consensus_threshold,
                    final_consensus=final_content,
                    confidence_score=final_confidence,
                    rounds=self.rounds,
                    participating_agents=[agent.name for agent in self.participating_agents],
                    total_duration_ms=total_duration,
                    total_cost_usd=self.total_cost,
                    total_tokens=self.total_tokens,
                    metadata={
                        "final_consensus_score": final_consensus_score,
                        "total_arguments": len(self.arguments),
                        "roles_assigned": {agent.name: role.value for agent, role in roles.items()}
                    }
                )
                
                # Convert to TaskResult
                task_result = TaskResult(
                    task_id=self.task.task_id,
                    success=debate_result.success,
                    content=debate_result.final_consensus,
                    execution_time_ms=debate_result.total_duration_ms,
                    total_cost_usd=debate_result.total_cost_usd,
                    total_tokens=debate_result.total_tokens,
                    strategy_used=OrchestrationStrategy.DEBATE,
                    metadata={
                        "debate_result": debate_result,
                        "consensus_score": final_consensus_score,
                        "rounds_completed": len(self.rounds),
                        "agents_participated": debate_result.participating_agents
                    }
                )
                
                self.logger.info(
                    f"Debate completed - Success: {debate_result.success}, "
                    f"Consensus: {final_consensus_score:.2f}, "
                    f"Rounds: {len(self.rounds)}, "
                    f"Cost: ${self.total_cost:.4f}"
                )
                
                return task_result
                
            except Exception as e:
                total_duration = (time.time() - self.start_time) * 1000
                self.logger.error(f"Debate execution failed: {e}")
                
                return TaskResult(
                    task_id=self.task.task_id,
                    success=False,
                    content="",
                    execution_time_ms=total_duration,
                    total_cost_usd=self.total_cost,
                    total_tokens=self.total_tokens,
                    strategy_used=OrchestrationStrategy.DEBATE,
                    error_message=str(e),
                    metadata={
                        "debate_id": self.debate_id,
                        "rounds_completed": len(self.rounds),
                        "arguments_made": len(self.arguments)
                    }
                )