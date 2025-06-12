"""
Display components for AngelaMCP Rich terminal UI.

This module provides specialized display components for visualizing different
aspects of multi-agent collaboration including debates, voting, and performance.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.align import Align

from src.orchestration.debate import DebateResult, DebateArgument, DebateRole, ArgumentType
from src.orchestration.voting import VoteResult, Vote, VoteType
from src.agents.base import BaseAgent, AgentResponse
from src.utils.logger import get_logger

logger = get_logger("ui.display")


@dataclass
class DisplayTheme:
    """Theme configuration for display components."""
    primary: str = "bright_blue"
    secondary: str = "bright_cyan"
    success: str = "bright_green"
    warning: str = "bright_yellow"
    error: str = "bright_red"
    info: str = "bright_white"
    muted: str = "dim white"
    
    # Agent-specific colors
    claude_color: str = "bright_magenta"
    openai_color: str = "bright_green"
    gemini_color: str = "bright_blue"


class DisplayManager:
    """
    Manager for creating Rich display components.
    
    I'm providing specialized display components for different aspects
    of the multi-agent system with consistent styling and formatting.
    """
    
    def __init__(self, console: Optional[Console] = None, theme: Optional[DisplayTheme] = None):
        self.console = console or Console()
        self.theme = theme or DisplayTheme()
        self.logger = get_logger("ui.display_manager")
    
    def get_agent_color(self, agent_type: str) -> str:
        """Get color for an agent type."""
        color_map = {
            "claude_code": self.theme.claude_color,
            "openai": self.theme.openai_color,
            "gemini": self.theme.gemini_color
        }
        return color_map.get(agent_type, self.theme.info)
    
    def format_timestamp(self, timestamp: float) -> str:
        """Format timestamp for display."""
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    
    def format_duration(self, duration_ms: float) -> str:
        """Format duration for display."""
        if duration_ms < 1000:
            return f"{duration_ms:.0f}ms"
        elif duration_ms < 60000:
            return f"{duration_ms/1000:.1f}s"
        else:
            return f"{duration_ms/60000:.1f}m"
    
    def create_agent_panel(self, agent: BaseAgent, detailed: bool = False) -> Panel:
        """Create a panel displaying agent information."""
        content = []
        
        # Basic info
        info_table = Table(show_header=False, box=None)
        info_table.add_column(style="bold")
        info_table.add_column()
        
        info_table.add_row("Name:", agent.name)
        info_table.add_row("Type:", agent.agent_type.value)
        
        # Performance metrics
        metrics = agent.performance_metrics
        info_table.add_row("Requests:", str(metrics["total_requests"]))
        info_table.add_row("Cost:", f"${metrics['total_cost_usd']:.4f}")
        info_table.add_row("Uptime:", f"{metrics['uptime_seconds']:.0f}s")
        
        content.append(info_table)
        
        # Capabilities if detailed
        if detailed and agent.capabilities:
            cap_tree = Tree("Capabilities", style=self.theme.secondary)
            for cap in agent.capabilities:
                cap_node = cap_tree.add(f"📋 {cap.name}")
                cap_node.add(f"Description: {cap.description}")
                if cap.supported_formats:
                    cap_node.add(f"Formats: {', '.join(cap.supported_formats)}")
                if cap.cost_per_request:
                    cap_node.add(f"Cost: ${cap.cost_per_request:.4f}")
            
            content.append(cap_tree)
        
        agent_color = self.get_agent_color(agent.agent_type.value)
        return Panel(
            Group(*content),
            title=f"Agent: {agent.name}",
            style=agent_color
        )
    
    def create_debate_panel(self, debate_result: DebateResult, detailed: bool = True) -> Panel:
        """Create a panel displaying debate results."""
        content = []
        
        # Debate summary
        summary_table = Table(title="Debate Summary", style=self.theme.primary)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", style=self.theme.success)
        
        summary_table.add_row("Debate ID", debate_result.debate_id[:8])
        summary_table.add_row("Success", "✅ Yes" if debate_result.success else "❌ No")
        summary_table.add_row("Confidence", f"{debate_result.confidence_score:.2f}")
        summary_table.add_row("Rounds", str(len(debate_result.rounds)))
        summary_table.add_row("Duration", self.format_duration(debate_result.total_duration_ms))
        summary_table.add_row("Cost", f"${debate_result.total_cost_usd:.4f}")
        summary_table.add_row("Participants", ", ".join(debate_result.participating_agents))
        
        content.append(summary_table)
        
        if detailed:
            # Debate flow
            debate_tree = Tree("Debate Flow", style=self.theme.secondary)
            
            for round_data in debate_result.rounds:
                round_node = debate_tree.add(
                    f"Round {round_data.round_number} "
                    f"(consensus: {round_data.consensus_score:.2f})"
                )
                
                for arg in round_data.arguments:
                    agent_color = self.get_agent_color(arg.agent_type)
                    arg_text = f"{arg.agent_name} ({arg.role.value}): {arg.argument_type.value}"
                    arg_node = round_node.add(arg_text, style=agent_color)
                    
                    # Add truncated content
                    preview = arg.content[:100] + "..." if len(arg.content) > 100 else arg.content
                    arg_node.add(Text(preview, style="dim"))
                    
                    if arg.confidence_score:
                        arg_node.add(f"Confidence: {arg.confidence_score:.2f}")
            
            content.append(debate_tree)
            
            # Final consensus
            if debate_result.final_consensus:
                consensus_text = debate_result.final_consensus
                if len(consensus_text) > 300:
                    consensus_text = consensus_text[:300] + "..."
                
                content.append(
                    Panel(
                        Text(consensus_text, style=self.theme.info),
                        title="Final Consensus",
                        style=self.theme.success if debate_result.success else self.theme.warning
                    )
                )
        
        return Panel(
            Group(*content),
            title="Debate Results",
            style=self.theme.primary
        )
    
    def create_voting_panel(self, vote_result: VoteResult, detailed: bool = True) -> Panel:
        """Create a panel displaying voting results."""
        content = []
        
        # Voting summary
        summary_table = Table(title="Vote Summary", style=self.theme.primary)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", style=self.theme.success)
        
        summary_table.add_row("Vote ID", vote_result.vote_id[:8])
        summary_table.add_row("Decision", vote_result.final_decision.value.title())
        summary_table.add_row("Confidence", f"{vote_result.confidence_score:.2f}")
        summary_table.add_row("Method", vote_result.voting_method.value.replace("_", " ").title())
        summary_table.add_row("Duration", self.format_duration(vote_result.duration_ms))
        
        if vote_result.has_veto:
            summary_table.add_row("Veto", "⚠️ Yes", style=self.theme.warning)
        
        content.append(summary_table)
        
        # Vote breakdown
        breakdown_table = Table(title="Vote Breakdown", style=self.theme.secondary)
        breakdown_table.add_column("Vote Type", style="bold")
        breakdown_table.add_column("Weight", style=self.theme.success)
        breakdown_table.add_column("Percentage", style=self.theme.info)
        
        total_weight = vote_result.total_weight or 1.0
        
        breakdown_table.add_row(
            "Approve", 
            f"{vote_result.approve_weight:.2f}",
            f"{(vote_result.approve_weight/total_weight)*100:.1f}%"
        )
        breakdown_table.add_row(
            "Reject",
            f"{vote_result.reject_weight:.2f}", 
            f"{(vote_result.reject_weight/total_weight)*100:.1f}%"
        )
        breakdown_table.add_row(
            "Abstain",
            f"{vote_result.abstain_weight:.2f}",
            f"{(vote_result.abstain_weight/total_weight)*100:.1f}%"
        )
        
        content.append(breakdown_table)
        
        if detailed and vote_result.votes:
            # Individual votes
            votes_tree = Tree("Individual Votes", style=self.theme.secondary)
            
            for vote in vote_result.votes:
                agent_color = self.get_agent_color(vote.agent_type)
                vote_icon = {
                    VoteType.APPROVE: "✅",
                    VoteType.REJECT: "❌", 
                    VoteType.ABSTAIN: "🤷",
                    VoteType.VETO: "⛔"
                }.get(vote.vote_type, "❓")
                
                vote_text = f"{vote_icon} {vote.agent_name} - {vote.vote_type.value.title()}"
                vote_node = votes_tree.add(vote_text, style=agent_color)
                
                vote_node.add(f"Weight: {vote.weight:.2f}")
                vote_node.add(f"Confidence: {vote.confidence:.2f}")
                
                if vote.reasoning:
                    reasoning_preview = vote.reasoning[:150] + "..." if len(vote.reasoning) > 150 else vote.reasoning
                    vote_node.add(Text(reasoning_preview, style="dim"))
            
            content.append(votes_tree)
            
            # Veto reasoning if present
            if vote_result.has_veto and vote_result.veto_reason:
                veto_text = vote_result.veto_reason
                if len(veto_text) > 200:
                    veto_text = veto_text[:200] + "..."
                
                content.append(
                    Panel(
                        Text(veto_text, style=self.theme.warning),
                        title="⛔ Veto Reasoning",
                        style=self.theme.error
                    )
                )
        
        return Panel(
            Group(*content),
            title="Voting Results", 
            style=self.theme.primary
        )
    
    def create_task_progress_panel(self, task_id: str, progress_info: Dict[str, Any]) -> Panel:
        """Create a panel showing task progress."""
        content = []
        
        # Task info
        info_table = Table(show_header=False, box=None)
        info_table.add_column(style="bold")
        info_table.add_column()
        
        info_table.add_row("Task ID:", task_id[:8])
        info_table.add_row("Type:", progress_info.get("type", "Unknown"))
        info_table.add_row("Strategy:", progress_info.get("strategy", "Unknown"))
        info_table.add_row("Status:", progress_info.get("status", "Unknown"))
        
        content.append(info_table)
        
        # Progress bar if available
        if "progress" in progress_info:
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
            )
            
            task = progress.add_task(
                "Processing...", 
                total=100, 
                completed=progress_info["progress"]
            )
            
            content.append(progress)
        
        # Active agents
        if "agents" in progress_info:
            agents_text = Text("Active Agents: ", style="bold")
            for i, agent in enumerate(progress_info["agents"]):
                if i > 0:
                    agents_text.append(", ")
                agents_text.append(agent, style=self.get_agent_color(agent))
            
            content.append(agents_text)
        
        return Panel(
            Group(*content),
            title="Task Progress",
            style=self.theme.info
        )
    
    def create_performance_panel(self, metrics: Dict[str, Any]) -> Panel:
        """Create a panel displaying performance metrics."""
        content = []
        
        # Key metrics table
        metrics_table = Table(title="Performance Metrics", style=self.theme.primary)
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", style=self.theme.success)
        metrics_table.add_column("Trend", style=self.theme.info)
        
        metrics_table.add_row(
            "Total Tasks",
            str(metrics.get("total_tasks", 0)),
            "📈"  # Could show actual trend
        )
        metrics_table.add_row(
            "Success Rate", 
            f"{metrics.get('success_rate', 0):.1%}",
            "📊"
        )
        metrics_table.add_row(
            "Avg Cost/Task",
            f"${metrics.get('average_cost_per_task', 0):.4f}",
            "💰"
        )
        metrics_table.add_row(
            "Tasks/Min",
            f"{metrics.get('tasks_per_minute', 0):.1f}",
            "⚡"
        )
        
        content.append(metrics_table)
        
        # Additional metrics if available
        if "agent_performance" in metrics:
            agent_perf = metrics["agent_performance"]
            agent_table = Table(title="Agent Performance", style=self.theme.secondary)
            agent_table.add_column("Agent", style="bold")
            agent_table.add_column("Requests", style=self.theme.info)
            agent_table.add_column("Success Rate", style=self.theme.success)
            agent_table.add_column("Avg Cost", style=self.theme.warning)
            
            for agent_name, perf in agent_perf.items():
                agent_table.add_row(
                    agent_name,
                    str(perf.get("requests", 0)),
                    f"{perf.get('success_rate', 0):.1%}",
                    f"${perf.get('avg_cost', 0):.4f}"
                )
            
            content.append(agent_table)
        
        return Panel(
            Group(*content),
            title="Performance Dashboard",
            style=self.theme.primary
        )
    
    def create_error_panel(self, error_message: str, details: Optional[Dict[str, Any]] = None) -> Panel:
        """Create a panel displaying error information."""
        content = [Text(error_message, style=self.theme.error)]
        
        if details:
            details_table = Table(show_header=False, box=None)
            details_table.add_column(style="bold")
            details_table.add_column(style=self.theme.muted)
            
            for key, value in details.items():
                details_table.add_row(f"{key}:", str(value))
            
            content.append(details_table)
        
        return Panel(
            Group(*content),
            title="⚠️ Error",
            style=self.theme.error
        )
