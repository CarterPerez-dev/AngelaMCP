"""
Rich Terminal UI for AngelaMCP Collaboration.

This module provides a real-time visual interface showing multi-agent
collaboration in action. Users can see agents debating, voting, and
reaching consensus live in the terminal.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.align import Align
from rich import box

from src.orchestrator.collaboration import CollaborationOrchestrator, CollaborationRequest, CollaborationResult
from src.utils.logger import get_logger

logger = get_logger("ui.collaboration")


@dataclass
class AgentStatus:
    """Status information for an agent in the UI."""
    name: str
    emoji: str
    status: str = "idle"
    current_task: str = ""
    last_update: float = field(default_factory=time.time)
    messages: List[str] = field(default_factory=list)
    is_active: bool = False


class CollaborationUI:
    """
    Rich terminal UI for real-time multi-agent collaboration display.
    
    Shows three agent panels with live updates as they debate, critique,
    and vote on solutions. Makes the collaboration process visually engaging.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the collaboration UI."""
        self.console = console or Console()
        self.layout = self._create_layout()
        
        # Agent status tracking
        self.agents = {
            "claude_code": AgentStatus("Claude Code", "🔧", status="ready"),
            "openai": AgentStatus("OpenAI", "🧠", status="ready"), 
            "gemini": AgentStatus("Gemini", "✨", status="ready")
        }
        
        # UI state
        self.current_phase = "idle"
        self.progress_message = ""
        self.collaboration_id = ""
        self.start_time = 0.0
        
        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=True
        )
        
        self.logger = get_logger("ui")
    
    def _create_layout(self) -> Layout:
        """Create the main terminal layout."""
        layout = Layout()
        
        # Split into header, body, and footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into agent panels and status
        layout["body"].split_row(
            Layout(name="agents", ratio=3),
            Layout(name="status", ratio=2)
        )
        
        # Split agents into three columns
        layout["agents"].split_row(
            Layout(name="claude", ratio=1),
            Layout(name="openai", ratio=1),
            Layout(name="gemini", ratio=1)
        )
        
        # Split status into progress and results
        layout["status"].split_column(
            Layout(name="progress", ratio=1),
            Layout(name="results", ratio=2)
        )
        
        return layout
    
    def update_header(self) -> None:
        """Update the header panel."""
        if self.collaboration_id:
            elapsed = time.time() - self.start_time if self.start_time else 0
            title = f"🎭 AngelaMCP Collaboration [{self.collaboration_id[:8]}] - {elapsed:.1f}s"
        else:
            title = "🎭 AngelaMCP - Multi-Agent Collaboration Platform"
        
        header_panel = Panel(
            Align.center(title),
            style="bold blue",
            box=box.ROUNDED
        )
        self.layout["header"].update(header_panel)
    
    def update_agent_panel(self, agent_key: str) -> None:
        """Update an individual agent panel."""
        agent = self.agents[agent_key]
        
        # Create status indicator
        status_color = "green" if agent.status == "ready" else "yellow" if agent.is_active else "red"
        status_text = f"[{status_color}]●[/{status_color}] {agent.status.title()}"
        
        # Create content
        content_lines = [
            f"{agent.emoji} [bold]{agent.name}[/bold]",
            f"Status: {status_text}",
            ""
        ]
        
        if agent.current_task:
            content_lines.append(f"[italic]{agent.current_task}[/italic]")
            content_lines.append("")
        
        # Add recent messages (last 5)
        if agent.messages:
            content_lines.append("[dim]Recent activity:[/dim]")
            for msg in agent.messages[-5:]:
                content_lines.append(f"[dim]• {msg}[/dim]")
        
        content = "\n".join(content_lines)
        
        # Style based on activity
        border_style = "green" if agent.is_active else "blue" if agent.status == "ready" else "yellow"
        
        panel = Panel(
            content,
            title=f" {agent.name} ",
            border_style=border_style,
            box=box.ROUNDED
        )
        
        self.layout[agent_key.replace("_", "")].update(panel)
    
    def update_progress_panel(self) -> None:
        """Update the progress panel."""
        content_lines = [
            f"[bold]Current Phase:[/bold] {self.current_phase.replace('_', ' ').title()}",
            ""
        ]
        
        if self.progress_message:
            content_lines.append(f"[italic]{self.progress_message}[/italic]")
        
        content_lines.extend([
            "",
            "[dim]Process Flow:[/dim]",
            "[dim]1. 💡 Agents propose solutions[/dim]",
            "[dim]2. 🔍 Agents critique each other[/dim]", 
            "[dim]3. ✨ Agents refine proposals[/dim]",
            "[dim]4. 🗳️ Weighted voting[/dim]",
            "[dim]5. 🏆 Final decision[/dim]"
        ])
        
        panel = Panel(
            "\n".join(content_lines),
            title=" Progress ",
            border_style="yellow",
            box=box.ROUNDED
        )
        
        self.layout["progress"].update(panel)
    
    def update_results_panel(self, result: Optional[CollaborationResult] = None) -> None:
        """Update the results panel."""
        if result is None:
            content = "[dim]Collaboration results will appear here...[/dim]"
        else:
            content_lines = []
            
            if result.success:
                content_lines.extend([
                    f"[green]✅ Success![/green]",
                    f"[bold]Winner:[/bold] {result.chosen_agent}",
                    f"[bold]Duration:[/bold] {result.total_duration:.1f}s",
                    f"[bold]Consensus:[/bold] {'Yes' if result.consensus_reached else 'No'}",
                    ""
                ])
                
                if result.voting_result:
                    content_lines.append("[bold]Vote Breakdown:[/bold]")
                    for score in result.voting_result.proposal_scores:
                        status_emoji = "🏆" if score.proposal.agent_name == result.chosen_agent else "🚫" if score.claude_vetoed else ""
                        content_lines.append(
                            f"• {score.proposal.agent_name}: {score.weighted_score:.1f} "
                            f"(✅{score.approval_count} ❌{score.rejection_count}) {status_emoji}"
                        )
            else:
                content_lines.extend([
                    "[red]❌ Collaboration Failed[/red]",
                    f"[bold]Error:[/bold] {result.error_message}",
                    f"[bold]Duration:[/bold] {result.total_duration:.1f}s"
                ])
            
            content = "\n".join(content_lines)
        
        panel = Panel(
            content,
            title=" Results ",
            border_style="green" if result and result.success else "red" if result else "blue",
            box=box.ROUNDED
        )
        
        self.layout["results"].update(panel)
    
    def update_footer(self) -> None:
        """Update the footer panel."""
        footer_text = "Press Ctrl+C to stop | 🔧 Claude Code (Senior Dev) | 🧠 OpenAI (Reviewer) | ✨ Gemini (Researcher)"
        
        footer_panel = Panel(
            Align.center(footer_text),
            style="dim",
            box=box.ROUNDED
        )
        self.layout["footer"].update(footer_panel)
    
    def refresh_display(self) -> None:
        """Refresh the entire display."""
        self.update_header()
        
        for agent_key in self.agents.keys():
            self.update_agent_panel(agent_key)
        
        self.update_progress_panel()
        self.update_results_panel()
        self.update_footer()
    
    def add_agent_message(self, agent_name: str, message: str) -> None:
        """Add a message to an agent's activity log."""
        agent_key = agent_name.lower().replace(" ", "_")
        if agent_key in self.agents:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.agents[agent_key].messages.append(f"[{timestamp}] {message}")
            self.agents[agent_key].last_update = time.time()
    
    def set_agent_status(self, agent_name: str, status: str, task: str = "") -> None:
        """Set an agent's status and current task."""
        agent_key = agent_name.lower().replace(" ", "_")
        if agent_key in self.agents:
            self.agents[agent_key].status = status
            self.agents[agent_key].current_task = task
            self.agents[agent_key].is_active = status in ["working", "thinking", "voting", "critiquing"]
            self.agents[agent_key].last_update = time.time()
    
    def set_phase(self, phase: str, message: str = "") -> None:
        """Set the current collaboration phase."""
        self.current_phase = phase
        self.progress_message = message
    
    async def run_collaboration_with_ui(
        self,
        orchestrator: CollaborationOrchestrator,
        request: CollaborationRequest
    ) -> CollaborationResult:
        """
        Run a collaboration with live UI updates.
        
        Args:
            orchestrator: The collaboration orchestrator
            request: The collaboration request
            
        Returns:
            CollaborationResult
        """
        self.collaboration_id = ""
        self.start_time = time.time()
        
        # Set up status callback for real-time updates
        def status_callback(message: str) -> None:
            # Parse status messages to update UI appropriately
            if "Starting collaboration" in message:
                self.collaboration_id = message.split()[2] if len(message.split()) > 2 else "unknown"
                self.set_phase("initializing", "Setting up agents...")
            elif "debate mode" in message:
                self.set_phase("debate_setup", "Preparing for debate...")
            elif "Getting proposal from" in message:
                agent_name = message.split("from")[1].strip() if "from" in message else "unknown"
                self.set_agent_status(agent_name, "thinking", "Creating proposal...")
                self.add_agent_message(agent_name, "Generating solution proposal")
            elif "critique" in message.lower():
                if "reviewing" in message:
                    parts = message.split()
                    if len(parts) >= 2:
                        critic = parts[1]
                        self.set_agent_status(critic, "critiquing", "Reviewing proposals...")
                        self.add_agent_message(critic, "Analyzing and critiquing solutions")
            elif "voting" in message.lower():
                if "voting on" in message:
                    agent_name = message.split()[1] if len(message.split()) > 1 else "unknown"
                    self.set_agent_status(agent_name, "voting", "Evaluating proposals...")
                    self.add_agent_message(agent_name, "Casting weighted vote")
                elif "Starting voting" in message:
                    self.set_phase("voting", "Agents voting on proposals...")
            elif "completed" in message.lower():
                winner = message.split(":")[-1].strip() if ":" in message else "unknown"
                self.set_phase("completed", f"Winner: {winner}")
                # Reset all agents to ready
                for agent_name in ["Claude Code", "OpenAI", "Gemini"]:
                    self.set_agent_status(agent_name, "ready", "")
        
        # Set the callback
        orchestrator.status_callback = status_callback
        
        # Initialize display
        self.refresh_display()
        
        # Run collaboration with live display
        with Live(self.layout, console=self.console, refresh_per_second=4) as live:
            try:
                # Start collaboration
                result = await orchestrator.collaborate(request)
                
                # Update final display
                self.update_results_panel(result)
                
                # Show completion message
                if result.success:
                    self.add_agent_message(result.chosen_agent, f"🏆 Solution selected by consensus!")
                else:
                    self.set_phase("failed", "Collaboration failed")
                
                self.refresh_display()
                
                return result
                
            except KeyboardInterrupt:
                self.set_phase("cancelled", "Collaboration cancelled by user")
                self.refresh_display()
                raise
            except Exception as e:
                self.set_phase("error", f"Error: {str(e)}")
                self.refresh_display()
                raise
    
    def create_summary_table(self, result: CollaborationResult) -> Table:
        """Create a summary table for the collaboration result."""
        table = Table(title="🎭 Collaboration Summary", box=box.ROUNDED)
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Task", result.request.task_description[:50] + "..." if len(result.request.task_description) > 50 else result.request.task_description)
        table.add_row("Winner", result.chosen_agent or "None")
        table.add_row("Duration", f"{result.total_duration:.1f}s")
        table.add_row("Consensus", "✅ Yes" if result.consensus_reached else "❌ No")
        table.add_row("Mode", result.request.mode.value.replace("_", " ").title())
        
        if result.voting_result:
            table.add_row("Total Votes", str(sum(len(score.votes) for score in result.voting_result.proposal_scores)))
            if result.voting_result.claude_used_veto:
                table.add_row("Claude Veto", "🚫 Used")
        
        return table
    
    async def demo_collaboration(self, task: str = "Create a Python function to calculate Fibonacci numbers") -> None:
        """Run a demo collaboration for testing the UI."""
        # Create a mock orchestrator for demo
        from src.orchestrator.collaboration import CollaborationOrchestrator
        
        orchestrator = CollaborationOrchestrator()
        request = CollaborationRequest(
            task_description=task,
            timeout_minutes=5
        )
        
        try:
            result = await self.run_collaboration_with_ui(orchestrator, request)
            
            # Show final summary
            self.console.print("\n")
            summary_table = self.create_summary_table(result)
            self.console.print(summary_table)
            
            if result.success and result.final_solution:
                self.console.print(f"\n[bold green]📋 Final Solution:[/bold green]")
                self.console.print(Panel(result.final_solution[:500] + "..." if len(result.final_solution) > 500 else result.final_solution))
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️ Demo cancelled by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]❌ Demo failed: {e}[/red]")


# Utility functions for quick UI operations
def quick_collaboration(task: str, timeout_minutes: int = 5) -> None:
    """Quick function to run a collaboration with UI."""
    async def _run():
        ui = CollaborationUI()
        await ui.demo_collaboration(task)
    
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Demo the UI
    quick_collaboration("Create a REST API for managing todo items")