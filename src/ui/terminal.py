"""
Rich-based terminal UI for AngelaMCP.

This module implements a sophisticated terminal interface using Rich that provides
real-time visualization of multi-agent collaboration, including agent status,
task progress, debate visualization, and performance metrics.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.tree import Tree
from rich.status import Status
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown

from src.agents.base import BaseAgent, AgentResponse, agent_registry
from src.orchestration.orchestrator import OrchestrationTask, TaskResult, OrchestrationStrategy
from src.orchestration.debate import DebateResult, DebateArgument
from src.orchestration.voting import VoteResult, Vote
from src.utils.logger import get_logger

logger = get_logger("ui.terminal")


class UIMode(str, Enum):
    """UI display modes."""
    OVERVIEW = "overview"
    AGENTS = "agents"
    TASKS = "tasks"
    DEBATE = "debate"
    VOTING = "voting"
    LOGS = "logs"
    PERFORMANCE = "performance"


@dataclass
class UIState:
    """Current state of the UI."""
    mode: UIMode = UIMode.OVERVIEW
    active_task_id: Optional[str] = None
    active_debate_id: Optional[str] = None
    active_vote_id: Optional[str] = None
    show_details: bool = False
    auto_refresh: bool = True
    refresh_rate: float = 1.0
    
    # Data caches
    agent_statuses: Dict[str, Dict[str, Any]] = None
    task_results: Dict[str, TaskResult] = None
    debate_results: Dict[str, DebateResult] = None
    vote_results: Dict[str, VoteResult] = None
    
    def __post_init__(self):
        if self.agent_statuses is None:
            self.agent_statuses = {}
        if self.task_results is None:
            self.task_results = {}
        if self.debate_results is None:
            self.debate_results = {}
        if self.vote_results is None:
            self.vote_results = {}


class TerminalUI:
    """
    Rich-based terminal user interface for AngelaMCP.
    
    I'm implementing a comprehensive TUI that provides real-time monitoring
    and interaction with the multi-agent collaboration system.
    """
    
    def __init__(self, orchestration_engine):
        self.engine = orchestration_engine
        self.console = Console(record=True)
        self.state = UIState()
        self.logger = get_logger("ui.terminal")
        
        # UI components
        self.layout = self._create_layout()
        self.live = None
        self._running = False
        self._update_task = None
        
        # Color scheme
        self.colors = {
            "primary": "bright_blue",
            "secondary": "bright_cyan", 
            "success": "bright_green",
            "warning": "bright_yellow",
            "error": "bright_red",
            "info": "bright_white",
            "muted": "dim white"
        }
        
        # Status tracking
        self._last_update = time.time()
        self._update_count = 0
        
        self.logger.info("Terminal UI initialized")
    
    def _create_layout(self) -> Layout:
        """Create the main UI layout."""
        layout = Layout()
        
        # Split into header, body, and footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into sidebar and main content
        layout["body"].split_row(
            Layout(name="sidebar", size=30),
            Layout(name="main")
        )
        
        # Split main content into primary and secondary panels
        layout["main"].split_column(
            Layout(name="primary"),
            Layout(name="secondary", size=15)
        )
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create the header panel."""
        title = Text("AngelaMCP - Multi-Agent Collaboration Platform", style="bold bright_blue")
        subtitle = Text(f"Mode: {self.state.mode.value.title()}", style="dim")
        
        # Status indicators
        agent_count = len(agent_registry.get_all_agents())
        active_tasks = len(self.engine.orchestrator.get_active_tasks())
        
        status_text = Text()
        status_text.append(f"Agents: {agent_count} ", style="bright_green")
        status_text.append(f"Active Tasks: {active_tasks} ", style="bright_yellow")
        status_text.append(f"Updated: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        
        header_content = Group(
            Align.center(title),
            Columns([subtitle, Align.right(status_text)], expand=True)
        )
        
        return Panel(header_content, style="bright_blue")
    
    def _create_sidebar(self) -> Panel:
        """Create the sidebar with navigation and agent status."""
        sidebar_content = []
        
        # Navigation menu
        nav_tree = Tree("🎯 Navigation", style="bold bright_cyan")
        modes = [
            ("📊 Overview", UIMode.OVERVIEW),
            ("🤖 Agents", UIMode.AGENTS),
            ("📋 Tasks", UIMode.TASKS),
            ("💬 Debate", UIMode.DEBATE),
            ("🗳️  Voting", UIMode.VOTING),
            ("📈 Performance", UIMode.PERFORMANCE),
            ("📝 Logs", UIMode.LOGS)
        ]
        
        for label, mode in modes:
            style = "bold bright_green" if mode == self.state.mode else "dim"
            nav_tree.add(label, style=style)
        
        sidebar_content.append(nav_tree)
        
        # Agent status summary
        agents_tree = Tree("🤖 Agent Status", style="bold bright_cyan")
        
        for agent in agent_registry.get_all_agents():
            metrics = agent.performance_metrics
            status_icon = "🟢" if metrics["total_requests"] > 0 else "🟡"
            status_text = f"{status_icon} {agent.name}"
            
            agent_node = agents_tree.add(status_text)
            agent_node.add(f"Requests: {metrics['total_requests']}")
            agent_node.add(f"Cost: ${metrics['total_cost_usd']:.3f}")
            
        sidebar_content.append(agents_tree)
        
        # Quick actions
        actions_tree = Tree("⚡ Quick Actions", style="bold bright_cyan")
        actions_tree.add("🔄 Refresh (r)")
        actions_tree.add("🎯 Switch Mode (m)")
        actions_tree.add("📋 New Task (n)")
        actions_tree.add("🚪 Exit (q)")
        
        sidebar_content.append(actions_tree)
        
        return Panel(Group(*sidebar_content), title="Navigation", style="bright_cyan")
    
    def _create_overview_panel(self) -> Panel:
        """Create the main overview panel."""
        content = []
        
        # System overview table
        overview_table = Table(title="System Overview", style="bright_blue")
        overview_table.add_column("Metric", style="bold")
        overview_table.add_column("Value", style="bright_green")
        overview_table.add_column("Details", style="dim")
        
        # Get orchestrator metrics
        metrics = self.engine.orchestrator.get_performance_metrics()
        
        overview_table.add_row(
            "Total Tasks", 
            str(metrics["total_tasks"]),
            f"Success Rate: {metrics['success_rate']:.1%}"
        )
        overview_table.add_row(
            "Active Tasks",
            str(metrics["active_tasks_count"]),
            f"{metrics['tasks_per_minute']:.1f}/min"
        )
        overview_table.add_row(
            "Total Cost",
            f"${metrics['total_cost_usd']:.4f}",
            f"${metrics['average_cost_per_task']:.4f}/task"
        )
        overview_table.add_row(
            "Uptime",
            f"{metrics['uptime_seconds']:.0f}s",
            f"{metrics['uptime_seconds']/3600:.1f} hours"
        )
        
        content.append(overview_table)
        
        # Recent tasks
        if self.state.task_results:
            recent_table = Table(title="Recent Tasks", style="bright_yellow")
            recent_table.add_column("Task ID", style="dim")
            recent_table.add_column("Strategy", style="bright_cyan")
            recent_table.add_column("Status", style="bold")
            recent_table.add_column("Cost", style="bright_green")
            recent_table.add_column("Time", style="bright_magenta")
            
            # Show last 5 tasks
            recent_tasks = list(self.state.task_results.values())[-5:]
            for task in recent_tasks:
                status_icon = "✅" if task.success else "❌"
                recent_table.add_row(
                    task.task_id[:8],
                    task.strategy_used.value if task.strategy_used else "unknown",
                    f"{status_icon} {'Success' if task.success else 'Failed'}",
                    f"${task.total_cost_usd:.4f}",
                    f"{task.execution_time_ms:.0f}ms"
                )
            
            content.append(recent_table)
        
        return Panel(Group(*content), title="Overview", style="bright_blue")
    
    def _create_agents_panel(self) -> Panel:
        """Create the agents detail panel."""
        agents_table = Table(title="Agent Details", style="bright_green")
        agents_table.add_column("Agent", style="bold")
        agents_table.add_column("Type", style="bright_cyan")
        agents_table.add_column("Status", style="bold")
        agents_table.add_column("Requests", style="bright_yellow")
        agents_table.add_column("Success Rate", style="bright_green")
        agents_table.add_column("Cost", style="bright_magenta")
        agents_table.add_column("Avg Time", style="bright_blue")
        
        for agent in agent_registry.get_all_agents():
            metrics = agent.performance_metrics
            
            # Calculate success rate
            success_rate = 1.0
            if metrics["total_requests"] > 0:
                failed_requests = metrics.get("failed_requests", 0)
                success_rate = (metrics["total_requests"] - failed_requests) / metrics["total_requests"]
            
            # Status based on recent activity
            if metrics["total_requests"] == 0:
                status = "🟡 Idle"
            elif success_rate > 0.9:
                status = "🟢 Excellent"
            elif success_rate > 0.7:
                status = "🟠 Good"
            else:
                status = "🔴 Issues"
            
            # Average response time
            avg_time = "N/A"
            if metrics["total_requests"] > 0:
                # This would need to be tracked in agent metrics
                avg_time = "~1.5s"  # Placeholder
            
            agents_table.add_row(
                agent.name,
                agent.agent_type.value,
                status,
                str(metrics["total_requests"]),
                f"{success_rate:.1%}",
                f"${metrics['total_cost_usd']:.4f}",
                avg_time
            )
        
        # Agent capabilities
        capabilities_content = []
        for agent in agent_registry.get_all_agents():
            cap_tree = Tree(f"{agent.name} Capabilities", style="bright_cyan")
            for capability in agent.capabilities:
                cap_node = cap_tree.add(f"📋 {capability.name}")
                cap_node.add(f"Description: {capability.description}")
                cap_node.add(f"Formats: {', '.join(capability.supported_formats)}")
                if capability.cost_per_request:
                    cap_node.add(f"Cost: ${capability.cost_per_request:.4f}")
            
            capabilities_content.append(cap_tree)
        
        content = [agents_table]
        if capabilities_content:
            content.extend(capabilities_content)
        
        return Panel(Group(*content), title="Agents", style="bright_green")
    
    def _create_tasks_panel(self) -> Panel:
        """Create the tasks detail panel."""
        content = []
        
        # Active tasks
        active_tasks = self.engine.orchestrator.get_active_tasks()
        if active_tasks:
            active_table = Table(title="Active Tasks", style="bright_yellow")
            active_table.add_column("Task ID", style="dim")
            active_table.add_column("Type", style="bright_cyan")
            active_table.add_column("Strategy", style="bright_magenta")
            active_table.add_column("Priority", style="bold")
            active_table.add_column("Started", style="bright_blue")
            
            for task in active_tasks:
                active_table.add_row(
                    task.task_id[:8],
                    task.task_type.value,
                    task.strategy.value,
                    task.priority.value,
                    "Just now"  # Would need start time tracking
                )
            
            content.append(active_table)
        else:
            content.append(Text("No active tasks", style="dim"))
        
        # Task history
        if self.state.task_results:
            history_table = Table(title="Task History", style="bright_blue")
            history_table.add_column("Task ID", style="dim")
            history_table.add_column("Strategy", style="bright_cyan")
            history_table.add_column("Status", style="bold")
            history_table.add_column("Execution Time", style="bright_magenta")
            history_table.add_column("Cost", style="bright_green")
            history_table.add_column("Agents Used", style="bright_yellow")
            
            for task_id, result in list(self.state.task_results.items())[-10:]:
                status_icon = "✅" if result.success else "❌"
                agents_used = "N/A"
                if result.agent_responses:
                    agents_used = ", ".join(set(resp.agent_type for resp in result.agent_responses))
                
                history_table.add_row(
                    task_id[:8],
                    result.strategy_used.value if result.strategy_used else "unknown",
                    f"{status_icon} {'Success' if result.success else 'Failed'}",
                    f"{result.execution_time_ms:.0f}ms",
                    f"${result.total_cost_usd:.4f}",
                    agents_used
                )
            
            content.append(history_table)
        
        return Panel(Group(*content), title="Tasks", style="bright_yellow")
    
    def _create_footer(self) -> Panel:
        """Create the footer panel."""
        help_text = Text()
        help_text.append("Controls: ", style="bold")
        help_text.append("(r) Refresh ", style="bright_green")
        help_text.append("(m) Mode ", style="bright_cyan")
        help_text.append("(n) New Task ", style="bright_yellow")
        help_text.append("(q) Quit ", style="bright_red")
        help_text.append("(h) Help", style="bright_magenta")
        
        return Panel(Align.center(help_text), style="dim")
    
    async def _update_data(self):
        """Update UI data from the system."""
        try:
            # Update agent statuses
            for agent in agent_registry.get_all_agents():
                self.state.agent_statuses[agent.name] = {
                    "type": agent.agent_type.value,
                    "metrics": agent.performance_metrics,
                    "capabilities": len(agent.capabilities)
                }
            
            # Update orchestrator metrics
            self.state.orchestrator_metrics = self.engine.orchestrator.get_performance_metrics()
            
            self._last_update = time.time()
            self._update_count += 1
            
        except Exception as e:
            self.logger.error(f"Error updating UI data: {e}")
    
    def _render_layout(self):
        """Render the current layout based on UI state."""
        # Update header
        self.layout["header"].update(self._create_header())
        
        # Update sidebar
        self.layout["sidebar"].update(self._create_sidebar())
        
        # Update main content based on mode
        if self.state.mode == UIMode.OVERVIEW:
            self.layout["primary"].update(self._create_overview_panel())
        elif self.state.mode == UIMode.AGENTS:
            self.layout["primary"].update(self._create_agents_panel())
        elif self.state.mode == UIMode.TASKS:
            self.layout["primary"].update(self._create_tasks_panel())
        else:
            # Placeholder for other modes
            self.layout["primary"].update(
                Panel(f"Mode: {self.state.mode.value.title()}", title="Coming Soon")
            )
        
        # Update secondary panel (status/logs)
        status_content = []
        status_content.append(f"Updates: {self._update_count}")
        status_content.append(f"Last: {datetime.fromtimestamp(self._last_update).strftime('%H:%M:%S')}")
        status_content.append(f"Rate: {self.state.refresh_rate:.1f}s")
        
        self.layout["secondary"].update(
            Panel("\n".join(status_content), title="Status", style="dim")
        )
        
        # Update footer
        self.layout["footer"].update(self._create_footer())
    
    async def _auto_refresh_loop(self):
        """Auto-refresh loop for real-time updates."""
        while self._running and self.state.auto_refresh:
            try:
                await self._update_data()
                self._render_layout()
                await asyncio.sleep(self.state.refresh_rate)
            except Exception as e:
                self.logger.error(f"Error in auto-refresh loop: {e}")
                await asyncio.sleep(1.0)
    
    async def start(self):
        """Start the terminal UI."""
        self._running = True
        
        try:
            with Live(self.layout, console=self.console, refresh_per_second=2) as live:
                self.live = live
                
                # Initial data update and render
                await self._update_data()
                self._render_layout()
                
                # Start auto-refresh task
                if self.state.auto_refresh:
                    self._update_task = asyncio.create_task(self._auto_refresh_loop())
                
                # Keep running until stopped
                while self._running:
                    await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("UI interrupted by user")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the terminal UI."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Terminal UI stopped")
    
    def switch_mode(self, mode: UIMode):
        """Switch UI mode."""
        self.state.mode = mode
        self._render_layout()
        self.logger.info(f"Switched to {mode.value} mode")
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh."""
        self.state.auto_refresh = not self.state.auto_refresh
        self.logger.info(f"Auto-refresh {'enabled' if self.state.auto_refresh else 'disabled'}")
    
    def set_refresh_rate(self, rate: float):
        """Set refresh rate in seconds."""
        self.state.refresh_rate = max(0.1, rate)
        self.logger.info(f"Refresh rate set to {self.state.refresh_rate}s")


class UIManager:
    """
    High-level UI manager that coordinates terminal interface with user input.
    
    I'm providing a simplified interface for running the terminal UI with
    keyboard handling and command processing.
    """
    
    def __init__(self, orchestration_engine):
        self.engine = orchestration_engine
        self.ui = TerminalUI(orchestration_engine)
        self.logger = get_logger("ui.manager")
        self._input_task = None
    
    async def _handle_input(self):
        """Handle keyboard input in a separate task."""
        # This would need to be implemented with a proper async input handler
        # For now, we'll use a placeholder
        pass
    
    async def run(self):
        """Run the UI manager."""
        self.logger.info("Starting UI manager")
        
        try:
            # Start UI and input handling
            ui_task = asyncio.create_task(self.ui.start())
            
            await ui_task
            
        except Exception as e:
            self.logger.error(f"UI manager error: {e}")
            raise
        finally:
            await self.ui.stop()
    
    async def process_command(self, command: str) -> bool:
        """Process a user command. Returns True if should continue."""
        command = command.strip().lower()
        
        if command == 'q' or command == 'quit':
            return False
        elif command == 'r' or command == 'refresh':
            await self.ui._update_data()
            self.ui._render_layout()
        elif command == 'm' or command == 'mode':
            # Cycle through modes
            modes = list(UIMode)
            current_index = modes.index(self.ui.state.mode)
            next_index = (current_index + 1) % len(modes)
            self.ui.switch_mode(modes[next_index])
        elif command == 'a' or command == 'auto':
            self.ui.toggle_auto_refresh()
        elif command.startswith('rate '):
            try:
                rate = float(command.split()[1])
                self.ui.set_refresh_rate(rate)
            except (IndexError, ValueError):
                self.logger.warning("Invalid rate format. Use: rate <seconds>")
        else:
            self.logger.warning(f"Unknown command: {command}")
        
        return True