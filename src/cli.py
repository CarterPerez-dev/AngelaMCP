"""
Command Line Interface for AngelaMCP.
Provides interactive terminal interface for multi-agent collaboration.
"""

import asyncio
import argparse
import sys
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import shlex

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings
from src.orchestration.orchestrator import TaskOrchestrator
from src.models.database import TaskExecution, TaskStatus
from src.ui.terminal_interface import TerminalUI


class CLICommands:
    """CLI command handlers."""
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
    async def help_command(self, args: List[str] = None) -> None:
        """Show help information."""
        help_text = """
[bold cyan]AngelaMCP - Multi-AI Agent Collaboration Platform[/bold cyan]

[bold]Available Commands:[/bold]

[cyan]/help[/cyan]              Show this help message
[cyan]/status[/cyan]            Show system status and agent health
[cyan]/history[/cyan]           Show recent conversation history
[cyan]/debate <topic>[/cyan]    Start a structured debate between agents
[cyan]/vote <question>[/cyan]   Initiate a voting session
[cyan]/clear[/cyan]             Clear the terminal
[cyan]/settings[/cyan]          Show current configuration
[cyan]/agents[/cyan]            Show agent information and capabilities
[cyan]/metrics[/cyan]           Show performance metrics
[cyan]/session[/cyan]           Session management commands
[cyan]/exit[/cyan] or [cyan]/quit[/cyan]  Exit the application

[bold]Task Execution:[/bold]
Simply type your request and press Enter to execute tasks.
Examples:
  • "Create a REST API with authentication"
  • "Debug this Python function: [code]"
  • "Review my code for security issues"

[bold]Special Modes:[/bold]
[cyan]/interactive[/cyan]       Enter interactive mode with real-time collaboration
[cyan]/parallel[/cyan]          Execute multiple tasks in parallel
[cyan]/solo <agent>[/cyan]      Execute task with specific agent only

Use [cyan]Ctrl+C[/cyan] to interrupt long-running tasks.
        """
        self.console.print(Panel(help_text, title="AngelaMCP Help", border_style="cyan"))
    
    async def status_command(self, args: List[str] = None) -> None:
        """Show system status."""
        try:
            status = await self.orchestrator.get_system_status()
            
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details")
            
            for component, info in status.items():
                status_text = "✅ Healthy" if info.get("healthy", False) else "❌ Error"
                details = info.get("details", "")
                table.add_row(component, status_text, details)
                
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting status: {e}[/red]")
    
    async def history_command(self, args: List[str] = None) -> None:
        """Show conversation history."""
        try:
            limit = 10
            if args and args[0].isdigit():
                limit = int(args[0])
                
            history = await self.orchestrator.get_conversation_history(limit=limit)
            
            if not history:
                self.console.print("[yellow]No conversation history found.[/yellow]")
                return
                
            table = Table(title=f"Recent Conversations (Last {limit})")
            table.add_column("Time", style="dim")
            table.add_column("User", style="cyan")
            table.add_column("Task Summary")
            table.add_column("Status", style="green")
            
            for item in history:
                table.add_row(
                    item.created_at.strftime("%H:%M:%S"),
                    "User",
                    item.content[:50] + "..." if len(item.content) > 50 else item.content,
                    item.status
                )
                
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting history: {e}[/red]")
    
    async def debate_command(self, args: List[str]) -> None:
        """Start a structured debate."""
        if not args:
            self.console.print("[red]Usage: /debate <topic>[/red]")
            return
            
        topic = " ".join(args)
        self.console.print(f"[cyan]Starting debate on: {topic}[/cyan]")
        
        try:
            result = await self.orchestrator.start_debate(topic)
            self.console.print(Panel(result, title="Debate Result", border_style="green"))
        except Exception as e:
            self.console.print(f"[red]Error in debate: {e}[/red]")
    
    async def vote_command(self, args: List[str]) -> None:
        """Initiate a voting session."""
        if not args:
            self.console.print("[red]Usage: /vote <question>[/red]")
            return
            
        question = " ".join(args)
        self.console.print(f"[cyan]Starting vote on: {question}[/cyan]")
        
        try:
            result = await self.orchestrator.start_vote(question)
            self.console.print(Panel(result, title="Vote Result", border_style="green"))
        except Exception as e:
            self.console.print(f"[red]Error in vote: {e}[/red]")
    
    async def agents_command(self, args: List[str] = None) -> None:
        """Show agent information."""
        agents_info = await self.orchestrator.get_agents_info()
        
        table = Table(title="AI Agents")
        table.add_column("Agent", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status")
        table.add_column("Capabilities")
        
        for agent_name, info in agents_info.items():
            table.add_row(
                agent_name,
                info.get("type", "Unknown"),
                "🟢 Ready" if info.get("available", False) else "🔴 Offline",
                ", ".join(info.get("capabilities", []))
            )
            
        self.console.print(table)
    
    async def metrics_command(self, args: List[str] = None) -> None:
        """Show performance metrics."""
        try:
            metrics = await self.orchestrator.get_metrics()
            
            table = Table(title="Performance Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Unit")
            
            for metric_name, data in metrics.items():
                table.add_row(
                    metric_name,
                    str(data.get("value", "N/A")),
                    data.get("unit", "")
                )
                
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting metrics: {e}[/red]")
    
    async def settings_command(self, args: List[str] = None) -> None:
        """Show current configuration."""
        config_info = {
            "Environment": settings.app_env.value,
            "Debug Mode": settings.debug,
            "Log Level": settings.log_level.value,
            "Claude Code Path": str(settings.claude_code_path),
            "OpenAI Model": settings.openai_model,
            "Gemini Model": settings.gemini_model,
            "Database": "PostgreSQL" if settings.database_url else "Not configured",
            "Cache": "Redis" if settings.redis_url else "Not configured",
        }
        
        table = Table(title="Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for setting, value in config_info.items():
            table.add_row(setting, str(value))
            
        self.console.print(table)


class CLI:
    """Main CLI interface for AngelaMCP."""
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.console = Console()
        self.commands = CLICommands(orchestrator)
        self.ui = TerminalUI(orchestrator)
        self.logger = logging.getLogger(__name__)
        self.running = False
        
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="AngelaMCP - Multi-AI Agent Collaboration Platform",
            prog="macp"
        )
        
        parser.add_argument(
            "--version", 
            action="version", 
            version=f"AngelaMCP {settings.app_version}"
        )
        
        parser.add_argument(
            "--config", 
            type=Path, 
            help="Path to configuration file"
        )
        
        parser.add_argument(
            "--log-level", 
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set logging level"
        )
        
        parser.add_argument(
            "--debug", 
            action="store_true", 
            help="Enable debug mode"
        )
        
        parser.add_argument(
            "--non-interactive", 
            action="store_true", 
            help="Run in non-interactive mode"
        )
        
        parser.add_argument(
            "command", 
            nargs="*", 
            help="Command to execute"
        )
        
        return parser.parse_args()
    
    async def show_welcome(self) -> None:
        """Show welcome message."""
        welcome_text = f"""
[bold cyan]Welcome to AngelaMCP[/bold cyan]
[dim]Multi-AI Agent Collaboration Platform v{settings.app_version}[/dim]

[green]✓[/green] Claude Code Agent ready
[green]✓[/green] OpenAI Agent ({settings.openai_model}) ready  
[green]✓[/green] Gemini Agent ({settings.gemini_model}) ready

Type [cyan]/help[/cyan] for commands or start typing your request.
Use [cyan]Ctrl+C[/cyan] to exit.
        """
        self.console.print(Panel(welcome_text, border_style="cyan"))
    
    def parse_command(self, user_input: str) -> tuple[str, List[str]]:
        """Parse user input into command and arguments."""
        if not user_input.startswith("/"):
            return "task", [user_input]
            
        parts = shlex.split(user_input[1:])  # Remove leading /
        command = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        return command, args
    
    async def execute_command(self, command: str, args: List[str]) -> bool:
        """Execute a command. Returns False if should exit."""
        try:
            if command in ["exit", "quit", "q"]:
                return False
                
            elif command == "help" or command == "h":
                await self.commands.help_command(args)
                
            elif command == "status":
                await self.commands.status_command(args)
                
            elif command == "history":
                await self.commands.history_command(args)
                
            elif command == "debate":
                await self.commands.debate_command(args)
                
            elif command == "vote":
                await self.commands.vote_command(args)
                
            elif command == "agents":
                await self.commands.agents_command(args)
                
            elif command == "metrics":
                await self.commands.metrics_command(args)
                
            elif command == "settings":
                await self.commands.settings_command(args)
                
            elif command == "clear":
                self.console.clear()
                
            elif command == "task":
                # Execute task with all agents
                task_input = args[0] if args else ""
                if task_input.strip():
                    await self.execute_task(task_input)
                    
            else:
                self.console.print(f"[red]Unknown command: /{command}[/red]")
                self.console.print("Type [cyan]/help[/cyan] for available commands.")
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Command interrupted by user[/yellow]")
        except Exception as e:
            self.logger.error(f"Command execution error: {e}", exc_info=True)
            self.console.print(f"[red]Error executing command: {e}[/red]")
            
        return True
    
    async def execute_task(self, task_input: str) -> None:
        """Execute a task using the orchestrator."""
        try:
            self.console.print(f"\n[cyan]Executing task:[/cyan] {task_input}")
            
            # Show progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task_id = progress.add_task("Processing with AI agents...", total=None)
                
                # Execute task through orchestrator
                result = await self.orchestrator.execute_task(task_input)
                
                progress.update(task_id, description="Task completed!")
                
            # Display results using UI
            await self.ui.display_task_result(result)
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}", exc_info=True)
            self.console.print(f"[red]Error executing task: {e}[/red]")
    
    async def interactive_loop(self) -> None:
        """Main interactive loop."""
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = await asyncio.to_thread(
                        Prompt.ask,
                        "\n[cyan]MACP>[/cyan]",
                        console=self.console
                    )
                    
                    if not user_input.strip():
                        continue
                        
                    # Parse and execute command
                    command, args = self.parse_command(user_input.strip())
                    should_continue = await self.execute_command(command, args)
                    
                    if not should_continue:
                        break
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use /exit to quit properly[/yellow]")
                    
        except Exception as e:
            self.logger.error(f"Interactive loop error: {e}", exc_info=True)
            self.console.print(f"[red]CLI error: {e}[/red]")
    
    async def run(self) -> None:
        """Run the CLI interface."""
        try:
            self.running = True
            
            # Show welcome message
            await self.show_welcome()
            
            # Start interactive loop
            await self.interactive_loop()
            
        except Exception as e:
            self.logger.error(f"CLI run error: {e}", exc_info=True)
            raise
        finally:
            self.running = False
    
    async def cleanup(self) -> None:
        """Cleanup CLI resources."""
        self.running = False
        self.console.print("[dim]CLI shutting down...[/dim]")