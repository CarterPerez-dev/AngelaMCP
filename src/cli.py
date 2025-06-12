"""
CLI interface for AngelaMCP.

Rich terminal interface for multi-agent collaboration.
I'm implementing a production-grade CLI with real-time updates.
"""

import asyncio
import sys
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.spinner import Spinner
from rich.progress import Progress, TaskID
import typer
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from src.orchestrator import TaskOrchestrator, CollaborationStrategy
from src.agents import TaskContext, TaskType
from src.utils import get_logger
from config import settings

logger = get_logger("cli")

class CLI:
    """
    CLI interface for AngelaMCP multi-agent collaboration.
    
    Provides rich terminal interface with real-time collaboration updates.
    """
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.console = Console()
        self.current_session: Optional[str] = None
        self.is_running = False
        
        # Command completion
        self.commands = [
            "help", "exit", "quit", "debug", "status", "history",
            "collaborate", "debate", "agents", "config", "reset"
        ]
        self.completer = WordCompleter(self.commands, ignore_case=True)
    
    async def run(self):
        """Run the CLI main loop."""
        self.is_running = True
        
        # Welcome screen
        self.show_welcome()
        
        try:
            while self.is_running:
                try:
                    # Get user input
                    user_input = await self.get_user_input()
                    
                    if not user_input.strip():
                        continue
                    
                    # Process command
                    await self.process_command(user_input.strip())
                    
                except KeyboardInterrupt:
                    self.console.print("\n👋 Goodbye!")
                    break
                except EOFError:
                    break
                except Exception as e:
                    logger.error(f"CLI error: {e}")
                    self.console.print(f"[red]Error: {e}[/red]")
        
        finally:
            self.is_running = False
    
    def show_welcome(self):
        """Show welcome screen."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold blue]AngelaMCP[/bold blue]\n"
            "[italic]Multi-AI Agent Collaboration Platform[/italic]\n\n"
            f"[green]✓[/green] Claude Code Agent Ready\n"
            f"[green]✓[/green] OpenAI Agent Ready\n"
            f"[green]✓[/green] Gemini Agent Ready\n\n"
            "[dim]Type 'help' for commands or start typing a task...[/dim]",
            title="🤖 Welcome to AngelaMCP",
            border_style="blue"
        ))
    
    async def get_user_input(self) -> str:
        """Get user input with completion."""
        try:
            return await asyncio.to_thread(
                prompt,
                "MACP> ",
                completer=self.completer,
                complete_style="column"
            )
        except (KeyboardInterrupt, EOFError):
            raise
    
    async def process_command(self, user_input: str):
        """Process user command or task."""
        if user_input.lower() in ["exit", "quit"]:
            self.is_running = False
            return
        
        if user_input.lower() == "help":
            self.show_help()
            return
        
        if user_input.lower() == "status":
            await self.show_status()
            return
        
        if user_input.lower() == "agents":
            await self.show_agents()
            return
        
        if user_input.startswith("/debate "):
            topic = user_input[8:].strip()
            await self.start_debate(topic)
            return
        
        if user_input.startswith("/collaborate "):
            task = user_input[13:].strip()
            await self.collaborate_on_task(task)
            return
        
        # Default: treat as general task
        await self.handle_general_task(user_input)
    
    def show_help(self):
        """Show help information."""
        help_table = Table(title="AngelaMCP Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("help", "Show this help message")
        help_table.add_row("status", "Show system status")
        help_table.add_row("agents", "Show agent information")
        help_table.add_row("/debate <topic>", "Start structured debate")
        help_table.add_row("/collaborate <task>", "Multi-agent collaboration")
        help_table.add_row("exit/quit", "Exit the application")
        help_table.add_row("<task>", "Execute task with best strategy")
        
        self.console.print(help_table)
    
    async def show_status(self):
        """Show system status."""
        status_table = Table(title="System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Check database
        try:
            await self.orchestrator.db_manager.health_check()
            db_status = "✓ Connected"
        except Exception:
            db_status = "✗ Error"
        
        status_table.add_row("Database", db_status, "PostgreSQL + Redis")
        status_table.add_row("Claude Agent", "✓ Ready", "Claude Code integration")
        status_table.add_row("OpenAI Agent", "✓ Ready", f"Model: {settings.openai_model}")
        status_table.add_row("Gemini Agent", "✓ Ready", f"Model: {settings.gemini_model}")
        
        self.console.print(status_table)
    
    async def show_agents(self):
        """Show agent information."""
        agents_table = Table(title="AI Agents")
        agents_table.add_column("Agent", style="cyan")
        agents_table.add_column("Role", style="yellow")
        agents_table.add_column("Capabilities", style="white")
        agents_table.add_column("Vote Weight", style="green")
        
        agents_table.add_row(
            "Claude Code", 
            "Senior Developer", 
            "Code generation, file operations, execution",
            str(settings.claude_vote_weight)
        )
        agents_table.add_row(
            "OpenAI", 
            "Code Reviewer", 
            "Analysis, review, optimization",
            str(settings.openai_vote_weight)
        )
        agents_table.add_row(
            "Gemini", 
            "Research Specialist", 
            "Research, documentation, best practices",
            str(settings.gemini_vote_weight)
        )
        
        self.console.print(agents_table)
    
    async def start_debate(self, topic: str):
        """Start a structured debate."""
        self.console.print(f"\n🎭 Starting debate on: [bold]{topic}[/bold]")
        
        with self.console.status("[bold green]Orchestrating debate...") as status:
            try:
                context = TaskContext(
                    task_type=TaskType.DEBATE,
                    session_id=self.current_session
                )
                
                result = await self.orchestrator.start_debate(topic, context)
                
                if result.success:
                    self.console.print(f"\n[green]✓[/green] Debate completed successfully!")
                    self.console.print(Panel(
                        result.final_consensus or "No consensus reached",
                        title="Final Consensus",
                        border_style="green"
                    ))
                else:
                    self.console.print(f"\n[red]✗[/red] Debate failed: {result.error_message}")
                    
            except Exception as e:
                self.console.print(f"\n[red]Error during debate: {e}[/red]")
    
    async def collaborate_on_task(self, task: str):
        """Collaborate on a task."""
        self.console.print(f"\n🤝 Collaborating on: [bold]{task}[/bold]")
        
        with Progress() as progress:
            task_id = progress.add_task("[green]Collaboration in progress...", total=100)
            
            try:
                context = TaskContext(
                    task_type=TaskType.GENERAL,
                    session_id=self.current_session
                )
                
                # Update progress as collaboration proceeds
                progress.update(task_id, advance=25)
                
                result = await self.orchestrator.collaborate(
                    task, 
                    strategy=CollaborationStrategy.DEBATE,
                    context=context
                )
                
                progress.update(task_id, advance=75)
                
                if result.success:
                    self.console.print(f"\n[green]✓[/green] Collaboration completed!")
                    self.console.print(Panel(
                        result.final_solution,
                        title="Final Solution",
                        border_style="green"
                    ))
                    
                    # Show cost breakdown if available
                    if result.cost_breakdown:
                        cost_table = Table(title="Cost Breakdown")
                        cost_table.add_column("Agent", style="cyan")
                        cost_table.add_column("Cost (USD)", style="green")
                        
                        for agent, cost in result.cost_breakdown.items():
                            cost_table.add_row(agent, f"${cost:.4f}")
                        
                        self.console.print(cost_table)
                else:
                    self.console.print(f"\n[red]✗[/red] Collaboration failed")
                    
            except Exception as e:
                self.console.print(f"\n[red]Error during collaboration: {e}[/red]")
            finally:
                progress.update(task_id, completed=100)
    
    async def handle_general_task(self, task: str):
        """Handle a general task with automatic strategy selection."""
        self.console.print(f"\n🎯 Processing task: [bold]{task}[/bold]")
        
        with self.console.status("[bold green]Analyzing task and selecting strategy...") as status:
            try:
                context = TaskContext(
                    task_type=TaskType.GENERAL,
                    session_id=self.current_session
                )
                
                result = await self.orchestrator.execute_task(task, context)
                
                if result.success:
                    self.console.print(f"\n[green]✓[/green] Task completed!")
                    self.console.print(Panel(
                        result.final_solution,
                        title=f"Solution (Strategy: {result.strategy_used.value if result.strategy_used else 'auto'})",
                        border_style="green"
                    ))
                else:
                    self.console.print(f"\n[red]✗[/red] Task failed")
                    
            except Exception as e:
                self.console.print(f"\n[red]Error processing task: {e}[/red]")
