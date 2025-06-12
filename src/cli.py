"""
CLI interface for AngelaMCP standalone mode.

This provides an interactive terminal interface when not running as MCP server.
I'm implementing a rich terminal experience with real-time collaboration display.
"""

import asyncio
import sys
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.live import Live

from src.orchestrator.manager import TaskOrchestrator, CollaborationStrategy
from src.utils.logger import get_logger
from config.settings import settings


class CLI:
    """
    Command-line interface for AngelaMCP.
    
    Provides interactive terminal interface for multi-agent collaboration.
    """
    
    def __init__(self, orchestrator: TaskOrchestrator):
        self.orchestrator = orchestrator
        self.console = Console()
        self.logger = get_logger("cli")
        self.running = False
        
    async def run(self) -> None:
        """Run the CLI interface."""
        try:
            await self.show_welcome()
            self.running = True
            
            while self.running:
                try:
                    # Get user input
                    user_input = await self.get_user_input()
                    
                    if not user_input.strip():
                        continue
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        await self.handle_command(user_input)
                    else:
                        # Regular collaboration request
                        await self.handle_collaboration_request(user_input)
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use /exit to quit properly[/yellow]")
                except EOFError:
                    break
                except Exception as e:
                    self.logger.error(f"CLI error: {e}", exc_info=True)
                    self.console.print(f"[red]Error: {e}[/red]")
            
        except Exception as e:
            self.logger.error(f"CLI run failed: {e}", exc_info=True)
            self.console.print(f"[red]CLI failed: {e}[/red]")
        finally:
            await self.cleanup()
    
    async def show_welcome(self) -> None:
        """Show welcome message and status."""
        welcome_panel = Panel.fit(
            f"""[bold cyan]AngelaMCP - Multi-AI Agent Collaboration Platform[/bold cyan]
[dim]Version {settings.app_version} | Environment: {settings.app_env}[/dim]

[green]✓[/green] Claude Code Agent ready
[green]✓[/green] OpenAI Agent ({settings.openai_model}) ready  
[green]✓[/green] Gemini Agent ({settings.gemini_model}) ready

Type your task to start collaboration, or use commands:
[cyan]/help[/cyan] - Show all commands
[cyan]/debate <topic>[/cyan] - Start structured debate
[cyan]/status[/cyan] - Show agent status
[cyan]/exit[/cyan] - Quit application""",
            title="Welcome",
            border_style="bright_blue"
        )
        
        self.console.print(welcome_panel)
        self.console.print()
    
    async def get_user_input(self) -> str:
        """Get user input with proper prompt."""
        return await asyncio.to_thread(
            Prompt.ask, 
            "[bold blue]MACP>[/bold blue]",
            console=self.console
        )
    
    async def handle_command(self, command: str) -> None:
        """Handle CLI commands."""
        cmd_parts = command.strip().split(' ', 1)
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        if cmd == "/help":
            await self.show_help()
        elif cmd == "/exit" or cmd == "/quit":
            self.console.print("[yellow]Goodbye![/yellow]")
            self.running = False
        elif cmd == "/status":
            await self.show_status()
        elif cmd == "/debug":
            await self.show_debug_info()
        elif cmd == "/debate":
            if not args:
                self.console.print("[red]Usage: /debate <topic>[/red]")
                return
            await self.start_debate(args)
        elif cmd == "/analyze":
            if not args:
                self.console.print("[red]Usage: /analyze <task description>[/red]")
                return
            await self.analyze_task_complexity(args)
        elif cmd == "/config":
            await self.show_config()
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("Type [cyan]/help[/cyan] for available commands")
    
    async def handle_collaboration_request(self, task_description: str) -> None:
        """Handle a collaboration request."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                
                # Analyze task first
                progress.add_task("Analyzing task complexity...", total=None)
                analysis = await self.orchestrator.analyze_task_complexity(task_description)
                
                # Show analysis
                self.show_task_analysis(analysis)
                
                # Ask user for strategy preference
                strategy = self.get_strategy_preference(analysis["recommended_strategy"])
                
                # Start collaboration
                progress.add_task("Starting collaboration...", total=None)
                
                result = await self.orchestrator.collaborate_on_task(
                    task_description=task_description,
                    strategy=strategy,
                    max_rounds=3,
                    require_consensus=True
                )
                
                # Show results
                self.show_collaboration_result(result)
                
        except Exception as e:
            self.console.print(f"[red]Collaboration failed: {e}[/red]")
            self.logger.error(f"Collaboration error: {e}", exc_info=True)
    
    async def start_debate(self, topic: str) -> None:
        """Start a structured debate."""
        try:
            with self.console.status("[bold green]Starting debate...") as status:
                result = await self.orchestrator.start_debate(
                    topic=topic,
                    max_rounds=3,
                    timeout_seconds=300
                )
                
                self.show_debate_result(result)
                
        except Exception as e:
            self.console.print(f"[red]Debate failed: {e}[/red]")
    
    async def analyze_task_complexity(self, task_description: str) -> None:
        """Analyze and show task complexity."""
        try:
            analysis = await self.orchestrator.analyze_task_complexity(task_description)
            self.show_task_analysis(analysis)
        except Exception as e:
            self.console.print(f"[red]Analysis failed: {e}[/red]")
    
    def show_task_analysis(self, analysis: Dict[str, Any]) -> None:
        """Show task complexity analysis."""
        table = Table(title="Task Complexity Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Complexity Score", f"{analysis.get('complexity_score', 0):.1f}/10")
        table.add_row("Complexity Level", analysis.get('complexity_level', 'Unknown'))
        table.add_row("Recommended Strategy", analysis.get('recommended_strategy', 'Unknown'))
        table.add_row("Estimated Time", analysis.get('estimated_time', 'Unknown'))
        table.add_row("Collaboration Benefit", analysis.get('collaboration_benefit', 'Unknown'))
        
        self.console.print(table)
        
        if analysis.get('reasoning'):
            reasoning_panel = Panel(
                analysis['reasoning'],
                title="Analysis Reasoning",
                border_style="dim"
            )
            self.console.print(reasoning_panel)
    
    def get_strategy_preference(self, recommended: str) -> str:
        """Get user's strategy preference."""
        strategies = {
            "1": "single_agent",
            "2": "parallel", 
            "3": "debate",
            "4": "consensus"
        }
        
        self.console.print(f"\n[bold]Collaboration Strategies:[/bold]")
        self.console.print(f"1. Single Agent (fast)")
        self.console.print(f"2. Parallel (balanced)")
        self.console.print(f"3. Debate (thorough)")
        self.console.print(f"4. Consensus (comprehensive)")
        self.console.print(f"\nRecommended: [green]{recommended}[/green]")
        
        choice = Prompt.ask(
            "Choose strategy (1-4) or press Enter for recommended",
            choices=["1", "2", "3", "4", ""],
            default="",
            console=self.console
        )
        
        return strategies.get(choice, recommended)
    
    def show_collaboration_result(self, result) -> None:
        """Show collaboration results."""
        # Status panel
        status_color = "green" if result.success else "red"
        status_text = "✅ Success" if result.success else "❌ Failed"
        
        result_panel = Panel.fit(
            f"""[bold]{status_text}[/bold]
Strategy: {result.strategy_used.value if result.strategy_used else 'Unknown'}
Execution Time: {result.execution_time:.2f}s
Consensus Score: {result.consensus_score:.2f}
Agents: {len(result.agent_responses)}""",
            title="Collaboration Result",
            border_style=status_color
        )
        
        self.console.print(result_panel)
        
        # Final solution
        if result.final_solution:
            solution_panel = Panel(
                Markdown(result.final_solution),
                title="Final Solution",
                border_style="bright_blue"
            )
            self.console.print(solution_panel)
        
        # Agent responses
        if result.agent_responses:
            self.console.print("\n[bold]Agent Contributions:[/bold]")
            for i, response in enumerate(result.agent_responses, 1):
                agent_name = response.get('agent', 'Unknown')
                content = response.get('content', 'No response')[:200] + "..."
                confidence = response.get('confidence', 0)
                
                agent_panel = Panel(
                    f"[dim]{content}[/dim]\n\nConfidence: {confidence:.2f}",
                    title=f"{i}. {agent_name.title()} Agent",
                    border_style="dim"
                )
                self.console.print(agent_panel)
        
        # Debate summary
        if result.debate_summary:
            debate_panel = Panel(
                result.debate_summary,
                title="Debate Summary",
                border_style="yellow"
            )
            self.console.print(debate_panel)
    
    def show_debate_result(self, result: Dict[str, Any]) -> None:
        """Show debate results."""
        topic = result.get('topic', 'Unknown')
        rounds_completed = result.get('rounds_completed', 0)
        
        debate_panel = Panel.fit(
            f"""[bold]Debate Topic:[/bold] {topic}
[bold]Rounds Completed:[/bold] {rounds_completed}
[bold]Participants:[/bold] Claude, OpenAI, Gemini""",
            title="Debate Results",
            border_style="bright_magenta"
        )
        
        self.console.print(debate_panel)
        
        # Show rounds
        rounds = result.get('rounds', [])
        for round_data in rounds:
            round_num = round_data.get('round', 0)
            self.console.print(f"\n[bold]Round {round_num}:[/bold]")
            
            responses = round_data.get('responses', [])
            for response in responses:
                agent = response.get('agent', 'Unknown')
                content = response.get('content', 'No response')[:300] + "..."
                
                response_panel = Panel(
                    content,
                    title=f"{agent.title()} Position",
                    border_style="dim"
                )
                self.console.print(response_panel)
        
        # Final consensus
        consensus = result.get('consensus', {})
        if consensus.get('summary'):
            consensus_panel = Panel(
                f"{consensus['summary']}\n\nConsensus Score: {consensus.get('score', 0):.2f}",
                title="Final Consensus",
                border_style="green"
            )
            self.console.print(consensus_panel)
    
    async def show_status(self) -> None:
        """Show agent status."""
        try:
            # Get agent health
            claude_health = await self.orchestrator.claude_agent.health_check()
            openai_health = await self.orchestrator.openai_agent.health_check()
            gemini_health = await self.orchestrator.gemini_agent.health_check()
            
            # Create status table
            table = Table(title="Agent Status")
            table.add_column("Agent", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Response Time", style="green")
            table.add_column("Model", style="yellow")
            
            # Add rows
            claude_status = "✅ Healthy" if claude_health.get("status") == "healthy" else "❌ Unhealthy"
            table.add_row(
                "Claude Code",
                claude_status,
                f"{claude_health.get('response_time', 0):.3f}s",
                claude_health.get('version', 'Unknown')
            )
            
            openai_status = "✅ Healthy" if openai_health.get("status") == "healthy" else "❌ Unhealthy"
            table.add_row(
                "OpenAI",
                openai_status, 
                f"{openai_health.get('response_time', 0):.3f}s",
                openai_health.get('model', settings.openai_model)
            )
            
            gemini_status = "✅ Healthy" if gemini_health.get("status") == "healthy" else "❌ Unhealthy"
            table.add_row(
                "Gemini",
                gemini_status,
                f"{gemini_health.get('response_time', 0):.3f}s",
                settings.gemini_model
            )
            
            self.console.print(table)
            
        except Exception as e:
            self.console.print(f"[red]Failed to get status: {e}[/red]")
    
    async def show_debug_info(self) -> None:
        """Show debug information."""
        try:
            # Get database health
            db_health = await self.orchestrator.db_manager.health_check()
            
            debug_table = Table(title="Debug Information")
            debug_table.add_column("Component", style="cyan")
            debug_table.add_column("Status", style="green")
            debug_table.add_column("Details", style="dim")
            
            # Database status
            pg_status = db_health.get('postgres', {}).get('status', 'unknown')
            redis_status = db_health.get('redis', {}).get('status', 'unknown')
            
            debug_table.add_row("PostgreSQL", pg_status, str(db_health.get('postgres', {})))
            debug_table.add_row("Redis", redis_status, str(db_health.get('redis', {})))
            debug_table.add_row("Environment", settings.app_env, f"Debug: {settings.debug}")
            
            self.console.print(debug_table)
            
        except Exception as e:
            self.console.print(f"[red]Debug info failed: {e}[/red]")
    
    async def show_config(self) -> None:
        """Show configuration information."""
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Environment", settings.app_env)
        config_table.add_row("Debug Mode", str(settings.debug))
        config_table.add_row("Log Level", settings.log_level)
        config_table.add_row("OpenAI Model", settings.openai_model)
        config_table.add_row("Gemini Model", settings.gemini_model)
        config_table.add_row("Claude Path", str(settings.claude_code_path))
        
        self.console.print(config_table)
    
    async def show_help(self) -> None:
        """Show help information."""
        help_text = """[bold cyan]AngelaMCP Commands:[/bold cyan]

[bold]Basic Usage:[/bold]
• Type any task description to start collaboration
• Example: "Create a REST API with authentication"

[bold]Commands:[/bold]
• [cyan]/help[/cyan] - Show this help message
• [cyan]/status[/cyan] - Show agent status and health
• [cyan]/debug[/cyan] - Show debug information
• [cyan]/config[/cyan] - Show configuration
• [cyan]/exit[/cyan] - Quit the application

[bold]Advanced Commands:[/bold]
• [cyan]/debate <topic>[/cyan] - Start structured debate
• [cyan]/analyze <task>[/cyan] - Analyze task complexity

[bold]Examples:[/bold]
• [dim]create a calculator function[/dim]
• [dim]/debate Should we use TypeScript or Python?[/dim]
• [dim]/analyze building a microservices architecture[/dim]"""

        help_panel = Panel(
            help_text,
            title="Help",
            border_style="bright_blue"
        )
        
        self.console.print(help_panel)
    
    async def cleanup(self) -> None:
        """Cleanup CLI resources."""
        try:
            self.logger.info("CLI cleanup completed")
        except Exception as e:
            self.logger.error(f"CLI cleanup error: {e}")
    
    def __del__(self):
        """Destructor."""
        if hasattr(self, 'running') and self.running:
            asyncio.create_task(self.cleanup())
