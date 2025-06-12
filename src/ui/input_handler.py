"""
Input handling for AngelaMCP terminal UI.

This module provides keyboard input handling, command processing, and
interactive features for the Rich-based terminal interface.
"""

import asyncio
import sys
import termios
import tty
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.text import Text
from rich.panel import Panel

from src.orchestration.orchestrator import OrchestrationTask, OrchestrationStrategy, TaskType, TaskPriority
from src.agents.base import agent_registry
from src.utils.logger import get_logger

logger = get_logger("ui.input_handler")


class InputMode(str, Enum):
    """Input handling modes."""
    NORMAL = "normal"           # Normal navigation mode
    COMMAND = "command"         # Command entry mode
    TASK_ENTRY = "task_entry"   # Task creation mode
    SEARCH = "search"           # Search mode


@dataclass
class KeyBinding:
    """Represents a key binding."""
    key: str
    description: str
    action: Callable
    mode: InputMode = InputMode.NORMAL


class CommandProcessor:
    """
    Command processor for handling user commands.
    
    I'm implementing a command system that allows users to interact
    with the multi-agent system through text commands.
    """
    
    def __init__(self, orchestration_engine):
        self.engine = orchestration_engine
        self.logger = get_logger("ui.command_processor")
        
        # Command registry
        self.commands = {
            "help": self._cmd_help,
            "agents": self._cmd_agents,
            "tasks": self._cmd_tasks,
            "new": self._cmd_new_task,
            "debate": self._cmd_start_debate,
            "vote": self._cmd_start_vote,
            "status": self._cmd_status,
            "clear": self._cmd_clear,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }
    
    async def process_command(self, command_line: str) -> Dict[str, Any]:
        """Process a command line and return result."""
        if not command_line.strip():
            return {"success": True, "message": ""}
        
        parts = command_line.strip().split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if command not in self.commands:
            return {
                "success": False,
                "message": f"Unknown command: {command}. Type 'help' for available commands."
            }
        
        try:
            result = await self.commands[command](args)
            return {"success": True, "result": result}
        except Exception as e:
            self.logger.error(f"Command '{command}' failed: {e}")
            return {"success": False, "message": f"Command failed: {e}"}
    
    async def _cmd_help(self, args: List[str]) -> str:
        """Show help information."""
        help_text = """
Available Commands:

🤖 Agent Commands:
  agents              - List all agents and their status
  status              - Show system status and metrics

📋 Task Commands:
  tasks               - List recent tasks
  new <description>   - Create a new task
  debate <topic>      - Start a debate on a topic
  vote <proposal>     - Start a voting session

🎛️  System Commands:
  clear               - Clear the display
  help                - Show this help message
  exit/quit           - Exit the application

Examples:
  new "Write a Python function to sort a list"
  debate "Should we use microservices architecture?"
  vote "Implement the new feature using React"
"""
        return help_text
    
    async def _cmd_agents(self, args: List[str]) -> str:
        """List agents and their status."""
        agents = agent_registry.get_all_agents()
        
        if not agents:
            return "No agents are currently registered."
        
        result = "Registered Agents:\n\n"
        for agent in agents:
            metrics = agent.performance_metrics
            result += f"🤖 {agent.name} ({agent.agent_type.value})\n"
            result += f"   Requests: {metrics['total_requests']}\n"
            result += f"   Cost: ${metrics['total_cost_usd']:.4f}\n"
            result += f"   Capabilities: {len(agent.capabilities)}\n\n"
        
        return result
    
    async def _cmd_tasks(self, args: List[str]) -> str:
        """List recent tasks."""
        active_tasks = self.engine.orchestrator.get_active_tasks()
        metrics = self.engine.orchestrator.get_performance_metrics()
        
        result = f"Task Summary:\n\n"
        result += f"Total Tasks: {metrics['total_tasks']}\n"
        result += f"Success Rate: {metrics['success_rate']:.1%}\n"
        result += f"Active Tasks: {len(active_tasks)}\n\n"
        
        if active_tasks:
            result += "Active Tasks:\n"
            for task in active_tasks:
                result += f"  📋 {task.task_id[:8]} - {task.task_type.value} ({task.strategy.value})\n"
        else:
            result += "No active tasks.\n"
        
        return result
    
    async def _cmd_new_task(self, args: List[str]) -> str:
        """Create a new task."""
        if not args:
            return "Usage: new <task description>"
        
        description = " ".join(args)
        
        # Create task with intelligent routing
        result = await self.engine.analyze_and_route(description)
        
        if result.success:
            return f"✅ Task completed successfully!\n\nResult:\n{result.content}"
        else:
            return f"❌ Task failed: {result.error_message}"
    
    async def _cmd_start_debate(self, args: List[str]) -> str:
        """Start a debate session."""
        if not args:
            return "Usage: debate <topic>"
        
        topic = " ".join(args)
        
        # Create debate task
        result = await self.engine.process_request(
            topic,
            task_type=TaskType.ANALYSIS,
            strategy=OrchestrationStrategy.DEBATE
        )
        
        if result.success:
            return f"🗣️ Debate completed!\n\nConsensus:\n{result.content}"
        else:
            return f"❌ Debate failed: {result.error_message}"
    
    async def _cmd_start_vote(self, args: List[str]) -> str:
        """Start a voting session."""
        if not args:
            return "Usage: vote <proposal>"
        
        proposal = " ".join(args)
        
        # This would need integration with the voting system
        return f"🗳️ Voting on: {proposal}\n(Voting system integration needed)"
    
    async def _cmd_status(self, args: List[str]) -> str:
        """Show system status."""
        status = self.engine.get_status()
        
        result = "System Status:\n\n"
        result += f"Available Agents: {status['available_agents']}\n"
        
        metrics = status['orchestrator_metrics']
        result += f"Total Tasks: {metrics['total_tasks']}\n"
        result += f"Success Rate: {metrics['success_rate']:.1%}\n"
        result += f"Total Cost: ${metrics['total_cost_usd']:.4f}\n"
        result += f"Uptime: {metrics['uptime_seconds']:.0f}s\n"
        
        return result
    
    async def _cmd_clear(self, args: List[str]) -> str:
        """Clear the display."""
        return "CLEAR_DISPLAY"  # Special marker for UI to clear
    
    async def _cmd_exit(self, args: List[str]) -> str:
        """Exit the application."""
        return "EXIT_APPLICATION"  # Special marker for UI to exit


class InputHandler:
    """
    Advanced input handler for the terminal UI.
    
    I'm implementing a comprehensive input system that handles keyboard
    shortcuts, command processing, and interactive task creation.
    """
    
    def __init__(self, orchestration_engine, console: Optional[Console] = None):
        self.engine = orchestration_engine
        self.console = console or Console()
        self.command_processor = CommandProcessor(orchestration_engine)
        self.logger = get_logger("ui.input_handler")
        
        # Input state
        self.mode = InputMode.NORMAL
        self.command_buffer = ""
        self.history: List[str] = []
        self.history_index = -1
        
        # Key bindings
        self.key_bindings = self._setup_key_bindings()
        
        # Callbacks
        self.mode_change_callback: Optional[Callable] = None
        self.refresh_callback: Optional[Callable] = None
        self.exit_callback: Optional[Callable] = None
    
    def _setup_key_bindings(self) -> Dict[str, KeyBinding]:
        """Set up keyboard shortcuts."""
        bindings = {}
        
        # Navigation keys
        bindings['r'] = KeyBinding('r', "Refresh display", self._key_refresh)
        bindings['m'] = KeyBinding('m', "Switch mode", self._key_mode_switch)
        bindings['h'] = KeyBinding('h', "Show help", self._key_help)
        bindings['q'] = KeyBinding('q', "Quit application", self._key_quit)
        bindings['c'] = KeyBinding('c', "Clear display", self._key_clear)
        
        # Task management keys
        bindings['n'] = KeyBinding('n', "New task", self._key_new_task)
        bindings['d'] = KeyBinding('d', "Start debate", self._key_debate)
        bindings['v'] = KeyBinding('v', "Start vote", self._key_vote)
        bindings['s'] = KeyBinding('s', "Show status", self._key_status)
        
        # Command mode
        bindings[':'] = KeyBinding(':', "Enter command mode", self._key_command_mode)
        
        return bindings
    
    def set_callbacks(self, mode_change: Optional[Callable] = None,
                     refresh: Optional[Callable] = None,
                     exit_app: Optional[Callable] = None):
        """Set callback functions for UI events."""
        self.mode_change_callback = mode_change
        self.refresh_callback = refresh
        self.exit_callback = exit_app
    
    async def handle_key(self, key: str) -> bool:
        """Handle a single key press. Returns True if app should continue."""
        try:
            if self.mode == InputMode.COMMAND:
                return await self._handle_command_mode_key(key)
            else:
                return await self._handle_normal_mode_key(key)
        except Exception as e:
            self.logger.error(f"Error handling key '{key}': {e}")
            return True
    
    async def _handle_normal_mode_key(self, key: str) -> bool:
        """Handle key in normal mode."""
        if key in self.key_bindings:
            binding = self.key_bindings[key]
            return await binding.action()
        else:
            # Unknown key, ignore
            return True
    
    async def _handle_command_mode_key(self, key: str) -> bool:
        """Handle key in command mode."""
        if key == '\r' or key == '\n':  # Enter
            await self._execute_command()
            return True
        elif key == '\x1b':  # Escape
            self._exit_command_mode()
            return True
        elif key == '\x7f':  # Backspace
            if self.command_buffer:
                self.command_buffer = self.command_buffer[:-1]
            return True
        elif key == '\x03':  # Ctrl+C
            self._exit_command_mode()
            return True
        elif len(key) == 1 and ord(key) >= 32:  # Printable character
            self.command_buffer += key
            return True
        else:
            # Other special keys, ignore in command mode
            return True
    
    async def _execute_command(self):
        """Execute the current command buffer."""
        if self.command_buffer.strip():
            # Add to history
            self.history.append(self.command_buffer)
            
            # Process command
            result = await self.command_processor.process_command(self.command_buffer)
            
            # Handle special results
            if result.get("success") and "result" in result:
                command_result = result["result"]
                if command_result == "CLEAR_DISPLAY":
                    if self.refresh_callback:
                        self.refresh_callback()
                elif command_result == "EXIT_APPLICATION":
                    if self.exit_callback:
                        self.exit_callback()
                else:
                    # Display result
                    self.console.print(Panel(command_result, title="Command Result"))
            elif not result.get("success"):
                self.console.print(Panel(result.get("message", "Unknown error"), 
                                       title="Error", style="red"))
        
        self._exit_command_mode()
    
    def _exit_command_mode(self):
        """Exit command mode."""
        self.mode = InputMode.NORMAL
        self.command_buffer = ""
        if self.mode_change_callback:
            self.mode_change_callback(self.mode)
    
    # Key binding actions
    async def _key_refresh(self) -> bool:
        """Refresh display."""
        if self.refresh_callback:
            self.refresh_callback()
        return True
    
    async def _key_mode_switch(self) -> bool:
        """Switch UI mode."""
        # This would cycle through UI modes
        if self.mode_change_callback:
            self.mode_change_callback("next")
        return True
    
    async def _key_help(self) -> bool:
        """Show help."""
        help_result = await self.command_processor._cmd_help([])
        self.console.print(Panel(help_result, title="Help"))
        return True
    
    async def _key_quit(self) -> bool:
        """Quit application."""
        if Confirm.ask("Are you sure you want to quit?"):
            if self.exit_callback:
                self.exit_callback()
            return False
        return True
    
    async def _key_clear(self) -> bool:
        """Clear display."""
        if self.refresh_callback:
            self.refresh_callback()
        return True
    
    async def _key_new_task(self) -> bool:
        """Create new task interactively."""
        description = Prompt.ask("Enter task description")
        if description:
            result = await self.command_processor._cmd_new_task(description.split())
            self.console.print(Panel(result, title="Task Result"))
        return True
    
    async def _key_debate(self) -> bool:
        """Start debate interactively."""
        topic = Prompt.ask("Enter debate topic")
        if topic:
            result = await self.command_processor._cmd_start_debate(topic.split())
            self.console.print(Panel(result, title="Debate Result"))
        return True
    
    async def _key_vote(self) -> bool:
        """Start vote interactively."""
        proposal = Prompt.ask("Enter proposal to vote on")
        if proposal:
            result = await self.command_processor._cmd_start_vote(proposal.split())
            self.console.print(Panel(result, title="Vote Result"))
        return True
    
    async def _key_status(self) -> bool:
        """Show status."""
        result = await self.command_processor._cmd_status([])
        self.console.print(Panel(result, title="System Status"))
        return True
    
    async def _key_command_mode(self) -> bool:
        """Enter command mode."""
        self.mode = InputMode.COMMAND
        self.command_buffer = ""
        if self.mode_change_callback:
            self.mode_change_callback(self.mode)
        return True
    
    def get_command_prompt(self) -> str:
        """Get command prompt display."""
        if self.mode == InputMode.COMMAND:
            return f":{self.command_buffer}"
        return ""
    
    def get_status_line(self) -> str:
        """Get status line for display."""
        mode_indicator = {
            InputMode.NORMAL: "NORMAL",
            InputMode.COMMAND: "COMMAND",
            InputMode.TASK_ENTRY: "TASK",
            InputMode.SEARCH: "SEARCH"
        }.get(self.mode, "UNKNOWN")
        
        return f"Mode: {mode_indicator} | Press 'h' for help, 'q' to quit"


# Utility functions for cross-platform keyboard input
def get_key():
    """Get a single key press (cross-platform)."""
    try:
        if sys.platform == 'win32':
            import msvcrt
            return msvcrt.getch().decode('utf-8')
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        return None


async def async_input_loop(input_handler: InputHandler) -> None:
    """Async loop for handling keyboard input."""
    loop = asyncio.get_event_loop()
    
    def get_input():
        return get_key()
    
    while True:
        try:
            # Get key in a non-blocking way
            key = await loop.run_in_executor(None, get_input)
            if key:
                should_continue = await input_handler.handle_key(key)
                if not should_continue:
                    break
            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in input loop: {e}")
            await asyncio.sleep(0.1)
