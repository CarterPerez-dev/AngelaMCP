#!/usr/bin/env python3
"""
🚀 AngelaMCP One-Click Installer
================================

Streamlined setup script that takes you from bare Linux to fully working AngelaMCP
with just one command. Handles Docker, databases, environment, and MCP registration.

Usage:
    python install.py
    # or
    ./install.py

Author: AngelaMCP Team
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import shutil

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, TaskID
    from rich.text import Text
    from rich import print as rprint
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "typer"])
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, TaskID
    from rich.text import Text
    from rich import print as rprint


@dataclass
class SetupConfig:
    """Configuration for AngelaMCP setup."""
    # Required inputs
    openai_api_key: str = ""
    gemini_api_key: str = ""
    database_user: str = ""
    database_password: str = ""
    database_name: str = ""
    
    # Optional inputs with defaults
    claude_code_path: str = ""
    user_email: str = ""
    github_token: str = ""
    
    # Setup options
    use_docker: bool = True
    setup_databases: bool = True
    register_mcp: bool = True


class AngelaMCPInstaller:
    """One-click installer for AngelaMCP."""
    
    def __init__(self):
        self.console = Console()
        self.config = SetupConfig()
        self.project_root = Path.cwd()
        self.env_file = self.project_root / ".env"
        
    def run(self) -> None:
        """Run the complete installation process."""
        try:
            self.show_welcome()
            self.check_prerequisites()
            self.collect_user_input()
            self.show_setup_plan()
            
            if Confirm.ask("🚀 Start installation?", default=True):
                self.execute_installation()
                self.show_completion()
            else:
                self.console.print("❌ Installation cancelled", style="red")
                
        except KeyboardInterrupt:
            self.console.print("\n❌ Installation cancelled by user", style="red")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n💥 Installation failed: {e}", style="red bold")
            sys.exit(1)
    
    def show_welcome(self) -> None:
        """Show welcome message and introduction."""
        welcome_text = """
🤖 Welcome to AngelaMCP Setup!

This installer will set up a complete multi-AI agent collaboration platform.
You'll get:

✨ Multi-agent orchestration (Claude, OpenAI, Gemini)
🗄️  Database persistence (PostgreSQL + Redis)
🔗 MCP integration with Claude Code
🐳 Docker containerization
🎯 One-command deployment

Let's get started!
        """
        
        self.console.print(Panel(welcome_text.strip(), title="🚀 AngelaMCP Installer", 
                                title_align="left", style="cyan"))
    
    def check_prerequisites(self) -> None:
        """Check system prerequisites and install if needed."""
        self.console.print("\n🔍 Checking prerequisites...", style="yellow")
        
        # Check Python version
        if sys.version_info < (3, 9):
            raise RuntimeError("Python 3.9+ required")
        
        # Check for essential commands
        required_commands = ["git", "docker", "docker-compose"]
        missing_commands = []
        
        for cmd in required_commands:
            if not shutil.which(cmd):
                missing_commands.append(cmd)
        
        if missing_commands:
            self.console.print(f"❌ Missing commands: {', '.join(missing_commands)}", style="red")
            self.console.print("Installing missing dependencies...", style="yellow")
            self._install_system_dependencies(missing_commands)
        
        # Check for Claude Code
        claude_path = self._find_claude_code()
        if claude_path:
            self.config.claude_code_path = claude_path
            self.console.print(f"✅ Claude Code found: {claude_path}", style="green")
        else:
            self.console.print("⚠️  Claude Code not found - install it later", style="yellow")
        
        self.console.print("✅ Prerequisites checked", style="green")
    
    def collect_user_input(self) -> None:
        """Collect required configuration from user."""
        self.console.print("\n📝 Configuration Setup", style="cyan bold")
        self.console.print("We need some information to configure AngelaMCP:\n")
        
        # Required API keys
        self.console.print("🔑 [bold]API Keys[/bold] (required for AI agents)")
        self.config.openai_api_key = Prompt.ask(
            "OpenAI API Key",
            default="",
            show_default=False,
            password=True
        )
        
        self.config.gemini_api_key = Prompt.ask(
            "Google Gemini API Key", 
            default="",
            show_default=False,
            password=True
        )
        
        # Database configuration
        self.console.print("\n🗄️  [bold]Database Configuration[/bold]")
        self.config.database_user = Prompt.ask(
            "Database username",
            default="angelamcp"
        )
        
        self.config.database_password = Prompt.ask(
            "Database password",
            default="securepass123",
            password=True
        )
        
        self.config.database_name = Prompt.ask(
            "Database name",
            default="angeladb"
        )
        
        # Optional configurations - streamlined
        self.console.print("\n⚙️  [dim]Optional settings (press Enter to skip):[/dim]")
        
        self.config.user_email = Prompt.ask(
            "Email for alerts",
            default="",
            show_default=False
        )
        
        self.config.github_token = Prompt.ask(
            "GitHub token",
            default="",
            show_default=False,
            password=True
        )
        
        # Setup options
        self.console.print("\n🛠️  [bold]Setup Options[/bold]")
        self.config.use_docker = Confirm.ask(
            "Use Docker for databases? (recommended)",
            default=True
        )
        
        self.config.register_mcp = Confirm.ask(
            "Register with Claude Code MCP?",
            default=True
        ) if self.config.claude_code_path else False
    
    def show_setup_plan(self) -> None:
        """Show what will be installed/configured."""
        table = Table(title="🎯 Setup Plan")
        table.add_column("Component", style="cyan")
        table.add_column("Action", style="green")
        table.add_column("Details", style="dim")
        
        table.add_row("Environment", "Create .env file", f"Database: {self.config.database_name}")
        table.add_row("Python Dependencies", "Install packages", "OpenAI, Gemini, FastAPI, etc.")
        
        if self.config.use_docker:
            table.add_row("Docker", "Start containers", "PostgreSQL + Redis")
        else:
            table.add_row("System Databases", "Install & configure", "PostgreSQL + Redis")
        
        table.add_row("Database", "Initialize schema", "Tables, indexes, test data")
        
        if self.config.register_mcp:
            table.add_row("MCP Integration", "Register with Claude", "angelamcp server")
        
        table.add_row("Verification", "Test all components", "End-to-end testing")
        
        self.console.print(table)
    
    def execute_installation(self) -> None:
        """Execute the complete installation process."""
        with Progress() as progress:
            main_task = progress.add_task("🚀 Installing AngelaMCP...", total=100)
            
            # Step 1: Create environment file (15%)
            progress.update(main_task, description="📝 Creating environment file...")
            self._create_env_file()
            progress.update(main_task, advance=15)
            
            # Step 2: Install Python dependencies (25%)
            progress.update(main_task, description="🐍 Installing Python dependencies...")
            self._run_command("make install")
            progress.update(main_task, advance=25)
            
            # Step 3: Setup databases (30%)
            progress.update(main_task, description="🗄️  Setting up databases...")
            if self.config.use_docker:
                self._setup_docker_databases()
            else:
                self._setup_system_databases()
            progress.update(main_task, advance=30)
            
            # Step 4: Initialize database schema (15%)
            progress.update(main_task, description="🏗️  Initializing database schema...")
            self._initialize_database()
            progress.update(main_task, advance=15)
            
            # Step 5: Register MCP (10%)
            if self.config.register_mcp:
                progress.update(main_task, description="🔗 Registering MCP server...")
                self._register_mcp()
            progress.update(main_task, advance=10)
            
            # Step 6: Verify installation (5%)
            progress.update(main_task, description="✅ Verifying installation...")
            self._verify_installation()
            progress.update(main_task, advance=5)
            
            progress.update(main_task, description="🎉 Installation complete!")
    
    def _create_env_file(self) -> None:
        """Create the .env file with user configuration."""
        env_template = self.project_root / ".env.example"
        
        if not env_template.exists():
            # Create a basic template
            env_content = self._generate_env_content()
        else:
            # Use existing template and replace values
            env_content = env_template.read_text()
            env_content = self._replace_env_values(env_content)
        
        self.env_file.write_text(env_content)
        self.console.print(f"✅ Created {self.env_file}", style="green")
    
    def _replace_env_values(self, env_content: str) -> str:
        """Replace placeholder values in existing .env content."""
        # Read current .env and only replace what user provided
        lines = env_content.split('\n')
        updated_lines = []
        
        for line in lines:
            if '=' in line and not line.strip().startswith('#'):
                key, current_value = line.split('=', 1)
                key = key.strip()
                
                # Replace only the keys we have new values for
                if key == "OPENAI_API_KEY" and self.config.openai_api_key:
                    line = f"OPENAI_API_KEY={self.config.openai_api_key}"
                elif key == "GOOGLE_API_KEY" and self.config.gemini_api_key:
                    line = f"GOOGLE_API_KEY={self.config.gemini_api_key}"
                elif key == "DATABASE_USER" and self.config.database_user:
                    line = f"DATABASE_USER={self.config.database_user}"
                elif key == "DATABASE_PASSWORD" and self.config.database_password:
                    line = f"DATABASE_PASSWORD={self.config.database_password}"
                elif key == "DATABASE_NAME" and self.config.database_name:
                    line = f"DATABASE_NAME={self.config.database_name}"
                elif key == "ALERT_EMAIL" and self.config.user_email:
                    line = f"ALERT_EMAIL={self.config.user_email}"
                elif key == "GITHUB_TOKEN" and self.config.github_token:
                    line = f"GITHUB_TOKEN={self.config.github_token}"
                elif key == "CLAUDE_CODE_PATH" and self.config.claude_code_path:
                    line = f"CLAUDE_CODE_PATH={self.config.claude_code_path}"
                    
            updated_lines.append(line)
        
        return '\n'.join(updated_lines)
    
    def _generate_env_content(self) -> str:
        """Generate .env file content from user configuration."""
        return f"""# AngelaMCP Environment Configuration
# Generated by installer on {time.strftime('%Y-%m-%d %H:%M:%S')}

# ============================================
# Application Settings
# ============================================
APP_NAME=AngelaMCP
APP_ENV=development
DEBUG=false
LOG_LEVEL=INFO

# ============================================
# Claude Code Configuration
# ============================================
CLAUDE_CODE_PATH={self.config.claude_code_path or '~/.claude/local/claude'}

# ============================================
# OpenAI Configuration
# ============================================
OPENAI_API_KEY={self.config.openai_api_key}
OPENAI_MODEL=gpt-4o-mini

# ============================================
# Google Gemini Configuration
# ============================================
GOOGLE_API_KEY={self.config.gemini_api_key}
GEMINI_MODEL=gemini-2.5-pro-preview-06-05

# ============================================
# Database Configuration
# ============================================
DATABASE_USER={self.config.database_user}
DATABASE_PASSWORD={self.config.database_password}
DATABASE_NAME={self.config.database_name}
DATABASE_HOST={"postgres" if self.config.use_docker else "localhost"}

DATABASE_URL=postgresql+asyncpg://{self.config.database_user}:{self.config.database_password}@{"postgres" if self.config.use_docker else "localhost"}:5432/{self.config.database_name}

# Redis connection
REDIS_URL=redis://{"redis" if self.config.use_docker else "localhost"}:6379/0

# ============================================
# Optional Configuration
# ============================================
ALERT_EMAIL={self.config.user_email}
GITHUB_TOKEN={self.config.github_token}

# ============================================
# Feature Flags
# ============================================
ENABLE_COST_TRACKING=true
ENABLE_PARALLEL_EXECUTION=true
ENABLE_DEBATE_MODE=true
"""
    
    def _setup_docker_databases(self) -> None:
        """Setup databases using Docker."""
        self._run_command("make docker-build")
        self._run_command("make docker-up")
        
        # Wait for databases to be ready
        self.console.print("⏳ Waiting for databases to start...", style="yellow")
        time.sleep(10)
    
    def _setup_system_databases(self) -> None:
        """Setup databases on system."""
        self._run_command("make deps")
    
    def _initialize_database(self) -> None:
        """Initialize database schema and test connectivity."""
        try:
            self._run_command("make db-setup")
        except subprocess.CalledProcessError:
            # Retry once after a delay
            time.sleep(5)
            self._run_command("make db-setup")
    
    def _register_mcp(self) -> None:
        """Register MCP server with Claude Code."""
        if self.config.claude_code_path:
            try:
                # Use clean register to remove any old servers first
                self._run_command("make mcp-clean-register")
            except subprocess.CalledProcessError:
                # Fallback to direct registration
                self.console.print("Trying direct registration...", style="yellow")
                self._run_command('claude mcp remove angelamcp 2>/dev/null || true')
                self._run_command('claude mcp remove multi-ai-collab 2>/dev/null || true')
                self._run_command('claude mcp add angelamcp "python -m src.main mcp-server"')
    
    def _verify_installation(self) -> None:
        """Verify the complete installation."""
        self._run_command("make verify")
    
    def _run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            return result
        except subprocess.CalledProcessError as e:
            self.console.print(f"❌ Command failed: {command}", style="red")
            self.console.print(f"Error: {e.stderr}", style="red dim")
            if check:
                raise
            return e
    
    def _find_claude_code(self) -> Optional[str]:
        """Find Claude Code installation."""
        possible_paths = [
            f"{Path.home()}/.claude/local/claude",
            "/usr/local/bin/claude",
            "/opt/claude/claude",
        ]
        
        # Also check PATH
        claude_in_path = shutil.which("claude")
        if claude_in_path:
            possible_paths.insert(0, claude_in_path)
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return None
    
    def _install_system_dependencies(self, missing_commands: List[str]) -> None:
        """Install missing system dependencies."""
        if "docker" in missing_commands:
            self.console.print("Installing Docker...", style="yellow")
            # Install Docker based on the system
            if shutil.which("apt"):
                self._run_command("sudo apt update && sudo apt install -y docker.io docker-compose")
            elif shutil.which("yum"):
                self._run_command("sudo yum install -y docker docker-compose")
        
        if "git" in missing_commands:
            self.console.print("Installing Git...", style="yellow")
            if shutil.which("apt"):
                self._run_command("sudo apt install -y git")
            elif shutil.which("yum"):
                self._run_command("sudo yum install -y git")
    
    def show_completion(self) -> None:
        """Show completion message and next steps."""
        completion_text = f"""
🎉 AngelaMCP Installation Complete!

✅ Environment configured
✅ Dependencies installed  
✅ Databases {"(Docker)" if self.config.use_docker else "(System)"} ready
✅ MCP server {"registered" if self.config.register_mcp else "ready"}

🚀 Next Steps:

1. Test the installation:
   make verify

2. Start AngelaMCP:
   make run

3. Start MCP server:
   make run-mcp

4. Connect from Claude Code:
   claude "Use AngelaMCP to help with a task"

📁 Configuration saved to: {self.env_file}
🐳 Docker databases: make docker-logs
📖 Documentation: README.md

Happy collaborating! 🤖✨
        """
        
        self.console.print(Panel(completion_text.strip(), title="🎉 Installation Complete", 
                                title_align="left", style="green"))


def main():
    """Main entry point."""
    installer = AngelaMCPInstaller()
    installer.run()


if __name__ == "__main__":
    main()