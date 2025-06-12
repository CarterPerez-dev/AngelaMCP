#!/usr/bin/env python3
"""
End-to-End Test for AngelaMCP Multi-Agent Collaboration.

This script demonstrates the full collaboration pipeline:
1. Three AI agents (Claude Code, OpenAI, Gemini) working together
2. Structured debate protocol
3. Weighted voting with Claude's veto power
4. Real-time Rich terminal UI

Run this to see AngelaMCP in action!
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.orchestrator.collaboration import CollaborationOrchestrator, CollaborationRequest, CollaborationMode
from src.ui.collaboration_ui import CollaborationUI
from src.agents.claude_agent import ClaudeCodeAgent
from src.agents.openai_agent import OpenAIAgent  
from src.agents.gemini_agent import GeminiAgent
from config.settings import settings

console = Console()


def print_banner():
    """Print the AngelaMCP banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     🎭 AngelaMCP - Multi-Agent Collaboration Platform        ║
║                                                               ║
║  Claude Code (🔧) + OpenAI (🧠) + Gemini (✨) = Better AI   ║
║                                                               ║
║  Watch AI agents debate, vote, and reach consensus!          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold blue")


def check_environment():
    """Check if the environment is properly configured."""
    console.print("🔍 Checking environment configuration...", style="yellow")
    
    issues = []
    
    # Check Claude Code path
    try:
        claude_path = Path(settings.claude_code_path)
        if not claude_path.exists():
            issues.append(f"❌ Claude Code not found at {claude_path}")
        else:
            console.print(f"✅ Claude Code found at {claude_path}")
    except Exception as e:
        issues.append(f"❌ Claude Code path error: {e}")
    
    # Check API keys
    try:
        if not settings.openai_api_key.get_secret_value():
            issues.append("❌ OpenAI API key not configured")
        else:
            console.print("✅ OpenAI API key configured")
    except Exception as e:
        issues.append(f"❌ OpenAI API key error: {e}")
    
    try:
        if not settings.google_api_key.get_secret_value():
            issues.append("❌ Google API key not configured")
        else:
            console.print("✅ Google API key configured")
    except Exception as e:
        issues.append(f"❌ Google API key error: {e}")
    
    if issues:
        console.print("\n⚠️ Environment Issues Found:", style="bold red")
        for issue in issues:
            console.print(f"  {issue}")
        console.print(f"\n💡 Please check your .env file and ensure all API keys are set.")
        return False
    
    console.print("✅ Environment looks good!\n", style="bold green")
    return True


async def test_individual_agents():
    """Test each agent individually."""
    console.print("🧪 Testing individual agents...", style="yellow")
    
    # Test Claude Code
    try:
        console.print("  Testing Claude Code agent...")
        claude = ClaudeCodeAgent()
        health = await claude.health_check()
        if health.get("status") == "healthy":
            console.print("  ✅ Claude Code agent working")
        else:
            console.print(f"  ❌ Claude Code agent issue: {health.get('error', 'Unknown')}")
    except Exception as e:
        console.print(f"  ❌ Claude Code agent failed: {e}")
    
    # Test OpenAI
    try:
        console.print("  Testing OpenAI agent...")
        openai = OpenAIAgent()
        health = await openai.health_check()
        if health.get("status") == "healthy":
            console.print("  ✅ OpenAI agent working")
        else:
            console.print(f"  ❌ OpenAI agent issue: {health.get('error', 'Unknown')}")
    except Exception as e:
        console.print(f"  ❌ OpenAI agent failed: {e}")
    
    # Test Gemini
    try:
        console.print("  Testing Gemini agent...")
        gemini = GeminiAgent()
        health = await gemini.health_check()
        if health.get("status") == "healthy":
            console.print("  ✅ Gemini agent working")
        else:
            console.print(f"  ❌ Gemini agent issue: {health.get('error', 'Unknown')}")
    except Exception as e:
        console.print(f"  ❌ Gemini agent failed: {e}")
    
    console.print()


async def demo_simple_collaboration():
    """Run a simple collaboration demo without full UI."""
    console.print("🤝 Running simple collaboration demo...", style="yellow")
    
    try:
        # Create orchestrator
        orchestrator = CollaborationOrchestrator()
        
        # Create a simple request
        request = CollaborationRequest(
            task_description="Create a Python function that calculates the factorial of a number",
            mode=CollaborationMode.CLAUDE_LEAD,  # Use Claude lead for faster demo
            timeout_minutes=3
        )
        
        console.print("📋 Task: Create a Python factorial function")
        console.print("🎯 Mode: Claude Lead (Claude implements, others review)")
        console.print("⏱️ Timeout: 3 minutes\n")
        
        # Run collaboration
        result = await orchestrator.collaborate(request)
        
        # Display results
        if result.success:
            console.print("🎉 Collaboration successful!", style="bold green")
            console.print(f"Winner: {result.chosen_agent}")
            console.print(f"Duration: {result.total_duration:.1f}s")
            console.print(f"Consensus: {result.consensus_reached}")
            
            if result.final_solution:
                console.print("\n📋 Final Solution:")
                solution_preview = result.final_solution[:300] + "..." if len(result.final_solution) > 300 else result.final_solution
                console.print(Panel(solution_preview, title="Generated Code", border_style="green"))
        else:
            console.print("❌ Collaboration failed!", style="bold red")
            console.print(f"Error: {result.error_message}")
    
    except Exception as e:
        console.print(f"❌ Demo failed: {e}", style="bold red")
    
    console.print()


async def demo_full_ui_collaboration():
    """Run the full collaboration with Rich UI."""
    console.print("🎭 Starting full UI collaboration demo...", style="bold yellow")
    console.print("This will show all agents working together in real-time!\n")
    
    # Get task from user or use default
    task_options = [
        "Create a Python function to calculate Fibonacci numbers",
        "Build a simple REST API for managing todo items",
        "Write a function to parse and validate email addresses",
        "Create a basic web scraper for extracting article titles",
        "Implement a binary search algorithm with proper error handling"
    ]
    
    console.print("📝 Choose a task for the agents to collaborate on:")
    for i, task in enumerate(task_options, 1):
        console.print(f"  {i}. {task}")
    
    try:
        choice = Prompt.ask("Enter choice (1-5) or press Enter for default", default="1")
        task_index = int(choice) - 1
        if 0 <= task_index < len(task_options):
            selected_task = task_options[task_index]
        else:
            selected_task = task_options[0]
    except (ValueError, KeyboardInterrupt):
        selected_task = task_options[0]
    
    console.print(f"\n🎯 Selected task: {selected_task}")
    console.print("Press Enter to continue...")
    input()
    
    try:
        # Create UI and orchestrator
        ui = CollaborationUI()
        orchestrator = CollaborationOrchestrator()
        
        # Create collaboration request
        request = CollaborationRequest(
            task_description=selected_task,
            mode=CollaborationMode.FULL_DEBATE,
            timeout_minutes=8
        )
        
        # Run with UI
        result = await ui.run_collaboration_with_ui(orchestrator, request)
        
        # Show final summary
        console.print("\n" + "="*60)
        console.print("🎭 COLLABORATION COMPLETE", style="bold cyan", justify="center")
        console.print("="*60)
        
        summary_table = ui.create_summary_table(result)
        console.print(summary_table)
        
        if result.success and result.final_solution:
            console.print(f"\n📋 Final Solution Preview:")
            solution_preview = result.final_solution[:500] + "..." if len(result.final_solution) > 500 else result.final_solution
            console.print(Panel(solution_preview, title=f"Solution by {result.chosen_agent}", border_style="green"))
        
        # Show what makes this special
        console.print("\n🌟 What Just Happened:")
        console.print("• Three AI agents worked together on your task")
        console.print("• Claude Code used its MCP tools for file system access")
        console.print("• OpenAI provided technical review and critique")
        console.print("• Gemini offered research and alternative approaches")
        console.print("• They debated, voted, and reached consensus")
        console.print("• You saw the whole process happen in real-time!")
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Collaboration cancelled by user", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Full demo failed: {e}", style="bold red")


async def main():
    """Main test runner."""
    print_banner()
    
    # Check environment
    if not check_environment():
        console.print("🛠️ Please fix environment issues before running the demo.", style="bold red")
        return
    
    # Menu
    while True:
        console.print("🎮 Choose an option:", style="bold cyan")
        console.print("1. Test individual agents")
        console.print("2. Simple collaboration demo (fast)")
        console.print("3. Full UI collaboration demo (recommended)")
        console.print("4. Exit")
        
        try:
            choice = Prompt.ask("Enter choice", choices=["1", "2", "3", "4"], default="3")
            
            if choice == "1":
                await test_individual_agents()
            elif choice == "2":
                await demo_simple_collaboration()
            elif choice == "3":
                await demo_full_ui_collaboration()
            elif choice == "4":
                console.print("👋 Thanks for trying AngelaMCP!", style="bold green")
                break
                
            if choice != "4":
                console.print("\nPress Enter to continue...")
                input()
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold yellow")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}", style="bold red")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="bold yellow")
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="bold red")