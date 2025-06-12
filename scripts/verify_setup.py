#!/usr/bin/env python3
"""
Setup verification script for AngelaMCP.

This checks all components and dependencies to ensure the Dockerized
environment is working correctly.
"""

import asyncio
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import inspect # We need this to check for async functions

# --- Add project root to path ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# --- Color codes for terminal output ---
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# --- Main Verifier Class ---
class AngelaMCPVerifier:
    def __init__(self):
        self.results = []
        # Import settings here so path is set
        try:
            from config.settings import settings
            self.settings = settings
        except Exception as e:
            print(f"{Colors.RED}❌ Critical Error: Could not load settings. Check .env and config/settings.py.{Colors.END}")
            print(f"   Details: {e}")
            sys.exit(1)

    # --- THIS IS THE CORRECTED run_check METHOD ---
    async def run_check(self, name: str, check_func, *args, **kwargs) -> bool:
        """Runs a single check and prints the result, handling both sync and async functions."""
        print(f"  {Colors.CYAN}Checking {name}...{Colors.END}", end='', flush=True)
        padding = 40 - len(name)
        print("." * padding, end='', flush=True)
        
        try:
            # Check if the function is async and await it if so
            if inspect.iscoroutinefunction(check_func):
                status, message = await check_func(*args, **kwargs)
            else:
                status, message = check_func(*args, **kwargs) # Run it synchronously
            
            if status == "PASS":
                print(f" {Colors.GREEN}[  OK  ]{Colors.END}")
                if message: print(f"    {Colors.GREEN}└─> {message}{Colors.END}")
                self.results.append(True)
                return True
            elif status == "WARN":
                print(f" {Colors.YELLOW}[ WARN ]{Colors.END}")
                if message: print(f"    {Colors.YELLOW}└─> {message}{Colors.END}")
                self.results.append(True) # Warnings don't cause failure
                return True
            else: # FAIL
                print(f" {Colors.RED}[ FAIL ]{Colors.END}")
                if message: print(f"    {Colors.RED}└─> {message}{Colors.END}")
                self.results.append(False)
                return False
        except Exception as e:
            print(f" {Colors.RED}[ ERROR ]{Colors.END}")
            print(f"    {Colors.RED}└─> An unexpected error occurred: {e}{Colors.END}")
            self.results.append(False)
            return False

    def check_python_version(self) -> Tuple[str, str]:
        """Checks Python version."""
        v = sys.version_info
        if v >= (3, 10):
            return "PASS", f"Python {v.major}.{v.minor}.{v.micro}"
        return "FAIL", f"Python version is {v.major}.{v.minor}. Require 3.10+"

    def check_dependencies(self) -> Tuple[str, str]:
        """Checks if required packages are installed in the venv."""
        try:
            with open(project_root / "requirements.txt") as f:
                reqs = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
            
            import_map = {'psycopg2-binary': 'psycopg2', 'google-generativeai': 'google.genai'}
            
            missing = []
            for req in reqs:
                try:
                    __import__(import_map.get(req, req))
                except ImportError:
                    missing.append(req)

            if not missing:
                return "PASS", f"{len(reqs)} dependencies are installed."
            return "FAIL", f"Missing packages: {', '.join(missing)}. Run 'make install'."
        except FileNotFoundError:
            return "FAIL", "requirements.txt not found."
    
    def check_env_file(self) -> Tuple[str, str]:
        """Checks for .env file."""
        if not (project_root / ".env").exists():
            return "FAIL", "'.env' file not found. Copy '.env.example' and fill it out."
        return "PASS", "'.env' file found."

    def check_api_keys(self) -> Tuple[str, str]:
        """Checks if API keys seem to be configured."""
        missing_keys = []
        if 'your-' in str(self.settings.openai_api_key) or not str(self.settings.openai_api_key):
            missing_keys.append("OPENAI_API_KEY")
        if 'your-' in str(self.settings.google_api_key) or not str(self.settings.google_api_key):
            missing_keys.append("GOOGLE_API_KEY")
        
        if not missing_keys:
            return "PASS", "OpenAI & Google API keys seem to be set."
        return "WARN", f"API keys might be missing or placeholders: {', '.join(missing_keys)}"

    def check_docker_running(self) -> Tuple[str, str]:
        """Checks if the Docker daemon is running."""
        try:
            result = subprocess.run(["docker", "ps"], capture_output=True, check=True)
            return "PASS", "Docker daemon is running."
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "FAIL", "Docker daemon is not running or 'docker' command is not in PATH."

    def check_project_containers(self) -> Tuple[str, str]:
        """Checks if the required project containers are running."""
        required = ["angelamcp", "angelamcp_postgres", "angelamcp_redis"]
        try:
            result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
            running_containers = result.stdout.strip().split('\n')
            
            missing = [name for name in required if name not in running_containers]
            
            if not missing:
                return "PASS", "All project containers are running."
            return "FAIL", f"Missing containers: {', '.join(missing)}. Run 'make docker-up'."
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "FAIL", "Could not check Docker containers. Is Docker running?"

    # --- THIS IS THE CORRECTED DB CONNECTION CHECK ---
    async def check_db_connection(self) -> Tuple[str, str]:
        """Checks database connectivity via asyncpg."""
        try:
            # The asyncpg library does not understand "+asyncpg", so we remove it for the test.
            # SQLAlchemy still needs it, but this direct test does not.
            connect_url = str(self.settings.database_url).replace("+asyncpg", "")
            
            import asyncpg
            conn = await asyncio.wait_for(asyncpg.connect(connect_url), timeout=5)
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            return "PASS", f"Connected to PostgreSQL ({version.split(' ')[1]})."
        except Exception as e:
            return "FAIL", f"Could not connect to database in Docker. Details: {e}"

    async def check_redis_connection(self) -> Tuple[str, str]:
        """Checks Redis connectivity."""
        try:
            import redis.asyncio as redis
            r = redis.from_url(str(self.settings.redis_url))
            await asyncio.wait_for(r.ping(), timeout=5)
            await r.aclose() # Use aclose() for newer versions
            return "PASS", "Connected to Redis."
        except Exception as e:
            return "FAIL", f"Could not connect to Redis in Docker. Details: {e}"

def print_header():
    """Prints the cool header."""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⠤⠶⠶⠶⠤⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠦⣄⠀⠀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⣠⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⣰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣆⠀⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⢠⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⠀⠀⠀")
    print("AngelaMCP ─ │⠀⢀⣠⠤⠶⠒⠒⠒⠒⠶⠥⠤⣀⡀⠀⢀⣀⠠⠤⠶⠒⠒⠒⠶⠬⣑⡀⠀│ ─ Verifier")
    print("⠀⠀⠀⠀⠀⠸⡄⠸⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⣬⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⢈⡗⠀⢸⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠙⠦⣝⠒⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠚⣁⡤⠚⠀⠀⠀⠀")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠒⠒⠦⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠴⠒⠋⠉⠀⠀⠀⠀⠀⠀⠀")
    print(f"{Colors.END}")

def print_success_footer():
    """Prints the very cool success message."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}==============================================================={Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}█▀▀ █░█ █▀▀ █▀▀ ▄▀█ █▀ █▀   █▀ ▄▀█ █▄█ █▀▀   █▀▄▀█ █▀▀ ▀█▀ █▀█{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}█▄▄ █▀█ ██▄ ██▄ █▀█ ▄█ ▄█   ▄█ █▀█ ░█░ ██▄   █░█░█ ██▄ ░█░ █▄█{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}==============================================================={Colors.END}")
    print(f"\n{Colors.CYAN}All systems operational. AngelaMCP is ready for deployment.{Colors.END}\n")
    print("  To start the application, run: " + f"{Colors.YELLOW}make run{Colors.END}")
    print("  To run as an MCP server, use: " + f"{Colors.YELLOW}make run-mcp{Colors.END}\n")

def print_failure_footer():
    """Prints the failure message."""
    print(f"\n{Colors.BOLD}{Colors.RED}==============================================================={Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}                             SETUP FAILED                             {Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}==============================================================={Colors.END}")
    print(f"\n{Colors.YELLOW}Please fix the [ FAIL ] checks above before proceeding.{Colors.END}\n")


async def main():
    """Main verification function."""
    print_header()
    time.sleep(1) # For dramatic effect
    
    verifier = AngelaMCPVerifier()

    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}Phase 1: Local Environment Checks{Colors.END}")
    await verifier.run_check("Python Version", verifier.check_python_version)
    await verifier.run_check("Python Dependencies", verifier.check_dependencies)
    await verifier.run_check("Environment File (.env)", verifier.check_env_file)
    await verifier.run_check("API Key Configuration", verifier.check_api_keys)
    
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}Phase 2: Docker Environment Checks{Colors.END}")
    await verifier.run_check("Docker Service", verifier.check_docker_running)
    await verifier.run_check("Project Containers", verifier.check_project_containers)

    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}Phase 3: Service Connectivity Checks{Colors.END}")
    await verifier.run_check("Database (PostgreSQL) Connection", verifier.check_db_connection)
    await verifier.run_check("Cache (Redis) Connection", verifier.check_redis_connection)

    if all(verifier.results):
        print_success_footer()
        return True
    else:
        print_failure_footer()
        return False


if __name__ == "__main__":
    is_success = False
    try:
        is_success = asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Verification interrupted by user.{Colors.END}")
        sys.exit(1)
    
    sys.exit(0 if is_success else 1)
