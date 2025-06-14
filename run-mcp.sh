#!/bin/bash
# AngelaMCP MCP Server Launcher
# This script ensures the MCP server runs from the correct directory with the right environment

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CLAUDE_CODE_PATH="${CLAUDE_CODE_PATH:-$HOME/.claude/local/claude}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating a template..."
    cp .env.example .env 2>/dev/null || echo "Please create a .env file with your API keys"
fi

# Run the MCP server
exec python -m src.main mcp-server