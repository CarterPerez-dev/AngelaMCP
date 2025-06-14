#!/bin/bash
# AngelaMCP MCP Server Launcher
# This script ensures the MCP server runs from the correct directory with the right environment

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run setup first:" >&2
    echo "   ./setup" >&2
    exit 1
fi

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CLAUDE_CODE_PATH="${CLAUDE_CODE_PATH:-$HOME/.claude/local/claude}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please run setup first:" >&2
    echo "   ./setup" >&2
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Verify key dependencies
if ! python -c "import openai, google.genai" 2>/dev/null; then
    echo "❌ Required Python packages not found. Please run:" >&2
    echo "   pip install -r requirements.txt" >&2
    exit 1
fi

# Start Docker services if they're not running
if command -v docker-compose >/dev/null 2>&1; then
    if ! docker-compose -f docker/docker-compose.yml ps | grep -q "Up"; then
        echo "🐳 Starting Docker services..."
        docker-compose -f docker/docker-compose.yml up -d
        sleep 5  # Give services time to start
    fi
fi

# Run the MCP server
echo "🚀 Starting AngelaMCP MCP Server..."
exec python -m src.main mcp-server
