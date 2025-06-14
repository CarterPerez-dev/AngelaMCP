#!/bin/bash
# clear_cache_and_verify.sh - Complete cache clearing and verification script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧹 Clear All Caches and Verify Fix${NC}"
echo "=================================="

# 1. Kill ALL processes first
echo -e "\n${YELLOW}1. Killing all MCP processes...${NC}"
pkill -f "mcp-server" 2>/dev/null || true
pkill -f "src.main" 2>/dev/null || true
pkill -f "run-mcp.sh" 2>/dev/null || true
pkill -f "python.*mcp" 2>/dev/null || true

# Wait for processes to die
sleep 3

# 2. Clear Python cache
echo -e "\n${YELLOW}2. Clearing Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# 3. Clear any Python import cache
echo -e "\n${YELLOW}3. Clearing Python import cache...${NC}"
export PYTHONDONTWRITEBYTECODE=1

# 4. Verify the file was actually updated
echo -e "\n${YELLOW}4. Verifying mcp_server.py has the fix...${NC}"
if grep -q "_handle_collaborate" src/mcp_server.py; then
    echo -e "${GREEN}✅ File contains new handler methods${NC}"
else
    echo -e "${RED}❌ File still has old structure - need to update it${NC}"
    echo "The file wasn't properly updated. Let me show you what to do:"
    echo ""
    echo "1. Open src/mcp_server.py in your editor"
    echo "2. Replace the entire content with the fixed version"
    echo "3. Save the file"
    exit 1
fi

# 5. Check for the specific problematic pattern
if grep -q "async def get_agent_status():" src/mcp_server.py; then
    echo -e "${RED}❌ Found old function signature without arguments parameter${NC}"
    echo "The file still has the old problematic function signature"
    exit 1
else
    echo -e "${GREEN}✅ No old function signatures found${NC}"
fi

# 6. Verify the new structure exists
if grep -q "async def _handle_get_agent_status" src/mcp_server.py; then
    echo -e "${GREEN}✅ New handler method structure found${NC}"
else
    echo -e "${RED}❌ New handler methods not found${NC}"
    exit 1
fi

# 7. Clear any Claude Code cache
echo -e "\n${YELLOW}5. Clearing Claude Code cache...${NC}"
rm -rf ~/.claude/cache 2>/dev/null || true
rm -rf ~/.claude/mcp_cache 2>/dev/null || true

# 8. Remove any .mcp.json to force re-registration
echo -e "\n${YELLOW}6. Backing up and clearing MCP registration...${NC}"
if [ -f ".mcp.json" ]; then
    cp .mcp.json .mcp.json.backup
    echo "Backed up .mcp.json to .mcp.json.backup"
fi

# 9. Clean restart environment
echo -e "\n${YELLOW}7. Setting up clean environment...${NC}"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# 10. Re-register MCP server with clean state
echo -e "\n${YELLOW}8. Re-registering MCP server...${NC}"
if command -v claude >/dev/null 2>&1; then
    # Remove old registration
    claude mcp remove angelamcp 2>/dev/null || true
    
    # Re-add with explicit path
    claude mcp add angelamcp "$(pwd)/run-mcp.sh"
    echo -e "${GREEN}✅ MCP server re-registered${NC}"
else
    echo -e "${YELLOW}⚠️  Claude Code not found in PATH${NC}"
fi

# 11. Start the server fresh
echo -e "\n${YELLOW}9. Starting fresh MCP server...${NC}"
echo "Starting server in background..."

# Make sure run-mcp.sh is executable
chmod +x run-mcp.sh

# Start the server
./run-mcp.sh &
SERVER_PID=$!

# Wait a moment for startup
sleep 5

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${GREEN}✅ MCP server started successfully (PID: $SERVER_PID)${NC}"
else
    echo -e "${RED}❌ MCP server failed to start${NC}"
    exit 1
fi

echo -e "\n${GREEN}🎉 Cache cleared and server restarted!${NC}"
echo ""
echo "Now try testing the MCP tools:"
echo "1. angelamcp:get_agent_status"
echo "2. angelamcp:debate (topic: \"test topic\")"
echo ""
echo "If it still fails, the file might not have been updated correctly."
