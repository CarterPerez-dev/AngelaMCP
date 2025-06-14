# AngelaMCP
# Multi-AI Agent Collaboration Claude-Code MCP

## 🎯 Overview

AngelaMCP enables AI collaboration through:
- **Claude Code MCP** as the primary agent with file system access and code execution
- **OpenAI o3** for code review and alternative perspectives
- **Gemini 2.5-pro** for research, documentation, and parallel analysis
- **Debate & Voting System** for collaborative decision-making
- **Async Task Management** for efficient parallel execution

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AngelaMCP Orchestrator                │
├─────────────────┬────────────────┬──────────────────────┤
│   Claude Code   │  OpenAI API    │    Gemini API        │
│      (MCP)      │                │                      │
│                 │                │                      │
│ • File System   │ • Code Review  │ • Research           │
│ • Code Exec     │ • Critique     │ • Documentation      │
│ • Main Tasks    │ • Validation   │ • Parallel Analysis  │
└─────────────────┴────────────────┴──────────────────────┘
                           │
                    PostgreSQL + Redis
                    (Persistence Layer)
```

## 📋 Prerequisites

- **Debian/Ubuntu Linux**
- **Python 3.10+**
- **PostgreSQL 14+**
- **Redis 6+**
- **Claude Code** installed and configured
- API keys for OpenAI and Google Gemini


## 🚀 Quick Start

### One-Command Setup
```bash
git clone <your-repo-url> AngelaMCP
cd AngelaMCP
./setup
```

The setup script will:
1. ✅ Install Python dependencies
2. 🐳 Start Docker services (PostgreSQL + Redis)
3. 📝 Collect your API keys interactively
4. 🗄️ Initialize database schema
5. 📄 Create `.mcp.json` with correct paths
6. 🔗 Register with Claude Code
7. ✅ Verify everything works

### Manual Setup (Alternative)
```bash
# 1. Install dependencies
make install

# 2. Copy environment template
cp .env.example .env
# Edit .env with your API keys

# 3. Start Docker services
make docker-up

# 4. Initialize database
make db-setup

# 5. Register MCP server
make mcp-register

# 6. Verify setup
make verify
```

### Testing Your Setup
```bash
# Run integration tests
python test_integration.py

# Test with Claude Code
claude "Use AngelaMCP to collaborate on building a calculator app"
```

## 🔧 Configuration

### Required API Keys
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **Google Gemini API Key**: Get from [makersuite.google.com](https://makersuite.google.com/app/apikey)

### Database Configuration
AngelaMCP uses Docker containers for databases by default:
- **PostgreSQL**: `localhost:5432` (angelamcp_db)
- **Redis**: `localhost:6379`

For local installation, update `.env` file accordingly.

### MCP Integration
The setup automatically creates `.mcp.json` with correct paths and registers with Claude Code.

## 📋 Available Commands

```bash
# Development
make run          # Start standalone CLI
make run-mcp      # Start MCP server
make test         # Run test suite
make lint         # Code linting

# Docker
make docker-up    # Start containers
make docker-down  # Stop containers
make docker-logs  # View logs

# MCP
make mcp-register       # Register with Claude Code
make mcp-test          # Test MCP integration
make mcp-clean-register # Clean & re-register

# Maintenance
make status       # Show system status
make clean        # Clean temp files
make reset        # Reset database
```

## 📖 Usage Examples

### Basic Collaboration
```bash
claude "Use AngelaMCP to help design a REST API for a blog platform"
```

### Structured Debate
```bash
claude "Use AngelaMCP to debate the best database choice for a high-traffic web app"
```

### Code Review
```bash
claude "Use AngelaMCP to review this Python function for performance issues"
```

### Example Output
```
**AngelaMCP Collaboration Result**

**Strategy Used:** DEBATE
**Success:** True
**Execution Time:** 12.34s
**Consensus Score:** 0.85

**Final Solution:**
After collaborative analysis, we recommend using PostgreSQL with Redis caching:

1. **PostgreSQL** (Primary Database)
   - Handles complex queries and transactions
   - ACID compliance for data integrity
   - Excellent performance for read-heavy workloads

2. **Redis** (Caching Layer)
   - Cache frequently accessed data
   - Session storage
   - Real-time analytics

**Cost Breakdown:**
- claude: $0.0234
- openai: $0.0156
- gemini: $0.0089

**Debate Summary:**
All agents agreed on PostgreSQL as the primary choice. OpenAI suggested 
additional optimization strategies, while Gemini provided comprehensive 
scaling considerations.
```

### Sample Collaboration Flow
```
[Debate Initiated] Topic: "Best authentication method for a financial API"
[Claude Code]: I recommend JWT with RS256 for stateless authentication...
[OpenAI]: Consider session-based for better security in financial contexts...
[Gemini]: Here's a comprehensive comparison of both approaches...
[Voting]: JWT (2 votes) vs Sessions (1 vote) - JWT selected
```

## 🏗️ Architecture Details

### Agent Roles
- **Claude Code**: Primary agent with file system access, code execution, and task leadership
- **OpenAI (GPT-4)**: Code review, optimization suggestions, and alternative perspectives  
- **Gemini**: Research, documentation, best practices, and parallel analysis

### Collaboration Strategies
1. **Single Agent**: Simple tasks handled by Claude Code alone
2. **Debate**: Structured discussion for complex decisions
3. **Consensus**: All agents must agree on the solution
4. **Parallel**: Tasks split and executed simultaneously
5. **Auto**: Intelligent strategy selection based on task complexity

### Data Persistence
- **PostgreSQL**: Stores conversation history, agent responses, and collaboration results
- **Redis**: Caches session data and provides real-time coordination

## 🔍 Troubleshooting

### Common Issues

**1. MCP Registration Failed**
```bash
# Check Claude Code installation
claude --version

# Manual registration
claude mcp add angelamcp "$(pwd)/run-mcp.sh"
```

**2. Database Connection Issues**
```bash
# Check Docker services
docker-compose -f docker/docker-compose.yml ps

# Restart services
make docker-down && make docker-up
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
echo $PYTHONPATH
```

**4. API Key Issues**
```bash
# Verify .env file
cat .env | grep -E "(OPENAI|GOOGLE)_API_KEY"

# Test API connections
python -c "import openai; print('OpenAI OK')"
python -c "import google.genai; print('Gemini OK')"
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true LOG_LEVEL=DEBUG
make run-mcp
```

### Health Check
```bash
make health    # Check all services
make status    # Show configuration
```

## 🤝 Contributing

Please see [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🔮 Roadmap

- [ ] Web UI interface
- [ ] Plugin system for custom agents

---
Built with ❤️ using Claude Code, OpenAI, and Gemini
