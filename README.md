# AngelaMCP
# Multi-AI Agent Collaboration Platform

A production-grade terminal-based platform that orchestrates collaboration between multiple AI agents, featuring Claude Code as the senior developer with OpenAI and Gemini as supporting agents.

## 🎯 Overview

AngelaMCP enables sophisticated AI collaboration through:
- **Claude Code** as the primary agent with file system access and code execution
- **OpenAI o3-mini** for code review and alternative perspectives
- **Gemini 2.5-pro** for research, documentation, and parallel analysis
- **Debate & Voting System** for collaborative decision-making
- **Async Task Management** for efficient parallel execution
- **Rich Terminal UI** with real-time streaming

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AngelaMCP Orchestrator                      │
├─────────────────┬────────────────┬──────────────────────┤
│   Claude Code   │  OpenAI API   │    Gemini API        │
│   (via CLI)     │   (o3-mini)   │ (2.5-pro-preview)    │
│                 │               │                       │
│ • File System   │ • Code Review │ • Research           │
│ • Code Exec     │ • Critique    │ • Documentation      │
│ • Main Tasks    │ • Validation  │ • Parallel Analysis  │
└─────────────────┴────────────────┴──────────────────────┘
                           │
                    PostgreSQL + Redis
                    (Persistence Layer)
```

## ✨ Key Features

### 1. **Intelligent Task Distribution**
- Claude Code handles primary development tasks
- Parallel task execution for supporting agents
- Smart routing based on task type

### 2. **Collaborative Decision Making**
- Agents propose solutions independently
- Structured debate protocol for critiques
- Weighted voting system (Claude Code has senior vote)
- Consensus building with override capabilities

### 3. **Production-Ready Infrastructure**
- PostgreSQL for conversation persistence
- Redis for session caching
- Comprehensive error handling
- Rate limiting and cost tracking
- Async/await for optimal performance

### 4. **Rich Terminal Experience**
- Real-time streaming of agent thoughts
- Interactive command interface
- Progress indicators and status updates
- Color-coded agent outputs

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/CarterPerez-dev/AngelaMCP.git
cd AngelaMCP

# Set up the environment
make setup

# Configure your API keys
cp .env.example .env
# Edit .env with your API keys

# Initialize the database
make db-init

# Run the platform
make run
```

## 📋 Prerequisites

- **Debian/Ubuntu Linux** (tested on Debian 11+)
- **Python 3.10+**
- **PostgreSQL 14+**
- **Redis 6+**
- **Claude Code** installed and configured
- API keys for OpenAI and Google Gemini

## 🔧 Configuration

The platform uses environment variables for configuration:

```bash
# Claude Code (uses existing Claude account)
CLAUDE_CODE_PATH=/usr/local/bin/claude

# OpenAI
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=o3-mini # Or change it (Use smartest model is my recommendation)

# Google Gemini
GOOGLE_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.5-pro-preview-06-05

# Database
DATABASE_URL=postgresql://user:pass@localhost/angelamcp
REDIS_URL=redis://localhost:6379
```

## 📖 Usage Examples

### Basic Task Execution
```bash
# Start MACP
$ macp

MACP> Create a REST API for a todo application with authentication

[Claude Code]: Analyzing requirements...
[Gemini]: Researching best practices for REST API design...
[OpenAI]: Preparing to review implementation...
```

### Guided Collaboration
```bash
MACP> /debate Should we use JWT or session-based auth?

[Debate Initiated]
[Claude Code]: I recommend JWT for stateless authentication...
[OpenAI]: Consider session-based for better security...
[Gemini]: Here's a comparison of both approaches...
[Voting]: JWT (2 votes) vs Sessions (1 vote)
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration

# Test agent connectivity
python scripts/test_agents.py
```

## 📚 Documentation

- [SETUP.md](./SETUP.md) - Detailed installation instructions
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design and flow
- [API.md](./docs/API.md) - API reference
- [Examples](./docs/examples/) - Usage examples

## 🤝 Contributing

Please see [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🔮 Roadmap

- [ ] Web UI interface
- [ ] Plugin system for custom agents
- [ ] Advanced memory and learning
- [ ] Multi-project workspace support
- [ ] Team collaboration features

---

Built with ❤️ using Claude Code, OpenAI, and Gemini
