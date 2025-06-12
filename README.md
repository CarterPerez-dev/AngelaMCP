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


# Setup
TODO:




## 📖 Usage Examples

[Debate Initiated]
[Claude Code]: I recommend JWT for stateless authentication...
[OpenAI]: Consider session-based for better security...
[Gemini]: Here's a comparison of both approaches...
[Voting]: JWT (2 votes) vs Sessions (1 vote)
```

## 📚 Documentation

TODO:

## 🤝 Contributing

Please see [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🔮 Roadmap

- [ ] Web UI interface
- [ ] Plugin system for custom agents

---
Built with ❤️ using Claude Code, OpenAI, and Gemini
