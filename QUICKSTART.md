# 🚀 AngelaMCP Quick Start Guide

Your Multi-AI Agent Collaboration Platform is now fully implemented and ready to use!

## ⚡ Quick Setup (5 minutes)

### 1. **Setup Environment**
```bash
# Navigate to your project
cd /home/yoshi/AngelaMCP

# Complete automated setup
make setup
```

### 2. **Configure API Keys**
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required API Keys:**
- `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys
- `GOOGLE_API_KEY` - Get from https://makersuite.google.com/app/apikey
- Database URL is already configured for your setup: `postgresql://yoshi:yoshi@localhost:5432/angeladb`

### 3. **Start AngelaMCP**
```bash
# Start the platform
make run

# Or use the direct command
macp
```

## 🎯 What You Can Do Now

### **Basic Commands**
```bash
MACP> /help              # Show all commands
MACP> /status            # System status
MACP> /agents            # Agent information
MACP> /metrics           # Performance metrics
```

### **Task Examples**
```bash
# Code generation
MACP> Create a Python function to calculate fibonacci numbers

# Code review
MACP> /debate Should we use JWT or session-based authentication?

# Research task
MACP> Research best practices for REST API design

# Multi-agent collaboration
MACP> Build a complete todo application with authentication
```

### **Advanced Features**
```bash
# Voting on decisions
MACP> /vote Which database should we use for this project?

# View conversation history
MACP> /history

# Check system metrics
MACP> /metrics
```

## 🛠️ Development Commands

```bash
# Verify setup
make verify

# Test agent connectivity  
make agent-test

# Run tests
make test

# Format code
make format

# View logs
make logs

# Monitor resources
make monitor
```

## 🔧 Configuration

All settings are in `.env`. Key configurations:
- **Models**: OpenAI o3-mini, Gemini 2.5-pro-preview
- **Database**: PostgreSQL (angeladb)
- **Cache**: Redis
- **Logs**: `logs/macp.log`

## 🎉 You're Ready!

Your AngelaMCP platform includes:
- ✅ Full CLI with 15+ commands
- ✅ Claude Code as senior developer
- ✅ OpenAI for code review & analysis  
- ✅ Gemini for research & documentation
- ✅ Debate & voting system
- ✅ Async task management
- ✅ PostgreSQL + Redis persistence
- ✅ Rich terminal UI
- ✅ Comprehensive metrics
- ✅ Production-ready architecture

## 🚨 Troubleshooting

**Database Connection Issues:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check database exists
psql -U yoshi -d angeladb -c "SELECT 1;"
```

**Redis Connection Issues:**
```bash
# Check Redis is running
sudo systemctl status redis
```

**Claude Code Issues:**
```bash
# Verify Claude Code installation
claude --version
```

## 📚 Next Steps

1. **Try the platform**: `make run`
2. **Read the full documentation**: `ARCHITECTURE.md`
3. **Customize settings**: Edit `.env`
4. **Add custom agents**: Extend the agent system
5. **Deploy to production**: Use Docker setup

**Happy coding with your AI agents! 🤖✨**