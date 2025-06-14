# 🚀 AngelaMCP Quick Start

**From bare Linux to fully working multi-AI collaboration in ONE command.**

## ⚡ Ultra-Quick Setup (1 Command!)

```bash
# Clone and setup everything automatically
git clone <repo-url> AngelaMCP && cd AngelaMCP && ./setup
```

**That's literally it!** The setup script will:
- ✅ Install all dependencies (Python, Docker, etc.)
- ✅ Configure databases (PostgreSQL + Redis)
- ✅ Interactively collect your API keys
- ✅ Initialize database schema
- ✅ Register MCP server with Claude Code
- ✅ Verify everything works
- ✅ Start the system

## 🎯 Alternative Methods

### If you already cloned the repo:
```bash
./setup              # One command does everything
```

### If you have an existing .env:
```bash
./setup              # Detects existing config, skips prompts
```

### Just the essentials:
```bash
make setup           # Same as ./setup
```

## 🔧 What You'll Need

The installer will prompt you for:

**Required:**
- OpenAI API Key
- Google Gemini API Key  
- Database username/password

**Optional:**
- GitHub token
- Email for alerts
- Custom Claude Code path

## 🎮 Usage

After setup:

```bash
# Start standalone mode
make run

# Start MCP server (for Claude Code)
make run-mcp

# Check status
make status

# View logs
make docker-logs
```

## 🔗 Connect with Claude Code

The installer automatically registers the MCP server. Just use:

```
claude "Use AngelaMCP to collaborate on this task"
```

## 🆘 Troubleshooting

```bash
# Check health
make health

# Verify setup
make verify

# Reset if needed
make reset

# Clean reinstall
make clean && make setup
```

## 📁 Key Files

- `.env` - Your configuration (auto-generated)
- `docker/docker-compose.yml` - Database containers
- `logs/` - Application logs
- `scripts/` - Utility scripts

---

**Need help?** Run `make help` for all available commands.

**From bare Linux to working AngelaMCP in 2 commands:**
```bash
git clone <repo> && cd AngelaMCP && make setup
make run
```

🎉 **That's it!** You now have a fully functional multi-AI collaboration platform!