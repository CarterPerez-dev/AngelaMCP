# AngelaMCP - Multi-AI Agent Collaboration Platform
# Production-grade Makefile for complete setup and management

SHELL = /bin/bash

PYTHON = venv/bin/python

.PHONY: help setup install db-setup verify run run-mcp test clean lint format \
        docker-build docker-up docker-down docker-logs docker-clean docker-prune-all \
		mcp-register mcp-test status

PYTHON_VERSION := $(shell python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')


# Default target
help:
	@echo "👻 AngelaMCP - Multi-AI Agent Collaboration Platform"
	@echo "=================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  ./setup           - ONE command setup (RECOMMENDED)"
	@echo "  make setup        - Same as above"
	@echo "  make docker-mode  - Docker-only databases"
	@echo ""
	@echo "📋 Manual Setup Commands:"
	@echo "  make setup-manual - Manual step-by-step setup"
	@echo "  make install      - Install Python dependencies only"
	@echo "  make deps         - Install system dependencies"
	@echo "  make db-setup     - Setup database"
	@echo "  make verify       - Verify entire setup is working"
	@echo ""
	@echo "Development Commands:"
	@echo "  make run          - Run AngelaMCP standalone CLI"
	@echo "  make run-mcp      - Run as MCP server"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with ruff"
	@echo ""
	@echo "MCP Commands:"
	@echo "  make mcp-register       - Register with Claude Code"
	@echo "  make mcp-clean-register - Remove old servers + register fresh"
	@echo "  make mcp-test           - Test MCP integration"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker containers"
	@echo "  make docker-up    - Start with Docker Compose"
	@echo "  make docker-down  - Stop Docker containers"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Clean temporary files"
	@echo "  make reset        - Reset database and logs"

# Interactive one-click setup
setup:
	@echo "🚀 Starting AngelaMCP interactive setup..."
	@bash setup

# Complete setup for new installations (manual mode)
setup-manual: deps install env-setup db-setup verify
	@echo "✅ Manual setup finished!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run: make verify"
	@echo "3. Run: make mcp-register"
	@echo "4. Test: make run"

# Install system dependencies
deps:
	@echo "📦 Installing system dependencies..."
	sudo apt update
	sudo apt install -y postgresql postgresql-contrib redis-server python$(PYTHON_VERSION)-dev libpq-dev
	sudo systemctl start postgresql redis-server
	sudo systemctl enable postgresql redis-server
	@echo "✅ System dependencies installed"

# Install Python dependencies
install:
	@echo "🐍 Installing Python dependencies..."
	@test -d venv || python3 -m venv venv 
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Python dependencies installed"
        
# Setup environment file
env-setup:
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env with your API keys"; \
	else \
		echo "✅ .env file already exists"; \
	fi

# Complete database setup
db-setup: db-init db-verify
	@echo "✅ Database setup complete"

# Initialize database
db-init:
	@echo "🗄️ Initializing database..."
	$(PYTHON) scripts/init_db.py

# Verify database setup
db-verify:
	@echo "🔍 Verifying database..."
	$(PYTHON) -c "import asyncio; from src.persistence.database import DatabaseManager; asyncio.run(DatabaseManager().initialize())"

# Verify complete setup
verify:
	@echo "🔍 Verifying AngelaMCP setup..."
	$(PYTHON) scripts/verify_setup.py

# Run AngelaMCP standalone
run:
	@echo "🕵️‍♀️ Starting AngelaMCP standalone..."
	$(PYTHON) -m src.main

# Run as MCP server
run-mcp:
	@echo "🧙‍♀️ Starting AngelaMCP MCP server..."
	@chmod +x run-mcp.sh
	./run-mcp.sh

# Run development mode with debug
run-dev:
	@echo "👩‍🎨 Starting AngelaMCP in development mode..."
	DEBUG=true LOG_LEVEL=DEBUG python -m src.main

# Register MCP server with Claude Code
mcp-register:
	@echo "💃 Registering AngelaMCP with Claude Code..."
	@if command -v claude >/dev/null 2>&1; then \
		chmod +x $(PWD)/run-mcp.sh && \
		claude mcp add angelamcp "$(PWD)/run-mcp.sh" && \
		echo "✅ MCP server registered successfully" && \
		echo "Test with: claude 'Use AngelaMCP to help with a task'"; \
	else \
		echo "❌ Claude Code not found. Install Claude Code first."; \
		exit 1; \
	fi

# Clean up old MCP servers and register fresh
mcp-clean-register:
	@echo "🧹 Cleaning up old MCP servers..."
	@if command -v claude >/dev/null 2>&1; then \
		claude mcp remove angelamcp 2>/dev/null || true; \
		claude mcp remove multi-ai-collab 2>/dev/null || true; \
		chmod +x $(PWD)/run-mcp.sh; \
		echo "✅ Old servers removed"; \
		$(MAKE) mcp-register; \
	else \
		echo "❌ Claude Code not found"; \
	fi

# Test MCP integration
mcp-test:
	@echo "🧪 Testing MCP integration..."
	@if command -v claude >/dev/null 2>&1; then \
		claude mcp list | grep angelamcp || echo "❌ AngelaMCP not registered"; \
		echo "Run 'make mcp-register' to register the server"; \
	else \
		echo "❌ Claude Code not found"; \
	fi

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
	@echo "🔍 Linting code..."
	ruff check src/ config/ tests/
	mypy src/ --ignore-missing-imports

# Format code
format:
	@echo "✨ Formatting code..."
	ruff format src/ config/ tests/
	ruff check --fix src/ config/ tests/

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	@echo "✅ Cleanup complete"

# Reset database and logs
reset: clean
	@echo "🔄 Resetting AngelaMCP..."
	rm -rf logs/*.log
	@echo "Reset database? [y/N]" && read ans && [ $${ans:-N} = y ] && \
		python -c "from sqlalchemy import create_engine; from config.settings import settings; from src.persistence.models import Base; engine = create_engine(str(settings.database_url)); Base.metadata.drop_all(engine); Base.metadata.create_all(engine); print('✅ Database reset')" || \
		echo "Database reset skipped"


docker-build:
	@echo "🐳 Building Docker containers..."
	@docker-compose --env-file .env -f docker/docker-compose.yml build

docker-up:
	@echo "🐳 Starting Docker containers..."
	@docker-compose --env-file .env -f docker/docker-compose.yml up -d
	@echo "✅ Containers started. Check status with: docker-compose ps"

docker-down:
	@echo "🐳 Stopping Docker containers..."
	@docker-compose --env-file .env -f docker/docker-compose.yml down
	
docker-logs:
	@echo "📋 Showing Docker logs..."
	@docker-compose --env-file .env -f docker/docker-compose.yml logs -f

docker-clean:
	@echo "🧹 Cleaning up all containers, networks, and volumes for THIS project..."
	@docker-compose --env-file .env -f docker/docker-compose.yml down --volumes
	@echo "✅ Project cleanup complete."

docker-prune-all:
	@echo "☢️  WARNING: This will remove ALL unused Docker assets on your entire system."
	@echo "This affects ALL projects, not just this one. This cannot be undone."
	@read -p "Are you absolutely sure you want to continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Pruning system..."; \
		docker system prune -a --volumes -f; \
		echo "✅ Docker system prune complete."; \
	else \
		echo "Prune cancelled."; \
	fi


# Health check
health:
	@echo "🏥 AngelaMCP Health Check..."
	@echo "PostgreSQL:" && systemctl is-active postgresql && echo "✅" || echo "❌"
	@echo "Redis:" && systemctl is-active redis-server && echo "✅" || echo "❌"
	@echo "Claude Code:" && claude --version >/dev/null 2>&1 && echo "✅" || echo "❌"
	@echo "Python deps:" && python -c "import openai, google.genai; print('✅')" 2>/dev/null || echo "❌"

# Show status
status:
	@echo "🕵️‍♀️ AngelaMCP Status"
	@echo "=================="
	@echo "Project Root: $(PWD)"
	@echo "Python: $(shell python --version)"
	@echo "Environment: $(shell [ -f .env ] && echo '✅ Configured' || echo '❌ Missing')"
	@echo "Database: $(shell python -c "from config.settings import settings; print(settings.database_url)" 2>/dev/null | sed 's/:.*/.../' || echo '❌ Not configured')"
	@echo "MCP Server: $(shell claude mcp list 2>/dev/null | grep -q angelamcp && echo '✅ Registered' || echo '❌ Not registered')"

# Development shortcuts
dev: run-dev
prod: run
server: run-mcp

# All-in-one commands for different use cases
first-time: setup
	@echo "👻 First-time setup complete!"
	@echo "Try: make run"

quick-start: install verify run

# Docker-only mode (databases in Docker, MCP on host)
docker-mode: docker-up
	@echo "🐳 Docker databases started"
	@echo "Set environment: export CLAUDE_CODE_PATH=\"$$HOME/.claude/local/claude\""
	@echo "Run MCP server: make run-mcp"

# Everything in one command (recommended path)
easy-setup: setup
	@echo "🎉 AngelaMCP is ready!"

# Backup and restore (for production)
backup:
	@echo "💾 Creating backup..."
	mkdir -p backups
	pg_dump $(shell python -c "from config.settings import settings; print(settings.database_url)") > backups/angelamcp_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/"

# Show environment info
env-info:
	@echo "🔧 Environment Information"
	@echo "========================="
	@python -c "from config.settings import settings; import json; print(json.dumps({k: '***' if 'key' in k.lower() or 'password' in k.lower() else str(v) for k, v in settings.__dict__.items() if not k.startswith('_')}, indent=2))"
