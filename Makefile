# AngelaMCP Makefile
# Build automation for Multi-AI Agent Collaboration MCP Platfrorm

.PHONY: help setup install clean test run dev lint format db-init db-migrate docker-build docker-up

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3.10
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
PROJECT_NAME := macp
DB_NAME := angeladb
DB_USER := yoshi

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo '$(GREEN)AngelaMCP - Multi-AI Agent Collaboration Platform$(NC)'
	@echo ''
	@echo 'Usage:'
	@echo '  make $(YELLOW)<target>$(NC)'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Complete setup (venv, deps, db)
	@echo "$(GREEN)Setting up MACP...$(NC)"
	@make venv
	@make install
	@make db-init
	@echo "$(GREEN)Setup complete! Run 'make run' to start.$(NC)"

venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip setuptools wheel
	@echo "$(GREEN)Virtual environment created.$(NC)"

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed.$(NC)"

install-dev: install ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	@$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Development dependencies installed.$(NC)"

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(NC)"

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	@$(PYTHON_VENV) -m ruff check src/ tests/
	@$(PYTHON_VENV) -m mypy src/

format: ## Format code with black and ruff
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(PYTHON_VENV) -m black src/ tests/
	@$(PYTHON_VENV) -m ruff check --fix src/ tests/
	@echo "$(GREEN)Code formatted.$(NC)"

run: ## Run MACP
	@echo "$(GREEN)Starting AngelaMCP...$(NC)"
	@$(PYTHON_VENV) -m src.main

dev: ## Run in development mode with auto-reload
	@echo "$(GREEN)Starting AngelaMCP in development mode...$(NC)"
	@$(PYTHON_VENV) -m src.main --dev --debug

db-init: ## Initialize database
	@echo "$(GREEN)Initializing database...$(NC)"
	@$(PYTHON_VENV) scripts/init_db.py
	@echo "$(GREEN)Database initialized.$(NC)"

db-migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	@$(PYTHON_VENV) -m alembic upgrade head
	@echo "$(GREEN)Migrations complete.$(NC)"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data in the database!$(NC)"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	@sudo -u postgres psql -c "DROP DATABASE IF EXISTS $(DB_NAME);"
	@sudo -u postgres psql -c "CREATE DATABASE $(DB_NAME) OWNER $(DB_USER);"
	@make db-init
	@echo "$(GREEN)Database reset complete.$(NC)"

verify: ## Verify installation
	@echo "$(GREEN)Verifying AngelaMCP installation...$(NC)"
	@$(PYTHON_VENV) scripts/verify_setup.py

agent-test: ## Test agent connectivity
	@echo "$(GREEN)Testing agent connectivity...$(NC)"
	@$(PYTHON_VENV) scripts/test_agents.py

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	@docker build -t $(PROJECT_NAME):latest -f docker/Dockerfile .
	@echo "$(GREEN)Docker image built.$(NC)"

docker-up: ## Start Docker services
	@echo "$(GREEN)Starting Docker services...$(NC)"
	@docker-compose -f docker/docker-compose.yml up -d
	@echo "$(GREEN)Services started.$(NC)"

docker-down: ## Stop Docker services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	@docker-compose -f docker/docker-compose.yml down
	@echo "$(GREEN)Services stopped.$(NC)"

docker-logs: ## View Docker logs
	@docker-compose -f docker/docker-compose.yml logs -f

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	@$(PYTHON_VENV) -m mkdocs build
	@echo "$(GREEN)Documentation built in site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	@$(PYTHON_VENV) -m mkdocs serve

version: ## Show version information
	@echo "$(GREEN)MACP Version Information$(NC)"
	@echo "Python: $(shell $(PYTHON_VENV) --version)"
	@echo "Claude: $(shell claude --version 2>/dev/null || echo 'Not installed')"
	@echo "PostgreSQL: $(shell psql --version 2>/dev/null || echo 'Not installed')"
	@echo "Redis: $(shell redis-cli --version 2>/dev/null || echo 'Not installed')"

logs: ## Tail application logs
	@echo "$(GREEN)Tailing AngelaMCP logs...$(NC)"
	@tail -f logs/macp.log

monitor: ## Monitor system resources
	@echo "$(GREEN)Monitoring AngelaMCP resources...$(NC)"
	@watch -n 1 '$(PYTHON_VENV) scripts/monitor.py'

backup: ## Backup database
	@echo "$(GREEN)Backing up database...$(NC)"
	@mkdir -p backups
	@pg_dump -U $(DB_USER) -d $(DB_NAME) -f backups/$(DB_NAME)_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backed up to backups/$(NC)"

restore: ## Restore database from latest backup
	@echo "$(YELLOW)Restoring from latest backup...$(NC)"
	@psql -U $(DB_USER) -d $(DB_NAME) -f $(shell ls -t backups/*.sql | head -1)
	@echo "$(GREEN)Database restored.$(NC)"

# Development shortcuts
.PHONY: d r t l f
d: dev ## Shortcut for 'make dev'
r: run ## Shortcut for 'make run'
t: test ## Shortcut for 'make test'
l: lint ## Shortcut for 'make lint'
f: format ## Shortcut for 'make format'
