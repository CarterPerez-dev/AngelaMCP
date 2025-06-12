# MACP Project Structure

```
AngelaMCP/
├── README.md                 # Main project documentation
├── SETUP.md                  # Detailed setup instructions
├── ARCHITECTURE.md           # System architecture documentation
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment configuration
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Project configuration (ruff, pytest)
├── Makefile                 # Build automation
│
├── config/
│   ├── __init__.py
│   ├── settings.py          # Pydantic settings management
│   ├── models.py            # Model configurations
│   └── prompts/             # System prompts for agents
│       ├── debate.yaml
│       └── templates.yaml
│
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── cli.py               # CLI interface
│   │
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base agent interface
│   │   ├── claude_agent.py  # Claude Code wrapper
│   │   ├── openai_agent.py  # OpenAI o3-mini
│   │   └── gemini_agent.py  # Gemini 2.5-pro
│   │
│   ├── orchestrator/        # Core orchestration logic
│   │   ├── __init__.py
│   │   ├── manager.py       # Main orchestrator
│   │   ├── debate.py        # Debate protocol
│   │   ├── voting.py        # Voting mechanism
│   │   └── task_queue.py    # Async task management
│   │
│   ├── persistence/         # Database layer
│   │   ├── __init__.py
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── database.py      # DB connection management
│   │   └── repositories.py  # Data access layer
│   │
│   ├── ui/                  # Terminal interface
│   │   ├── __init__.py
│   │   ├── display.py       # Rich terminal UI
│   │   ├── streaming.py     # Real-time output
│   │   └── input_handler.py # User input management
│   │
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── logger.py        # Logging configuration
│       ├── exceptions.py    # Custom exceptions
│       └── helpers.py       # Helper functions
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest configuration
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test data
│
├── scripts/                 # Utility scripts
│   ├── setup_db.py         # Database initialization
│   ├── migrate.py          # Database migrations
│   └── test_agents.py      # Agent connectivity test
│
├── docker/                  # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── postgres/
│       └── init.sql        # DB initialization
│
└── docs/                    # Additional documentation
    ├── API.md              # API documentation
    ├── CONTRIBUTING.md     # Contribution guidelines
    └── examples/           # Usage examples
```
