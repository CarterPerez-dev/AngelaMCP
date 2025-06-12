"""
AngelaMCP Configuration Settings

Centralized configuration management using Pydantic Settings.
All settings are loaded from environment variables with validation.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import Field, validator, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings
from pydantic.types import SecretStr


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level options."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class UITheme(str, Enum):
    """UI theme options."""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class Settings(BaseSettings):
    """Main application settings."""

    # Application Settings
    app_name: str = Field(default="ANGELAMCP", env="APP_NAME")
    app_env: Environment = Field(default=Environment.DEVELOPMENT, env="APP_ENV")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_file: Path = Field(default=Path("logs/macp.log"), env="LOG_FILE")
    log_max_size: int = Field(default=10485760, env="LOG_MAX_SIZE")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    # Claude Code Configuration
    claude_code_path: Path = Field(default=Path("/usr/local/bin/claude"), env="CLAUDE_CODE_PATH")
    claude_code_timeout: int = Field(default=300, env="CLAUDE_CODE_TIMEOUT")
    claude_code_max_turns: int = Field(default=10, env="CLAUDE_CODE_MAX_TURNS")
    claude_code_output_format: str = Field(default="json", env="CLAUDE_CODE_OUTPUT_FORMAT")
    claude_session_persist: bool = Field(default=True, env="CLAUDE_SESSION_PERSIST")
    claude_session_dir: Path = Field(default=Path("~/.macp/claude_sessions").expanduser(), env="CLAUDE_SESSION_DIR")

    # OpenAI Configuration
    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="o3-mini", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")
    openai_top_p: float = Field(default=0.9, env="OPENAI_TOP_P")
    openai_frequency_penalty: float = Field(default=0.0, env="OPENAI_FREQUENCY_PENALTY")
    openai_presence_penalty: float = Field(default=0.0, env="OPENAI_PRESENCE_PENALTY")
    openai_timeout: int = Field(default=120, env="OPENAI_TIMEOUT")
    openai_rate_limit: int = Field(default=60, env="OPENAI_RATE_LIMIT")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    openai_retry_delay: int = Field(default=1, env="OPENAI_RETRY_DELAY")

    # Google Gemini Configuration
    google_api_key: SecretStr = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-pro-preview-06-05", env="GEMINI_MODEL")
    gemini_max_output_tokens: int = Field(default=4096, env="GEMINI_MAX_OUTPUT_TOKENS")
    gemini_temperature: float = Field(default=0.7, env="GEMINI_TEMPERATURE")
    gemini_top_p: float = Field(default=0.9, env="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=40, env="GEMINI_TOP_K")
    gemini_timeout: int = Field(default=120, env="GEMINI_TIMEOUT")
    gemini_harm_block_threshold: str = Field(default="BLOCK_MEDIUM_AND_ABOVE", env="GEMINI_HARM_BLOCK_THRESHOLD")
    gemini_rate_limit: int = Field(default=60, env="GEMINI_RATE_LIMIT")
    gemini_max_retries: int = Field(default=3, env="GEMINI_MAX_RETRIES")
    gemini_retry_delay: int = Field(default=1, env="GEMINI_RETRY_DELAY")

    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=40, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # Redis Configuration
    redis_url: str = Field(..., env="REDIS_URL")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    redis_decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    redis_connection_timeout: int = Field(default=5, env="REDIS_CONNECTION_TIMEOUT")

    # Session Configuration
    session_timeout: int = Field(default=3600, env="SESSION_TIMEOUT")
    session_cleanup_interval: int = Field(default=300, env="SESSION_CLEANUP_INTERVAL")
    max_conversation_length: int = Field(default=100, env="MAX_CONVERSATION_LENGTH")
    max_concurrent_sessions: int = Field(default=10, env="MAX_CONCURRENT_SESSIONS")

    # Task Execution Configuration
    task_queue_max_size: int = Field(default=100, env="TASK_QUEUE_MAX_SIZE")
    task_execution_timeout: int = Field(default=1800, env="TASK_EXECUTION_TIMEOUT")
    parallel_task_limit: int = Field(default=5, env="PARALLEL_TASK_LIMIT")

    # Debate Protocol Settings
    debate_timeout: int = Field(default=300, env="DEBATE_TIMEOUT")
    debate_max_rounds: int = Field(default=3, env="DEBATE_MAX_ROUNDS")
    debate_min_participants: int = Field(default=2, env="DEBATE_MIN_PARTICIPANTS")

    # Voting Settings
    voting_timeout: int = Field(default=60, env="VOTING_TIMEOUT")
    claude_vote_weight: float = Field(default=2.0, env="CLAUDE_VOTE_WEIGHT")
    openai_vote_weight: float = Field(default=1.0, env="OPENAI_VOTE_WEIGHT")
    gemini_vote_weight: float = Field(default=1.0, env="GEMINI_VOTE_WEIGHT")
    claude_veto_enabled: bool = Field(default=True, env="CLAUDE_VETO_ENABLED")

    # Feature Flags
    enable_cost_tracking: bool = Field(default=True, env="ENABLE_COST_TRACKING")
    enable_parallel_execution: bool = Field(default=True, env="ENABLE_PARALLEL_EXECUTION")
    enable_debate_mode: bool = Field(default=True, env="ENABLE_DEBATE_MODE")
    enable_auto_save: bool = Field(default=True, env="ENABLE_AUTO_SAVE")
    enable_memory_persistence: bool = Field(default=True, env="ENABLE_MEMORY_PERSISTENCE")
    enable_analytics: bool = Field(default=False, env="ENABLE_ANALYTICS")
    enable_telemetry: bool = Field(default=False, env="ENABLE_TELEMETRY")

    # Cost Tracking Configuration (USD per 1K tokens)
    openai_input_cost: float = Field(default=0.003, env="OPENAI_INPUT_COST")
    openai_output_cost: float = Field(default=0.006, env="OPENAI_OUTPUT_COST")
    gemini_input_cost: float = Field(default=0.00025, env="GEMINI_INPUT_COST")
    gemini_output_cost: float = Field(default=0.0005, env="GEMINI_OUTPUT_COST")
    daily_budget_limit: float = Field(default=10.00, env="DAILY_BUDGET_LIMIT")
    monthly_budget_limit: float = Field(default=250.00, env="MONTHLY_BUDGET_LIMIT")
    budget_warning_threshold: float = Field(default=0.8, env="BUDGET_WARNING_THRESHOLD")

    # UI Configuration
    ui_theme: UITheme = Field(default=UITheme.DARK, env="UI_THEME")
    ui_refresh_rate: int = Field(default=100, env="UI_REFRESH_RATE")
    ui_max_output_lines: int = Field(default=1000, env="UI_MAX_OUTPUT_LINES")
    ui_show_timestamps: bool = Field(default=True, env="UI_SHOW_TIMESTAMPS")
    ui_show_agent_icons: bool = Field(default=True, env="UI_SHOW_AGENT_ICONS")
    ui_enable_colors: bool = Field(default=True, env="UI_ENABLE_COLORS")
    ui_enable_animations: bool = Field(default=True, env="UI_ENABLE_ANIMATIONS")

    # File System Configuration
    workspace_dir: Path = Field(default=Path("~/.angelamcp/workspace").expanduser(), env="WORKSPACE_DIR")
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_file_extensions: List[str] = Field(
        default=[".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash"],
        env="ALLOWED_FILE_EXTENSIONS"
    )
    auto_save_interval: int = Field(default=60, env="AUTO_SAVE_INTERVAL")

    # Security Configuration
    enable_input_validation: bool = Field(default=True, env="ENABLE_INPUT_VALIDATION")
    enable_output_sanitization: bool = Field(default=True, env="ENABLE_OUTPUT_SANITIZATION")
    max_input_length: int = Field(default=10000, env="MAX_INPUT_LENGTH")
    blocked_commands: List[str] = Field(
        default=["rm -rf", "sudo rm", "format", "del /f"],
        env="BLOCKED_COMMANDS"
    )
    sandbox_mode: bool = Field(default=False, env="SANDBOX_MODE")

    @validator("claude_code_path")
    def validate_claude_path(cls, v: Path) -> Path:
        """Ensure Claude Code executable exists."""
        if not v.exists():
            # Try to find claude in PATH
            import shutil
            claude_path = shutil.which("claude")
            if claude_path:
                return Path(claude_path)
            raise ValueError(f"Claude Code not found at {v}. Please install Claude Code or update CLAUDE_CODE_PATH.")
        return v

    @validator("workspace_dir", "claude_session_dir", "log_file")
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        if v.suffix:  # It's a file
            v.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            v.mkdir(parents=True, exist_ok=True)
        return v

    @validator("allowed_file_extensions")
    def parse_file_extensions(cls, v: Any) -> List[str]:
        """Parse file extensions from string or list."""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v

    @validator("blocked_commands")
    def parse_blocked_commands(cls, v: Any) -> List[str]:
        """Parse blocked commands from string or list."""
        if isinstance(v, str):
            return [cmd.strip() for cmd in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == Environment.DEVELOPMENT

    def get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent type."""
        if agent_type == "claude_code":
            return {
                "path": str(self.claude_code_path),
                "timeout": self.claude_code_timeout,
                "max_turns": self.claude_code_max_turns,
                "output_format": self.claude_code_output_format,
            }
        elif agent_type == "openai":
            return {
                "api_key": self.openai_api_key.get_secret_value(),
                "model": self.openai_model,
                "max_tokens": self.openai_max_tokens,
                "temperature": self.openai_temperature,
                "timeout": self.openai_timeout,
            }
        elif agent_type == "gemini":
            return {
                "api_key": self.google_api_key.get_secret_value(),
                "model": self.gemini_model,
                "max_output_tokens": self.gemini_max_output_tokens,
                "temperature": self.gemini_temperature,
                "timeout": self.gemini_timeout,
            }
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        # Allow extra fields for forward compatibility
        extra = "allow"


# Create global settings instance
settings = Settings()
