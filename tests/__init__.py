"""Test package for AngelaMCP."""

import os
import sys
from pathlib import Path

# Add src to path for testing
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATABASE_URL = "sqlite:///./test_angelamcp.db"
TEST_REDIS_URL = "redis://localhost:6379/1" 

# Common test fixtures and utilities
def get_test_config():
    """Get test configuration."""
    return {
        "database_url": TEST_DATABASE_URL,
        "redis_url": TEST_REDIS_URL,
        "debug": True,
        "log_level": "DEBUG"
    }

__all__ = ["get_test_config", "TEST_DATABASE_URL", "TEST_REDIS_URL"]
