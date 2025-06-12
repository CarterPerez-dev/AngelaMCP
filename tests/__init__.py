"""
AngelaMCP Test Suite

This package contains comprehensive tests for the AngelaMCP multi-agent
collaboration platform. The test suite is organized as follows:

Structure:
    unit/                   - Unit tests for individual components
        test_agents/        - Agent system tests
        test_orchestrator/  - Orchestration system tests  
        test_persistence/   - Database and persistence tests
        test_ui/           - User interface tests
    integration/           - Integration tests for full system
    fixtures/             - Shared test fixtures and data

Test Categories:
    - unit: Fast, isolated tests for individual components
    - integration: Tests that verify components work together
    - slow: Tests that take longer to run (marked separately)
    - ui: User interface specific tests
    - database: Tests requiring database setup

Running Tests:
    # All tests
    python scripts/run_tests.py --all
    
    # Unit tests only
    python scripts/run_tests.py --unit
    
    # Integration tests only  
    python scripts/run_tests.py --integration
    
    # Fast tests (excluding slow)
    python scripts/run_tests.py --fast
    
    # With coverage
    python scripts/run_tests.py --cov-html
    
    # Specific test
    python scripts/run_tests.py --test tests/unit/test_agents/test_base_agent.py

Coverage Target: >80%

The test suite uses pytest with async support and includes:
- Comprehensive fixtures for all components
- Mock agents and services for isolated testing
- Integration tests for end-to-end workflows
- Performance and stress tests
- UI component testing
- Database migration and persistence testing
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test configuration constants
TEST_TIMEOUT = 30  # Default test timeout in seconds
INTEGRATION_TIMEOUT = 60  # Integration test timeout
SLOW_TEST_TIMEOUT = 300  # Slow test timeout

# Export commonly used test utilities
from .conftest import *
