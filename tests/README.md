# AngelaMCP Tests

This directory contains tests for the AngelaMCP multi-agent collaboration platform.

## Structure

```
tests/
├── unit/                              # Unit tests
│   ├── test_collaboration_orchestrator.py  # Core orchestrator tests
│   └── test_debate_system.py              # Debate & voting system tests
├── integration/                        # Integration tests
│   └── test_collaboration.py           # End-to-end collaboration test
├── conftest.py                        # Pytest configuration
└── README.md                          # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution, no external dependencies
- Mock external services and APIs

### Integration Tests (`tests/integration/`)
- Test complete workflows end-to-end
- May require API keys for full functionality
- Test real agent collaboration scenarios

## Running Tests

### Prerequisites
1. Install test dependencies:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

2. Set up environment (for integration tests):
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   ```

### Run All Tests
```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Run Specific Tests
```bash
# Test specific file
pytest tests/unit/test_collaboration_orchestrator.py

# Test specific test
pytest tests/unit/test_collaboration_orchestrator.py::TestCollaborationOrchestrator::test_orchestrator_initialization
```

## Test Scope

This test suite focuses on the **core collaboration functionality**:
- Multi-agent orchestration
- Debate protocol
- Weighted voting system
- Real-time UI integration

### What's Tested
✅ Collaboration orchestrator  
✅ Debate and voting systems  
✅ Data model structures  
✅ End-to-end collaboration flows  

### What's Not Tested
❌ Database persistence (not used in core system)  
❌ Complex caching (not part of core collaboration)  
❌ Legacy orchestration components  

## Writing New Tests

### Unit Test Example
```python
import pytest
from src.orchestrator.collaboration import CollaborationOrchestrator

class TestMyComponent:
    @pytest.fixture
    def component(self):
        return CollaborationOrchestrator()
    
    def test_basic_functionality(self, component):
        assert component is not None
```

### Integration Test Example
```python
import pytest
from src.orchestrator.collaboration import CollaborationRequest

@pytest.mark.asyncio
async def test_full_workflow():
    request = CollaborationRequest(task_description="Test task")
    # Test end-to-end flow
```

## Contributing

When adding new features:
1. Add unit tests for individual components
2. Add integration tests for end-to-end workflows
3. Keep tests focused on current architecture
4. Remove tests for deprecated components