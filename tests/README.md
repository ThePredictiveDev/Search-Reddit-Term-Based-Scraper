# 🧪 Reddit Mention Tracker - Testing Guide

This directory contains comprehensive tests for the Reddit Mention Tracker system. The testing framework is designed to ensure reliability, performance, and security across all components.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── pytest.ini                 # Pytest configuration
├── test_runner.py             # Advanced test runner script
├── README.md                  # This documentation
├── test_database.py           # Database and models tests
├── test_scraper.py            # Reddit scraper tests
├── test_analytics.py          # Analytics and metrics tests
├── test_ui.py                 # UI and visualization tests
├── test_integration.py        # Integration tests
└── test_reports/              # Generated test reports
```

## 🚀 Quick Start

### 1. Setup Test Environment

```bash
# Install test dependencies
pip install -r requirements.txt

# Setup test environment
python tests/test_runner.py --setup-only
```

### 2. Run Basic Tests

```bash
# Run all unit tests
python tests/test_runner.py --category unit

# Run specific test file
python tests/test_runner.py --file test_database.py

# Run smoke tests (quick validation)
python tests/test_runner.py --smoke
```

### 3. Run Full Test Suite

```bash
# Run complete test suite with coverage
python tests/test_runner.py --full
```

## 📊 Test Categories

### Unit Tests
- **Database Tests** (`test_database.py`): Database models, operations, and integrity
- **Scraper Tests** (`test_scraper.py`): Reddit scraping functionality and edge cases
- **Analytics Tests** (`test_analytics.py`): Metrics calculation and data validation
- **UI Tests** (`test_ui.py`): Visualization and interface components

### Integration Tests
- **System Integration** (`test_integration.py`): End-to-end workflows and component interaction
- **API Integration**: RESTful endpoints and WebSocket functionality
- **Database Integration**: Cross-component data consistency
- **Cache Integration**: Caching system performance and reliability

### Performance Tests
- **Load Testing**: System behavior under high load
- **Stress Testing**: Breaking point identification
- **Memory Testing**: Memory usage and leak detection
- **Concurrency Testing**: Multi-threaded operation validation

### Security Tests
- **Input Validation**: SQL injection and XSS prevention
- **Authentication**: API security and access control
- **Data Sanitization**: Content filtering and validation
- **Rate Limiting**: DoS protection mechanisms

## 🛠️ Test Runner Usage

The `test_runner.py` script provides advanced testing capabilities:

### Basic Commands

```bash
# Run unit tests
python tests/test_runner.py --category unit

# Run integration tests
python tests/test_runner.py --category integration

# Run performance tests
python tests/test_runner.py --category performance

# Run security tests
python tests/test_runner.py --category security
```

### Advanced Options

```bash
# Run with specific markers
python tests/test_runner.py --markers "database and not slow"

# Run without parallel execution
python tests/test_runner.py --no-parallel

# Run without coverage reporting
python tests/test_runner.py --no-coverage

# Run specific test file
python tests/test_runner.py --file test_scraper.py
```

### Full Test Suite

```bash
# Run complete test suite
python tests/test_runner.py --full

# This runs:
# 1. Unit tests
# 2. Integration tests  
# 3. Performance tests
# 4. Security tests
```

## 📈 Test Reports

Test execution generates comprehensive reports in the `test_reports/` directory:

### Report Types

1. **HTML Report**: Interactive test results with detailed information
2. **JUnit XML**: CI/CD compatible test results
3. **Coverage Report**: Code coverage analysis with line-by-line details
4. **JSON Summary**: Machine-readable test statistics

### Report Files

```
test_reports/
├── test_report_YYYYMMDD_HHMMSS.html      # Interactive HTML report
├── junit_report_YYYYMMDD_HHMMSS.xml      # JUnit XML for CI/CD
├── coverage_report_YYYYMMDD_HHMMSS.html  # Coverage analysis
├── coverage_YYYYMMDD_HHMMSS.json         # Coverage data
└── test_summary_YYYYMMDD_HHMMSS.json     # Test statistics
```

## 🎯 Test Markers

Tests are organized using pytest markers for flexible execution:

### Available Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.smoke`: Quick validation tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.asyncio`: Asynchronous tests
- `@pytest.mark.database`: Database-related tests
- `@pytest.mark.scraper`: Scraper-related tests
- `@pytest.mark.analytics`: Analytics-related tests
- `@pytest.mark.ui`: UI-related tests
- `@pytest.mark.api`: API-related tests
- `@pytest.mark.cache`: Cache-related tests
- `@pytest.mark.monitoring`: Monitoring-related tests

### Using Markers

```bash
# Run only database tests
pytest -m database

# Run fast tests only
pytest -m "not slow"

# Run unit tests excluding performance
pytest -m "unit and not performance"

# Run smoke tests for quick validation
pytest -m smoke
```

## 🔧 Configuration

### Pytest Configuration (`pytest.ini`)

The pytest configuration includes:
- Test discovery patterns
- Marker definitions
- Async test support
- Logging configuration
- Warning filters
- Timeout settings

### Fixtures (`conftest.py`)

Shared fixtures provide:
- Sample test data
- Mock objects
- Database setup/teardown
- Performance monitoring
- Temporary file management

## 📝 Writing Tests

### Test Structure

```python
"""
Test module docstring explaining what is being tested.
"""
import pytest
from unittest.mock import Mock, patch

class TestComponentName:
    """Test class for specific component."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic functionality with descriptive name."""
        # Arrange
        input_data = "test_input"
        expected_result = "expected_output"
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test asynchronous functionality."""
        result = await async_function()
        assert result is not None
    
    @pytest.mark.performance
    def test_performance(self, performance_monitor):
        """Test performance characteristics."""
        performance_monitor.start()
        
        # Execute operation
        result = expensive_operation()
        
        metrics = performance_monitor.stop()
        assert metrics['duration'] < 5.0  # 5 second limit
```

### Best Practices

1. **Descriptive Names**: Use clear, descriptive test names
2. **Arrange-Act-Assert**: Follow the AAA pattern
3. **Single Responsibility**: One assertion per test when possible
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Edge Cases**: Include boundary conditions and error cases
6. **Performance Assertions**: Include timing and resource checks
7. **Cleanup**: Ensure proper cleanup of resources

### Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        'id': 'test123',
        'title': 'Test Post',
        'score': 42
    }

@pytest.fixture
def mock_database(temp_database):
    """Provide mock database for testing."""
    db = DatabaseManager(f"sqlite:///{temp_database}")
    db.create_tables()
    yield db
    db.close()
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH includes project root
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Database Connection Issues**
   ```bash
   # Check database permissions and paths
   ls -la data/
   ```

3. **Async Test Failures**
   ```bash
   # Ensure pytest-asyncio is installed
   pip install pytest-asyncio
   ```

4. **Coverage Issues**
   ```bash
   # Install coverage dependencies
   pip install pytest-cov coverage
   ```

### Debug Mode

```bash
# Run tests with debug output
pytest -v -s --tb=long

# Run single test with debugging
pytest -v -s tests/test_database.py::TestDatabaseManager::test_initialization
```

### Performance Issues

```bash
# Run tests without coverage for better performance
python tests/test_runner.py --no-coverage

# Run tests sequentially
python tests/test_runner.py --no-parallel
```

## 📊 Coverage Goals

### Target Coverage Levels

- **Overall Coverage**: ≥ 90%
- **Critical Components**: ≥ 95%
  - Database operations
  - Data validation
  - Security functions
- **UI Components**: ≥ 80%
- **Integration Tests**: ≥ 85%

### Coverage Reports

```bash
# Generate detailed coverage report
python tests/test_runner.py --category unit

# View coverage in browser
open test_reports/coverage_report_YYYYMMDD_HHMMSS.html
```

## 🔄 Continuous Integration

### GitHub Actions Integration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python tests/test_runner.py --full
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Async Testing Guide](https://pytest-asyncio.readthedocs.io/)

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure
2. Add appropriate markers
3. Include docstrings
4. Update this documentation if needed
5. Ensure tests pass in CI/CD pipeline

## 📞 Support

For testing-related questions:
1. Check this documentation
2. Review existing test examples
3. Check the troubleshooting section
4. Create an issue with test logs 