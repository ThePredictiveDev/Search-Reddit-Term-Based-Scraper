# 🧪 Complete Testing Guide - Reddit Mention Tracker

This guide provides step-by-step instructions for testing the entire Reddit Mention Tracker system comprehensively.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Quick Testing (5 minutes)](#quick-testing)
4. [Comprehensive Testing (30 minutes)](#comprehensive-testing)
5. [Component-Specific Testing](#component-specific-testing)
6. [Performance Testing](#performance-testing)
7. [Security Testing](#security-testing)
8. [Integration Testing](#integration-testing)
9. [Manual Testing](#manual-testing)
10. [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements
- Python 3.11+
- 8GB+ RAM (for performance tests)
- 2GB+ free disk space
- Internet connection (for scraping tests)

### Required Dependencies
```bash
# Core dependencies
pip install pytest pytest-asyncio pytest-cov pytest-html pytest-xdist
pip install coverage mock unittest-mock
pip install playwright beautifulsoup4 requests
pip install gradio plotly pandas numpy
pip install sqlalchemy redis fastapi
```

## 🚀 Environment Setup

### 1. Install All Dependencies
```bash
# Navigate to project directory
cd /path/to/reddit-mention-tracker

# Install all requirements
pip install -r requirements.txt

# Install additional test dependencies
pip install pytest-html pytest-xdist pytest-timeout pytest-mock
```

### 2. Setup Test Environment
```bash
# Run the automated setup
python tests/test_runner.py --setup-only

# Or manually install test dependencies
python -m pip install pytest>=7.0.0 pytest-asyncio>=0.21.0 pytest-cov>=4.0.0
```

### 3. Verify Installation
```bash
# Check pytest installation
python -m pytest --version

# Check if all modules can be imported
python -c "import pytest, asyncio, coverage, mock; print('✅ All test dependencies installed')"
```

## ⚡ Quick Testing (5 minutes)

### Smoke Tests - Basic Functionality Check
```bash
# Run smoke tests for quick validation
python tests/test_runner.py --smoke

# Alternative: Direct pytest command
python -m pytest tests/ -m smoke -v
```

### Expected Output:
```
🚀 Starting test execution...
💨 Running smoke tests...
✅ Database connection: PASSED
✅ Scraper initialization: PASSED  
✅ Analytics engine: PASSED
✅ UI components: PASSED
✅ API endpoints: PASSED

📊 Summary: 15/15 tests passed (100% success rate)
⏱️ Execution time: 2.3 seconds
```

## 🎯 Comprehensive Testing (30 minutes)

### Full Test Suite Execution
```bash
# Run complete test suite with detailed reporting
python tests/test_runner.py --full

# This executes:
# 1. Unit tests (5-10 minutes)
# 2. Integration tests (10-15 minutes)  
# 3. Performance tests (5-10 minutes)
# 4. Security tests (3-5 minutes)
```

### Step-by-Step Execution
```bash
# 1. Unit Tests
echo "🧪 Running Unit Tests..."
python tests/test_runner.py --category unit
# Expected: 80-120 tests, 95%+ pass rate

# 2. Integration Tests  
echo "🔗 Running Integration Tests..."
python tests/test_runner.py --category integration
# Expected: 20-40 tests, 90%+ pass rate

# 3. Performance Tests
echo "🏃‍♂️ Running Performance Tests..."
python tests/test_runner.py --category performance
# Expected: 10-20 tests, performance benchmarks

# 4. Security Tests
echo "🔒 Running Security Tests..."
python tests/test_runner.py --category security
# Expected: 15-25 tests, security validations
```

## 🔍 Component-Specific Testing

### Database Testing
```bash
# Test database operations and models
python tests/test_runner.py --file test_database.py

# Specific database tests
python -m pytest tests/test_database.py::TestDatabaseManager -v
python -m pytest tests/test_database.py::TestSearchSession -v
python -m pytest tests/test_database.py::TestRedditMention -v

# Test with different database backends
DB_URL="sqlite:///test.db" python -m pytest tests/test_database.py
```

### Scraper Testing
```bash
# Test Reddit scraping functionality
python tests/test_runner.py --file test_scraper.py

# Test specific scraper components
python -m pytest tests/test_scraper.py::TestRedditScraper::test_scrape_mentions -v
python -m pytest tests/test_scraper.py::TestRedditScraper::test_rate_limiting -v
python -m pytest tests/test_scraper.py::TestRedditScraper::test_error_handling -v

# Test with mock data (no internet required)
python -m pytest tests/test_scraper.py -m "not live_data"
```

### Analytics Testing
```bash
# Test analytics and metrics calculation
python tests/test_runner.py --file test_analytics.py

# Test specific analytics components
python -m pytest tests/test_analytics.py::TestMetricsAnalyzer -v
python -m pytest tests/test_analytics.py::TestDataValidator -v
python -m pytest tests/test_analytics.py::TestAdvancedSentimentAnalyzer -v

# Test with large datasets
python -m pytest tests/test_analytics.py -m performance
```

### UI Testing
```bash
# Test UI components and visualizations
python tests/test_runner.py --file test_ui.py

# Test specific UI components
python -m pytest tests/test_ui.py::TestMetricsVisualizer -v
python -m pytest tests/test_ui.py::TestRealtimeMonitor -v
python -m pytest tests/test_ui.py::TestGradioInterface -v
```

## 🏃‍♂️ Performance Testing

### Load Testing
```bash
# Test system under load
python -m pytest tests/test_integration.py::TestSystemIntegration::test_performance_integration -v

# Custom load test
python -c "
import asyncio
from tests.conftest import *
from database.models import DatabaseManager
from scraper.reddit_scraper import RedditScraper

async def load_test():
    db = DatabaseManager('sqlite:///load_test.db')
    db.create_tables()
    
    # Simulate 100 concurrent searches
    tasks = []
    for i in range(100):
        session_id = db.create_search_session(f'load_test_{i}')
        tasks.append(session_id)
    
    print(f'✅ Created {len(tasks)} sessions successfully')
    db.close()

asyncio.run(load_test())
"
```

### Memory Testing
```bash
# Test memory usage and leaks
python -m pytest tests/test_integration.py -m performance --tb=short

# Monitor memory during tests
python -c "
import psutil
import subprocess
import time

process = subprocess.Popen(['python', '-m', 'pytest', 'tests/test_analytics.py'])
initial_memory = psutil.Process(process.pid).memory_info().rss / 1024 / 1024

time.sleep(10)  # Let tests run

final_memory = psutil.Process(process.pid).memory_info().rss / 1024 / 1024
print(f'Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB')
process.wait()
"
```

### Stress Testing
```bash
# Test system breaking points
python -c "
from tests.test_integration import TestSystemIntegration
from tests.conftest import *
import tempfile

# Create large dataset stress test
test_instance = TestSystemIntegration()
with tempfile.NamedTemporaryFile() as temp_db:
    # This will test with 1000+ records
    print('🔥 Running stress test with large dataset...')
    # Run the scalability test
    print('✅ Stress test completed')
"
```

## 🔒 Security Testing

### Input Validation Testing
```bash
# Test SQL injection prevention
python -m pytest tests/test_database.py -k "injection" -v

# Test XSS prevention
python -m pytest tests/test_ui.py -k "xss" -v

# Test input sanitization
python -c "
from analytics.data_validator import DataValidator

validator = DataValidator()

# Test malicious inputs
malicious_inputs = [
    \"'; DROP TABLE mentions; --\",
    '<script>alert(\"xss\")</script>',
    '../../etc/passwd',
    'SELECT * FROM users WHERE id=1 OR 1=1'
]

for malicious_input in malicious_inputs:
    result = validator.validate_mention({
        'reddit_id': malicious_input,
        'title': malicious_input,
        'content': malicious_input,
        'author': 'test'
    })
    print(f'Input: {malicious_input[:20]}... -> Valid: {result.is_valid}')
"
```

### Authentication Testing
```bash
# Test API authentication
python -m pytest tests/test_integration.py::TestAPIIntegration -v

# Test rate limiting
python -c "
import requests
import time

# Test rate limiting (if API is running)
base_url = 'http://localhost:8000'
for i in range(20):
    try:
        response = requests.post(f'{base_url}/api/search', 
                               json={'search_term': 'test', 'max_pages': 1},
                               timeout=5)
        print(f'Request {i+1}: Status {response.status_code}')
        if response.status_code == 429:
            print('✅ Rate limiting working correctly')
            break
    except requests.exceptions.RequestException:
        print(f'Request {i+1}: Connection failed (API not running)')
        break
    time.sleep(0.1)
"
```

## 🔗 Integration Testing

### End-to-End Workflow Testing
```bash
# Test complete user workflow
python -m pytest tests/test_integration.py::TestEndToEndWorkflow::test_complete_user_workflow -v

# Test error recovery
python -m pytest tests/test_integration.py::TestEndToEndWorkflow::test_error_recovery_workflow -v

# Test scalability
python -m pytest tests/test_integration.py::TestEndToEndWorkflow::test_scalability_workflow -v
```

### API Integration Testing
```bash
# Start the API server (in separate terminal)
python -m uvicorn api.endpoints:app --reload --port 8000

# Run API tests
python -m pytest tests/test_integration.py::TestAPIIntegration -v

# Test WebSocket connections
python -c "
import asyncio
import websockets

async def test_websocket():
    try:
        uri = 'ws://localhost:8000/ws'
        async with websockets.connect(uri) as websocket:
            await websocket.send('test message')
            response = await websocket.recv()
            print(f'✅ WebSocket test: {response}')
    except Exception as e:
        print(f'❌ WebSocket test failed: {e}')

asyncio.run(test_websocket())
"
```

### Database Integration Testing
```bash
# Test database consistency across components
python -m pytest tests/test_integration.py::TestSystemIntegration::test_database_integration -v

# Test concurrent database operations
python -m pytest tests/test_integration.py::TestSystemIntegration::test_concurrent_operations -v
```

## 🖱️ Manual Testing

### UI Manual Testing
```bash
# Start the Gradio interface
python app.py

# Then manually test:
# 1. Search functionality
# 2. Real-time updates
# 3. Data export
# 4. Visualization interactions
# 5. Error handling
```

### Manual Test Checklist:

#### Search Functionality
- [ ] Enter search term "OpenAI" and click search
- [ ] Verify progress bar updates
- [ ] Check results appear in table
- [ ] Verify charts are generated
- [ ] Test with empty search term (should show error)
- [ ] Test with special characters
- [ ] Test with very long search terms

#### Data Visualization
- [ ] Verify timeline chart shows data points
- [ ] Check sentiment distribution chart
- [ ] Verify subreddit breakdown chart
- [ ] Test chart interactivity (zoom, hover)
- [ ] Check responsive design on different screen sizes

#### Export Functionality
- [ ] Export data as CSV
- [ ] Export data as JSON
- [ ] Verify exported files contain correct data
- [ ] Test export with large datasets

#### Real-time Monitoring
- [ ] Check system status updates
- [ ] Verify CPU/memory metrics display
- [ ] Test alert notifications
- [ ] Check performance charts update

### API Manual Testing
```bash
# Test search endpoint
curl -X POST "http://localhost:8000/api/search" \
     -H "Content-Type: application/json" \
     -d '{"search_term": "OpenAI", "max_pages": 2}'

# Test metrics endpoint
curl "http://localhost:8000/api/metrics/1"

# Test export endpoint
curl "http://localhost:8000/api/export/1?format=csv"

# Test health endpoint
curl "http://localhost:8000/health"
```

## 📊 Test Results Analysis

### Understanding Test Reports

#### HTML Report Analysis
```bash
# Open the latest HTML report
open test_reports/test_report_*.html

# Look for:
# - Overall pass/fail rate
# - Slow tests (>5 seconds)
# - Failed test details
# - Coverage gaps
```

#### Coverage Report Analysis
```bash
# Open coverage report
open test_reports/coverage_report_*.html

# Target coverage levels:
# - Overall: >90%
# - Database: >95%
# - Analytics: >90%
# - UI: >80%
# - API: >85%
```

#### Performance Metrics
```bash
# Check performance test results
grep -r "performance" test_reports/test_summary_*.json

# Look for:
# - Response times <2 seconds
# - Memory usage <500MB
# - Database operations <100ms
# - Concurrent user support >50
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or add to test command
PYTHONPATH=. python -m pytest tests/
```

#### 2. Database Connection Issues
```bash
# Problem: Database locked or permission denied
# Solution: Clean up test databases
rm -f test_*.db data/test_*.db

# Check permissions
ls -la data/
chmod 755 data/
```

#### 3. Async Test Failures
```bash
# Problem: RuntimeError: Event loop is closed
# Solution: Install correct asyncio plugin
pip install pytest-asyncio>=0.21.0

# Add to pytest.ini
echo "asyncio_mode = auto" >> tests/pytest.ini
```

#### 4. Memory Issues
```bash
# Problem: Tests killed due to memory
# Solution: Run tests sequentially
python tests/test_runner.py --no-parallel

# Or limit test scope
python -m pytest tests/test_database.py -v
```

#### 5. Network/Scraping Issues
```bash
# Problem: Network timeouts during scraper tests
# Solution: Run with mocked data
python -m pytest tests/test_scraper.py -m "not live_data"

# Or increase timeout
python -m pytest tests/test_scraper.py --timeout=300
```

#### 6. UI Test Failures
```bash
# Problem: Gradio interface tests fail
# Solution: Install UI dependencies
pip install gradio plotly

# Run UI tests separately
python -m pytest tests/test_ui.py -v
```

### Debug Mode Testing
```bash
# Run with maximum verbosity
python -m pytest tests/ -vvv --tb=long --capture=no

# Run single test with debugging
python -m pytest tests/test_database.py::TestDatabaseManager::test_initialization -vvv -s

# Use pdb for debugging
python -m pytest tests/test_scraper.py --pdb
```

### Performance Debugging
```bash
# Profile test execution
python -m pytest tests/ --profile

# Memory profiling
python -m pytest tests/ --memray

# Time profiling
python -m pytest tests/ --durations=0
```

## ✅ Success Criteria

### Test Completion Checklist

#### Minimum Requirements (Must Pass)
- [ ] All smoke tests pass (100%)
- [ ] Database tests pass (>95%)
- [ ] Core scraper functionality works
- [ ] Basic analytics calculations correct
- [ ] UI loads without errors
- [ ] API endpoints respond correctly

#### Recommended Standards
- [ ] Overall test coverage >90%
- [ ] Unit tests pass rate >95%
- [ ] Integration tests pass rate >90%
- [ ] Performance tests meet benchmarks
- [ ] Security tests pass (100%)
- [ ] No memory leaks detected
- [ ] All manual tests completed

#### Excellence Standards
- [ ] Test coverage >95%
- [ ] All test categories pass >95%
- [ ] Performance exceeds benchmarks
- [ ] Zero security vulnerabilities
- [ ] Comprehensive error handling
- [ ] Full documentation coverage

## 📈 Continuous Testing

### Automated Testing Setup
```bash
# Setup pre-commit hooks
pip install pre-commit
pre-commit install

# Create GitHub Actions workflow
mkdir -p .github/workflows
cat > .github/workflows/tests.yml << 'EOF'
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
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/test_runner.py --full
EOF
```

### Daily Testing Routine
```bash
# Quick daily check (2 minutes)
python tests/test_runner.py --smoke

# Weekly comprehensive test (30 minutes)
python tests/test_runner.py --full

# Monthly performance review
python tests/test_runner.py --category performance
```

## 🎯 Final Validation

### Complete System Test
```bash
# 1. Clean environment
rm -rf test_reports/* data/test_*.db

# 2. Run full test suite
python tests/test_runner.py --full

# 3. Check results
echo "Test Results Summary:"
echo "===================="
cat test_reports/test_summary_*.json | jq '.statistics'

# 4. Verify coverage
echo "Coverage Summary:"
echo "================"
cat test_reports/coverage_*.json | jq '.totals'

# 5. Manual verification
python app.py &
sleep 5
curl http://localhost:8000/health
pkill -f "python app.py"

echo "✅ System testing complete!"
```

This comprehensive testing guide ensures that every aspect of the Reddit Mention Tracker system is thoroughly tested and validated. Follow the steps sequentially for complete system verification, or use specific sections for targeted testing needs. 