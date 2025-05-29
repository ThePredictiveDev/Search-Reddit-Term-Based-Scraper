# 🚀 Quick Start Guide - Reddit Mention Tracker

This guide shows you how to quickly test and run the Reddit Mention Tracker system using the two main scripts.

## 📋 Two Main Scripts

### 1. `run_all_tests.py` - Complete Testing Script
**Purpose**: Tests every component of the system at once with comprehensive validation.

### 2. `run_system.py` - System Runner Script  
**Purpose**: Starts the complete system and opens the web interface for actual use.

---

## 🧪 Testing Everything (run_all_tests.py)

### Quick Test (5 minutes)
```bash
python run_all_tests.py --quick
```
This runs essential tests only to verify the system works.

### Full Test Suite (30-45 minutes)
```bash
python run_all_tests.py --full
```
This runs comprehensive testing including:
- Unit tests (database, scraper, analytics, UI)
- Integration tests (component interaction)
- Performance tests (load and stress testing)
- Security tests (vulnerability assessment)

### Other Testing Options
```bash
# Run without performance tests (faster)
python run_all_tests.py --no-performance

# Run with detailed output
python run_all_tests.py --verbose

# Just check if everything can be imported
python run_all_tests.py --quick
```

### What the Test Script Does:
1. ✅ Checks all prerequisites and dependencies
2. 🧪 Runs smoke tests for quick validation
3. 📊 Executes comprehensive test suite
4. 📈 Generates detailed reports with coverage analysis
5. 🎯 Provides final system status assessment

### Test Results:
- **HTML Report**: Interactive test results
- **Coverage Report**: Code coverage analysis  
- **JSON Summary**: Machine-readable statistics
- **Console Output**: Real-time progress and summary

---

## 🎨 Running the System (run_system.py)

### Start Complete System
```bash
python run_system.py
```
This will:
1. ✅ Check system requirements
2. 📦 Install dependencies automatically
3. 🗄️ Setup database
4. 🌐 Start API server (port 8000)
5. 🎨 Start web interface (port 7860)
6. 🌐 Auto-open browser to the interface

### Custom Options
```bash
# Use custom port for web interface
python run_system.py --port 8080

# Start only the web interface (no API)
python run_system.py --ui-only

# Start only the API server
python run_system.py --api-only

# Don't auto-open browser
python run_system.py --no-browser

# Enable debug mode
python run_system.py --debug

# Create public share link (Gradio)
python run_system.py --share
```

### What the System Runner Does:
1. 🔍 Validates all system requirements
2. 📦 Installs missing dependencies
3. 🗄️ Initializes database
4. 🧪 Runs quick system test
5. 🌐 Starts API server (if enabled)
6. 🎨 Starts Gradio web interface
7. 🖥️ Opens browser automatically
8. 📊 Monitors system resources
9. 🛑 Handles graceful shutdown

### Access Points:
- **Web Interface**: http://localhost:7860
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 🎯 Recommended Workflow

### First Time Setup:
```bash
# 1. Test everything first
python run_all_tests.py --quick

# 2. If tests pass, run the system
python run_system.py
```

### Regular Usage:
```bash
# Just start the system
python run_system.py
```

### Before Deployment:
```bash
# Run full test suite
python run_all_tests.py --full
```

---

## 🔧 Troubleshooting

### If Tests Fail:
1. Check the HTML test report in `test_reports/`
2. Look at console output for specific errors
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check Python version: `python --version` (needs 3.11+)

### If System Won't Start:
1. Run quick test first: `python run_system.py --quick-test`
2. Check for missing files or dependencies
3. Try different ports: `python run_system.py --port 8080`
4. Check console output for specific errors

### Common Issues:
- **Port already in use**: Use `--port` to specify different port
- **Import errors**: Run `pip install -r requirements.txt`
- **Database errors**: Delete `data/` folder and restart
- **Browser won't open**: Use `--no-browser` and open manually

---

## 📊 What You Can Do With The System

Once running, the web interface provides:

### 🔍 Search Functionality
- Enter search terms (e.g., "OpenAI", "ChatGPT")
- Set quality thresholds
- Configure number of pages to scrape

### 📈 Analytics Dashboard
- Real-time mention tracking
- Sentiment analysis
- Subreddit breakdown
- Temporal patterns
- Engagement metrics

### 📊 Visualizations
- Interactive timeline charts
- Sentiment distribution
- Subreddit popularity
- Trending topics
- Quality metrics

### 💾 Data Export
- Export results as CSV
- Export as JSON
- Download charts as images

### 🖥️ System Monitoring
- Real-time CPU/memory usage
- Active session tracking
- Performance metrics
- Alert notifications

---

## 🎉 Success Indicators

### Tests Passed Successfully:
- ✅ Overall success rate >95%
- ✅ All critical components working
- ✅ No security vulnerabilities
- ✅ Performance within benchmarks

### System Running Successfully:
- ✅ Web interface loads at http://localhost:7860
- ✅ Can perform searches and get results
- ✅ Charts and visualizations display
- ✅ Data export works
- ✅ No error messages in console

---

## 📞 Need Help?

1. **Check test reports** in `test_reports/` folder
2. **Review console output** for specific error messages
3. **Try quick test** first: `python run_all_tests.py --quick`
4. **Check system test**: `python run_system.py --quick-test`
5. **Verify dependencies**: `pip install -r requirements.txt`

The system is designed to be self-diagnosing and will provide helpful error messages to guide you through any issues. 