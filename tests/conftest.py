"""
Pytest configuration and shared fixtures for Reddit Mention Tracker tests.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import os
import time

# Test data fixtures
@pytest.fixture
def sample_mention_data():
    """Sample mention data for testing."""
    return {
        'reddit_id': 'test123',
        'title': 'Test post about OpenAI',
        'content': 'This is a test post discussing OpenAI technology.',
        'author': 'test_user',
        'subreddit': 'technology',
        'url': 'https://reddit.com/r/technology/comments/test123',
        'score': 10,
        'num_comments': 5,
        'created_utc': datetime.utcnow(),
        'post_type': 'submission',
        'scraped_at': datetime.utcnow()
    }

@pytest.fixture
def sample_mentions_list():
    """Sample list of mentions for testing."""
    return [
        {
            'reddit_id': f'test{i}',
            'title': f'Test post {i} about OpenAI',
            'content': f'This is a test post discussing OpenAI technology and its impact.',
            'author': 'test_user',
            'subreddit': 'technology',
            'url': f'https://reddit.com/r/technology/comments/test123',
            'score': 10,
            'num_comments': 15,
            'post_type': 'submission',
            'created_utc': datetime.utcnow(),
            'scraped_at': datetime.utcnow(),
            'sentiment_score': -0.5 if i % 2 else 0.5,
            'relevance_score': 0.8
        }
        for i in range(10)
    ]

@pytest.fixture
def temp_database():
    """Create a temporary database for testing."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    db_url = f"sqlite:///{db_path}"
    
    yield db_url
    
    # Clean up - close any active connections first
    try:
        import gc
        gc.collect()  # Force garbage collection to close connections
        time.sleep(0.1)  # Brief pause to allow connections to close
        shutil.rmtree(temp_dir)
    except (PermissionError, OSError) as e:
        # If we can't delete immediately, try again after a longer pause
        try:
            time.sleep(1)
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError):
            # Log the issue but don't fail the test
            print(f"Warning: Could not clean up temporary database: {e}")
            pass

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    
    # Create a simple in-memory storage to simulate Redis behavior
    _storage = {}
    
    def mock_get(key):
        return _storage.get(key)
    
    def mock_set(key, value):
        _storage[key] = value
        return True
    
    def mock_setex(key, ttl, value):
        _storage[key] = value
        return True
    
    def mock_delete(*keys):
        for key in keys:
            _storage.pop(key, None)
        return len(keys)
    
    def mock_exists(key):
        return key in _storage
    
    mock_redis.ping.return_value = True
    mock_redis.get = Mock(side_effect=mock_get)
    mock_redis.set = Mock(side_effect=mock_set)
    mock_redis.setex = Mock(side_effect=mock_setex)
    mock_redis.delete = Mock(side_effect=mock_delete)
    mock_redis.exists = Mock(side_effect=mock_exists)
    mock_redis.ttl.return_value = -1
    
    return mock_redis

@pytest.fixture
def mock_playwright_page():
    """Mock Playwright page for testing."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.query_selector_all = AsyncMock(return_value=[])
    mock_page.query_selector = AsyncMock(return_value=None)
    return mock_page

@pytest.fixture
def mock_browser_context():
    """Mock browser context for testing."""
    mock_context = AsyncMock()
    mock_page = Mock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    return mock_context

@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        'environment': 'testing',
        'debug': True,
        'log_level': 'DEBUG',
        'database': {
            'url': 'sqlite:///:memory:',
            'echo': False
        },
        'scraping': {
            'max_pages_per_search': 2,
            'max_concurrent_requests': 1,
            'request_delay_min': 0.1,
            'request_delay_max': 0.2,
            'timeout_seconds': 5,
            'retry_attempts': 1,
            'headless_mode': True,
            'quality_threshold': 0.3
        },
        'features': {
            'advanced_sentiment': True,
            'real_time_monitoring': False,
            'data_validation': True,
            'caching': False,
            'api_endpoints': True
        }
    }

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def create_reddit_post_html(title, author, score, comments):
        """Create mock Reddit post HTML."""
        return f"""
        <div data-testid="post-container">
            <h3>{title}</h3>
            <a data-click-id="body" href="/r/test/comments/123/{title.lower().replace(' ', '_')}"></a>
            <span data-testid="subreddit-name">r/test</span>
            <span data-testid="post_author_link">u/{author}</span>
            <button aria-label="upvote. {score} points"></button>
            <span data-testid="comment-count">{comments} comments</span>
            <div data-testid="post-content"><p>Test content</p></div>
        </div>
        """
    
    @staticmethod
    def create_search_results_html(posts):
        """Create mock search results HTML."""
        html = "<html><body>"
        for post in posts:
            html += TestDataGenerator.create_reddit_post_html(**post)
        html += "</body></html>"
        return html

@pytest.fixture
def test_data_generator():
    """Test data generator utility."""
    return TestDataGenerator()

# Database fixtures
@pytest.fixture
def db_manager(temp_database, test_config):
    """Database manager for testing."""
    from database.models import DatabaseManager
    
    # Use the temp_database URL directly (it's already a sqlite:/// URL)
    db_manager = DatabaseManager(temp_database)
    db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    db_manager.close()

# Component fixtures
@pytest.fixture
def metrics_analyzer(db_manager):
    """Metrics analyzer for testing."""
    from analytics.metrics_analyzer import MetricsAnalyzer
    return MetricsAnalyzer(db_manager)

@pytest.fixture
def data_validator():
    """Data validator for testing."""
    from analytics.data_validator import DataValidator
    return DataValidator()

@pytest.fixture
def visualizer():
    """Metrics visualizer for testing."""
    from ui.visualization import MetricsVisualizer
    return MetricsVisualizer()

# Mock external services
@pytest.fixture
def mock_textblob():
    """Mock TextBlob for sentiment analysis."""
    mock_blob = Mock()
    mock_blob.sentiment.polarity = 0.5
    mock_blob.sentiment.subjectivity = 0.6
    return mock_blob

@pytest.fixture
def mock_requests():
    """Mock requests library for HTTP testing."""
    mock_requests = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'success'}
    mock_response.text = '<html><body>Mock response</body></html>'
    mock_requests.get.return_value = mock_response
    mock_requests.post.return_value = mock_response
    return mock_requests

@pytest.fixture
def scraper(db_manager):
    """Reddit scraper for testing."""
    from scraper.reddit_scraper import RedditScraper
    return RedditScraper(db_manager)

@pytest.fixture
def monitor():
    """Real-time monitor for testing."""
    from ui.realtime_monitor import RealTimeMonitor
    return RealTimeMonitor(host="localhost", port=8765)

@pytest.fixture
def cache_manager(mock_redis):
    """Cache manager for testing."""
    from database.cache_manager import CacheManager
    cache_manager = CacheManager()
    cache_manager.redis_client = mock_redis
    return cache_manager

@pytest.fixture
def api_client():
    """API client for testing."""
    from fastapi.testclient import TestClient
    from api.endpoints import create_app
    
    # Create app with test configuration
    app = create_app()
    return TestClient(app)

@pytest.fixture
def sentiment_analyzer():
    """Advanced sentiment analyzer for testing."""
    try:
        from analytics.advanced_sentiment import AdvancedSentimentAnalyzer
        analyzer = AdvancedSentimentAnalyzer()
        if not analyzer.is_available():
            # Return mock if not available
            mock_analyzer = Mock()
            mock_analyzer.is_available.return_value = False
            mock_analyzer.analyze_sentiment = AsyncMock(return_value={'composite_sentiment': 0.0})
            mock_analyzer.analyze_batch.return_value = [{'composite_sentiment': 0.0}]
            return mock_analyzer
        return analyzer
    except ImportError:
        # Return mock if import fails
        mock_analyzer = Mock()
        mock_analyzer.is_available.return_value = False
        mock_analyzer.analyze_sentiment = AsyncMock(return_value={'composite_sentiment': 0.0})
        mock_analyzer.analyze_batch.return_value = [{'composite_sentiment': 0.0}]
        return mock_analyzer

@pytest.fixture
def performance_monitor():
    """Performance monitor for testing."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.start_cpu = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.virtual_memory().used
            self.start_cpu = psutil.cpu_percent()
        
        def stop(self):
            if self.start_time:
                duration = time.time() - self.start_time
                memory_used = psutil.virtual_memory().used - self.start_memory
                cpu_usage = psutil.cpu_percent()
                
                return {
                    'duration': duration,
                    'memory_used': memory_used,
                    'peak_cpu': max(self.start_cpu, cpu_usage),
                    'final_cpu': cpu_usage
                }
            return {'duration': 0, 'memory_used': 0, 'peak_cpu': 0, 'final_cpu': 0}
    
    return PerformanceMonitor()

@pytest.fixture
async def async_test_client():
    """Async test client for API testing."""
    from httpx import AsyncClient
    from api.endpoints import create_app
    
    class AsyncTestClient:
        def __init__(self, app):
            self.app = app
            self.client = AsyncClient(app=app, base_url="http://test")
        
        async def get(self, *args, **kwargs):
            return await self.client.get(*args, **kwargs)
        
        async def post(self, *args, **kwargs):
            return await self.client.post(*args, **kwargs)
        
        async def put(self, *args, **kwargs):
            return await self.client.put(*args, **kwargs)
        
        async def delete(self, *args, **kwargs):
            return await self.client.delete(*args, **kwargs)
        
        async def close(self):
            await self.client.aclose()
    
    app = create_app()
    async_client = AsyncTestClient(app)
    yield async_client
    await async_client.close()

@pytest.fixture
def integrated_system(db_manager, metrics_analyzer, visualizer):
    """Create integrated system for end-to-end testing."""
    from analytics.data_validator import DataValidator
    data_validator = DataValidator()
    
    return {
        'db_manager': db_manager,
        'metrics_analyzer': metrics_analyzer,
        'data_validator': data_validator,
        'visualizer': visualizer
    } 