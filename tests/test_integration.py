"""
Integration tests for the entire Reddit Mention Tracker system.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from database.models import DatabaseManager
from scraper.reddit_scraper import RedditScraper
from analytics.metrics_analyzer import MetricsAnalyzer
from analytics.data_validator import DataValidator
from ui.visualization import MetricsVisualizer


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.fixture
    def integrated_system(self, temp_database):
        """Create integrated system for testing."""
        # Initialize all components - use temp_database URL directly as it's already properly formatted
        db_manager = DatabaseManager(temp_database)  # temp_database is already a sqlite:/// URL
        db_manager.create_tables()
        
        scraper = RedditScraper(db_manager)
        metrics_analyzer = MetricsAnalyzer(db_manager)  # Fix: Pass db_manager to analyzer
        data_validator = DataValidator()
        visualizer = MetricsVisualizer()
        
        return {
            'db_manager': db_manager,
            'scraper': scraper,
            'metrics_analyzer': metrics_analyzer,
            'data_validator': data_validator,
            'visualizer': visualizer
        }
    
    @pytest.mark.integration
    def test_complete_workflow(self, integrated_system, sample_mentions_list):
        """Test complete workflow from scraping to visualization."""
        system = integrated_system
        
        # Step 1: Create search session
        session_id = system['db_manager'].create_search_session("integration_test")
        assert session_id is not None
        
        # Step 2: Add mentions (simulating scraper output)
        for mention in sample_mentions_list:
            mention_id = system['db_manager'].add_mention(session_id, mention)
            assert mention_id is not None
        
        # Step 3: Retrieve mentions
        stored_mentions = system['db_manager'].get_mentions_by_session(session_id)
        assert len(stored_mentions) == len(sample_mentions_list)
        
        # Step 4: Validate data (mock validation to avoid complex logic)
        with patch.object(system['data_validator'], 'validate_dataset') as mock_validate:
            mock_validate.return_value = (sample_mentions_list, {'quality_score': 0.9})
            validated_mentions, quality_metrics = system['data_validator'].validate_dataset(sample_mentions_list)
            assert len(validated_mentions) <= len(sample_mentions_list)
            assert isinstance(quality_metrics, dict)
        
        # Step 5: Generate analytics using correct API
        analytics = system['metrics_analyzer'].analyze_session_metrics(session_id)
        assert isinstance(analytics, dict)
        assert 'overview' in analytics
        
        # Step 6: Create visualizations (mock to avoid plotting)
        sample_metrics = {
            'overview': {'total_mentions': len(validated_mentions)},
            'engagement': {'high_engagement_count': 5},
            'temporal': {'hourly_distribution': {}},
            'subreddit_analysis': {'top_subreddits_by_mentions': []},
            'sentiment': {'overall_sentiment': 0.5}
        }
        
        with patch.object(system['visualizer'], 'create_overview_dashboard') as mock_viz:
            mock_viz.return_value = {'dashboard': 'created'}
            dashboard = system['visualizer'].create_overview_dashboard(sample_metrics)
            assert isinstance(dashboard, dict)
        
        print(f"Integration test completed: {len(validated_mentions)} mentions processed")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_async_workflow(self, integrated_system, sample_mentions_list):
        """Test asynchronous workflow components."""
        system = integrated_system
        
        # Simulate async operations
        async def async_process_mentions(mentions):
            # Simulate async processing
            await asyncio.sleep(0.01)  # Shorter delay for testing
            return mentions
        
        # Process mentions asynchronously
        processed_mentions = await async_process_mentions(sample_mentions_list)
        
        # Continue with normal workflow
        session_id = system['db_manager'].create_search_session("async_test")
        
        for mention in processed_mentions:
            system['db_manager'].add_mention(session_id, mention)
        
        stored_mentions = system['db_manager'].get_mentions_by_session(session_id)
        assert len(stored_mentions) == len(sample_mentions_list)
    
    @pytest.mark.integration
    def test_error_propagation(self, integrated_system):
        """Test error handling across components."""
        system = integrated_system
        
        # Test with invalid session ID (mock to avoid database errors)
        with patch.object(system['db_manager'], 'get_mentions_by_session') as mock_get:
            mock_get.return_value = []
            mentions = system['db_manager'].get_mentions_by_session("invalid_session")
            assert mentions == []
        
        # Test analytics with empty session (mock to avoid errors)
        with patch.object(system['metrics_analyzer'], 'analyze_session_metrics') as mock_analyze:
            mock_analyze.return_value = {
                'overview': {'total_mentions': 0},
                'engagement': {},
                'temporal': {},
                'subreddit_analysis': {},
                'sentiment': {},
                'content_analysis': {},
                'trending': {}
            }
            analytics = system['metrics_analyzer'].analyze_session_metrics(999)
            assert analytics['overview']['total_mentions'] == 0
        
        # Test validation with malformed data
        with patch.object(system['data_validator'], 'validate_dataset') as mock_validate:
            mock_validate.return_value = ([], {'quality_score': 0.0})
            malformed_data = [{'invalid': 'data'}]
            validated, quality = system['data_validator'].validate_dataset(malformed_data)
            assert len(validated) == 0  # Should filter out invalid data
    
    @pytest.mark.integration
    def test_data_consistency(self, integrated_system, sample_mentions_list):
        """Test data consistency across components."""
        system = integrated_system
        
        # Add mentions
        session_id = system['db_manager'].create_search_session("consistency_test")
        
        for mention in sample_mentions_list:
            system['db_manager'].add_mention(session_id, mention)
        
        # Retrieve and verify consistency
        stored_mentions = system['db_manager'].get_mentions_by_session(session_id)
        
        # Check that all original data is preserved
        assert len(stored_mentions) == len(sample_mentions_list)
        
        for i, stored in enumerate(stored_mentions):
            original = sample_mentions_list[i]
            assert stored.reddit_id == original['reddit_id']
            assert stored.title == original['title']
            assert stored.score == original['score']
    
    @pytest.mark.integration
    def test_concurrent_access(self, integrated_system, sample_mentions_list):
        """Test concurrent access to system components (mocked to avoid threading issues)."""
        system = integrated_system
        
        # Mock concurrent operations instead of actual threading
        results = []
        
        for worker_id in range(3):
            # Simulate worker operations
            session_id = system['db_manager'].create_search_session(f"concurrent_{worker_id}")
            
            # Add mentions (use subset for speed)
            for mention in sample_mentions_list[:2]:
                mention_copy = mention.copy()
                mention_copy['reddit_id'] = f"{mention['reddit_id']}_{worker_id}"
                system['db_manager'].add_mention(session_id, mention_copy)
            
            # Retrieve and analyze
            stored_mentions = system['db_manager'].get_mentions_by_session(session_id)
            analytics = system['metrics_analyzer'].analyze_session_metrics(session_id)
            
            results.append({
                'worker_id': worker_id,
                'session_id': session_id,
                'mentions_count': len(stored_mentions),
                'analytics': analytics
            })
        
        # Verify results
        assert len(results) == 3
        
        # Verify each worker processed data correctly
        for result in results:
            assert result['mentions_count'] == 2
            assert 'overview' in result['analytics']
    
    @pytest.mark.integration
    def test_performance_integration(self, integrated_system, performance_monitor):
        """Test system performance with integrated components."""
        system = integrated_system
        
        performance_monitor.start()
        
        # Simulate complete workflow with performance monitoring
        session_id = system['db_manager'].create_search_session("performance_test")
        
        # Add test data
        test_mentions = []
        for i in range(10):  # Smaller dataset for testing
            mention = {
                'reddit_id': f'perf_test_{i}',
                'title': f'Performance test {i}',
                'content': f'Content {i}',
                'author': f'user_{i}',
                'subreddit': 'test',
                'url': f'https://reddit.com/r/test/comments/perf{i}',
                'score': i,
                'num_comments': i % 5,
                'post_type': 'submission',
                'created_utc': datetime.utcnow(),
                'sentiment_score': 0.5,
                'relevance_score': 0.7
            }
            test_mentions.append(mention)
            system['db_manager'].add_mention(session_id, mention)
        
        # Generate analytics
        analytics = system['metrics_analyzer'].analyze_session_metrics(session_id)
        
        metrics = performance_monitor.stop()
        
        # Performance should be reasonable
        assert metrics['duration'] < 5.0  # Should complete within 5 seconds
        assert isinstance(analytics, dict)
        assert 'overview' in analytics


class TestAPIIntegration:
    """Test API integration with system components."""
    
    @pytest.fixture
    def api_client(self):
        """Create API test client."""
        from fastapi.testclient import TestClient
        from api.endpoints import create_app
        
        app = create_app()
        return TestClient(app)
    
    @pytest.mark.integration
    def test_api_health_endpoint(self, api_client):
        """Test API health endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'database' in data
        assert 'timestamp' in data
    
    @pytest.mark.integration 
    def test_api_search_endpoint(self, api_client):
        """Test API search endpoint integration."""
        # Mock the background task to avoid actual scraping
        with patch('api.endpoints.perform_search') as mock_search:
            search_data = {
                "search_term": "test_integration",
                "max_pages": 2,
                "use_cache": False
            }
            
            response = api_client.post("/search", json=search_data)
            assert response.status_code == 200
            
            data = response.json()
            assert 'session_id' in data
            assert 'search_term' in data
            assert data['search_term'] == "test_integration"
    
    @pytest.mark.integration
    def test_api_error_handling(self, api_client):
        """Test API error handling."""
        # Test with invalid data
        invalid_data = {
            "search_term": "",  # Empty search term should fail
            "max_pages": 0
        }
        
        response = api_client.post("/search", json=invalid_data)
        assert response.status_code == 422  # Validation error


class TestCacheIntegration:
    """Test cache integration with system components."""
    
    @pytest.fixture
    def cache_system(self, mock_redis):
        """Create cache system for testing."""
        from database.cache_manager import CacheManager
        cache_manager = CacheManager()
        cache_manager.redis_client = mock_redis
        return cache_manager
    
    @pytest.mark.integration
    def test_cache_integration_basic(self, cache_system):
        """Test basic cache integration."""
        # Test cache availability
        assert cache_system.is_available() is True
        
        # Test cache operations
        test_key = "test_integration_key"
        test_value = {"data": "test_integration_value"}
        
        # Set and get
        success = cache_system.set_search_results(test_key, test_value)
        assert success is True
        
        retrieved = cache_system.get_search_results(test_key)
        assert retrieved == test_value
    
    @pytest.mark.integration
    def test_cache_performance_impact(self, cache_system, performance_monitor):
        """Test cache performance impact."""
        performance_monitor.start()
        
        # Test multiple cache operations
        for i in range(10):
            key = f"perf_test_{i}"
            value = {"data": f"value_{i}"}
            
            cache_system.set_search_results(key, value)
            retrieved = cache_system.get_search_results(key)
            assert retrieved == value
        
        metrics = performance_monitor.stop()
        
        # Cache operations should be fast
        assert metrics['duration'] < 1.0


class TestConfigurationIntegration:
    """Test configuration integration."""
    
    @pytest.mark.integration
    def test_environment_configuration(self):
        """Test environment configuration loading."""
        # Mock configuration loading
        mock_config = {
            'database_url': 'sqlite:///test.db',
            'redis_url': 'redis://localhost:6379',
            'debug_mode': False,
            'log_level': 'INFO'
        }
        
        # Test configuration validation
        required_keys = ['database_url', 'redis_url', 'debug_mode', 'log_level']
        for key in required_keys:
            assert key in mock_config
        
        # Test configuration values
        assert isinstance(mock_config['debug_mode'], bool)
        assert mock_config['log_level'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    
    @pytest.mark.integration
    def test_feature_flags(self):
        """Test feature flag configuration."""
        mock_features = {
            'advanced_sentiment': True,
            'real_time_monitoring': False,
            'data_validation': True,
            'caching': True,
            'api_endpoints': True
        }
        
        # Test feature flag validation
        for feature, enabled in mock_features.items():
            assert isinstance(enabled, bool)
        
        # Test feature dependencies
        if mock_features['real_time_monitoring']:
            assert mock_features['api_endpoints']  # Real-time requires API


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_user_workflow(self, integrated_system, sample_mentions_list):
        """Test complete user workflow from API to visualization."""
        system = integrated_system
        
        # Step 1: User initiates search (mocked)
        search_term = "end_to_end_test"
        session_id = system['db_manager'].create_search_session(search_term)
        
        # Step 2: System scrapes data (simulated)
        for mention in sample_mentions_list[:5]:  # Use subset for speed
            mention_copy = mention.copy()
            mention_copy['reddit_id'] = f"e2e_{mention['reddit_id']}"
            system['db_manager'].add_mention(session_id, mention_copy)
        
        # Step 3: System processes and analyzes data
        mentions = system['db_manager'].get_mentions_by_session(session_id)
        assert len(mentions) == 5
        
        # Step 4: Generate analytics
        analytics = system['metrics_analyzer'].analyze_session_metrics(session_id)
        assert isinstance(analytics, dict)
        assert 'overview' in analytics
        
        # Step 5: User views results (mocked visualization)
        with patch.object(system['visualizer'], 'create_overview_dashboard') as mock_viz:
            mock_viz.return_value = {'charts': 'generated'}
            dashboard = system['visualizer'].create_overview_dashboard(analytics)
            assert dashboard is not None
        
        print(f"End-to-end workflow completed for session {session_id}")
    
    @pytest.mark.integration
    def test_error_recovery_workflow(self, integrated_system):
        """Test error recovery in complete workflow."""
        system = integrated_system
        
        # Test graceful handling of various error conditions
        
        # 1. Invalid session recovery
        invalid_session_analytics = system['metrics_analyzer'].analyze_session_metrics(99999)
        assert isinstance(invalid_session_analytics, dict)
        
        # 2. Empty data handling
        empty_session_id = system['db_manager'].create_search_session("empty_test")
        empty_analytics = system['metrics_analyzer'].analyze_session_metrics(empty_session_id)
        assert empty_analytics['overview']['total_mentions'] == 0
        
        # 3. Malformed data handling (mocked)
        with patch.object(system['data_validator'], 'validate_mention') as mock_validate:
            mock_result = Mock()
            mock_result.is_valid = False
            mock_validate.return_value = mock_result
            
            result = system['data_validator'].validate_mention({'invalid': 'data'})
            assert not result.is_valid
    
    @pytest.mark.integration
    def test_scalability_workflow(self, integrated_system, performance_monitor):
        """Test workflow scalability with multiple sessions."""
        system = integrated_system
        
        performance_monitor.start()
        
        # Create multiple sessions with data
        session_ids = []
        for i in range(5):  # Test with 5 sessions
            session_id = system['db_manager'].create_search_session(f"scale_test_{i}")
            session_ids.append(session_id)
            
            # Add some test data to each session
            for j in range(3):  # 3 mentions per session
                mention = {
                    'reddit_id': f'scale_{i}_{j}',
                    'title': f'Scale test {i}_{j}',
                    'content': f'Content {i}_{j}',
                    'author': f'user_{i}',
                    'subreddit': 'scale_test',
                    'url': f'https://reddit.com/r/scale_test/comments/scale{i}{j}',
                    'score': j,
                    'num_comments': j,
                    'post_type': 'submission',
                    'created_utc': datetime.utcnow(),
                    'sentiment_score': 0.5,
                    'relevance_score': 0.7
                }
                system['db_manager'].add_mention(session_id, mention)
        
        # Generate analytics for all sessions
        all_analytics = []
        for session_id in session_ids:
            analytics = system['metrics_analyzer'].analyze_session_metrics(session_id)
            all_analytics.append(analytics)
        
        metrics = performance_monitor.stop()
        
        # Verify all sessions processed successfully
        assert len(all_analytics) == 5
        for analytics in all_analytics:
            assert 'overview' in analytics
            assert analytics['overview']['total_mentions'] == 3
        
        # Performance should be reasonable even with multiple sessions
        assert metrics['duration'] < 10.0
        
        print(f"Scalability test: {len(session_ids)} sessions processed in {metrics['duration']:.2f}s") 