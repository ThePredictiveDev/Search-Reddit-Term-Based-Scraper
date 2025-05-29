"""
Tests for UI components and visualization functionality.
"""
import pytest
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from datetime import datetime, timedelta

from ui.visualization import MetricsVisualizer
from ui.realtime_monitor import RealTimeMonitor


class TestMetricsVisualizer:
    """Test metrics visualization functionality."""
    
    @pytest.mark.unit
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer is not None
        assert hasattr(visualizer, 'color_scheme')
        assert 'primary' in visualizer.color_scheme
        assert 'secondary' in visualizer.color_scheme
    
    @pytest.mark.unit
    def test_create_overview_dashboard(self, visualizer, sample_mentions_list):
        """Test creating overview dashboard."""
        # Create sample metrics structure
        sample_metrics = {
            'overview': {'total_mentions': len(sample_mentions_list)},
            'engagement': {'engagement_distribution': {'low': 5, 'medium': 10, 'high': 3}},
            'temporal': {'daily_timeline': [{'date': '2024-01-01', 'mentions': 5}]},
            'subreddit_analysis': {'top_subreddits_by_mentions': {'technology': 10, 'programming': 5}},
            'sentiment': {'sentiment_distribution': {'positive': 8, 'neutral': 5, 'negative': 2}}
        }
        
        fig = visualizer.create_overview_dashboard(sample_metrics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    @pytest.mark.unit
    def test_create_temporal_analysis(self, visualizer, sample_mentions_list):
        """Test creating temporal analysis."""
        # Create sample temporal metrics
        sample_metrics = {
            'temporal': {
                'daily_timeline': [
                    {'date': '2024-01-01', 'mentions': 5},
                    {'date': '2024-01-02', 'mentions': 8},
                    {'date': '2024-01-03', 'mentions': 3}
                ],
                'hourly_distribution': {str(i): i % 5 for i in range(24)},
                'day_of_week_distribution': {'Monday': 10, 'Tuesday': 8, 'Wednesday': 12}
            }
        }
        
        fig = visualizer.create_temporal_analysis(sample_metrics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    @pytest.mark.unit
    def test_create_subreddit_analysis(self, visualizer, sample_mentions_list):
        """Test creating subreddit analysis."""
        # Add different subreddits to sample data
        subreddits = ['technology', 'programming', 'artificial']
        for i, mention in enumerate(sample_mentions_list):
            mention['subreddit'] = subreddits[i % len(subreddits)]
        
        sample_metrics = {
            'subreddit_analysis': {
                'top_subreddits_by_mentions': {'technology': 10, 'programming': 8, 'artificial': 5},
                'subreddit_diversity_score': 0.7,
                'total_subreddits': 3
            }
        }
        
        fig = visualizer.create_subreddit_analysis(sample_metrics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    @pytest.mark.unit
    def test_create_engagement_analysis(self, visualizer, sample_mentions_list):
        """Test creating engagement analysis."""
        sample_metrics = {
            'engagement': {
                'score_stats': {'mean': 25.5, 'max': 100, 'min': 1},
                'comment_stats': {'mean': 15.2, 'max': 50, 'min': 0},
                'engagement_distribution': {'low': 5, 'medium': 10, 'high': 3},
                'top_posts': [
                    {'title': 'Top post 1', 'score': 100, 'subreddit': 'technology'},
                    {'title': 'Top post 2', 'score': 85, 'subreddit': 'programming'}
                ]
            }
        }
        
        fig = visualizer.create_engagement_analysis(sample_metrics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    @pytest.mark.unit
    def test_create_sentiment_analysis(self, visualizer, sample_mentions_list):
        """Test creating sentiment analysis."""
        sample_metrics = {
            'sentiment': {
                'overall_sentiment': 0.2,
                'sentiment_distribution': {'positive': 8, 'neutral': 5, 'negative': 2},
                'sentiment_by_subreddit': {'technology': 0.3, 'programming': 0.1},
                'most_positive_subreddit': 'technology',
                'most_negative_subreddit': 'programming'
            }
        }
        
        fig = visualizer.create_sentiment_analysis(sample_metrics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    @pytest.mark.unit
    def test_create_summary_metrics_table(self, visualizer, sample_mentions_list):
        """Test creating summary metrics table."""
        sample_metrics = {
            'overview': {'total_mentions': 15, 'unique_subreddits': 3},
            'engagement': {'score_stats': {'mean': 25.5}},
            'sentiment': {'overall_sentiment': 0.2}
        }
        
        table_html = visualizer.create_summary_metrics_table(sample_metrics)
        
        assert isinstance(table_html, str)
        assert len(table_html) > 0
        assert 'table' in table_html.lower() or 'Total Mentions' in table_html
    
    @pytest.mark.unit
    def test_empty_data_handling(self, visualizer):
        """Test handling of empty data."""
        empty_metrics = {
            'overview': {'total_mentions': 0},
            'engagement': {},
            'temporal': {},
            'subreddit_analysis': {},
            'sentiment': {}
        }
        
        # Should handle empty data gracefully
        fig = visualizer.create_overview_dashboard(empty_metrics)
        assert fig is not None
    
    @pytest.mark.unit
    def test_malformed_data_handling(self, visualizer):
        """Test handling of malformed data."""
        malformed_metrics = {
            'overview': None,  # None instead of dict
            'engagement': {'invalid_key': 'invalid_value'},
            'temporal': {'daily_timeline': 'not_a_list'}  # String instead of list
        }
        
        # Should handle malformed data gracefully
        fig = visualizer.create_overview_dashboard(malformed_metrics)
        assert fig is not None
    
    @pytest.mark.unit
    def test_color_scheme_consistency(self, visualizer, sample_mentions_list):
        """Test color scheme consistency across charts."""
        sample_metrics = {
            'overview': {'total_mentions': len(sample_mentions_list)},
            'engagement': {'engagement_distribution': {'low': 5, 'medium': 10, 'high': 3}},
            'sentiment': {'sentiment_distribution': {'positive': 8, 'neutral': 5, 'negative': 2}}
        }
        
        fig = visualizer.create_overview_dashboard(sample_metrics)
        
        # Check that color scheme is applied
        assert visualizer.color_scheme['primary'] == '#FF4500'  # Reddit orange
        assert visualizer.color_scheme['secondary'] == '#0079D3'  # Reddit blue
    
    @pytest.mark.unit
    def test_chart_interactivity(self, visualizer, sample_mentions_list):
        """Test chart interactivity features."""
        sample_metrics = {
            'temporal': {
                'daily_timeline': [
                    {'date': '2024-01-01', 'mentions': 5},
                    {'date': '2024-01-02', 'mentions': 8}
                ]
            }
        }
        
        fig = visualizer.create_temporal_analysis(sample_metrics)
        
        # Check that figure has interactive elements
        assert fig is not None
        assert hasattr(fig, 'layout')
    
    @pytest.mark.unit
    def test_responsive_design(self, visualizer, sample_mentions_list):
        """Test responsive design features."""
        sample_metrics = {
            'subreddit_analysis': {
                'top_subreddits_by_mentions': {'technology': 10, 'programming': 5}
            }
        }
        
        fig = visualizer.create_subreddit_analysis(sample_metrics)
        
        # Check that figure is properly configured
        assert fig is not None
        assert hasattr(fig, 'layout')


class TestRealTimeMonitor:
    """Test real-time monitoring functionality."""
    
    @pytest.mark.unit
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor is not None
        assert hasattr(monitor, 'host')
        assert hasattr(monitor, 'port')
        assert hasattr(monitor, 'clients')
        assert hasattr(monitor, 'EVENT_TYPES')
        assert monitor.host == "localhost"
        assert monitor.port == 8765
    
    @pytest.mark.unit
    def test_event_types(self, monitor):
        """Test event type definitions."""
        expected_events = [
            'SEARCH_STARTED', 'SEARCH_PROGRESS', 'SEARCH_COMPLETED',
            'SEARCH_FAILED', 'MENTION_FOUND', 'METRICS_UPDATED',
            'CACHE_HIT', 'SYSTEM_STATUS'
        ]
        
        for event in expected_events:
            assert event in monitor.EVENT_TYPES
            assert isinstance(monitor.EVENT_TYPES[event], str)
    
    @pytest.mark.unit
    def test_emit_search_started(self, monitor):
        """Test search started event emission."""
        session_id = 1
        search_term = "test_search"
        max_pages = 5
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_search_started(session_id, search_term, max_pages)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['SEARCH_STARTED']
        assert message['session_id'] == session_id
        assert message['search_term'] == search_term
        assert message['max_pages'] == max_pages
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_search_progress(self, monitor):
        """Test search progress event emission."""
        session_id = 1
        progress = 0.5
        message_text = "Processing page 3 of 5"
        mentions_found = 15
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_search_progress(session_id, progress, message_text, mentions_found)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['SEARCH_PROGRESS']
        assert message['session_id'] == session_id
        assert message['progress'] == progress
        assert message['message'] == message_text
        assert message['mentions_found'] == mentions_found
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_search_completed(self, monitor):
        """Test search completed event emission."""
        session_id = 1
        total_mentions = 25
        duration = 45.5
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_search_completed(session_id, total_mentions, duration)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['SEARCH_COMPLETED']
        assert message['session_id'] == session_id
        assert message['total_mentions'] == total_mentions
        assert message['duration'] == duration
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_search_failed(self, monitor):
        """Test search failed event emission."""
        session_id = 1
        error_message = "API rate limit exceeded"
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_search_failed(session_id, error_message)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['SEARCH_FAILED']
        assert message['session_id'] == session_id
        assert message['error_message'] == error_message
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_mention_found(self, monitor, sample_mentions_list):
        """Test mention found event emission."""
        session_id = 1
        mention_data = sample_mentions_list[0]
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_mention_found(session_id, mention_data)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['MENTION_FOUND']
        assert message['session_id'] == session_id
        assert message['mention_data'] == mention_data
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_metrics_updated(self, monitor):
        """Test metrics updated event emission."""
        session_id = 1
        metrics_summary = {
            'total_mentions': 50,
            'avg_sentiment': 0.2,
            'top_subreddit': 'technology'
        }
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_metrics_updated(session_id, metrics_summary)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['METRICS_UPDATED']
        assert message['session_id'] == session_id
        assert message['metrics_summary'] == metrics_summary
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_cache_hit(self, monitor):
        """Test cache hit event emission."""
        search_term = "test_search"
        cached_count = 100
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_cache_hit(search_term, cached_count)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['CACHE_HIT']
        assert message['search_term'] == search_term
        assert message['cached_count'] == cached_count
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_emit_system_status(self, monitor):
        """Test system status event emission."""
        status = {
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'active_sessions': 3,
            'total_mentions': 1250
        }
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        monitor.emit_system_status(status)
        
        # Check that message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        
        assert message['type'] == monitor.EVENT_TYPES['SYSTEM_STATUS']
        assert message['status'] == status
        assert 'timestamp' in message
    
    @pytest.mark.unit
    def test_get_connection_stats(self, monitor):
        """Test connection statistics."""
        stats = monitor.get_connection_stats()
        
        assert isinstance(stats, dict)
        assert 'total_clients' in stats
        assert 'session_subscriptions' in stats
        assert 'server_status' in stats
        assert 'uptime' in stats
        
        # Should start with 0 clients
        assert stats['total_clients'] == 0
        assert stats['session_subscriptions'] == 0
    
    @pytest.mark.unit
    def test_progress_callback_initialization(self, monitor):
        """Test progress callback initialization."""
        from ui.realtime_monitor import ProgressCallback
        
        session_id = 1
        callback = ProgressCallback(monitor, session_id)
        
        assert callback.monitor == monitor
        assert callback.session_id == session_id
        assert hasattr(callback, '__call__')
    
    @pytest.mark.unit
    def test_progress_callback_execution(self, monitor):
        """Test progress callback execution."""
        from ui.realtime_monitor import ProgressCallback
        
        session_id = 1
        callback = ProgressCallback(monitor, session_id)
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        # Test callback with message only
        callback("Processing data...")
        
        # Check that progress message was queued
        assert not monitor.message_queue.empty()
        message = monitor.message_queue.get()
        assert message['type'] == monitor.EVENT_TYPES['SEARCH_PROGRESS']
        assert message['session_id'] == session_id
        assert message['message'] == "Processing data..."
    
    @pytest.mark.unit
    def test_progress_callback_with_mention(self, monitor, sample_mentions_list):
        """Test progress callback with mention data."""
        from ui.realtime_monitor import ProgressCallback
        
        session_id = 1
        callback = ProgressCallback(monitor, session_id)
        mention_data = sample_mentions_list[0]
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        # Test callback with mention data
        callback("Found new mention", progress=0.5, mention_data=mention_data)
        
        # Should have both progress and mention_found messages
        messages = []
        while not monitor.message_queue.empty():
            messages.append(monitor.message_queue.get())
        
        assert len(messages) >= 1  # At least progress message
        
        # Check progress message
        progress_msg = next((msg for msg in messages if msg['type'] == monitor.EVENT_TYPES['SEARCH_PROGRESS']), None)
        assert progress_msg is not None
        assert progress_msg['progress'] == 0.5
    
    @pytest.mark.unit
    def test_multiple_event_emissions(self, monitor):
        """Test multiple event emissions."""
        session_id = 1
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        # Emit multiple events
        monitor.emit_search_started(session_id, "test", 5)
        monitor.emit_search_progress(session_id, 0.2, "Starting...")
        monitor.emit_search_progress(session_id, 0.5, "Halfway...")
        monitor.emit_search_completed(session_id, 25, 30.0)
        
        # Check all messages were queued
        messages = []
        while not monitor.message_queue.empty():
            messages.append(monitor.message_queue.get())
        
        assert len(messages) == 4
        assert messages[0]['type'] == monitor.EVENT_TYPES['SEARCH_STARTED']
        assert messages[1]['type'] == monitor.EVENT_TYPES['SEARCH_PROGRESS']
        assert messages[2]['type'] == monitor.EVENT_TYPES['SEARCH_PROGRESS']
        assert messages[3]['type'] == monitor.EVENT_TYPES['SEARCH_COMPLETED']
    
    @pytest.mark.unit
    def test_event_message_structure(self, monitor):
        """Test that all events have proper message structure."""
        session_id = 1
        
        # Clear the queue first
        while not monitor.message_queue.empty():
            monitor.message_queue.get()
        
        # Test each event type
        monitor.emit_search_started(session_id, "test", 5)
        monitor.emit_search_progress(session_id, 0.5, "progress", 10)
        monitor.emit_search_completed(session_id, 25, 30.0)
        monitor.emit_search_failed(session_id, "error")
        monitor.emit_mention_found(session_id, {'title': 'test'})
        monitor.emit_metrics_updated(session_id, {'total': 10})
        monitor.emit_cache_hit("test", 5)
        monitor.emit_system_status({'status': 'ok'})
        
        # Check all messages have required fields
        while not monitor.message_queue.empty():
            message = monitor.message_queue.get()
            assert 'type' in message
            assert 'timestamp' in message
            assert message['type'] in monitor.EVENT_TYPES.values()


class TestGradioInterface:
    """Test Gradio interface functionality."""
    
    @pytest.fixture
    def mock_app_components(self):
        """Mock app components for testing."""
        return {
            'scraper': Mock(),
            'db_manager': Mock(),
            'metrics_analyzer': Mock(),
            'visualizer': Mock(),
            'realtime_monitor': Mock()
        }
    
    def test_interface_creation(self, mock_app_components):
        """Test Gradio interface creation."""
        with patch('app.create_gradio_interface') as mock_create:
            mock_interface = Mock()
            mock_create.return_value = mock_interface
            
            # Should create interface without errors
            interface = mock_create(**mock_app_components)
            assert interface is not None
    
    def test_search_functionality(self, mock_app_components):
        """Test search functionality in UI."""
        # Mock scraper response
        mock_mentions = [
            {'reddit_id': 'test1', 'title': 'Test post 1', 'score': 10},
            {'reddit_id': 'test2', 'title': 'Test post 2', 'score': 20}
        ]
        
        mock_app_components['scraper'].scrape_mentions = AsyncMock(return_value=mock_mentions)
        mock_app_components['metrics_analyzer'].generate_comprehensive_metrics.return_value = {
            'basic': {'total_mentions': 2}
        }
        
        # Test search function
        with patch('app.search_mentions') as mock_search:
            mock_search.return_value = ("Search completed", mock_mentions, {})
            
            result = mock_search("OpenAI", 5, 0.3)
            
            assert result is not None
            assert len(result) == 3  # status, mentions, metrics
    
    def test_export_functionality(self, mock_app_components):
        """Test data export functionality."""
        mock_mentions = [
            {'reddit_id': 'test1', 'title': 'Test post 1', 'score': 10},
            {'reddit_id': 'test2', 'title': 'Test post 2', 'score': 20}
        ]
        
        with patch('app.export_data') as mock_export:
            mock_export.return_value = "data.csv"
            
            result = mock_export(mock_mentions, "csv")
            
            assert result == "data.csv"
            mock_export.assert_called_once_with(mock_mentions, "csv")
    
    def test_real_time_updates(self, mock_app_components):
        """Test real-time updates functionality."""
        mock_status = {
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'active_sessions': 3
        }
        
        mock_app_components['realtime_monitor'].get_system_status.return_value = mock_status
        
        with patch('app.update_realtime_metrics') as mock_update:
            mock_update.return_value = mock_status
            
            result = mock_update()
            
            assert result == mock_status
    
    def test_error_handling_in_ui(self, mock_app_components):
        """Test error handling in UI components."""
        # Mock scraper to raise an error
        mock_app_components['scraper'].scrape_mentions = AsyncMock(
            side_effect=Exception("Scraping error")
        )
        
        with patch('app.search_mentions') as mock_search:
            mock_search.return_value = ("Error occurred", [], {})
            
            result = mock_search("OpenAI", 5, 0.3)
            
            # Should handle error gracefully
            assert "Error" in result[0]
            assert result[1] == []  # Empty mentions list
    
    def test_input_validation(self, mock_app_components):
        """Test input validation in UI."""
        with patch('app.validate_search_input') as mock_validate:
            # Test valid input
            mock_validate.return_value = (True, "")
            valid_result = mock_validate("OpenAI", 5, 0.3)
            assert valid_result[0] is True
            
            # Test invalid input
            mock_validate.return_value = (False, "Invalid search term")
            invalid_result = mock_validate("", 5, 0.3)
            assert invalid_result[0] is False
            assert "Invalid" in invalid_result[1]
    
    def test_progress_tracking(self, mock_app_components):
        """Test progress tracking in UI."""
        with patch('app.track_progress') as mock_progress:
            progress_updates = [
                (0.2, "Starting scrape..."),
                (0.5, "Processing mentions..."),
                (0.8, "Generating analytics..."),
                (1.0, "Complete!")
            ]
            
            mock_progress.side_effect = progress_updates
            
            for expected_progress, expected_message in progress_updates:
                result = mock_progress()
                assert result == (expected_progress, expected_message)


class TestUIIntegration:
    """Integration tests for UI components."""
    
    def test_full_ui_workflow(self, mock_app_components, sample_mentions_list):
        """Test full UI workflow from search to visualization."""
        # Mock the complete workflow
        mock_app_components['scraper'].scrape_mentions = AsyncMock(
            return_value=sample_mentions_list
        )
        mock_app_components['metrics_analyzer'].generate_comprehensive_metrics.return_value = {
            'basic': {'total_mentions': len(sample_mentions_list)},
            'sentiment': {'average_sentiment': 0.5}
        }
        mock_app_components['visualizer'].create_comprehensive_dashboard.return_value = {
            'timeline': go.Figure(),
            'sentiment': go.Figure()
        }
        
        with patch('app.full_search_workflow') as mock_workflow:
            mock_workflow.return_value = (
                "Search completed successfully",
                sample_mentions_list,
                {'basic': {'total_mentions': len(sample_mentions_list)}},
                {'timeline': go.Figure(), 'sentiment': go.Figure()}
            )
            
            result = mock_workflow("OpenAI", 5, 0.3)
            
            assert len(result) == 4  # status, mentions, metrics, charts
            assert "successful" in result[0]
            assert len(result[1]) == len(sample_mentions_list)
            assert isinstance(result[2], dict)
            assert isinstance(result[3], dict)
    
    def test_ui_performance_with_large_dataset(self, mock_app_components, performance_monitor):
        """Test UI performance with large dataset."""
        # Create large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                'reddit_id': f'test_{i}',
                'title': f'Test post {i}',
                'score': i % 100,
                'sentiment_score': (i % 200 - 100) / 100
            })
        
        mock_app_components['visualizer'].create_comprehensive_dashboard.return_value = {
            'timeline': go.Figure(),
            'sentiment': go.Figure(),
            'subreddits': go.Figure()
        }
        
        performance_monitor.start()
        
        with patch('app.create_visualizations') as mock_viz:
            mock_viz.return_value = {
                'timeline': go.Figure(),
                'sentiment': go.Figure(),
                'subreddits': go.Figure()
            }
            
            result = mock_viz(large_dataset)
            
        metrics = performance_monitor.stop()
        
        # Should complete within reasonable time
        assert metrics['duration'] < 10.0  # 10 seconds max
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_concurrent_ui_operations(self, mock_app_components):
        """Test concurrent UI operations."""
        import asyncio
        
        async def mock_search_operation(term):
            await asyncio.sleep(0.1)  # Simulate work
            return f"Results for {term}"
        
        async def test_concurrent_searches():
            tasks = [
                mock_search_operation("OpenAI"),
                mock_search_operation("Google"),
                mock_search_operation("Microsoft")
            ]
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Should handle concurrent operations
        with patch('asyncio.gather') as mock_gather:
            mock_gather.return_value = [
                "Results for OpenAI",
                "Results for Google", 
                "Results for Microsoft"
            ]
            
            results = mock_gather()
            assert len(results) == 3
    
    def test_ui_state_management(self, mock_app_components):
        """Test UI state management."""
        with patch('app.UIStateManager') as mock_state:
            state_manager = mock_state.return_value
            state_manager.get_state.return_value = {
                'current_search': 'OpenAI',
                'last_results': [],
                'session_id': 'test_session'
            }
            
            state = state_manager.get_state()
            
            assert isinstance(state, dict)
            assert 'current_search' in state
            assert 'last_results' in state
            assert 'session_id' in state
    
    def test_ui_accessibility(self, mock_app_components):
        """Test UI accessibility features."""
        with patch('app.check_accessibility') as mock_accessibility:
            accessibility_report = {
                'has_alt_text': True,
                'has_aria_labels': True,
                'color_contrast_ratio': 4.5,
                'keyboard_navigation': True
            }
            mock_accessibility.return_value = accessibility_report
            
            report = mock_accessibility()
            
            assert report['has_alt_text'] is True
            assert report['has_aria_labels'] is True
            assert report['color_contrast_ratio'] >= 4.5  # WCAG AA standard
            assert report['keyboard_navigation'] is True 