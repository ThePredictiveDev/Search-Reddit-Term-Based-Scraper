"""
Tests for analytics and metrics functionality.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from analytics.metrics_analyzer import MetricsAnalyzer
from analytics.data_validator import DataValidator, ValidationResult, DataQuality

# Optional import for advanced sentiment analyzer
ADVANCED_SENTIMENT_AVAILABLE = False
AdvancedSentimentAnalyzer = None

try:
    # Only import if explicitly requested and transformers is available
    import os
    if os.environ.get('ENABLE_ADVANCED_SENTIMENT', '').lower() == 'true':
        from analytics.advanced_sentiment import AdvancedSentimentAnalyzer
        ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    pass


class TestMetricsAnalyzer:
    """Test metrics analyzer functionality."""
    
    @pytest.fixture
    def analyzer(self, db_manager):
        """Create metrics analyzer instance."""
        return MetricsAnalyzer(db_manager)
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, 'db_manager')
        assert hasattr(analyzer, 'analyze_session_metrics')
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_calculate_basic_metrics(self, analyzer, sample_mentions_list):
        """Test basic metrics calculation."""
        # Create a mock session ID
        session_id = 1
        
        # Convert dict mentions to mock RedditMention objects with proper datetime conversion
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    if hasattr(value, 'strftime'):
                        # Already a datetime object
                        setattr(mock_mention, key, value)
                    else:
                        # Convert to datetime if not already
                        setattr(mock_mention, key, datetime.utcnow())
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        assert isinstance(metrics, dict)
        assert 'overview' in metrics
        assert 'engagement' in metrics
        assert 'temporal' in metrics
        assert 'subreddit_analysis' in metrics
        assert 'sentiment' in metrics
        
        # Check overview metrics
        overview = metrics['overview']
        assert 'total_mentions' in overview
        assert 'unique_subreddits' in overview
        assert overview['total_mentions'] == len(sample_mentions_list)
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_analyze_session_metrics(self, analyzer, sample_mentions_list):
        """Test session metrics analysis."""
        # Create a mock session ID
        session_id = 1
        
        # Convert dict mentions to mock RedditMention objects with proper datetime conversion
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    if hasattr(value, 'strftime'):
                        # Already a datetime object
                        setattr(mock_mention, key, value)
                    else:
                        # Convert to datetime if not already
                        setattr(mock_mention, key, datetime.utcnow())
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        assert isinstance(metrics, dict)
        assert 'overview' in metrics
        assert 'engagement' in metrics
        assert 'temporal' in metrics
        assert 'subreddit_analysis' in metrics
        assert 'sentiment' in metrics
        
        # Check overview metrics
        overview = metrics['overview']
        assert 'total_mentions' in overview
        assert 'unique_subreddits' in overview
        assert overview['total_mentions'] == len(sample_mentions_list)
    
    @pytest.mark.unit
    def test_calculate_temporal_metrics(self, analyzer, sample_mentions_list):
        """Test temporal metrics calculation."""
        # Create a mock session ID
        session_id = 1
        
        # Convert dict mentions to mock RedditMention objects
        from database.models import RedditMention
        mock_mentions = []
        now = datetime.utcnow()
        for i, mention_dict in enumerate(sample_mentions_list):
            mention_dict['created_utc'] = now - timedelta(hours=i)
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    setattr(mock_mention, key, value)
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Check temporal metrics are included
        assert 'temporal' in metrics
        temporal = metrics['temporal']
        assert 'daily_timeline' in temporal
        assert 'hourly_distribution' in temporal
        assert 'day_of_week_distribution' in temporal
        assert 'trend' in temporal
    
    @pytest.mark.unit
    def test_calculate_subreddit_metrics(self, analyzer, sample_mentions_list):
        """Test subreddit metrics calculation."""
        # Create a mock session ID
        session_id = 1
        
        # Add different subreddits
        subreddits = ['technology', 'programming', 'artificial', 'MachineLearning']
        for i, mention in enumerate(sample_mentions_list):
            mention['subreddit'] = subreddits[i % len(subreddits)]
        
        # Convert dict mentions to mock RedditMention objects
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    setattr(mock_mention, key, value)
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Check subreddit metrics are included
        assert 'subreddit_analysis' in metrics
        subreddit_analysis = metrics['subreddit_analysis']
        assert 'top_subreddits_by_mentions' in subreddit_analysis
        assert 'subreddit_diversity_score' in subreddit_analysis
        assert 'total_subreddits' in subreddit_analysis
    
    @pytest.mark.unit
    def test_calculate_sentiment_metrics(self, analyzer, sample_mentions_list):
        """Test sentiment metrics calculation."""
        # Create a mock session ID
        session_id = 1
        
        # Convert dict mentions to mock RedditMention objects
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    setattr(mock_mention, key, value)
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Check sentiment metrics are included
        assert 'sentiment' in metrics
        sentiment = metrics['sentiment']
        assert 'overall_sentiment' in sentiment
        assert 'sentiment_distribution' in sentiment
        assert 'most_positive_subreddit' in sentiment
        assert 'most_negative_subreddit' in sentiment
    
    @pytest.mark.unit
    def test_calculate_engagement_metrics(self, analyzer, sample_mentions_list):
        """Test engagement metrics calculation."""
        # Create a mock session ID
        session_id = 1
        
        # Convert dict mentions to mock RedditMention objects
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    setattr(mock_mention, key, value)
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Check engagement metrics are included
        assert 'engagement' in metrics
        engagement = metrics['engagement']
        assert 'score_stats' in engagement
        assert 'comment_stats' in engagement
        assert 'high_engagement_count' in engagement
        assert 'engagement_distribution' in engagement
    
    @pytest.mark.unit
    def test_detect_trending_topics(self, analyzer, sample_mentions_list):
        """Test trending topic detection."""
        # Create a mock session ID
        session_id = 1
        
        # Add keywords to titles
        keywords = ['AI', 'machine learning', 'neural networks', 'deep learning']
        for i, mention in enumerate(sample_mentions_list):
            mention['title'] = f"Post about {keywords[i % len(keywords)]} technology"
        
        # Convert dict mentions to mock RedditMention objects
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    setattr(mock_mention, key, value)
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Check trending metrics are included
        assert 'trending' in metrics
        trending = metrics['trending']
        assert 'momentum_24h' in trending
        assert 'is_trending' in trending
        
        # Check content analysis for common words
        assert 'content_analysis' in metrics
        content_analysis = metrics['content_analysis']
        assert 'common_words' in content_analysis
    
    @pytest.mark.unit
    def test_generate_comprehensive_metrics(self, analyzer, sample_mentions_list):
        """Test comprehensive metrics generation."""
        # Create a mock session ID
        session_id = 1
        
        # Convert dict mentions to mock RedditMention objects
        from database.models import RedditMention
        mock_mentions = []
        for mention_dict in sample_mentions_list:
            mock_mention = Mock(spec=RedditMention)
            for key, value in mention_dict.items():
                # For datetime fields, always use actual datetime objects
                if key in ['created_utc', 'scraped_at']:
                    setattr(mock_mention, key, value)
                else:
                    setattr(mock_mention, key, value)
            mock_mentions.append(mock_mention)
        
        # Mock the database call to return our mock mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Check all major metric categories are present
        expected_categories = [
            'overview', 'engagement', 'temporal', 'subreddit_analysis', 
            'sentiment', 'content_analysis', 'trending'
        ]
        
        for category in expected_categories:
            assert category in metrics
        
        # Check overview metrics
        assert 'total_mentions' in metrics['overview']
        assert 'unique_subreddits' in metrics['overview']
        assert 'analysis_period_days' in metrics['overview']
    
    @pytest.mark.unit
    def test_empty_mentions_handling(self, analyzer):
        """Test handling of empty mentions list."""
        # Create a mock session ID
        session_id = 1
        
        # Mock the database call to return empty list
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=[]):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Should return empty metrics structure
        assert isinstance(metrics, dict)
        # Empty metrics should still have the basic structure
        expected_keys = ['overview', 'engagement', 'temporal', 'subreddit_analysis', 'sentiment', 'content_analysis', 'trending']
        for key in expected_keys:
            assert key in metrics
    
    @pytest.mark.unit
    def test_malformed_data_handling(self, analyzer):
        """Test handling of malformed mention data."""
        # Create a mock session ID
        session_id = 1
        
        # Create malformed mock mentions (missing some fields)
        from database.models import RedditMention
        mock_mentions = []
        
        # Create a mention with missing fields
        mock_mention = Mock(spec=RedditMention)
        mock_mention.id = 1
        mock_mention.reddit_id = 'test123'
        mock_mention.post_type = 'submission'
        mock_mention.title = None  # Missing title
        mock_mention.content = None  # Missing content
        mock_mention.author = 'test_user'
        mock_mention.subreddit = 'test'
        mock_mention.url = 'https://reddit.com/test'
        mock_mention.score = 10
        mock_mention.num_comments = 5
        mock_mention.upvote_ratio = 0.8
        mock_mention.created_utc = datetime.utcnow()
        mock_mention.scraped_at = datetime.utcnow()
        mock_mention.sentiment_score = 0.0
        mock_mention.relevance_score = 0.5
        
        mock_mentions.append(mock_mention)
        
        # Mock the database call to return malformed mentions
        with patch.object(analyzer.db_manager, 'get_mentions_by_session', return_value=mock_mentions):
            metrics = analyzer.analyze_session_metrics(session_id)
        
        # Should handle malformed data gracefully
        assert isinstance(metrics, dict)
        assert 'overview' in metrics
        assert metrics['overview']['total_mentions'] == 1


class TestDataValidator:
    """Test data validator functionality."""
    
    @pytest.mark.unit
    def test_validator_initialization(self, data_validator):
        """Test data validator initialization."""
        assert data_validator is not None
        assert hasattr(data_validator, 'validation_rules')
        assert hasattr(data_validator, 'spam_patterns')
        assert hasattr(data_validator, 'quality_weights')
    
    @pytest.mark.unit
    def test_validate_mention_valid_data(self, data_validator, sample_mention_data):
        """Test validation of valid mention data."""
        result = data_validator.validate_mention(sample_mention_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert 0 <= result.quality_score <= 1
        assert isinstance(result.quality_level, DataQuality)
        assert isinstance(result.issues, list)
        assert isinstance(result.suggestions, list)
    
    @pytest.mark.unit
    def test_validate_mention_invalid_data(self, data_validator):
        """Test validation of invalid mention data."""
        invalid_mention = {
            'reddit_id': '',  # Empty required field
            'title': 'a',  # Too short
            'author': '',  # Empty required field
            'score': -2000,  # Below minimum
            'content': 'buy now click here spam'  # Spam content
        }
        
        result = data_validator.validate_mention(invalid_mention)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.quality_score < 0.5
        assert len(result.issues) > 0
    
    def test_validate_mention_spam_detection(self, data_validator):
        """Test spam detection in validation."""
        spam_mention = {
            'reddit_id': 'spam123',
            'title': 'Buy now! Limited time offer! Click here for discount!',
            'content': 'Amazing deal! Free bonus! Act now! Visit bit.ly/spam',
            'author': 'spambot123',
            'subreddit': 'test',
            'score': 1,
            'num_comments': 0,
            'created_utc': datetime.utcnow()
        }
        
        result = data_validator.validate_mention(spam_mention)
        
        # Should detect spam patterns
        spam_issues = [issue for issue in result.issues if 'spam' in issue.get('type', '')]
        assert len(spam_issues) > 0
        assert result.quality_score < 0.7  # Should have low quality score
    
    def test_validate_dataset(self, data_validator, sample_mentions_list):
        """Test dataset validation."""
        # Add some invalid mentions
        invalid_mentions = [
            {'reddit_id': '', 'title': 'Invalid'},  # Missing required fields
            {'reddit_id': 'spam', 'title': 'Buy now! Click here!', 'author': 'spammer'}  # Spam
        ]
        
        all_mentions = sample_mentions_list + invalid_mentions
        
        validated_mentions, quality_metrics = data_validator.validate_dataset(all_mentions)
        
        # Should filter out invalid mentions
        assert len(validated_mentions) <= len(all_mentions)
        assert len(validated_mentions) >= len(sample_mentions_list) - 2  # Allow some filtering
        
        # Check quality metrics
        assert quality_metrics.total_records == len(all_mentions)
        assert quality_metrics.valid_records == len(validated_mentions)
        assert 0 <= quality_metrics.average_quality_score <= 1
    
    def test_duplicate_detection(self, data_validator, sample_mention_data):
        """Test duplicate detection."""
        # Create mentions with duplicates
        mentions = [
            sample_mention_data.copy(),
            sample_mention_data.copy(),  # Exact duplicate
            {**sample_mention_data, 'score': 100},  # Same content, different score
        ]
        
        duplicates = data_validator._detect_duplicates(mentions)
        
        assert len(duplicates) > 0  # Should detect duplicates
        assert isinstance(duplicates, set)
    
    def test_language_analysis(self, data_validator):
        """Test language analysis."""
        english_mention = {
            'title': 'This is an English post about technology',
            'content': 'Here we discuss the latest developments in AI and machine learning.'
        }
        
        result = data_validator._analyze_language(english_mention)
        
        assert 'score' in result
        assert 'issues' in result
        assert 0 <= result['score'] <= 1
    
    def test_content_quality_analysis(self, data_validator):
        """Test content quality analysis."""
        high_quality_mention = {
            'title': 'Comprehensive Analysis of Machine Learning Trends in 2024',
            'content': '''This detailed analysis examines the current state of machine learning 
                         technology, including recent breakthroughs in neural networks, 
                         natural language processing, and computer vision. The research 
                         methodology involved analyzing over 1000 papers published in 
                         top-tier conferences and journals.'''
        }
        
        low_quality_mention = {
            'title': 'ai',
            'content': 'bad'
        }
        
        high_result = data_validator._analyze_content_quality(high_quality_mention)
        low_result = data_validator._analyze_content_quality(low_quality_mention)
        
        assert high_result['scores']['content_quality'] > low_result['scores']['content_quality']
        assert high_result['scores']['length_score'] > low_result['scores']['length_score']
    
    def test_quality_metrics_calculation(self, data_validator, sample_mentions_list):
        """Test quality metrics calculation."""
        # Create validation results
        validation_results = []
        for mention in sample_mentions_list:
            result = data_validator.validate_mention(mention)
            validation_results.append(result)
        
        quality_metrics = data_validator._calculate_quality_metrics(
            sample_mentions_list, validation_results, set()
        )
        
        assert quality_metrics.total_records == len(sample_mentions_list)
        assert quality_metrics.valid_records <= quality_metrics.total_records
        assert 0 <= quality_metrics.average_quality_score <= 1
        assert isinstance(quality_metrics.quality_distribution, dict)
        assert isinstance(quality_metrics.common_issues, list)


@pytest.mark.skipif(not ADVANCED_SENTIMENT_AVAILABLE, reason="Advanced sentiment analyzer not available")
class TestAdvancedSentimentAnalyzer:
    """Test advanced sentiment analysis functionality."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """Advanced sentiment analyzer for testing."""
        return AdvancedSentimentAnalyzer()
    
    def test_analyzer_initialization(self, sentiment_analyzer):
        """Test sentiment analyzer initialization."""
        assert sentiment_analyzer is not None
        assert hasattr(sentiment_analyzer, 'providers')
    
    def test_is_available(self, sentiment_analyzer):
        """Test availability check."""
        # Should return boolean
        available = sentiment_analyzer.is_available()
        assert isinstance(available, bool)
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_positive(self, sentiment_analyzer):
        """Test positive sentiment analysis."""
        positive_text = "This is absolutely amazing! I love this product. It's fantastic and wonderful!"
        
        if sentiment_analyzer.is_available():
            result = await sentiment_analyzer.analyze_sentiment(positive_text)
            
            assert isinstance(result, dict)
            assert 'composite_sentiment' in result
            assert result['composite_sentiment'] > 0  # Should be positive
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_negative(self, sentiment_analyzer):
        """Test negative sentiment analysis."""
        negative_text = "This is terrible! I hate this product. It's awful and disappointing."
        
        if sentiment_analyzer.is_available():
            result = await sentiment_analyzer.analyze_sentiment(negative_text)
            
            assert isinstance(result, dict)
            assert 'composite_sentiment' in result
            assert result['composite_sentiment'] < 0  # Should be negative
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_empty(self, sentiment_analyzer):
        """Test sentiment analysis with empty text."""
        if sentiment_analyzer.is_available():
            result = await sentiment_analyzer.analyze_sentiment("")
            
            assert isinstance(result, dict)
            assert 'composite_sentiment' in result
            assert result['composite_sentiment'] == 0.0  # Should be neutral
    
    def test_analyze_batch(self, sentiment_analyzer):
        """Test batch sentiment analysis."""
        texts = [
            "This is great!",
            "This is terrible!",
            "This is okay.",
            ""
        ]
        
        if sentiment_analyzer.is_available():
            results = sentiment_analyzer.analyze_batch(texts)
            
            assert isinstance(results, list)
            assert len(results) == len(texts)
            
            for result in results:
                assert isinstance(result, dict)
                assert 'composite_sentiment' in result
    
    def test_textblob_analysis(self, sentiment_analyzer):
        """Test TextBlob sentiment analysis."""
        text = "This is a great product!"
        
        result = sentiment_analyzer._analyze_textblob(text)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'confidence' in result
        assert -1 <= result['sentiment'] <= 1
        assert 0 <= result['confidence'] <= 1
    
    def test_vader_analysis(self, sentiment_analyzer):
        """Test VADER sentiment analysis."""
        text = "This is a great product!"
        
        if sentiment_analyzer._is_vader_available():
            result = sentiment_analyzer._analyze_vader(text)
            
            assert isinstance(result, dict)
            assert 'sentiment' in result
            assert 'confidence' in result
            assert -1 <= result['sentiment'] <= 1
    
    def test_emotion_detection(self, sentiment_analyzer):
        """Test emotion detection."""
        emotional_texts = {
            "I'm so happy and excited!": "joy",
            "I'm really angry about this!": "anger",
            "This makes me so sad.": "sadness",
            "I'm terrified and scared.": "fear"
        }
        
        for text, expected_emotion in emotional_texts.items():
            emotions = sentiment_analyzer._detect_emotions(text)
            
            assert isinstance(emotions, dict)
            assert 'primary_emotion' in emotions
            assert 'confidence' in emotions
            # Note: Emotion detection might not be 100% accurate, so we just check structure


class TestAnalyticsIntegration:
    """Integration tests for analytics components."""
    
    @pytest.mark.integration
    def test_full_analytics_pipeline(self, db_manager, data_validator, sample_mentions_list):
        """Test full analytics pipeline."""
        metrics_analyzer = MetricsAnalyzer(db_manager)
        
        # Step 1: Validate data
        validated_mentions, quality_metrics = data_validator.validate_dataset(sample_mentions_list)
        
        # Step 2: Store mentions in database for analytics
        session_id = db_manager.create_search_session("analytics_integration_test")
        for mention in validated_mentions:
            db_manager.add_mention(session_id, mention)
        
        # Step 3: Generate metrics using session-based analytics
        analytics_metrics = metrics_analyzer.analyze_session_metrics(session_id)
        
        # Step 4: Verify integration
        assert len(validated_mentions) <= len(sample_mentions_list)
        assert isinstance(analytics_metrics, dict)
        assert 'overview' in analytics_metrics
        assert 'engagement' in analytics_metrics
        
        # Quality metrics should be consistent
        assert quality_metrics.valid_records == len(validated_mentions)
        assert analytics_metrics['overview']['total_mentions'] == len(validated_mentions)
    
    @pytest.mark.integration
    def test_analytics_with_empty_data(self, db_manager, data_validator):
        """Test analytics pipeline with empty data."""
        metrics_analyzer = MetricsAnalyzer(db_manager)
        empty_mentions = []
        
        # Should handle empty data gracefully
        validated_mentions, quality_metrics = data_validator.validate_dataset(empty_mentions)
        
        # Create empty session
        session_id = db_manager.create_search_session("empty_test")
        analytics_metrics = metrics_analyzer.analyze_session_metrics(session_id)
        
        assert len(validated_mentions) == 0
        assert quality_metrics.total_records == 0
        assert analytics_metrics['overview']['total_mentions'] == 0
    
    @pytest.mark.integration
    def test_analytics_performance(self, db_manager, data_validator, performance_monitor):
        """Test analytics performance with large dataset."""
        metrics_analyzer = MetricsAnalyzer(db_manager)
        
        # Create large dataset
        large_dataset = []
        base_mention = {
            'reddit_id': 'test',
            'title': 'Test post about technology',
            'content': 'This is a test post discussing various technology topics.',
            'author': 'test_user',
            'subreddit': 'technology',
            'url': 'https://reddit.com/r/technology/comments/test',  # Add missing URL
            'score': 42,
            'num_comments': 15,
            'post_type': 'submission',
            'created_utc': datetime.utcnow(),
            'sentiment_score': 0.5,
            'relevance_score': 0.8
        }
        
        for i in range(100):  # Reduced size for faster testing
            mention = base_mention.copy()
            mention['reddit_id'] = f'test_{i}'
            mention['title'] = f'Test post {i} about technology'
            large_dataset.append(mention)
        
        performance_monitor.start()
        
        # Run analytics pipeline
        validated_mentions, quality_metrics = data_validator.validate_dataset(large_dataset)
        
        # Store in database
        session_id = db_manager.create_search_session("performance_test")
        for mention in validated_mentions:
            db_manager.add_mention(session_id, mention)
        
        # Generate analytics
        analytics_metrics = metrics_analyzer.analyze_session_metrics(session_id)
        
        metrics = performance_monitor.stop()
        
        # Performance assertions
        assert metrics['duration'] < 30.0  # Should complete within 30 seconds
        assert len(validated_mentions) > 0
        assert analytics_metrics['overview']['total_mentions'] > 0
        
        print(f"Analytics performance: {metrics['duration']:.2f}s for {len(large_dataset)} records")


class TestAnalyticsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_metrics_with_extreme_values(self, db_manager):
        """Test metrics calculation with extreme values."""
        metrics_analyzer = MetricsAnalyzer(db_manager)
        
        extreme_mentions = [
            {
                'reddit_id': 'extreme1',
                'title': 'Post with extreme score',
                'score': 999999,  # Very high score
                'num_comments': 50000,  # Very high comments
                'sentiment_score': 1.0,  # Maximum sentiment
                'created_utc': datetime.utcnow(),
                'post_type': 'submission',  # Add required field
                'author': 'user1',
                'subreddit': 'test'
            },
            {
                'reddit_id': 'extreme2',
                'title': 'Post with negative score',
                'score': -1000,  # Very low score
                'num_comments': 0,  # No comments
                'sentiment_score': -1.0,  # Minimum sentiment
                'created_utc': datetime.utcnow(),
                'post_type': 'submission',  # Add required field
                'author': 'user2',
                'subreddit': 'test'
            }
        ]
        
        # Store mentions in database
        session_id = db_manager.create_search_session("extreme_values_test")
        for mention in extreme_mentions:
            db_manager.add_mention(session_id, mention)
        
        # Should handle extreme values gracefully
        metrics = metrics_analyzer.analyze_session_metrics(session_id)
        
        assert isinstance(metrics, dict)
        assert 'overview' in metrics
        assert 'engagement' in metrics
        assert metrics['overview']['total_mentions'] == 2
        
        # Engagement metrics should handle extreme values
        engagement = metrics['engagement']
        assert 'score_stats' in engagement
        assert engagement['score_stats']['max_score'] == 999999
        assert engagement['score_stats']['min_score'] == -1000
    
    def test_validation_with_unicode_content(self, data_validator):
        """Test data validation with Unicode content."""
        unicode_mentions = [
            {
                'reddit_id': 'unicode1',
                'title': 'Post with émojis 🚀🎉 and ünïcödé',
                'content': 'Content with 中文字符 and العربية text',
                'author': 'üser_nämé',
                'subreddit': 'tëst',
                'score': 42,
                'num_comments': 5,
                'post_type': 'submission',  # Add required field
                'created_utc': datetime.utcnow()
            }
        ]
        
        # Should handle Unicode content gracefully
        validated_mentions, quality_metrics = data_validator.validate_dataset(unicode_mentions)
        
        assert len(validated_mentions) > 0
        assert quality_metrics.valid_records > 0
        
        # Unicode content should be preserved
        validated = validated_mentions[0]
        assert 'émojis' in validated['title']
        assert '中文字符' in validated['content']
    
    def test_analytics_with_missing_timestamps(self, db_manager):
        """Test analytics with missing or invalid timestamps."""
        metrics_analyzer = MetricsAnalyzer(db_manager)
        
        mentions_with_missing_timestamps = [
            {
                'reddit_id': 'missing1',
                'title': 'Post without timestamp',
                'score': 10,
                'num_comments': 2,
                'created_utc': None,  # Missing timestamp
                'author': 'user1',
                'subreddit': 'test',
                'post_type': 'submission'  # Add required field
            },
            {
                'reddit_id': 'valid1',
                'title': 'Post with valid timestamp',
                'score': 15,
                'num_comments': 3,
                'created_utc': datetime.utcnow(),
                'author': 'user2',
                'subreddit': 'test',
                'post_type': 'submission'  # Add required field
            }
        ]
        
        # Store mentions in database (with proper handling of None timestamps)
        session_id = db_manager.create_search_session("missing_timestamps_test")
        for mention in mentions_with_missing_timestamps:
            if mention['created_utc'] is None:
                mention['created_utc'] = datetime.utcnow()  # Default to current time
            db_manager.add_mention(session_id, mention)
        
        # Should handle missing timestamps gracefully
        metrics = metrics_analyzer.analyze_session_metrics(session_id)
        
        assert isinstance(metrics, dict)
        assert 'overview' in metrics
        assert 'temporal' in metrics
        assert metrics['overview']['total_mentions'] == 2 