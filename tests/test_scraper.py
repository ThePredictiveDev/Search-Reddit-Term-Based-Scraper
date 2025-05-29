"""
Tests for Reddit scraper functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from scraper.reddit_scraper import RedditScraper, ScrapingError, RateLimitError


class TestRedditScraper:
    """Test Reddit scraper functionality."""
    
    @pytest.fixture
    def scraper(self, db_manager):
        """Create scraper instance for testing."""
        return RedditScraper(db_manager)
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_scraper_initialization(self, scraper):
        """Test scraper initialization."""
        assert scraper.db_manager is not None
        assert scraper.logger is not None
        assert scraper.throttler is not None
        assert hasattr(scraper, 'search_patterns')
        assert hasattr(scraper, 'subreddit_categories')
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_sanitize_search_term(self, scraper):
        """Test search term sanitization."""
        # Valid search terms
        assert scraper._sanitize_search_term("OpenAI") == "OpenAI"
        assert scraper._sanitize_search_term("  OpenAI  ") == "OpenAI"
        assert scraper._sanitize_search_term("Open AI") == "Open AI"
        
        # Invalid search terms
        assert scraper._sanitize_search_term("") == ""
        assert scraper._sanitize_search_term("   ") == ""
        assert scraper._sanitize_search_term(None) == ""
        
        # Dangerous characters
        assert scraper._sanitize_search_term("OpenAI<script>") == "OpenAI"
        assert scraper._sanitize_search_term('OpenAI"test') == "OpenAItest"
    
    def test_generate_cache_key(self, scraper):
        """Test cache key generation."""
        key1 = scraper._generate_cache_key("OpenAI", 5, 0.3)
        key2 = scraper._generate_cache_key("OpenAI", 5, 0.3)
        key3 = scraper._generate_cache_key("OpenAI", 3, 0.3)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        assert key1 != key3
        
        # Keys should be strings
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    def test_extract_reddit_id(self, scraper):
        """Test Reddit ID extraction from URLs."""
        # Valid URLs
        url1 = "https://reddit.com/r/technology/comments/abc123/test_post"
        assert scraper._extract_reddit_id(url1) == "abc123"
        
        url2 = "https://www.reddit.com/r/programming/comments/xyz789/another_post/"
        assert scraper._extract_reddit_id(url2) == "xyz789"
        
        # Invalid URLs
        assert scraper._extract_reddit_id("") is None
        assert scraper._extract_reddit_id("https://google.com") is None
        assert scraper._extract_reddit_id(None) is None
    
    def test_parse_score(self, scraper):
        """Test score parsing from Reddit text."""
        assert scraper._parse_score("upvote. 42 points") == 42
        assert scraper._parse_score("upvote. 1 point") == 1
        assert scraper._parse_score("upvote. 1,234 points") == 1
        assert scraper._parse_score("") == 0
        assert scraper._parse_score("no numbers here") == 0
    
    def test_parse_comment_count(self, scraper):
        """Test comment count parsing."""
        assert scraper._parse_comment_count("15 comments") == 15
        assert scraper._parse_comment_count("1 comment") == 1
        assert scraper._parse_comment_count("") == 0
        assert scraper._parse_comment_count("no comments") == 0
    
    @pytest.mark.unit
    def test_calculate_relevance_score(self, scraper):
        """Test relevance score calculation."""
        # High relevance case
        high_relevance_title = "OpenAI releases new GPT model"
        high_relevance_content = "OpenAI has announced a breakthrough in artificial intelligence"
        
        score = scraper._calculate_relevance_score(high_relevance_title, high_relevance_content, "OpenAI")
        assert score > 0.5
        
        # Low relevance case
        low_relevance_title = "Random post about cats"
        low_relevance_content = "This is completely unrelated content about pets"
        
        score = scraper._calculate_relevance_score(low_relevance_title, low_relevance_content, "OpenAI")
        assert score < 0.3
        
        # Medium relevance case (partial match)
        medium_relevance_title = "AI developments in 2024"
        medium_relevance_content = "Various companies including OpenAI are making progress"
        
        score = scraper._calculate_relevance_score(medium_relevance_title, medium_relevance_content, "OpenAI")
        assert 0.3 <= score <= 0.8
    
    def test_calculate_basic_sentiment(self, scraper):
        """Test basic sentiment calculation."""
        # Positive sentiment
        positive_score = scraper._calculate_basic_sentiment(
            "Great news!", 
            "This is amazing and wonderful"
        )
        assert positive_score > 0
        
        # Negative sentiment
        negative_score = scraper._calculate_basic_sentiment(
            "Terrible news", 
            "This is awful and disappointing"
        )
        assert negative_score < 0
        
        # Empty content
        neutral_score = scraper._calculate_basic_sentiment("", "")
        assert neutral_score == 0.0
    
    def test_is_relevant(self, scraper):
        """Test relevance filtering."""
        relevant_mention = {
            'relevance_score': 0.8,
            'title': 'OpenAI news'
        }
        assert scraper._is_relevant(relevant_mention, "OpenAI") is True
        
        irrelevant_mention = {
            'relevance_score': 0.05,
            'title': 'Random post'
        }
        assert scraper._is_relevant(irrelevant_mention, "OpenAI") is False
    
    def test_remove_duplicates(self, scraper):
        """Test duplicate removal."""
        mentions = [
            {'reddit_id': 'abc123', 'title': 'Post 1'},
            {'reddit_id': 'def456', 'title': 'Post 2'},
            {'reddit_id': 'abc123', 'title': 'Post 1 duplicate'},
            {'reddit_id': 'ghi789', 'title': 'Post 3'}
        ]
        
        unique_mentions = scraper._remove_duplicates(mentions)
        
        assert len(unique_mentions) == 3
        reddit_ids = [m['reddit_id'] for m in unique_mentions]
        assert len(set(reddit_ids)) == 3  # All unique


class TestScrapingWithMocks:
    """Test scraping functionality with mocked dependencies."""
    
    @pytest.fixture
    def scraper_with_mocks(self, db_manager):
        """Create scraper with mocked external dependencies."""
        scraper = RedditScraper(db_manager)
        
        # Mock cache manager
        scraper.cache_manager = Mock()
        scraper.cache_manager.get_search_results.return_value = None
        scraper.cache_manager.set_search_results.return_value = True
        
        # Mock advanced sentiment
        scraper.advanced_sentiment = Mock()
        scraper.advanced_sentiment.is_available.return_value = True
        scraper.advanced_sentiment.analyze_batch.return_value = [
            {'composite_sentiment': 0.5, 'transformer_confidence': 0.8}
        ]
        
        return scraper
    
    @pytest.mark.asyncio
    async def test_extract_post_data_success(self, scraper_with_mocks, test_data_generator):
        """Test successful post data extraction."""
        # Create mock post element
        mock_post = AsyncMock()
        
        # Mock title element
        mock_title = AsyncMock()
        mock_title.inner_text.return_value = "Test OpenAI Post"
        mock_post.query_selector.side_effect = lambda selector: {
            'h3': mock_title,
            'a[data-click-id="body"]': AsyncMock(**{
                'get_attribute.return_value': '/r/test/comments/abc123/test_post'
            }),
            '[data-testid="subreddit-name"]': AsyncMock(**{
                'inner_text.return_value': 'r/technology'
            }),
            '[data-testid="post_author_link"]': AsyncMock(**{
                'inner_text.return_value': 'u/test_user'
            }),
            '[data-testid="vote-arrows"] button': AsyncMock(**{
                'get_attribute.return_value': 'upvote. 42 points'
            }),
            '[data-testid="comment-count"]': AsyncMock(**{
                'inner_text.return_value': '15 comments'
            }),
            '[data-testid="post-content"] p': AsyncMock(**{
                'inner_text.return_value': 'Test content about OpenAI'
            })
        }.get(selector)
        
        result = await scraper_with_mocks._extract_post_data(mock_post, "OpenAI", 1)
        
        assert result is not None
        assert result['reddit_id'] == 'abc123'
        assert result['title'] == 'Test OpenAI Post'
        assert result['subreddit'] == 'technology'
        assert result['author'] == 'test_user'
        assert result['score'] == 42
        assert result['num_comments'] == 15
        assert result['content'] == 'Test content about OpenAI'
        assert 'relevance_score' in result
        assert 'sentiment_score' in result
    
    @pytest.mark.asyncio
    async def test_extract_post_data_missing_elements(self, scraper_with_mocks):
        """Test post data extraction with missing elements."""
        # Create mock post element with missing data
        mock_post = AsyncMock()
        mock_post.query_selector.return_value = None  # All selectors return None
        
        result = await scraper_with_mocks._extract_post_data(mock_post, "OpenAI", 1)
        
        # Should handle missing elements gracefully
        assert result is None or result['title'] == ""
    
    @pytest.mark.asyncio
    async def test_handle_popups(self, scraper_with_mocks):
        """Test popup handling."""
        mock_page = AsyncMock()
        
        # Mock popup elements
        mock_cookie_button = AsyncMock()
        mock_app_banner = AsyncMock()
        mock_login_close = AsyncMock()
        
        mock_page.query_selector.side_effect = lambda selector: {
            'button[data-testid="cookie-banner-accept"]': mock_cookie_button,
            '[data-testid="app-banner-close"]': mock_app_banner,
            'button[aria-label="Close"]': mock_login_close
        }.get(selector)
        
        # Should not raise any exceptions
        await scraper_with_mocks._handle_popups(mock_page)
        
        # Verify popup elements were clicked
        mock_cookie_button.click.assert_called_once()
        mock_app_banner.click.assert_called_once()
        mock_login_close.click.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scrape_mentions_with_cache_hit(self, scraper_with_mocks):
        """Test scraping with cache hit."""
        # Mock cache hit
        cached_data = [
            {'reddit_id': 'cached123', 'title': 'Cached post', 'relevance_score': 0.8}
        ]
        scraper_with_mocks.cache_manager.get_search_results.return_value = cached_data
        
        result = await scraper_with_mocks.scrape_mentions("OpenAI", 1)
        
        assert result == cached_data
        # Should not have called actual scraping
        scraper_with_mocks.cache_manager.get_search_results.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scrape_mentions_error_handling(self, scraper_with_mocks):
        """Test error handling during scraping."""
        # Mock cache miss
        scraper_with_mocks.cache_manager.get_search_results.return_value = None
        
        # Mock playwright to raise an error
        with patch('scraper.reddit_scraper.async_playwright') as mock_playwright:
            mock_playwright.side_effect = Exception("Playwright error")
            
            with pytest.raises(ScrapingError):
                await scraper_with_mocks.scrape_mentions("OpenAI", 1)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, scraper_with_mocks):
        """Test circuit breaker pattern."""
        # Simulate multiple failures to trigger circuit breaker
        for _ in range(6):  # Exceed failure threshold
            scraper_with_mocks._record_failure()
        
        # Circuit breaker should be open
        assert not scraper_with_mocks._check_circuit_breaker()
        
        # Test recovery
        scraper_with_mocks._record_success()
        # Should still be open until timeout
        assert not scraper_with_mocks._check_circuit_breaker()


class TestScrapingPerformance:
    """Test scraping performance and optimization."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, db_manager, performance_monitor):
        """Test rate limiting functionality."""
        scraper = RedditScraper(db_manager)
        
        performance_monitor.start()
        
        # Make multiple throttled requests
        for _ in range(5):
            async with scraper.throttler:
                await asyncio.sleep(0.1)  # Simulate work
        
        metrics = performance_monitor.stop()
        
        # Should take at least the rate limit time
        expected_min_time = 5 * 0.5  # 5 requests with 0.5s minimum interval
        assert metrics['duration'] >= expected_min_time * 0.8  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_concurrent_scraping_limits(self, db_manager):
        """Test concurrent scraping limits."""
        scraper = RedditScraper(db_manager)
        
        # Mock the actual scraping to avoid network calls
        async def mock_scrape_pattern(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [{'reddit_id': 'test', 'title': 'Test', 'relevance_score': 0.5}]
        
        scraper._scrape_pattern = mock_scrape_pattern
        
        # Test concurrent operations
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                scraper._scrape_pattern(None, "test_pattern", f"term_{i}", 1, 1, None, None)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete successfully
        assert len(results) == 10
        assert all(not isinstance(r, Exception) for r in results)


class TestScrapingIntegration:
    """Integration tests for scraping functionality."""
    
    @pytest.mark.asyncio
    async def test_full_scraping_pipeline_mock(self, db_manager, sample_mentions_list):
        """Test full scraping pipeline with mocked browser."""
        scraper = RedditScraper(db_manager)
        
        # Mock the entire browser interaction
        with patch('scraper.reddit_scraper.async_playwright') as mock_playwright:
            # Setup mock browser
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright.return_value.__aenter__.return_value.chromium.launch.return_value = mock_browser
            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            
            # Mock page interactions
            mock_page.goto = AsyncMock()
            mock_page.wait_for_selector = AsyncMock()
            mock_page.query_selector_all.return_value = []  # No posts found
            
            # Run scraping
            result = await scraper.scrape_mentions("OpenAI", 1, max_pages=1)
            
            # Should complete without errors
            assert isinstance(result, list)
            # Verify browser was used
            mock_playwright.assert_called_once()
            mock_page.goto.assert_called()
    
    @pytest.mark.asyncio
    async def test_post_processing_pipeline(self, db_manager, sample_mentions_list):
        """Test post-processing pipeline."""
        scraper = RedditScraper(db_manager)
        
        # Test the post-processing method directly
        enhanced_mentions = await scraper._post_process_mentions(
            sample_mentions_list, "OpenAI", 0.3, None
        )
        
        assert isinstance(enhanced_mentions, list)
        assert len(enhanced_mentions) <= len(sample_mentions_list)  # May filter some out
        
        # Check that mentions have required fields
        for mention in enhanced_mentions:
            assert 'relevance_score' in mention
            assert 'sentiment_score' in mention
            assert mention['relevance_score'] >= 0.3  # Quality threshold applied
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, db_manager):
        """Test error recovery mechanisms."""
        scraper = RedditScraper(db_manager)
        
        # Test with invalid search term
        result = await scraper.scrape_mentions("", 1)
        assert result == []  # Should return empty list, not crash
        
        # Test with very long search term
        long_term = "a" * 1000
        result = await scraper.scrape_mentions(long_term, 1)
        assert isinstance(result, list)  # Should handle gracefully


class TestScrapingConfiguration:
    """Test scraping configuration and settings."""
    
    @pytest.fixture
    def scraper(self, db_manager):
        """Create scraper instance for testing."""
        return RedditScraper(db_manager)
    
    def test_proxy_configuration(self, db_manager):
        """Test proxy configuration."""
        scraper = RedditScraper(db_manager)
        
        # Test proxy list loading
        assert isinstance(scraper.proxy_list, list)
        
        # Test proxy rotation
        if scraper.proxy_list:
            proxy1 = scraper._get_next_proxy()
            proxy2 = scraper._get_next_proxy()
            
            assert proxy1 is not None
            assert proxy2 is not None
            # Should rotate through proxies
            if len(scraper.proxy_list) > 1:
                assert proxy1 != proxy2 or len(scraper.proxy_list) == 1
    
    def test_user_agent_rotation(self, db_manager):
        """Test user agent rotation."""
        scraper = RedditScraper(db_manager)
        
        assert len(scraper.user_agents) > 0
        assert all(isinstance(ua, str) for ua in scraper.user_agents)
        assert all(len(ua) > 0 for ua in scraper.user_agents)
    
    def test_search_patterns_configuration(self, db_manager):
        """Test search patterns configuration."""
        scraper = RedditScraper(db_manager)
        
        assert 'primary' in scraper.search_patterns
        assert 'secondary' in scraper.search_patterns
        assert 'fallback' in scraper.search_patterns
        
        # Verify patterns contain placeholders
        for pattern_type, patterns in scraper.search_patterns.items():
            assert isinstance(patterns, list)
            assert len(patterns) > 0
            for pattern in patterns:
                assert '{query}' in pattern
    
    def test_subreddit_categories(self, db_manager):
        """Test subreddit categories configuration."""
        scraper = RedditScraper(db_manager)
        
        assert isinstance(scraper.subreddit_categories, dict)
        assert len(scraper.subreddit_categories) > 0
        
        for category, subreddits in scraper.subreddit_categories.items():
            assert isinstance(subreddits, list)
            assert len(subreddits) > 0
            assert all(isinstance(sub, str) for sub in subreddits)

    @pytest.mark.unit
    def test_build_search_patterns(self, scraper):
        """Test search pattern building."""
        patterns = scraper._build_search_patterns("OpenAI")
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Should include exact match and variations
        assert any("OpenAI" in pattern for pattern in patterns)
    
    @pytest.mark.unit
    def test_filter_by_quality(self, scraper):
        """Test quality filtering."""
        mentions = [
            {'relevance_score': 0.8, 'score': 100, 'title': 'High quality post'},
            {'relevance_score': 0.2, 'score': 1, 'title': 'Low quality post'},
            {'relevance_score': 0.6, 'score': 50, 'title': 'Medium quality post'}
        ]
        
        # Filter with threshold 0.5
        filtered = scraper._filter_by_quality(mentions, threshold=0.5)
        
        assert len(filtered) == 2  # Should keep high and medium quality
        assert all(m['relevance_score'] >= 0.5 for m in filtered)
    
    @pytest.mark.unit
    def test_deduplicate_mentions(self, scraper):
        """Test mention deduplication."""
        mentions = [
            {'reddit_id': 'post1', 'title': 'First post'},
            {'reddit_id': 'post2', 'title': 'Second post'},
            {'reddit_id': 'post1', 'title': 'Duplicate post'},  # Duplicate
            {'reddit_id': 'post3', 'title': 'Third post'}
        ]
        
        deduplicated = scraper._deduplicate_mentions(mentions)
        
        assert len(deduplicated) == 3  # Should remove one duplicate
        reddit_ids = [m['reddit_id'] for m in deduplicated]
        assert len(set(reddit_ids)) == 3  # All unique IDs
    
    @pytest.mark.unit
    def test_validate_mention_data(self, scraper):
        """Test mention data validation."""
        # Valid mention
        valid_mention = {
            'reddit_id': 'valid123',
            'title': 'Valid title',
            'content': 'Valid content',
            'author': 'valid_user',
            'subreddit': 'technology',
            'score': 42,
            'num_comments': 15,
            'created_utc': datetime.utcnow(),
            'sentiment_score': 0.5,
            'relevance_score': 0.8
        }
        
        assert scraper._validate_mention_data(valid_mention) == True
        
        # Invalid mention - missing required fields
        invalid_mention = {
            'title': 'Missing reddit_id'
        }
        
        assert scraper._validate_mention_data(invalid_mention) == False
        
        # Invalid mention - wrong data types
        invalid_types = {
            'reddit_id': 'valid123',
            'title': 'Valid title',
            'score': 'not_a_number',  # Should be int
            'sentiment_score': 'not_a_float'  # Should be float
        }
        
        assert scraper._validate_mention_data(invalid_types) == False
    
    @pytest.mark.unit
    def test_error_handling(self, scraper):
        """Test error handling in scraper."""
        # Test with invalid search term
        with pytest.raises(ValueError):
            scraper._validate_search_parameters("", max_pages=5)
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            scraper._validate_search_parameters("OpenAI", max_pages=0)
        
        with pytest.raises(ValueError):
            scraper._validate_search_parameters("OpenAI", max_pages=101)  # Too many pages
    
    @pytest.mark.unit
    def test_rate_limiting_configuration(self, scraper):
        """Test rate limiting configuration."""
        # Check that throttler is properly configured
        assert hasattr(scraper, 'throttler')
        assert scraper.throttler is not None
        
        # Check throttler properties (asyncio-throttle library)
        assert hasattr(scraper.throttler, 'rate_limit')
        assert hasattr(scraper.throttler, 'period') 
        assert scraper.throttler.rate_limit > 0
        assert scraper.throttler.period > 0 