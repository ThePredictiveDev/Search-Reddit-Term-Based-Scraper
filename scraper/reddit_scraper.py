"""
Advanced Reddit scraper with comprehensive error handling, retry mechanisms, and edge case management.
"""
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Any, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
import random
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from textblob import TextBlob

from database.models import DatabaseManager

import requests

class ScrapingError(Exception):
    """Custom exception for scraping errors."""
    pass

class RateLimitError(ScrapingError):
    """Exception for rate limiting issues."""
    pass

class ContentNotFoundError(ScrapingError):
    """Exception when expected content is not found."""
    pass

class ScrapingStatus(Enum):
    """Enumeration for scraping status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"

@dataclass
class ScrapingMetrics:
    """Metrics for scraping performance."""
    start_time: datetime
    end_time: Optional[datetime] = None
    pages_scraped: int = 0
    mentions_found: int = 0
    errors_encountered: int = 0
    retries_performed: int = 0
    rate_limit_hits: int = 0
    cache_hits: int = 0
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        total_attempts = self.pages_scraped + self.errors_encountered
        if total_attempts == 0:
            return 0.0
        return self.pages_scraped / total_attempts

class RedditAPIClient:
    """Reddit API client for more reliable data collection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.APIClient')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RedditMentionTracker/1.0 (Educational Purpose)'
        })
        # Reddit allows up to 60 requests per minute for unauthenticated users
        self.rate_limit_delay = 1.1  # Slightly over 1 second between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting to respect Reddit's limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def search_reddit_comprehensive(self, query: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        """
        Comprehensive Reddit search using multiple strategies to find ALL mentions.
        Target: 1000 mentions from last 7 days only with comprehensive coverage.
        """
        try:
            self.logger.info(f"[SEARCH] Starting COMPREHENSIVE search for '{query}' targeting {max_results} mentions (LAST 7 DAYS ONLY)")
            
            all_mentions = []
            seen_ids = set()
            
            # Multiple search configurations for maximum coverage
            search_configs = [
                ('relevance', 'week', 200),
                ('top', 'week', 200), 
                ('hot', 'week', 200),
                ('new', 'week', 200),
                ('relevance', 'week', 100),  # Different limit for different results
                ('new', 'week', 100)
            ]
            
            self.logger.info(f"[EXEC] Executing {len(search_configs)} search configurations to maximize coverage...")
            
            # Execute each search configuration
            for i, (sort_type, time_filter, limit) in enumerate(search_configs, 1):
                self.logger.info(f"   [{i}/{len(search_configs)}] Searching with sort={sort_type}, time={time_filter}, limit={limit}")
                
                try:
                    results = await self._search_with_params(query, sort_type, time_filter, limit)
                    
                    # Add unique results and track which are actually new
                    new_results = 0
                    for post in results:
                        post_id = post.get('reddit_id')  # FIX: Use 'reddit_id' not 'id'
                        if post_id and post_id not in seen_ids:
                            seen_ids.add(post_id)
                            all_mentions.append(post)
                            new_results += 1
                    
                    self.logger.info(f"   [FOUND] Found {len(results)} posts with sort={sort_type}, time={time_filter}. New unique: {new_results}. Total unique: {len(all_mentions)}")
                    
                except Exception as e:
                    self.logger.warning(f"Search config {i} failed: {e}")
                    continue
                
                # Stop if we have enough results
                if len(all_mentions) >= max_results:
                    break
            
            # Final summary
            unique_subreddits = len(set(post.get('subreddit', 'unknown') for post in all_mentions))
            
            self.logger.info(f"""
[SUCCESS] COMPREHENSIVE SEARCH COMPLETED FOR '{query}':
[STATS] Total unique posts found: {len(all_mentions)}
[TARGET] Target was: {max_results} posts  
[PERIOD] Time period: Last 7 days only
[SUBS] Unique subreddits: {unique_subreddits}
[RESULT] SUCCESS: Found {len(all_mentions)} posts that will be processed and saved
""")
            
            return all_mentions
            
        except Exception as e:
            self.logger.error(f"Comprehensive search failed: {e}")
            return []
    
    async def _search_with_params(self, query: str, sort: str, time_filter: str, limit: int) -> List[Dict[str, Any]]:
        """Search with specific parameters."""
        try:
            self._rate_limit()
            
            url = "https://www.reddit.com/search.json"
            params = {
                'q': query,
                'sort': sort,
                't': time_filter,
                'limit': limit,
                'type': 'link'
            }
            
            self.logger.debug(f"[API] API Request: {url} with params: {params}")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            raw_posts = data.get('data', {}).get('children', [])
            
            self.logger.debug(f"[RAW] Reddit API returned {len(raw_posts)} raw posts")
            
            # Process and filter posts 
            processed_posts = []
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            
            for i, post_data in enumerate(raw_posts, 1):
                try:
                    post = post_data.get('data', {})
                    
                    # Extract key information
                    reddit_id = post.get('id', '')
                    title = post.get('title', '')
                    created_utc = post.get('created_utc', 0)
                    subreddit = post.get('subreddit', '')
                    
                    # Convert timestamp and check age
                    if created_utc:
                        post_time = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                        age_days = (datetime.now(timezone.utc) - post_time).days + \
                                  (datetime.now(timezone.utc) - post_time).seconds / 86400
                        
                        self.logger.debug(f"   [POST] Post {i}: ID={reddit_id}, Title='{title[:30]}...', Age={age_days:.1f} days, Subreddit=r/{subreddit}")
                        
                        # Only keep posts from last 7 days
                        if post_time >= seven_days_ago:
                            processed_post = {
                                'reddit_id': reddit_id,
                                'title': title,
                                'content': post.get('selftext', ''),
                                'url': f"https://reddit.com{post.get('permalink', '')}",
                                'author': post.get('author', 'unknown'),
                                'created_utc': created_utc,
                                'score': post.get('score', 0),
                                'num_comments': post.get('num_comments', 0),
                                'subreddit': subreddit,
                                'subreddit_subscribers': post.get('subreddit_subscribers', 0),
                                'upvote_ratio': post.get('upvote_ratio', 0.5),
                                'is_self': post.get('is_self', False),
                                'post_hint': post.get('post_hint', ''),
                                'domain': post.get('domain', ''),
                                'age_days': age_days
                            }
                            processed_posts.append(processed_post)
                            self.logger.debug(f"   [KEEP] Post {i} KEPT (within 7 days)")
                        else:
                            self.logger.debug(f"   [SKIP] Post {i} SKIPPED (older than 7 days)")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing post {i}: {e}")
                    continue
            
            self.logger.debug(f"[FILTER] Processed {len(processed_posts)} posts from {len(raw_posts)} raw posts (filtered to last 7 days)")
            return processed_posts
            
        except Exception as e:
            self.logger.error(f"API search failed: {e}")
            return []

class RedditScraper:
    """Simplified Reddit scraper using only API - no web scraping."""
    
    def __init__(self, db_manager):
        """Initialize Reddit scraper with enhanced configuration."""
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring (optional)
        try:
            from monitoring.system_monitor import get_system_monitor
            self.monitor = get_system_monitor()
        except ImportError:
            self.monitor = None
        
        # Initialize cache manager (optional)
        try:
            from database.cache_manager import CacheManager
            self.cache_manager = CacheManager()
        except ImportError:
            self.cache_manager = None
        
        # Circuit breaker configuration
        self.circuit_breaker = {
            'state': 'closed',  # closed, open, half_open
            'failure_count': 0,
            'failure_threshold': 5,
            'recovery_timeout': 300,  # 5 minutes
            'last_failure_time': None
        }
        
        # Initialize Reddit API client
        self.api_client = RedditAPIClient()
        
        # Session management for better performance
        self.session = None
        self._initialize_session()
        
        self.logger.info("Reddit scraper initialized with API-only configuration")
    
    def _initialize_session(self):
        """Initialize HTTP session for requests."""
        try:
            import requests
            self.session = requests.Session()
            
            # Set up session headers
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            })
            
            self.logger.info("HTTP session initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize session: {e}")
            self.session = None
    
    def _sanitize_log_message(self, message: str) -> str:
        """Sanitize log messages to avoid Unicode encoding errors."""
        try:
            # Replace problematic Unicode characters
            replacements = {
                '\u2192': '->',   # Right arrow
                '\u2190': '<-',   # Left arrow
                '\u2713': 'v',    # Check mark
                '\u2717': 'x',    # Cross mark
                '\u2022': '*',    # Bullet point
                '\u2026': '...',  # Ellipsis
            }
            
            sanitized = str(message)
            for unicode_char, replacement in replacements.items():
                sanitized = sanitized.replace(unicode_char, replacement)
            
            # Encode to ASCII safely
            sanitized = sanitized.encode('ascii', 'replace').decode('ascii')
            sanitized = sanitized.replace('?', '')  # Remove replacement chars
            
            return sanitized
        except Exception:
            return "Log message encoding error"
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests."""
        now = datetime.utcnow()
        
        if self.circuit_breaker['state'] == 'open':
            if (now - self.circuit_breaker['last_failure_time']).total_seconds() > self.circuit_breaker['recovery_timeout']:
                self.circuit_breaker['state'] = 'half_open'
                self.logger.info("Circuit breaker moved to half-open state")
                return True
            return False
        
        return True
    
    def _record_success(self):
        """Record successful operation for circuit breaker."""
        if self.circuit_breaker['state'] == 'half_open':
            self.circuit_breaker['state'] = 'closed'
            self.circuit_breaker['failure_count'] = 0
            self.logger.info("Circuit breaker closed after successful operation")
    
    def _record_failure(self):
        """Record failed operation for circuit breaker."""
        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure_time'] = datetime.utcnow()
        
        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'open'
            self.logger.warning("Circuit breaker opened due to repeated failures")
    
    async def scrape_mentions(
        self, 
        search_term: str, 
        session_id: int,
        max_pages: int = 5,
        progress_callback: Optional[Callable] = None,
        use_advanced_patterns: bool = True,
        quality_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Simplified scraping using ONLY Reddit API - no web scraping fallbacks.
        """
        # Input validation and sanitization
        search_term = self._sanitize_search_term(search_term)
        if not search_term:
            raise ValueError("Invalid search term after sanitization")
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise ScrapingError("Circuit breaker is open, scraping temporarily disabled")
        
        # Initialize metrics
        metrics = ScrapingMetrics(start_time=datetime.utcnow())
        
        # Check cache with enhanced key
        cache_key = self._generate_cache_key(search_term, max_pages, quality_threshold)
        if self.cache_manager:
            cached_mentions = self.cache_manager.get_search_results(cache_key)
            if cached_mentions:
                metrics.cache_hits = 1
                self.logger.info(f"Found {len(cached_mentions)} cached mentions for '{search_term}'")
                
                # IMPORTANT: Clean cached mentions to remove non-schema fields like quality_score
                cleaned_cached_mentions = self._clean_cached_mentions(cached_mentions)
                
                if self.monitor:
                    try:
                        # Only call if method exists
                        if hasattr(self.monitor, 'emit_cache_hit'):
                            self.monitor.emit_cache_hit(search_term, len(cleaned_cached_mentions))
                    except Exception:
                        pass  # Ignore monitoring errors
                return cleaned_cached_mentions
        
        try:
            # ONLY USE REDDIT API - no web scraping fallbacks
            if progress_callback:
                progress_callback("Searching using Reddit API...", 0.1)
            
            self.logger.info(f"Starting Reddit API search for: {search_term}")
            
            # Use comprehensive API search
            api_mentions = await self.api_client.search_reddit_comprehensive(
                search_term, 
                max_results=1000  # Get up to 1000 mentions
            )
            
            if not api_mentions:
                self.logger.warning(f"No mentions found for '{search_term}'")
                return []
            
            self.logger.info(f"Reddit API returned {len(api_mentions)} mentions")
            
            if progress_callback:
                progress_callback(f"Found {len(api_mentions)} mentions, processing...", 0.7)
            
            # Post-process mentions
            enhanced_mentions = await self._post_process_mentions(
                api_mentions, search_term, quality_threshold, metrics
            )
            
            # Record success
            self._record_success()
            metrics.end_time = datetime.utcnow()
            metrics.mentions_found = len(enhanced_mentions)
            
            # Cache results
            if self.cache_manager and enhanced_mentions:
                self.cache_manager.set_search_results(cache_key, enhanced_mentions)
            
            # Log metrics
            self._log_scraping_metrics(metrics, search_term)
            
            if progress_callback:
                progress_callback(f"[SUCCESS] Found {len(enhanced_mentions)} mentions via API", 1.0)
            
            return enhanced_mentions
            
        except Exception as e:
            self._record_failure()
            metrics.end_time = datetime.utcnow()
            metrics.errors_encountered += 1
            
            error_msg = f"Reddit API search failed for '{search_term}': {str(e)}"
            self.logger.error(error_msg)
            
            if progress_callback:
                progress_callback(f"[ERROR] {error_msg}", 1.0)
            
            raise ScrapingError(error_msg) from e
    
    def _sanitize_search_term(self, search_term: str) -> str:
        """Sanitize and validate search term."""
        if not search_term or not isinstance(search_term, str):
            return ""
        
        # Remove HTML tags first
        sanitized = re.sub(r'<[^>]*>', '', search_term.strip())
        
        # Remove dangerous characters and normalize
        sanitized = re.sub(r'["\'\\\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limit length
        sanitized = sanitized[:100]
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized
    
    def _generate_cache_key(self, search_term: str, max_pages: int, quality_threshold: float) -> str:
        """Generate enhanced cache key with parameters."""
        key_data = f"{search_term}:{max_pages}:{quality_threshold}:{datetime.utcnow().date()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _post_process_mentions(
        self, mentions: List[Dict[str, Any]], search_term: str, 
        quality_threshold: float, metrics: ScrapingMetrics
    ) -> List[Dict[str, Any]]:
        """Post-process mentions with deduplication, quality filtering, and enhancement."""
        if not mentions:
            return []
        
        self.logger.info(f"Post-processing {len(mentions)} mentions...")
        
        # Step 1: Remove duplicates
        deduplicated = self._deduplicate_mentions(mentions)
        self.logger.info(f"After deduplication: {len(deduplicated)} mentions")
        
        # Step 2: Validate mention data
        validated = self._validate_mention_data(deduplicated)
        self.logger.info(f"After validation: {len(validated)} mentions")
        
        # Step 3: Map Reddit API data to database schema - KEEP ALL VALID MENTIONS
        processed_mentions = []
        mention_quality_pairs = []
        
        for mention in validated:
            # Create a clean mention dict that only contains database fields
            clean_mention = self._map_to_database_schema(mention, search_term)
            
            # Calculate quality score for sorting only (DO NOT add to database mention)
            quality_score = self._calculate_quality_score(mention)
            
            # KEEP ALL MENTIONS - no quality threshold filtering
            processed_mentions.append(clean_mention)
            mention_quality_pairs.append((clean_mention, quality_score))
        
        self.logger.info(f"After database mapping: {len(processed_mentions)} mentions (ALL KEPT - no quality filtering)")
        
        # Step 4: Enhanced sentiment analysis if available
        if processed_mentions:
            try:
                # Use basic sentiment analysis only - remove advanced_sentiment reference
                for mention in processed_mentions:
                    if 'sentiment_score' not in mention or mention['sentiment_score'] is None:
                        mention['sentiment_score'] = self._calculate_basic_sentiment(
                            mention.get('title', ''), 
                            mention.get('content', '')
                        )
                self.logger.info(f"Enhanced {len(processed_mentions)} mentions with sentiment analysis")
            except Exception as e:
                self.logger.warning(f"Sentiment enhancement failed: {e}")
        
        # Step 5: Sort by quality score and relevance (but keep ALL mentions)
        mention_quality_pairs.sort(key=lambda x: (
            x[1],  # quality_score 
            x[0].get('relevance_score', 0), 
            x[0].get('score', 0)
        ), reverse=True)
        
        # Extract sorted mentions without quality_score
        final_mentions = [pair[0] for pair in mention_quality_pairs]
        
        self.logger.info(f"Final result: {len(final_mentions)} mentions (ALL CAPTURED, ZERO FILTERING)")
        
        return final_mentions
    
    def _log_scraping_metrics(self, metrics: ScrapingMetrics, search_term: str):
        """Log comprehensive scraping metrics."""
        self.logger.info(f"""
        Scraping completed for '{search_term}':
        - Duration: {metrics.duration:.2f}s
        - Pages scraped: {metrics.pages_scraped}
        - Mentions found: {metrics.mentions_found}
        - Success rate: {metrics.success_rate:.2%}
        - Errors: {metrics.errors_encountered}
        - Retries: {metrics.retries_performed}
        - Rate limit hits: {metrics.rate_limit_hits}
        - Cache hits: {metrics.cache_hits}
        """)
    
    def _deduplicate_mentions(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate mentions with much more lenient logic to preserve all valid mentions."""
        if not mentions:
            return []
        
        # Use a combination of fields for more intelligent deduplication
        seen_combinations = set()
        deduplicated = []
        
        for mention in mentions:
            # Create deduplication key using multiple fields (more lenient approach)
            reddit_id = mention.get('reddit_id', '')
            url = mention.get('url', '')
            title = mention.get('title', '')
            
            # Primary key: reddit_id (if available and not empty)
            if reddit_id and reddit_id.strip():
                dedup_key = f"id:{reddit_id}"
            # Secondary key: URL (if available and not empty)
            elif url and url.strip():
                dedup_key = f"url:{url}"
            # Tertiary key: Title + Subreddit combination (only for exact duplicates)
            else:
                subreddit = mention.get('subreddit', 'unknown')
                dedup_key = f"title_sub:{title[:50]}:{subreddit}"
            
            # Only remove if exact same key (much more lenient)
            if dedup_key not in seen_combinations:
                seen_combinations.add(dedup_key)
                deduplicated.append(mention)
            else:
                self.logger.debug(f"Skipping duplicate: {dedup_key[:100]}")
        
        self.logger.info(f"Deduplication: {len(mentions)} -> {len(deduplicated)} mentions (removed {len(mentions) - len(deduplicated)} exact duplicates)")
        return deduplicated
    
    def _validate_mention_data(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate mention data with EXTREMELY LENIENT criteria to keep ALL possible mentions."""
        if not mentions:
            return []
        
        validated = []
        
        for i, mention in enumerate(mentions):
            try:
                # ONLY reject mentions that are completely empty or invalid
                # Keep mentions even if they have missing fields - we'll fill them with defaults
                
                # Must have EITHER a title OR content (even minimal)
                has_title = bool(mention.get('title', '').strip())
                has_content = bool(mention.get('content', '').strip())
                has_url = bool(mention.get('url', '').strip())
                
                # Super lenient validation - keep if it has ANY content
                if has_title or has_content or has_url:
                    # Fill missing required fields with defaults to prevent database errors
                    validated_mention = {
                        'title': mention.get('title', '').strip() or f'[No Title #{i+1}]',
                        'content': mention.get('content', '').strip() or '[No content available]',
                        'url': mention.get('url', '').strip() or f'https://reddit.com/unknown_{i}',
                        'reddit_id': mention.get('reddit_id', '').strip() or f'generated_{i}_{hash(str(mention))%10000}',
                        'author': mention.get('author', '').strip() or '[Unknown Author]',
                        'subreddit': mention.get('subreddit', '').strip() or 'unknown',
                        'score': max(0, int(mention.get('score', 0))),  # Ensure non-negative
                        'created_utc': mention.get('created_utc', 0) or 0,
                        'post_type': mention.get('post_type', 'unknown'),
                        'num_comments': max(0, int(mention.get('num_comments', 0)))
                    }
                    
                    # Add optional fields if they exist
                    for optional_field in ['sentiment', 'sentiment_score', 'quality_score', 'engagement_score']:
                        if optional_field in mention:
                            validated_mention[optional_field] = mention[optional_field]
                    
                    validated.append(validated_mention)
                    
                else:
                    # Only reject completely empty mentions
                    self.logger.debug(f"Skipping completely empty mention: {mention}")
                    
            except (ValueError, TypeError) as e:
                # Even if there's an error, try to keep the mention with defaults
                self.logger.warning(f"Error validating mention, keeping with defaults: {e}")
                fallback_mention = {
                    'title': f'[Fallback Title #{i+1}]',
                    'content': '[Content extraction failed]',
                    'url': f'https://reddit.com/fallback_{i}',
                    'reddit_id': f'fallback_{i}_{hash(str(mention))%10000}',
                    'author': '[Unknown Author]',
                    'subreddit': 'unknown',
                    'score': 0,
                    'created_utc': 0,
                    'post_type': 'unknown',
                    'num_comments': 0
                }
                validated.append(fallback_mention)
        
        self.logger.info(f"Validation: {len(mentions)} raw mentions → {len(validated)} kept (EXTREMELY LENIENT - keeping everything possible)")
        return validated
    
    def _calculate_quality_score(self, mention: Dict[str, Any]) -> float:
        """Calculate quality score but DO NOT use it for filtering - just for sorting."""
        try:
            score = 0.0
            
            # Title quality (basic check)
            title_len = len(mention.get('title', ''))
            if title_len >= 10:
                score += 0.2
            elif title_len >= 5:
                score += 0.1
            
            # Content quality (basic check)
            content_len = len(mention.get('content', ''))
            if content_len >= 50:
                score += 0.2
            elif content_len >= 10:
                score += 0.1
            
            # Engagement score (Reddit karma/comments)
            reddit_score = mention.get('score', 0)
            comments = mention.get('num_comments', 0)
            
            if reddit_score >= 100:
                score += 0.3
            elif reddit_score >= 10:
                score += 0.2
            elif reddit_score >= 1:
                score += 0.1
            
            if comments >= 10:
                score += 0.2
            elif comments >= 1:
                score += 0.1
            
            # Relevance score boost
            relevance = mention.get('relevance_score', 0)
            if relevance > 0.7:
                score += 0.3
            elif relevance > 0.4:
                score += 0.2
            elif relevance > 0.1:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.5  # Default neutral score
    
    def _map_to_database_schema(self, mention: Dict[str, Any], search_term: str) -> Dict[str, Any]:
        """Map Reddit API data to database schema fields only."""
        # Start with a clean dict containing only database fields
        clean_mention = {}
        
        # Map Reddit API fields to database schema
        # Required fields
        clean_mention['reddit_id'] = mention.get('reddit_id', '')
        clean_mention['post_type'] = mention.get('post_type', 'post')
        clean_mention['subreddit'] = mention.get('subreddit', 'unknown')
        clean_mention['url'] = mention.get('url', '')
        
        # Optional text fields
        clean_mention['title'] = mention.get('title', '')
        clean_mention['content'] = mention.get('content', '')
        clean_mention['author'] = mention.get('author', '')
        
        # Numeric fields with defaults
        clean_mention['score'] = int(mention.get('score', 0))
        clean_mention['num_comments'] = int(mention.get('num_comments', 0))
        clean_mention['upvote_ratio'] = mention.get('upvote_ratio')  # Can be None
        
        # Analysis fields
        clean_mention['sentiment_score'] = self._calculate_basic_sentiment(
            clean_mention.get('title', ''), 
            clean_mention.get('content', '')
        )
        clean_mention['relevance_score'] = self._calculate_relevance_score(
            clean_mention.get('title', ''), 
            clean_mention.get('content', ''), 
            search_term
        )
        
        # Timestamps - ensure proper datetime objects
        if 'created_utc' in mention:
            created_utc = mention['created_utc']
            if isinstance(created_utc, (int, float)):
                clean_mention['created_utc'] = datetime.fromtimestamp(created_utc)
            elif isinstance(created_utc, datetime):
                clean_mention['created_utc'] = created_utc
            else:
                clean_mention['created_utc'] = datetime.utcnow()
        else:
            clean_mention['created_utc'] = datetime.utcnow()
        
        clean_mention['scraped_at'] = datetime.utcnow()
        
        return clean_mention
    
    def _clean_cached_mentions(self, cached_mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean cached mentions by removing non-database schema fields."""
        cleaned_mentions = []
        
        for mention in cached_mentions:
            # Use the same mapping function to ensure only schema fields are included
            cleaned_mention = {}
            
            # Only include database schema fields
            schema_fields = [
                'reddit_id', 'post_type', 'title', 'content', 'author', 
                'subreddit', 'url', 'score', 'num_comments', 'upvote_ratio',
                'created_utc', 'scraped_at', 'sentiment_score', 'relevance_score'
            ]
            
            for field in schema_fields:
                if field in mention:
                    cleaned_mention[field] = mention[field]
            
            # Ensure required fields exist with defaults
            if 'reddit_id' not in cleaned_mention or not cleaned_mention['reddit_id']:
                continue  # Skip mentions without reddit_id
            
            if 'scraped_at' not in cleaned_mention:
                cleaned_mention['scraped_at'] = datetime.utcnow()
            
            cleaned_mentions.append(cleaned_mention)
        
        self.logger.info(f"Cleaned {len(cached_mentions)} cached mentions -> {len(cleaned_mentions)} valid mentions")
        return cleaned_mentions
    
    def _calculate_basic_sentiment(self, title: str, content: str) -> float:
        """Calculate basic sentiment score using TextBlob."""
        try:
            text = f"{title} {content}"
            if not text.strip():
                return 0.0
            
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _calculate_relevance_score(self, title: str, content: str, search_term: str) -> float:
        """Calculate relevance score for a post."""
        title_lower = title.lower()
        content_lower = content.lower()
        search_lower = search_term.lower()
        
        score = 0.0
        
        # Exact match in title
        if search_lower in title_lower:
            score += 0.6
        
        # Exact match in content
        if search_lower in content_lower:
            score += 0.3
        
        # Word overlap
        search_words = set(search_lower.split())
        all_words = set((title_lower + ' ' + content_lower).split())
        
        overlap = len(search_words.intersection(all_words))
        if len(search_words) > 0:
            score += (overlap / len(search_words)) * 0.1
        
        return min(score, 1.0)
    
    def _is_relevant(self, mention_data: Dict[str, Any], search_term: str) -> bool:
        """Check if a mention is relevant to the search term."""
        relevance_threshold = 0.1
        return mention_data.get('relevance_score', 0.0) >= relevance_threshold 