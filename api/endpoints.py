"""
FastAPI endpoints for Reddit mention tracking API.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
from datetime import datetime
import logging

from database.models import DatabaseManager
from scraper.reddit_scraper import RedditScraper
from analytics.metrics_analyzer import MetricsAnalyzer
from database.cache_manager import CacheManager

# Optional advanced sentiment analyzer
try:
    from analytics.advanced_sentiment import AdvancedSentimentAnalyzer
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    ADVANCED_SENTIMENT_AVAILABLE = False
    AdvancedSentimentAnalyzer = None

# Pydantic models for API
class SearchRequest(BaseModel):
    search_term: str = Field(..., min_length=1, max_length=100, description="Term to search for")
    max_pages: int = Field(default=5, ge=1, le=20, description="Maximum pages to scrape")
    use_cache: bool = Field(default=True, description="Use cached results if available")

class SearchResponse(BaseModel):
    session_id: int
    search_term: str
    status: str
    total_mentions: int
    message: str

class MetricsResponse(BaseModel):
    session_id: int
    metrics: Dict[str, Any]
    cached: bool

class MentionModel(BaseModel):
    reddit_id: str
    title: Optional[str]
    content: Optional[str]
    author: Optional[str]
    subreddit: str
    url: str
    score: int
    num_comments: int
    created_utc: datetime
    sentiment_score: Optional[float]
    relevance_score: Optional[float]

class SentimentRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100, description="Texts to analyze")

class HealthResponse(BaseModel):
    status: str
    database: str
    cache: str
    sentiment_analyzer: str
    timestamp: datetime

# Initialize FastAPI app
app = FastAPI(
    title="Reddit Mention Tracker API",
    description="API for tracking and analyzing Reddit mentions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
cache_manager = CacheManager()
scraper = RedditScraper(db_manager)
analyzer = MetricsAnalyzer(db_manager)
sentiment_analyzer = AdvancedSentimentAnalyzer() if ADVANCED_SENTIMENT_AVAILABLE else None

logger = logging.getLogger(__name__)

# Dependency injection
def get_db_manager():
    return db_manager

def get_cache_manager():
    return cache_manager

def get_scraper():
    return scraper

def get_analyzer():
    return analyzer

def get_sentiment_analyzer():
    return sentiment_analyzer

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        database="connected" if db_manager else "disconnected",
        cache="connected" if cache_manager.is_available() else "disconnected",
        sentiment_analyzer="available" if sentiment_analyzer and sentiment_analyzer.is_available() else "unavailable",
        timestamp=datetime.utcnow()
    )

@app.post("/search", response_model=SearchResponse)
async def start_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager),
    cache: CacheManager = Depends(get_cache_manager),
    scraper: RedditScraper = Depends(get_scraper)
):
    """Start a new Reddit mention search."""
    try:
        # Check cache first if requested
        if request.use_cache:
            cached_results = cache.get_search_results(request.search_term)
            if cached_results:
                # Create session for cached results
                session_id = db.create_search_session(request.search_term)
                
                # Save cached results to database
                saved_count = 0
                for mention_data in cached_results:
                    try:
                        db.add_mention(session_id, mention_data)
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to save cached mention: {str(e)}")
                        continue
                
                db.update_search_session(
                    session_id,
                    status='completed',
                    completed_at=datetime.utcnow(),
                    total_mentions=saved_count
                )
                
                return SearchResponse(
                    session_id=session_id,
                    search_term=request.search_term,
                    status="completed",
                    total_mentions=saved_count,
                    message=f"Found {saved_count} cached mentions"
                )
        
        # Create new search session
        session_id = db.create_search_session(request.search_term)
        
        # Start background scraping task
        background_tasks.add_task(
            perform_search,
            session_id,
            request.search_term,
            request.max_pages,
            scraper,
            db,
            cache
        )
        
        # Increment search count for popularity tracking
        cache.increment_search_count(request.search_term)
        
        return SearchResponse(
            session_id=session_id,
            search_term=request.search_term,
            status="started",
            total_mentions=0,
            message="Search started in background"
        )
        
    except Exception as e:
        logger.error(f"Error starting search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def perform_search(
    session_id: int,
    search_term: str,
    max_pages: int,
    scraper: RedditScraper,
    db: DatabaseManager,
    cache: CacheManager
):
    """Background task to perform the actual search."""
    try:
        # Update session status
        db.update_search_session(session_id, status='running')
        
        # Scrape mentions
        mentions = await scraper.scrape_mentions(
            search_term=search_term,
            session_id=session_id,
            max_pages=max_pages
        )
        
        # Save mentions to database
        saved_count = 0
        for mention_data in mentions:
            try:
                db.add_mention(session_id, mention_data)
                saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to save mention: {str(e)}")
                continue
        
        # Cache results for future use
        cache.set_search_results(search_term, mentions)
        
        # Update session with results
        db.update_search_session(
            session_id,
            status='completed',
            completed_at=datetime.utcnow(),
            total_mentions=saved_count
        )
        
        logger.info(f"Search completed for '{search_term}': {saved_count} mentions")
        
    except Exception as e:
        logger.error(f"Error in background search: {str(e)}")
        db.update_search_session(session_id, status='failed')

@app.get("/search/{session_id}/status")
async def get_search_status(
    session_id: int,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get the status of a search session."""
    try:
        with db.get_session() as db_session:
            session = db_session.query(db.SearchSession).filter(
                db.SearchSession.id == session_id
            ).first()
            
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return {
                "session_id": session.id,
                "search_term": session.search_term,
                "status": session.status,
                "total_mentions": session.total_mentions,
                "created_at": session.created_at,
                "completed_at": session.completed_at
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting search status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{session_id}/metrics", response_model=MetricsResponse)
async def get_session_metrics(
    session_id: int,
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze"),
    use_cache: bool = Query(default=True, description="Use cached metrics if available"),
    db: DatabaseManager = Depends(get_db_manager),
    cache: CacheManager = Depends(get_cache_manager),
    analyzer: MetricsAnalyzer = Depends(get_analyzer)
):
    """Get analytics metrics for a search session."""
    try:
        # Check cache first
        cached_metrics = None
        if use_cache:
            cached_metrics = cache.get_metrics(session_id)
        
        if cached_metrics:
            return MetricsResponse(
                session_id=session_id,
                metrics=cached_metrics,
                cached=True
            )
        
        # Generate fresh metrics
        metrics = analyzer.analyze_session_metrics(session_id, days)
        
        # Cache the results
        cache.set_metrics(session_id, metrics)
        
        return MetricsResponse(
            session_id=session_id,
            metrics=metrics,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Error getting session metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{session_id}/mentions", response_model=List[MentionModel])
async def get_session_mentions(
    session_id: int,
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of mentions to return"),
    offset: int = Query(default=0, ge=0, description="Number of mentions to skip"),
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get mentions for a search session."""
    try:
        with db.get_session() as db_session:
            mentions = db_session.query(db.RedditMention).filter(
                db.RedditMention.session_id == session_id
            ).offset(offset).limit(limit).all()
            
            return [
                MentionModel(
                    reddit_id=mention.reddit_id,
                    title=mention.title,
                    content=mention.content,
                    author=mention.author,
                    subreddit=mention.subreddit,
                    url=mention.url,
                    score=mention.score,
                    num_comments=mention.num_comments,
                    created_utc=mention.created_utc,
                    sentiment_score=mention.sentiment_score,
                    relevance_score=mention.relevance_score
                )
                for mention in mentions
            ]
            
    except Exception as e:
        logger.error(f"Error getting session mentions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending")
async def get_trending_searches(
    limit: int = Query(default=10, ge=1, le=50, description="Number of trending terms to return"),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get trending search terms."""
    try:
        popular_searches = cache.get_popular_searches(limit)
        return {
            "trending_searches": popular_searches,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting trending searches: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/analyze")
async def analyze_sentiment(
    request: SentimentRequest,
    sentiment_analyzer: AdvancedSentimentAnalyzer = Depends(get_sentiment_analyzer) #type: ignore
):
    """Analyze sentiment for provided texts."""
    try:
        texts = request.texts
        
        if not sentiment_analyzer or not sentiment_analyzer.is_available():
            # Fallback to basic sentiment analysis
            from textblob import TextBlob
            results = []
            for text in texts:
                blob = TextBlob(text)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity,
                    'method': 'textblob'
                })
            return {"results": results}
        
        # Use advanced sentiment analysis
        results = sentiment_analyzer.analyze_batch(texts)
        
        formatted_results = []
        for text, result in zip(texts, results):
            formatted_results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'composite_sentiment': result.get('composite_sentiment', 0.0),
                'transformer_sentiment': result.get('transformer_sentiment', 0.0),
                'transformer_confidence': result.get('transformer_confidence', 0.0),
                'primary_emotion': result.get('primary_emotion', 'neutral'),
                'emotion_confidence': result.get('emotion_confidence', 0.0),
                'method': 'advanced'
            })
        
        return {"results": formatted_results}
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get cache statistics."""
    try:
        stats = cache.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache/clear")
async def clear_cache(
    pattern: Optional[str] = Query(default=None, description="Pattern to match for selective clearing"),
    cache: CacheManager = Depends(get_cache_manager)
):
    """Clear cache entries."""
    try:
        success = cache.clear_cache(pattern)
        return {
            "success": success,
            "message": f"Cache cleared{' for pattern: ' + pattern if pattern else ' completely'}",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def create_api_app(tracker_instance=None):
    """Create and configure the FastAPI application."""
    return app

def create_app():
    """Create and return the FastAPI application instance for testing."""
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 