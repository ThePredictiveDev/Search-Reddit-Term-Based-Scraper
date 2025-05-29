"""
Database models for Reddit mention tracking application.
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class SearchSession(Base):
    """Track search sessions and their parameters."""
    __tablename__ = 'search_sessions'
    
    id = Column(Integer, primary_key=True)
    search_term = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    total_mentions = Column(Integer, default=0)
    status = Column(String(50), default='pending')  # pending, running, completed, failed
    
    # Relationship to mentions
    mentions = relationship("RedditMention", back_populates="session")
    
    def to_dict(self):
        """Convert session to dictionary."""
        return {
            'id': self.id,
            'search_term': self.search_term,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_mentions': self.total_mentions,
            'status': self.status
        }

class RedditMention(Base):
    """Store individual Reddit mentions (posts and comments)."""
    __tablename__ = 'reddit_mentions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('search_sessions.id'), nullable=False)
    
    # Reddit-specific fields
    reddit_id = Column(String(50), unique=True, nullable=False)
    post_type = Column(String(20), nullable=False)  # 'post' or 'comment'
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    author = Column(String(100), nullable=True)
    subreddit = Column(String(100), nullable=False)
    url = Column(Text, nullable=False)
    
    # Engagement metrics
    score = Column(Integer, default=0)
    num_comments = Column(Integer, default=0)
    upvote_ratio = Column(Float, nullable=True)
    
    # Timestamps
    created_utc = Column(DateTime, nullable=False)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis fields
    sentiment_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    
    # Relationship to session
    session = relationship("SearchSession", back_populates="mentions")
    
    def to_dict(self):
        """Convert mention to dictionary."""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'reddit_id': self.reddit_id,
            'post_type': self.post_type,
            'title': self.title,
            'content': self.content,
            'author': self.author,
            'subreddit': self.subreddit,
            'url': self.url,
            'score': self.score,
            'num_comments': self.num_comments,
            'upvote_ratio': self.upvote_ratio,
            'created_utc': self.created_utc.isoformat() if self.created_utc else None,
            'scraped_at': self.scraped_at.isoformat() if self.scraped_at else None,
            'sentiment_score': self.sentiment_score,
            'relevance_score': self.relevance_score
        }

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, database_url: str = "sqlite:///reddit_mentions.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
    
    def create_search_session(self, search_term: str) -> int:
        """Create a new search session."""
        with self.get_session() as db:
            session = SearchSession(search_term=search_term)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id
    
    def update_search_session(self, session_id: int, **kwargs):
        """Update search session with new data."""
        with self.get_session() as db:
            session = db.query(SearchSession).filter(SearchSession.id == session_id).first()
            if session:
                for key, value in kwargs.items():
                    setattr(session, key, value)
                db.commit()
    
    def add_mention(self, session_id: int, mention_data: dict) -> int:
        """Add a new Reddit mention to the database with duplicate handling."""
        # Ensure mention_data has the session_id
        mention_data = mention_data.copy()
        mention_data['session_id'] = session_id
        
        with self.get_session() as db:
            # Check if mention already exists by reddit_id
            reddit_id = mention_data.get('reddit_id')
            if reddit_id:
                existing_mention = db.query(RedditMention).filter(
                    RedditMention.reddit_id == reddit_id
                ).first()
                
                if existing_mention:
                    # Update existing mention with new session_id if needed
                    if existing_mention.session_id != session_id:
                        existing_mention.session_id = session_id
                        db.commit()
                    return existing_mention.id
            
            try:
                mention = RedditMention(**mention_data)
                db.add(mention)
                db.commit()
                db.refresh(mention)
                return mention.id
            except Exception as e:
                db.rollback()
                if "UNIQUE constraint failed" in str(e):
                    # If it's a duplicate, try to find the existing mention
                    if reddit_id:
                        existing = db.query(RedditMention).filter(
                            RedditMention.reddit_id == reddit_id
                        ).first()
                        if existing:
                            return existing.id
                    return None  # Indicate duplicate was skipped
                else:
                    raise e
    
    def get_mentions_by_session(self, session_id: int, days: int = 7):
        """Get mentions for a session within the last N days."""
        with self.get_session() as db:
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            
            return db.query(RedditMention).filter(
                RedditMention.session_id == session_id,
                RedditMention.created_utc >= cutoff_date
            ).all()
    
    def get_latest_session(self, search_term: str) -> SearchSession:
        """Get the most recent search session for a term."""
        with self.get_session() as db:
            return db.query(SearchSession).filter(
                SearchSession.search_term == search_term
            ).order_by(SearchSession.created_at.desc()).first()
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, 'engine'):
            self.engine.dispose() 