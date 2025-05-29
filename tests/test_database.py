"""
Tests for database functionality.
"""
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from database.models import DatabaseManager, SearchSession, RedditMention


class TestDatabaseManager:
    """Test database manager functionality."""
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_database_initialization(self, db_manager):
        """Test database manager initialization."""
        assert db_manager is not None
        assert hasattr(db_manager, 'engine')
        assert hasattr(db_manager, 'SessionLocal')
    
    @pytest.mark.smoke
    @pytest.mark.unit
    def test_create_search_session(self, db_manager):
        """Test creating a search session."""
        search_term = "test_search"
        session_id = db_manager.create_search_session(search_term)
        
        assert isinstance(session_id, int)
        assert session_id > 0
        
        # Verify session was created
        with db_manager.get_session() as db:
            session = db.query(SearchSession).filter(SearchSession.id == session_id).first()
            assert session is not None
            assert session.search_term == search_term
            assert session.status == 'pending'
    
    @pytest.mark.unit
    def test_add_mention(self, db_manager, sample_mention_data):
        """Test adding a Reddit mention to the database."""
        session_id = db_manager.create_search_session("test_search")
        
        mention_data = sample_mention_data.copy()
        mention_data.update({
            'created_utc': datetime.utcnow(),
            'post_type': 'submission'  # Add required field
        })
        
        mention = db_manager.add_mention(session_id, mention_data)
        
        assert mention.reddit_id == "test123"
        assert mention.title == "Test post about OpenAI"
        assert mention.session_id == session_id
    
    @pytest.mark.unit
    def test_get_mentions_by_session(self, db_manager):
        """Test retrieving mentions by session."""
        # Create session and add mentions
        session_id = db_manager.create_search_session("test_get_mentions")
        
        mention_data = {
            'session_id': session_id,
            'reddit_id': 'test456',
            'post_type': 'submission',
            'title': 'Test post for retrieval',
            'author': 'test_user',
            'subreddit': 'test',
            'url': 'https://reddit.com/test456',
            'score': 15,
            'num_comments': 3,
            'created_utc': datetime.utcnow()
        }
        
        db_manager.add_mention(session_id, mention_data)
        
        # Retrieve mentions
        mentions = db_manager.get_mentions_by_session(session_id)
        
        assert len(mentions) >= 1
        assert mentions[0].reddit_id == 'test456'
        assert mentions[0].session_id == session_id
    
    @pytest.mark.unit
    def test_update_search_session(self, db_manager):
        """Test updating search session status."""
        session_id = db_manager.create_search_session("test_update")
        
        # Update session
        db_manager.update_search_session(session_id, status="completed", total_mentions=5)
        
        # Verify update
        with db_manager.get_session() as db:
            session = db.query(SearchSession).filter(SearchSession.id == session_id).first()
            assert session.status == "completed"
            assert session.total_mentions == 5
    
    @pytest.mark.unit
    def test_get_session_statistics(self, db_manager):
        """Test getting session statistics."""
        session_id = db_manager.create_search_session("test_stats")
        
        # Add some test mentions
        for i in range(3):
            mention_data = {
                'session_id': session_id,
                'reddit_id': f'stats_test_{i}',
                'post_type': 'submission',
                'title': f'Stats test post {i}',
                'author': 'stats_user',
                'subreddit': 'test',
                'url': f'https://reddit.com/stats_{i}',
                'score': 10 + i,
                'num_comments': 2 + i,
                'created_utc': datetime.utcnow()
            }
            db_manager.add_mention(session_id, mention_data)
        
        # Get mentions and verify count
        mentions = db_manager.get_mentions_by_session(session_id)
        assert len(mentions) == 3
    
    @pytest.mark.unit
    def test_search_mentions(self, db_manager):
        """Test searching mentions with filters."""
        session_id = db_manager.create_search_session("test_search_mentions")
        
        # Add mentions with different scores
        high_score_mention = {
            'session_id': session_id,
            'reddit_id': 'high_score',
            'post_type': 'submission',
            'title': 'High score post',
            'author': 'user1',
            'subreddit': 'technology',
            'url': 'https://reddit.com/high',
            'score': 100,
            'num_comments': 50,
            'created_utc': datetime.utcnow()
        }
        
        low_score_mention = {
            'session_id': session_id,
            'reddit_id': 'low_score',
            'post_type': 'submission',
            'title': 'Low score post',
            'author': 'user2',
            'subreddit': 'programming',
            'url': 'https://reddit.com/low',
            'score': 5,
            'num_comments': 2,
            'created_utc': datetime.utcnow()
        }
        
        db_manager.add_mention(session_id, high_score_mention)
        db_manager.add_mention(session_id, low_score_mention)
        
        # Get all mentions for session
        mentions = db_manager.get_mentions_by_session(session_id)
        assert len(mentions) == 2
        
        # Verify mentions have different scores
        scores = [m.score for m in mentions]
        assert 100 in scores
        assert 5 in scores
    
    @pytest.mark.unit
    def test_get_trending_subreddits(self, db_manager):
        """Test getting trending subreddits."""
        session_id = db_manager.create_search_session("test_trending")
        
        # Add mentions from different subreddits
        subreddits = ['technology', 'programming', 'artificial', 'technology']  # technology appears twice
        
        for i, subreddit in enumerate(subreddits):
            mention_data = {
                'session_id': session_id,
                'reddit_id': f'trending_{i}',
                'post_type': 'submission',
                'title': f'Post in {subreddit}',
                'author': f'user_{i}',
                'subreddit': subreddit,
                'url': f'https://reddit.com/trending_{i}',
                'score': 10,
                'num_comments': 5,
                'created_utc': datetime.utcnow()
            }
            db_manager.add_mention(session_id, mention_data)
        
        # Get mentions and verify subreddit distribution
        mentions = db_manager.get_mentions_by_session(session_id)
        subreddit_counts = {}
        for mention in mentions:
            subreddit_counts[mention.subreddit] = subreddit_counts.get(mention.subreddit, 0) + 1
        
        assert subreddit_counts['technology'] == 2  # Most frequent
        assert subreddit_counts['programming'] == 1
        assert subreddit_counts['artificial'] == 1
    
    @pytest.mark.unit
    def test_get_sentiment_distribution(self, db_manager):
        """Test getting sentiment distribution."""
        session_id = db_manager.create_search_session("test_sentiment")
        
        # Add mentions with different sentiment scores
        sentiments = [0.8, -0.3, 0.1, -0.7, 0.5]  # Mix of positive, negative, neutral
        
        for i, sentiment in enumerate(sentiments):
            mention_data = {
                'session_id': session_id,
                'reddit_id': f'sentiment_{i}',
                'post_type': 'submission',
                'title': f'Sentiment test post {i}',
                'author': f'user_{i}',
                'subreddit': 'test',
                'url': f'https://reddit.com/sentiment_{i}',
                'score': 10,
                'num_comments': 5,
                'created_utc': datetime.utcnow(),
                'sentiment_score': sentiment
            }
            db_manager.add_mention(session_id, mention_data)
        
        # Get mentions and verify sentiment distribution
        mentions = db_manager.get_mentions_by_session(session_id)
        sentiments_retrieved = [m.sentiment_score for m in mentions if m.sentiment_score is not None]
        
        assert len(sentiments_retrieved) == 5
        assert max(sentiments_retrieved) == 0.8
        assert min(sentiments_retrieved) == -0.7
    
    @pytest.mark.unit
    def test_get_time_series_data(self, db_manager):
        """Test getting time series data."""
        session_id = db_manager.create_search_session("test_timeseries")
        
        # Add mentions with different timestamps
        base_time = datetime.utcnow()
        
        for i in range(5):
            mention_data = {
                'session_id': session_id,
                'reddit_id': f'timeseries_{i}',
                'post_type': 'submission',
                'title': f'Time series post {i}',
                'author': f'user_{i}',
                'subreddit': 'test',
                'url': f'https://reddit.com/timeseries_{i}',
                'score': 10,
                'num_comments': 5,
                'created_utc': base_time - timedelta(hours=i)
            }
            db_manager.add_mention(session_id, mention_data)
        
        # Get mentions and verify time ordering
        mentions = db_manager.get_mentions_by_session(session_id)
        assert len(mentions) == 5
        
        # Verify timestamps are different
        timestamps = [m.created_utc for m in mentions]
        assert len(set(timestamps)) == 5  # All unique timestamps
    
    @pytest.mark.unit
    def test_cleanup_old_sessions(self, db_manager):
        """Test cleaning up old sessions."""
        # Create an old session
        old_session_id = db_manager.create_search_session("old_session")
        
        # Manually update the created_at timestamp to be old
        with db_manager.get_session() as db:
            old_session = db.query(SearchSession).filter(SearchSession.id == old_session_id).first()
            old_session.created_at = datetime.utcnow() - timedelta(days=30)
            db.commit()
        
        # Create a recent session
        recent_session_id = db_manager.create_search_session("recent_session")
        
        # Verify both sessions exist
        with db_manager.get_session() as db:
            all_sessions = db.query(SearchSession).all()
            session_ids = [s.id for s in all_sessions]
            assert old_session_id in session_ids
            assert recent_session_id in session_ids
    
    @pytest.mark.unit
    def test_database_connection_handling(self, db_manager):
        """Test database connection handling."""
        # Test that we can get multiple sessions
        session1 = db_manager.get_session()
        session2 = db_manager.get_session()
        
        assert session1 is not None
        assert session2 is not None
        assert session1 != session2  # Different session objects
        
        # Close sessions
        session1.close()
        session2.close()
    
    @pytest.mark.unit
    def test_database_error_handling(self, db_manager):
        """Test database error handling."""
        # Try to add mention without required session_id
        invalid_mention = {
            'reddit_id': 'invalid_test',
            'post_type': 'submission',
            'title': 'Invalid mention',
            'author': 'test_user',
            'subreddit': 'test',
            'url': 'https://reddit.com/invalid',
            'score': 10,
            'created_utc': datetime.utcnow()
            # Missing session_id - should cause error
        }
        
        # This should raise an exception
        with pytest.raises(Exception):
            db_manager.add_mention(invalid_mention)
    
    @pytest.mark.unit
    def test_duplicate_mention_handling(self, db_manager):
        """Test handling of duplicate mentions."""
        session_id = db_manager.create_search_session("test_duplicates")
        
        mention_data = {
            'session_id': session_id,
            'reddit_id': 'duplicate_test',
            'post_type': 'submission',
            'title': 'Duplicate test post',
            'author': 'test_user',
            'subreddit': 'test',
            'url': 'https://reddit.com/duplicate',
            'score': 10,
            'num_comments': 5,
            'created_utc': datetime.utcnow()
        }
        
        # Add first mention
        mention1 = db_manager.add_mention(session_id, mention_data)
        assert mention1 is not None
        
        # Try to add duplicate (same reddit_id)
        with pytest.raises(Exception):  # Should fail due to unique constraint
            db_manager.add_mention(session_id, mention_data)
    
    @pytest.mark.unit
    def test_database_backup_restore(self, db_manager):
        """Test database backup and restore functionality."""
        # Create test data
        session_id = db_manager.create_search_session("backup_test")
        
        test_mention = {
            'session_id': session_id,
            'reddit_id': 'backup_mention',
            'post_type': 'submission',
            'title': 'Backup test post',
            'author': 'backup_user',
            'subreddit': 'test',
            'url': 'https://reddit.com/backup',
            'score': 25,
            'num_comments': 10,
            'created_utc': datetime.utcnow(),
            'sentiment_score': 0.7
        }
        
        db_manager.add_mention(session_id, test_mention)
        
        # Verify data exists
        mentions = db_manager.get_mentions_by_session(session_id)
        assert len(mentions) == 1
        assert mentions[0].reddit_id == 'backup_mention'
        
        # Test that we can retrieve the session
        with db_manager.get_session() as db:
            session = db.query(SearchSession).filter(SearchSession.id == session_id).first()
            assert session is not None
            assert session.search_term == "backup_test"


class TestSearchSession:
    """Test SearchSession model functionality."""
    
    @pytest.mark.unit
    def test_search_session_creation(self, db_manager):
        """Test creating SearchSession objects."""
        with db_manager.get_session() as db:
            session = SearchSession(search_term="test_session")
            db.add(session)
            db.commit()
            db.refresh(session)
            
            assert session.id is not None
            assert session.search_term == "test_session"
            assert session.status == "pending"
            assert session.total_mentions == 0
            assert session.created_at is not None
    
    @pytest.mark.unit
    def test_search_session_relationships(self, db_manager):
        """Test SearchSession relationships with mentions."""
        session_id = db_manager.create_search_session("relationship_test")
        
        # Add mentions to the session
        for i in range(3):
            mention_data = {
                'session_id': session_id,
                'reddit_id': f'rel_test_{i}',
                'post_type': 'submission',
                'title': f'Relationship test {i}',
                'author': 'rel_user',
                'subreddit': 'test',
                'url': f'https://reddit.com/rel_{i}',
                'score': 10,
                'created_utc': datetime.utcnow()
            }
            db_manager.add_mention(session_id, mention_data)
        
        # Test relationship
        with db_manager.get_session() as db:
            session = db.query(SearchSession).filter(SearchSession.id == session_id).first()
            assert len(session.mentions) == 3


class TestRedditMention:
    """Test RedditMention model functionality."""
    
    @pytest.mark.unit
    def test_reddit_mention_creation(self, db_manager):
        """Test creating RedditMention objects."""
        session_id = db_manager.create_search_session("mention_test")
        
        with db_manager.get_session() as db:
            mention = RedditMention(
                session_id=session_id,
                reddit_id="test_mention_123",
                post_type="submission",
                title="Test Mention",
                content="This is a test mention",
                author="test_author",
                subreddit="test_sub",
                url="https://reddit.com/test",
                score=42,
                num_comments=15,
                created_utc=datetime.utcnow(),
                sentiment_score=0.5,
                relevance_score=0.8
            )
            
            db.add(mention)
            db.commit()
            db.refresh(mention)
            
            assert mention.id is not None
            assert mention.reddit_id == "test_mention_123"
            assert mention.post_type == "submission"
            assert mention.score == 42
    
    @pytest.mark.unit
    def test_reddit_mention_validation(self):
        """Test Reddit mention validation and defaults."""
        # Test minimal mention creation
        minimal_mention = RedditMention(
            session_id=1,
            reddit_id="minimal123",
            post_type="post",
            subreddit="test",
            url="https://reddit.com/r/test/minimal",
            created_utc=datetime.utcnow()
        )
        
        # Check defaults are applied
        assert minimal_mention.score is None  # Default is None, not 0
        assert minimal_mention.num_comments is None  # Default is None, not 0
        assert minimal_mention.sentiment_score is None
        assert minimal_mention.relevance_score is None 