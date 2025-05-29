"""
Analytics module for processing Reddit mentions and generating insights.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import re

from textblob import TextBlob
from database.models import DatabaseManager, RedditMention

class MetricsAnalyzer:
    """Analyze Reddit mentions and generate comprehensive metrics."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def analyze_session_metrics(self, session_id: int, days: int = 7) -> Dict:
        """
        Generate comprehensive metrics for a search session.
        
        Args:
            session_id: Database session ID
            days: Number of days to analyze (default: 7)
            
        Returns:
            Dictionary containing all metrics and insights
        """
        mentions = self.db_manager.get_mentions_by_session(session_id, days)
        
        if not mentions:
            return self._empty_metrics()
        
        # Convert to DataFrame for easier analysis
        df = self._mentions_to_dataframe(mentions)
        
        # Calculate all metrics
        metrics = {
            'overview': self._calculate_overview_metrics(df, days),
            'engagement': self._calculate_engagement_metrics(df),
            'temporal': self._calculate_temporal_metrics(df, days),
            'subreddit_analysis': self._calculate_subreddit_metrics(df),
            'sentiment': self._calculate_sentiment_metrics(df),
            'content_analysis': self._calculate_content_metrics(df),
            'trending': self._calculate_trending_metrics(df),
            'author_analysis': self._calculate_author_metrics(df),
            'performance_analysis': self._calculate_performance_metrics(df),
            'quality_analysis': self._calculate_quality_analysis(df),
            'competition_analysis': self._calculate_competition_metrics(df)
        }
        
        return metrics
    
    def _mentions_to_dataframe(self, mentions: List[RedditMention]) -> pd.DataFrame:
        """Convert list of mentions to pandas DataFrame."""
        data = []
        
        for mention in mentions:
            data.append({
                'id': mention.id,
                'reddit_id': mention.reddit_id,
                'post_type': mention.post_type,
                'title': mention.title or '',
                'content': mention.content or '',
                'author': mention.author,
                'subreddit': mention.subreddit,
                'url': mention.url,
                'score': mention.score,
                'num_comments': mention.num_comments,
                'upvote_ratio': mention.upvote_ratio,
                'created_utc': mention.created_utc,
                'scraped_at': mention.scraped_at,
                'sentiment_score': mention.sentiment_score,
                'relevance_score': mention.relevance_score
            })
        
        df = pd.DataFrame(data)
        
        # Ensure datetime columns are properly typed
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        
        return df
    
    def _calculate_overview_metrics(self, df: pd.DataFrame, days: int) -> Dict:
        """Calculate high-level overview metrics."""
        total_mentions = len(df)
        
        # Filter to last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_df = df[df['created_utc'] >= cutoff_date]
        recent_mentions = len(recent_df)
        
        # Calculate daily average
        daily_average = recent_mentions / days if days > 0 else 0
        
        # Unique subreddits and authors
        unique_subreddits = df['subreddit'].nunique()
        unique_authors = df['author'].nunique()
        
        # Engagement totals
        total_score = df['score'].sum()
        total_comments = df['num_comments'].sum()
        
        return {
            'total_mentions': total_mentions,
            'recent_mentions': recent_mentions,
            'daily_average': round(daily_average, 2),
            'unique_subreddits': unique_subreddits,
            'unique_authors': unique_authors,
            'total_score': total_score,
            'total_comments': total_comments,
            'analysis_period_days': days
        }
    
    def _calculate_engagement_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate engagement-related metrics."""
        if df.empty:
            return {}
        
        # Score statistics
        score_stats = {
            'mean_score': round(df['score'].mean(), 2),
            'median_score': round(df['score'].median(), 2),
            'max_score': int(df['score'].max()),
            'min_score': int(df['score'].min()),
            'std_score': round(df['score'].std(), 2)
        }
        
        # Comment statistics
        comment_stats = {
            'mean_comments': round(df['num_comments'].mean(), 2),
            'median_comments': round(df['num_comments'].median(), 2),
            'max_comments': int(df['num_comments'].max()),
            'total_comments': int(df['num_comments'].sum())
        }
        
        # High engagement posts (top 10% by score)
        score_threshold = df['score'].quantile(0.9)
        high_engagement = df[df['score'] >= score_threshold]
        
        # Engagement distribution
        engagement_buckets = {
            'low_engagement': len(df[df['score'] < 10]),
            'medium_engagement': len(df[(df['score'] >= 10) & (df['score'] < 100)]),
            'high_engagement': len(df[df['score'] >= 100])
        }
        
        return {
            'score_stats': score_stats,
            'comment_stats': comment_stats,
            'high_engagement_count': len(high_engagement),
            'engagement_distribution': engagement_buckets,
            'top_posts': self._get_top_posts(df, 5)
        }
    
    def _calculate_temporal_metrics(self, df: pd.DataFrame, days: int) -> Dict:
        """Calculate time-based metrics and trends."""
        if df.empty:
            return {}
        
        # Daily mention counts
        df['date'] = df['created_utc'].dt.date
        daily_counts = df.groupby('date').size().reset_index(name='mentions')
        
        # Fill missing dates with 0
        date_range = pd.date_range(
            start=datetime.utcnow().date() - timedelta(days=days-1),
            end=datetime.utcnow().date(),
            freq='D'
        ).date
        
        daily_timeline = []
        for date in date_range:
            count = daily_counts[daily_counts['date'] == date]['mentions'].sum()
            daily_timeline.append({
                'date': date.isoformat(),
                'mentions': int(count)
            })
        
        # Hourly distribution
        df['hour'] = df['created_utc'].dt.hour
        hourly_dist = df.groupby('hour').size().to_dict()
        
        # Day of week distribution
        df['day_of_week'] = df['created_utc'].dt.day_name()
        dow_dist = df.groupby('day_of_week').size().to_dict()
        
        # Trend analysis
        if len(daily_counts) > 1:
            trend = self._calculate_trend(daily_counts['mentions'].tolist())
        else:
            trend = 'insufficient_data'
        
        return {
            'daily_timeline': daily_timeline,
            'hourly_distribution': hourly_dist,
            'day_of_week_distribution': dow_dist,
            'trend': trend,
            'peak_day': daily_counts.loc[daily_counts['mentions'].idxmax()]['date'].isoformat() if not daily_counts.empty else None,
            'peak_hour': df['hour'].mode().iloc[0] if not df.empty else None
        }
    
    def _calculate_subreddit_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate subreddit-specific metrics."""
        if df.empty:
            return {}
        
        # Subreddit mention counts
        subreddit_counts = df['subreddit'].value_counts().head(10)
        
        # Subreddit engagement metrics
        subreddit_engagement = df.groupby('subreddit').agg({
            'score': ['mean', 'sum', 'count'],
            'num_comments': ['mean', 'sum']
        }).round(2)
        
        # Flatten column names
        subreddit_engagement.columns = ['_'.join(col).strip() for col in subreddit_engagement.columns]
        subreddit_engagement = subreddit_engagement.reset_index()
        
        # Top subreddits by different metrics
        top_by_mentions = subreddit_counts.head(5).to_dict()
        top_by_engagement = df.groupby('subreddit')['score'].sum().nlargest(5).to_dict()
        
        # Subreddit diversity (how spread out mentions are)
        total_mentions = len(df)
        diversity_score = 1 - (subreddit_counts.iloc[0] / total_mentions) if not subreddit_counts.empty else 0
        
        return {
            'top_subreddits_by_mentions': top_by_mentions,
            'top_subreddits_by_engagement': top_by_engagement,
            'subreddit_diversity_score': round(diversity_score, 3),
            'total_subreddits': df['subreddit'].nunique(),
            'subreddit_details': subreddit_engagement.head(10).to_dict('records')
        }
    
    def _calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate sentiment analysis metrics."""
        if df.empty:
            return {}
        
        # Calculate sentiment for posts without existing sentiment scores
        sentiments = []
        
        for _, row in df.iterrows():
            if pd.isna(row['sentiment_score']):
                text = f"{row['title']} {row['content']}"
                sentiment = self._analyze_sentiment(text)
                sentiments.append(sentiment)
            else:
                sentiments.append(row['sentiment_score'])
        
        df['calculated_sentiment'] = sentiments
        
        # Sentiment distribution
        sentiment_dist = {
            'positive': len(df[df['calculated_sentiment'] > 0.1]),
            'neutral': len(df[abs(df['calculated_sentiment']) <= 0.1]),
            'negative': len(df[df['calculated_sentiment'] < -0.1])
        }
        
        # Average sentiment
        avg_sentiment = df['calculated_sentiment'].mean()
        
        # Sentiment by subreddit
        subreddit_sentiment = df.groupby('subreddit')['calculated_sentiment'].mean().round(3).to_dict()
        
        return {
            'overall_sentiment': round(avg_sentiment, 3),
            'sentiment_distribution': sentiment_dist,
            'sentiment_by_subreddit': subreddit_sentiment,
            'most_positive_subreddit': max(subreddit_sentiment.items(), key=lambda x: x[1])[0] if subreddit_sentiment else None,
            'most_negative_subreddit': min(subreddit_sentiment.items(), key=lambda x: x[1])[0] if subreddit_sentiment else None
        }
    
    def _calculate_content_metrics(self, df: pd.DataFrame) -> Dict:
        """Analyze content characteristics."""
        if df.empty:
            return {}
        
        # Text length statistics
        df['title_length'] = df['title'].str.len()
        df['content_length'] = df['content'].str.len()
        
        # Common words analysis
        all_text = ' '.join(df['title'].fillna('') + ' ' + df['content'].fillna(''))
        common_words = self._extract_common_words(all_text)
        
        # Post type distribution
        post_type_dist = df['post_type'].value_counts().to_dict()
        
        # Content quality indicators
        has_content = len(df[df['content_length'] > 50])  # Posts with substantial content
        has_title = len(df[df['title_length'] > 10])  # Posts with meaningful titles
        
        return {
            'avg_title_length': round(df['title_length'].mean(), 1),
            'avg_content_length': round(df['content_length'].mean(), 1),
            'common_words': common_words,
            'post_type_distribution': post_type_dist,
            'posts_with_content': has_content,
            'posts_with_titles': has_title,
            'content_quality_score': round((has_content + has_title) / (2 * len(df)), 3)
        }
    
    def _calculate_trending_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate trending and momentum metrics."""
        if df.empty or len(df) < 2:
            return {}
        
        # Sort by creation time
        df_sorted = df.sort_values('created_utc')
        
        # Recent vs older mentions (last 24 hours vs previous period)
        now = datetime.utcnow()
        last_24h = df_sorted[df_sorted['created_utc'] >= now - timedelta(hours=24)]
        previous_24h = df_sorted[
            (df_sorted['created_utc'] >= now - timedelta(hours=48)) &
            (df_sorted['created_utc'] < now - timedelta(hours=24))
        ]
        
        # Momentum calculation
        recent_count = len(last_24h)
        previous_count = len(previous_24h)
        
        if previous_count > 0:
            momentum = (recent_count - previous_count) / previous_count
        else:
            momentum = 1.0 if recent_count > 0 else 0.0
        
        # Velocity (mentions per hour in last 24h)
        velocity = recent_count / 24 if recent_count > 0 else 0
        
        # Peak detection
        df_sorted['hour_bucket'] = df_sorted['created_utc'].dt.floor('H')
        hourly_counts = df_sorted.groupby('hour_bucket').size()
        peak_hour = hourly_counts.idxmax() if not hourly_counts.empty else None
        
        return {
            'momentum_24h': round(momentum, 3),
            'velocity_per_hour': round(velocity, 2),
            'recent_mentions_24h': recent_count,
            'previous_mentions_24h': previous_count,
            'peak_hour': peak_hour.isoformat() if peak_hour else None,
            'is_trending': momentum > 0.2 and recent_count > 5
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _extract_common_words(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Extract most common words from text."""
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'reddit', 'post', 'comment'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count and return top words
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)
    
    def _get_top_posts(self, df: pd.DataFrame, n: int = 5) -> List[Dict]:
        """Get top posts by engagement score."""
        top_posts = df.nlargest(n, 'score')[['title', 'subreddit', 'score', 'num_comments', 'url']]
        return top_posts.to_dict('records')
    
    def _calculate_trend(self, values: List[int]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.5:
            return 'increasing'
        elif slope < -0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure when no data is available."""
        return {
            'overview': {
                'total_mentions': 0,
                'recent_mentions': 0,
                'daily_average': 0,
                'unique_subreddits': 0,
                'unique_authors': 0,
                'total_score': 0,
                'total_comments': 0,
                'analysis_period_days': 0
            },
            'engagement': {},
            'temporal': {},
            'subreddit_analysis': {},
            'sentiment': {},
            'content_analysis': {},
            'trending': {},
            'author_analysis': {},
            'performance_analysis': {},
            'quality_analysis': {},
            'competition_analysis': {}
        }
    
    def _calculate_author_metrics(self, df: pd.DataFrame) -> Dict:
        """Analyze author patterns and behavior."""
        if df.empty:
            return {}
        
        # Author activity metrics
        author_counts = df['author'].value_counts()
        author_engagement = df.groupby('author').agg({
            'score': ['mean', 'sum', 'count'],
            'num_comments': ['mean', 'sum']
        }).round(2)
        
        # Flatten column names
        author_engagement.columns = ['_'.join(col).strip() for col in author_engagement.columns]
        author_engagement = author_engagement.reset_index()
        
        # Top authors by different metrics
        top_authors_by_posts = author_counts.head(5).to_dict()
        top_authors_by_score = df.groupby('author')['score'].sum().nlargest(5).to_dict()
        
        # Author diversity
        total_authors = df['author'].nunique()
        single_post_authors = len(author_counts[author_counts == 1])
        multi_post_authors = total_authors - single_post_authors
        
        return {
            'total_unique_authors': total_authors,
            'single_post_authors': single_post_authors,
            'multi_post_authors': multi_post_authors,
            'author_diversity_ratio': round(single_post_authors / total_authors, 3) if total_authors > 0 else 0,
            'top_authors_by_posts': top_authors_by_posts,
            'top_authors_by_engagement': top_authors_by_score,
            'avg_posts_per_author': round(len(df) / total_authors, 2) if total_authors > 0 else 0,
            'most_active_author': author_counts.index[0] if not author_counts.empty else None,
            'most_active_author_posts': int(author_counts.iloc[0]) if not author_counts.empty else 0
        }
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance benchmarks and comparisons."""
        if df.empty:
            return {}
        
        # Performance quartiles
        score_quartiles = df['score'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        comment_quartiles = df['num_comments'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        
        # High performers (top 10%)
        score_threshold_90 = df['score'].quantile(0.9)
        comment_threshold_90 = df['num_comments'].quantile(0.9)
        
        high_score_posts = df[df['score'] >= score_threshold_90]
        high_comment_posts = df[df['num_comments'] >= comment_threshold_90]
        viral_posts = df[(df['score'] >= score_threshold_90) & (df['num_comments'] >= comment_threshold_90)]
        
        # Performance by subreddit
        subreddit_performance = df.groupby('subreddit').agg({
            'score': ['mean', 'median', 'max'],
            'num_comments': ['mean', 'median', 'max']
        }).round(2)
        
        # Engagement rate (comments per score)
        df['engagement_rate'] = df['num_comments'] / (df['score'] + 1)  # +1 to avoid division by zero
        avg_engagement_rate = df['engagement_rate'].mean()
        
        return {
            'score_quartiles': {f'q{int(k*100)}': int(v) for k, v in score_quartiles.items()},
            'comment_quartiles': {f'q{int(k*100)}': int(v) for k, v in comment_quartiles.items()},
            'high_performers': {
                'high_score_posts': len(high_score_posts),
                'high_comment_posts': len(high_comment_posts),
                'viral_posts': len(viral_posts),
                'viral_rate': round(len(viral_posts) / len(df), 3)
            },
            'engagement_rate': round(avg_engagement_rate, 3),
            'best_performing_subreddit': subreddit_performance.xs('mean', axis=1, level=1)['score'].idxmax() if not subreddit_performance.empty else None,
            'performance_distribution': {
                'low_performers': len(df[df['score'] < score_quartiles[0.25]]),
                'average_performers': len(df[(df['score'] >= score_quartiles[0.25]) & (df['score'] < score_quartiles[0.75])]),
                'high_performers': len(df[df['score'] >= score_quartiles[0.75]])
            }
        }
    
    def _calculate_quality_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze content quality indicators."""
        if df.empty:
            return {}
        
        # Title quality metrics
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['has_question_mark'] = df['title'].str.contains(r'\?', na=False)
        df['has_numbers'] = df['title'].str.contains(r'\d+', na=False)
        df['has_caps'] = df['title'].str.contains(r'[A-Z]{2,}', na=False)
        
        # Content quality metrics
        df['content_length'] = df['content'].str.len()
        df['content_word_count'] = df['content'].str.split().str.len()
        df['has_links'] = df['content'].str.contains(r'http', na=False)
        
        # Quality scores
        df['title_quality_score'] = (
            (df['title_length'] > 10).astype(int) * 0.3 +
            (df['title_length'] < 100).astype(int) * 0.2 +
            (df['title_word_count'] > 3).astype(int) * 0.2 +
            (~df['has_caps']).astype(int) * 0.3  # Penalize excessive caps
        )
        
        df['content_quality_score'] = (
            (df['content_length'] > 50).astype(int) * 0.4 +
            (df['content_word_count'] > 10).astype(int) * 0.3 +
            (df['has_links']).astype(int) * 0.3
        )
        
        df['overall_quality_score'] = (df['title_quality_score'] + df['content_quality_score']) / 2
        
        # Quality distribution
        quality_distribution = {
            'high_quality': len(df[df['overall_quality_score'] >= 0.7]),
            'medium_quality': len(df[(df['overall_quality_score'] >= 0.4) & (df['overall_quality_score'] < 0.7)]),
            'low_quality': len(df[df['overall_quality_score'] < 0.4])
        }
        
        return {
            'avg_title_length': round(df['title_length'].mean(), 1),
            'avg_content_length': round(df['content_length'].mean(), 1),
            'avg_title_quality': round(df['title_quality_score'].mean(), 3),
            'avg_content_quality': round(df['content_quality_score'].mean(), 3),
            'avg_overall_quality': round(df['overall_quality_score'].mean(), 3),
            'quality_distribution': quality_distribution,
            'posts_with_questions': int(df['has_question_mark'].sum()),
            'posts_with_numbers': int(df['has_numbers'].sum()),
            'posts_with_links': int(df['has_links'].sum()),
            'quality_vs_engagement_correlation': round(df['overall_quality_score'].corr(df['score']), 3),
            'best_quality_posts': df.nlargest(3, 'overall_quality_score')[['title', 'subreddit', 'score', 'overall_quality_score']].to_dict('records')
        }
    
    def _calculate_competition_metrics(self, df: pd.DataFrame) -> Dict:
        """Analyze competitive landscape and market share."""
        if df.empty:
            return {}
        
        # Subreddit competition
        subreddit_counts = df['subreddit'].value_counts()
        total_mentions = len(df)
        
        # Market concentration (how concentrated mentions are in top subreddits)
        top_3_share = subreddit_counts.head(3).sum() / total_mentions
        top_5_share = subreddit_counts.head(5).sum() / total_mentions
        
        # Herfindahl-Hirschman Index for subreddit concentration
        subreddit_shares = subreddit_counts / total_mentions
        hhi = (subreddit_shares ** 2).sum()
        
        # Time-based competition (mentions over time by subreddit)
        df['date'] = df['created_utc'].dt.date
        subreddit_timeline = df.groupby(['date', 'subreddit']).size().unstack(fill_value=0)
        
        # Engagement competition
        subreddit_avg_scores = df.groupby('subreddit')['score'].mean().sort_values(ascending=False)
        
        return {
            'market_concentration': {
                'top_3_share': round(top_3_share, 3),
                'top_5_share': round(top_5_share, 3),
                'hhi_index': round(hhi, 3),
                'concentration_level': 'high' if hhi > 0.25 else 'medium' if hhi > 0.15 else 'low'
            },
            'dominant_subreddit': subreddit_counts.index[0] if not subreddit_counts.empty else None,
            'dominant_subreddit_share': round(subreddit_counts.iloc[0] / total_mentions, 3) if not subreddit_counts.empty else 0,
            'competitive_subreddits': len(subreddit_counts[subreddit_counts >= 3]),  # Subreddits with 3+ mentions
            'engagement_leaders': subreddit_avg_scores.head(3).to_dict(),
            'niche_subreddits': len(subreddit_counts[subreddit_counts == 1]),  # Subreddits with only 1 mention
            'market_diversity_score': round(1 - hhi, 3)  # Higher = more diverse
        } 