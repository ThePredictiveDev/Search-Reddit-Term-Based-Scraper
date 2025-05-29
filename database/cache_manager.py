"""
Redis cache manager for Reddit mention tracking application.
"""
import json
import redis
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pickle

class CacheManager:
    """Manage Redis caching for metrics and search results."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()  # Test connection
            self.logger = logging.getLogger(__name__)
            self.logger.info("Redis cache connected successfully")
        except redis.ConnectionError:
            self.redis_client = None
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Redis not available, caching disabled")
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self.redis_client is not None
    
    def set_metrics(self, session_id: int, metrics: Dict, ttl: int = 3600) -> bool:
        """Cache session metrics with TTL."""
        if not self.is_available():
            return False
        
        try:
            key = f"metrics:{session_id}"
            serialized_data = pickle.dumps(metrics)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            self.logger.error(f"Error caching metrics: {str(e)}")
            return False
    
    def get_metrics(self, session_id: int) -> Optional[Dict]:
        """Retrieve cached session metrics."""
        if not self.is_available():
            return None
        
        try:
            key = f"metrics:{session_id}"
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving cached metrics: {str(e)}")
            return None
    
    def set_search_results(self, search_term: str, results: List[Dict], ttl: int = 1800) -> bool:
        """Cache search results for 30 minutes."""
        if not self.is_available():
            return False
        
        try:
            key = f"search:{search_term.lower().replace(' ', '_')}"
            serialized_data = pickle.dumps({
                'results': results,
                'timestamp': datetime.utcnow().isoformat(),
                'count': len(results)
            })
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            self.logger.error(f"Error caching search results: {str(e)}")
            return False
    
    def get_search_results(self, search_term: str, max_age_minutes: int = 30) -> Optional[List[Dict]]:
        """Retrieve cached search results if not too old."""
        if not self.is_available():
            return None
        
        try:
            key = f"search:{search_term.lower().replace(' ', '_')}"
            cached_data = self.redis_client.get(key)
            if cached_data:
                data = pickle.loads(cached_data)
                cached_time = datetime.fromisoformat(data['timestamp'])
                if datetime.utcnow() - cached_time < timedelta(minutes=max_age_minutes):
                    return data['results']
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving cached search results: {str(e)}")
            return None
    
    def set_trending_terms(self, terms: List[str], ttl: int = 3600) -> bool:
        """Cache trending search terms."""
        if not self.is_available():
            return False
        
        try:
            key = "trending:terms"
            data = {
                'terms': terms,
                'timestamp': datetime.utcnow().isoformat()
            }
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            self.logger.error(f"Error caching trending terms: {str(e)}")
            return False
    
    def get_trending_terms(self) -> Optional[List[str]]:
        """Retrieve cached trending terms."""
        if not self.is_available():
            return None
        
        try:
            key = "trending:terms"
            cached_data = self.redis_client.get(key)
            if cached_data:
                data = pickle.loads(cached_data)
                return data['terms']
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving trending terms: {str(e)}")
            return None
    
    def increment_search_count(self, search_term: str) -> int:
        """Increment search count for popularity tracking."""
        if not self.is_available():
            return 0
        
        try:
            key = f"count:{search_term.lower().replace(' ', '_')}"
            return self.redis_client.incr(key)
        except Exception as e:
            self.logger.error(f"Error incrementing search count: {str(e)}")
            return 0
    
    def get_popular_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular search terms."""
        if not self.is_available():
            return []
        
        try:
            pattern = "count:*"
            keys = self.redis_client.keys(pattern)
            
            popular = []
            for key in keys:
                count = self.redis_client.get(key)
                if count:
                    term = key.decode('utf-8').replace('count:', '').replace('_', ' ')
                    popular.append({
                        'term': term,
                        'count': int(count)
                    })
            
            return sorted(popular, key=lambda x: x['count'], reverse=True)[:limit]
        except Exception as e:
            self.logger.error(f"Error getting popular searches: {str(e)}")
            return []
    
    def clear_cache(self, pattern: str = None) -> bool:
        """Clear cache entries matching pattern."""
        if not self.is_available():
            return False
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                self.redis_client.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.is_available():
            return {'status': 'unavailable'}
        
        try:
            info = self.redis_client.info()
            return {
                'status': 'available',
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def clear_all(self) -> bool:
        """Clear all cache entries."""
        return self.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (alias for get_cache_stats)."""
        stats = self.get_cache_stats()
        # Add key count
        if self.is_available():
            try:
                key_count = len(self.redis_client.keys('*'))
                stats['keys'] = key_count
            except Exception:
                stats['keys'] = 0
        else:
            stats['keys'] = 0
        return stats 