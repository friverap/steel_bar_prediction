"""
Cache Manager Service
Handles caching of predictions and data to improve API performance
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple in-memory cache manager
    In production, this would use Redis or similar
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
        # Don't start cleanup task in __init__ to avoid event loop issues
    
    def _start_cleanup_task(self):
        """Start background task to clean expired cache entries"""
        if not self._cleanup_task:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_expired())
            except RuntimeError:
                # No event loop running, cleanup will be handled manually
                pass
    
    async def _cleanup_expired(self):
        """Background task to remove expired cache entries"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, entry in self._cache.items():
                    if entry['expires_at'] <= current_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._cache[key]
                    logger.debug(f"Removed expired cache entry: {key}")
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry['expires_at'] <= datetime.utcnow():
                del self._cache[key]
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return entry['value']
            
        except Exception as e:
            logger.error(f"Error getting cache value for key {key}: {str(e)}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        expire_hours: Optional[int] = None
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            expire_hours: Expiration time in hours (default from settings)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if expire_hours is None:
                expire_hours = settings.CACHE_EXPIRY_HOURS
            
            expires_at = datetime.utcnow() + timedelta(hours=expire_hours)
            
            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.utcnow()
            }
            
            logger.debug(f"Cache set for key: {key}, expires at: {expires_at}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache value for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache entry deleted: {key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cache entries
        
        Returns:
            True if successful
        """
        try:
            self._cache.clear()
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = datetime.utcnow()
        total_entries = len(self._cache)
        expired_entries = 0
        
        for entry in self._cache.values():
            if entry['expires_at'] <= current_time:
                expired_entries += 1
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_type': 'in_memory'
        }
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache and is not expired
        
        Args:
            key: Cache key to check
            
        Returns:
            True if exists and not expired
        """
        return await self.get(key) is not None


class RedisCacheManager(CacheManager):
    """
    Redis-based cache manager for production use
    """
    
    def __init__(self):
        super().__init__()
        self.redis_client = None
        # In production, initialize Redis client here
        # import redis.asyncio as redis
        # self.redis_client = redis.from_url(settings.REDIS_URL)
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return await super().get(key)
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {str(e)}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        expire_hours: Optional[int] = None
    ) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return await super().set(key, value, expire_hours)
        
        try:
            expire_seconds = (expire_hours or settings.CACHE_EXPIRY_HOURS) * 3600
            await self.redis_client.setex(
                key, 
                expire_seconds, 
                json.dumps(value, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Redis set error: {str(e)}")
            return False
