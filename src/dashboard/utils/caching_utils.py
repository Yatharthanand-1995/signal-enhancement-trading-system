"""
Caching Utilities
Simple caching utilities for dashboard components
"""
from functools import lru_cache
from typing import Any, Dict

class CacheManager:
    """Simple cache manager"""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache = {}
    
    def get(self, key: str) -> Any:
        """Get value from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if len(self._cache) >= self.max_size:
            # Simple LRU: remove first item
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear cache"""
        self._cache.clear()