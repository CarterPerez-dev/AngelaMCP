"""
Cache module for AngelaMCP.

Simple cache manager implementation.
"""

class CacheManager:
    """Simple cache manager for development."""
    
    def __init__(self):
        self._cache = {}
    
    async def get(self, key: str):
        """Get value from cache."""
        return self._cache.get(key)
    
    async def set(self, key: str, value, ttl: int = None):
        """Set value in cache."""
        self._cache[key] = value
    
    async def delete(self, key: str):
        """Delete key from cache."""
        self._cache.pop(key, None)
    
    async def clear(self):
        """Clear all cache."""
        self._cache.clear()
