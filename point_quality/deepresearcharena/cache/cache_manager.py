"""
Cache manager for coordinating multiple cache instances
"""

import logging
from typing import Dict, Any, Optional, Union
from .file_cache import FileCache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manager for multiple cache instances used by evaluators
    """
    
    def __init__(self, cache_dir: str = "outputs/cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store all cache files
        """
        self.cache_dir = cache_dir
        self._caches: Dict[str, FileCache] = {}
        
        logger.info(f"Initialized cache manager with directory: {cache_dir}")
    
    def get_cache(self, cache_name: str) -> FileCache:
        """
        Get or create a cache instance
        
        Args:
            cache_name: Name of the cache
            
        Returns:
            FileCache instance
        """
        if cache_name not in self._caches:
            self._caches[cache_name] = FileCache(
                cache_dir=self.cache_dir,
                cache_name=cache_name
            )
            logger.info(f"Created new cache: {cache_name}")
        
        return self._caches[cache_name]
    
    def clear_cache(self, cache_name: str):
        """
        Clear a specific cache
        
        Args:
            cache_name: Name of the cache to clear
        """
        if cache_name in self._caches:
            self._caches[cache_name].clear()
            logger.info(f"Cleared cache: {cache_name}")
    
    def clear_all_caches(self):
        """Clear all caches"""
        for cache_name, cache in self._caches.items():
            cache.clear()
            logger.info(f"Cleared cache: {cache_name}")
    
    def get_cache_sizes(self) -> Dict[str, int]:
        """
        Get sizes of all caches
        
        Returns:
            Dictionary mapping cache names to their sizes
        """
        return {name: cache.size() for name, cache in self._caches.items()}
    
    def list_caches(self) -> list:
        """
        List all cache names
        
        Returns:
            List of cache names
        """
        return list(self._caches.keys())
    
    # Convenience methods for common cache operations
    def get_query_dimensions(self, query_id: int, default=None):
        """Get cached query dimensions"""
        return self.get_cache("query_dimensions").get(query_id, default)
    
    def set_query_dimensions(self, query_id: int, dimensions):
        """Set cached query dimensions"""
        self.get_cache("query_dimensions").set(query_id, dimensions)
    
    def get_query_weights(self, cache_key: str, default=None):
        """Get cached query weights"""
        return self.get_cache("query_weights").get(cache_key, default)
    
    def set_query_weights(self, cache_key: str, weights):
        """Set cached query weights"""
        self.get_cache("query_weights").set(cache_key, weights)
    
    def get_query_criteria(self, cache_key: str, default=None):
        """Get cached query criteria"""
        return self.get_cache("query_criteria").get(cache_key, default)
    
    def set_query_criteria(self, cache_key: str, criteria):
        """Set cached query criteria"""
        self.get_cache("query_criteria").set(cache_key, criteria)
    
    def get_model_results(self, query_id: int, default=None):
        """Get cached model results"""
        return self.get_cache("model_results").get(query_id, default)
    
    def set_model_results(self, query_id: int, results):
        """Set cached model results"""
        self.get_cache("model_results").set(query_id, results)
    
    def get_evaluation_result(self, cache_key: str, default=None):
        """Get cached evaluation result"""
        return self.get_cache("evaluation_results").get(cache_key, default)
    
    def set_evaluation_result(self, cache_key: str, result):
        """Set cached evaluation result"""
        self.get_cache("evaluation_results").set(cache_key, result)
    
    # Generic cache methods for flexible access
    def get(self, cache_name: str, key: str, default=None):
        """
        Generic get method for any cache
        
        Args:
            cache_name: Name of the cache
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        return self.get_cache(cache_name).get(key, default)
    
    def set(self, cache_name: str, key: str, value):
        """
        Generic set method for any cache
        
        Args:
            cache_name: Name of the cache
            key: Key to store
            value: Value to store
        """
        self.get_cache(cache_name).set(key, value)