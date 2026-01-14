"""
File-based cache implementation for evaluation data
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class FileCache:
    """File-based cache for storing evaluation data persistently"""
    
    def __init__(self, cache_dir: str = "outputs/cache", cache_name: str = "default"):
        """
        Initialize file cache
        
        Args:
            cache_dir: Directory to store cache files
            cache_name: Name of the cache (used in filename)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_name = cache_name
        self.cache_file = self.cache_dir / f"{cache_name}_cache.json"
        self._cache_data: Dict[str, Any] = {}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache_data = json.load(f)
                logger.info(f"Loaded {len(self._cache_data)} items from {self.cache_name} cache")
        except Exception as e:
            logger.warning(f"Failed to load {self.cache_name} cache: {e}")
            self._cache_data = {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {self.cache_name} cache: {e}")
    
    def get(self, key: Union[str, int], default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        str_key = str(key)
        return self._cache_data.get(str_key, default)
    
    def set(self, key: Union[str, int], value: Any):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        str_key = str(key)
        self._cache_data[str_key] = value
        self._save_cache()
    
    def has(self, key: Union[str, int]) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        str_key = str(key)
        return str_key in self._cache_data
    
    def remove(self, key: Union[str, int]) -> bool:
        """
        Remove key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key was removed, False if key didn't exist
        """
        str_key = str(key)
        if str_key in self._cache_data:
            del self._cache_data[str_key]
            self._save_cache()
            return True
        return False
    
    def clear(self):
        """Clear all cache data"""
        self._cache_data.clear()
        self._save_cache()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self._cache_data)
    
    def keys(self):
        """Get all cache keys"""
        return self._cache_data.keys()
    
    def items(self):
        """Get all cache items"""
        return self._cache_data.items()
    
    def update(self, data: Dict[str, Any]):
        """
        Update cache with multiple key-value pairs
        
        Args:
            data: Dictionary of key-value pairs to add to cache
        """
        self._cache_data.update({str(k): v for k, v in data.items()})
        self._save_cache()
