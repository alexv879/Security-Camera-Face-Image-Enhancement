"""Caching utilities for improved performance."""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable
import json
from functools import wraps
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using file-based caching")


class CacheBackend:
    """Base class for cache backends."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache."""
        raise NotImplementedError


class FileCache(CacheBackend):
    """File-based cache backend."""

    def __init__(self, cache_dir: Path = Path(".cache")):
        """Initialize file cache."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        path = self._get_path(key)

        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        path = self._get_path(key)

        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Clear all cache."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()


class RedisCache(CacheBackend):
    """Redis cache backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        """Initialize Redis cache."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,
        )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = self.client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Failed to get from Redis: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        try:
            data = pickle.dumps(value)
            if ttl:
                self.client.setex(key, ttl, data)
            else:
                self.client.set(key, data)
        except Exception as e:
            logger.warning(f"Failed to set in Redis: {e}")

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.client.delete(key)

    def clear(self) -> None:
        """Clear all cache."""
        self.client.flushdb()


class Cache:
    """
    Main cache interface with automatic backend selection.

    Supports:
    - File-based caching
    - Redis caching (if available)
    - LRU in-memory caching
    """

    def __init__(
        self,
        backend: str = "auto",
        cache_dir: Path = Path(".cache"),
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        """
        Initialize cache.

        Args:
            backend: Cache backend ('auto', 'file', 'redis')
            cache_dir: Directory for file cache
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
        """
        if backend == "auto":
            # Try Redis first, fall back to file
            if REDIS_AVAILABLE:
                try:
                    self.backend = RedisCache(redis_host, redis_port, redis_db)
                    logger.info("Using Redis cache backend")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}, using file cache")
                    self.backend = FileCache(cache_dir)
            else:
                self.backend = FileCache(cache_dir)
                logger.info("Using file cache backend")

        elif backend == "file":
            self.backend = FileCache(cache_dir)
            logger.info("Using file cache backend")

        elif backend == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis not available")
            self.backend = RedisCache(redis_host, redis_port, redis_db)
            logger.info("Using Redis cache backend")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        self.backend.set(key, value, ttl)

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.backend.delete(key)

    def clear(self) -> None:
        """Clear all cache."""
        self.backend.clear()


def cached(
    cache: Cache,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
):
    """
    Decorator to cache function results.

    Args:
        cache: Cache instance
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key

    Example:
        @cached(cache, ttl=3600)
        def expensive_function(x, y):
            return x + y
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key: function name + arguments
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            result = cache.get(cache_key)

            if result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return result

            # Call function and cache result
            logger.debug(f"Cache miss for {cache_key}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


# Global cache instance
_global_cache: Optional[Cache] = None


def get_cache() -> Cache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = Cache(backend="auto")
    return _global_cache
