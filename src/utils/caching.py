"""
Advanced Caching Layer for Trading System
Multi-level caching with Redis, intelligent TTL management, and cache warming.
"""
import json
import time
import redis
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Union, Callable
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass
import threading
import asyncio

# Use fallback imports for better compatibility
try:
    from config.enhanced_config import enhanced_config
except ImportError:
    # Fallback configuration
    class RedisConfig:
        host = 'localhost'
        port = 6379
        db = 0
    
    class EnhancedConfig:
        redis = RedisConfig()
    
    enhanced_config = EnhancedConfig()

try:
    from .error_handling import ErrorCategory, ErrorSeverity, handle_errors
    from .logging_setup import get_logger, perf_logger
except ImportError:
    # Fallback error handling and logging
    import logging
    
    class ErrorCategory:
        DATA_PROCESSING = "DATA_PROCESSING"
    
    class ErrorSeverity:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
    
    def handle_errors(category=None, severity=None):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in {func.__name__}: {e}")
                    return None
            return wrapper
        return decorator
    
    def get_logger(name):
        return logging.getLogger(name)
    
    class PerfLogger:
        def log_execution_time(self, operation, time_taken):
            logging.info(f"{operation} took {time_taken:.3f}s")
    
    perf_logger = PerfLogger()

logger = get_logger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration for different data types"""
    ttl_seconds: int
    max_memory_items: int
    compression: bool = False
    serialization: str = 'json'  # 'json', 'pickle'
    cache_level: str = 'both'  # 'memory', 'redis', 'both'

class CacheManager:
    """Advanced multi-level caching system"""
    
    # Cache configurations for different data types
    CACHE_CONFIGS = {
        'market_data': CacheConfig(ttl_seconds=60, max_memory_items=1000, compression=True, serialization='json'),
        'trading_signals': CacheConfig(ttl_seconds=300, max_memory_items=500, serialization='json'),
        'ml_predictions': CacheConfig(ttl_seconds=600, max_memory_items=200, serialization='pickle'),
        'user_sessions': CacheConfig(ttl_seconds=3600, max_memory_items=100, serialization='json'),
        'api_responses': CacheConfig(ttl_seconds=120, max_memory_items=200, compression=True, serialization='json'),
        'database_queries': CacheConfig(ttl_seconds=900, max_memory_items=300, serialization='json'),
        'performance_metrics': CacheConfig(ttl_seconds=30, max_memory_items=50, serialization='json')
    }
    
    def __init__(self):
        self._redis_client = None
        self._memory_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'memory_size': 0,
            'redis_connected': False
        }
        self._lock = threading.Lock()
        self._initialize_redis()
        self._start_cleanup_thread()
    
    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        try:
            redis_config = enhanced_config.redis
            self._redis_client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=getattr(redis_config, 'password', None),
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self._redis_client.ping()
            self._cache_stats['redis_connected'] = True
            
            logger.info(f"Redis cache initialized successfully at {redis_config.host}:{redis_config.port}")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Falling back to memory-only caching")
            self._redis_client = None
            self._cache_stats['redis_connected'] = False
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_memory_cache()
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.debug("Cache cleanup thread started")
    
    def _cleanup_memory_cache(self):
        """Clean up expired items from memory cache"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (data, expiry, config) in self._memory_cache.items():
                if expiry and current_time > expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                logger.debug(f"Expired memory cache key: {key}")
            
            if expired_keys:
                self._update_memory_size()
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _update_memory_size(self):
        """Update memory cache size statistics"""
        try:
            total_size = sum(
                len(pickle.dumps(data)) 
                for data, _, _ in self._memory_cache.values()
            )
            self._cache_stats['memory_size'] = total_size
        except Exception:
            pass  # Don't fail on size calculation errors
    
    def _generate_cache_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate consistent cache key"""
        if kwargs:
            key_parts = [str(k) + ":" + str(v) for k, v in sorted(kwargs.items())]
            key = key + ":" + ":".join(key_parts)
        
        # Hash long keys
        if len(key) > 200:
            key = hashlib.md5(key.encode()).hexdigest()
        
        return f"trading:{namespace}:{key}"
    
    def _serialize_data(self, data: Any, serialization: str, compression: bool = False) -> bytes:
        """Serialize data for storage"""
        try:
            if serialization == 'json':
                serialized = json.dumps(data, default=str).encode('utf-8')
            else:  # pickle
                serialized = pickle.dumps(data)
            
            if compression and len(serialized) > 1024:  # Compress if > 1KB
                import gzip
                serialized = gzip.compress(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    def _deserialize_data(self, data: bytes, serialization: str, compression: bool = False) -> Any:
        """Deserialize data from storage"""
        try:
            if compression:
                import gzip
                try:
                    data = gzip.decompress(data)
                except:
                    pass  # Data might not be compressed
            
            if serialization == 'json':
                return json.loads(data.decode('utf-8'))
            else:  # pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    @handle_errors(category=ErrorCategory.DATA_PROCESSING, severity=ErrorSeverity.MEDIUM)
    def get(self, namespace: str, key: str, **kwargs) -> Optional[Any]:
        """Get item from cache with multi-level lookup"""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        config = self.CACHE_CONFIGS.get(namespace, CacheConfig(ttl_seconds=300, max_memory_items=100))
        
        start_time = time.time()
        
        try:
            # Level 1: Memory cache
            if config.cache_level in ['memory', 'both']:
                with self._lock:
                    if cache_key in self._memory_cache:
                        data, expiry, _ = self._memory_cache[cache_key]
                        if not expiry or time.time() < expiry:
                            self._cache_stats['hits'] += 1
                            perf_logger.log_execution_time('cache_get_memory', time.time() - start_time)
                            logger.debug(f"Memory cache hit: {cache_key}")
                            return data
                        else:
                            # Expired, remove from memory
                            del self._memory_cache[cache_key]
            
            # Level 2: Redis cache
            if config.cache_level in ['redis', 'both'] and self._redis_client:
                try:
                    redis_data = self._redis_client.get(cache_key)
                    if redis_data:
                        deserialized_data = self._deserialize_data(
                            redis_data, config.serialization, config.compression
                        )
                        
                        # Warm memory cache if applicable
                        if config.cache_level == 'both':
                            self._set_memory_cache(cache_key, deserialized_data, config)
                        
                        self._cache_stats['hits'] += 1
                        perf_logger.log_execution_time('cache_get_redis', time.time() - start_time)
                        logger.debug(f"Redis cache hit: {cache_key}")
                        return deserialized_data
                        
                except Exception as e:
                    logger.warning(f"Redis get failed for {cache_key}: {e}")
            
            # Cache miss
            self._cache_stats['misses'] += 1
            perf_logger.log_execution_time('cache_get_miss', time.time() - start_time)
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for {cache_key}: {e}")
            return None
    
    @handle_errors(category=ErrorCategory.DATA_PROCESSING, severity=ErrorSeverity.MEDIUM)
    def set(self, namespace: str, key: str, value: Any, ttl_override: Optional[int] = None, **kwargs):
        """Set item in cache with multi-level storage"""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        config = self.CACHE_CONFIGS.get(namespace, CacheConfig(ttl_seconds=300, max_memory_items=100))
        ttl = ttl_override or config.ttl_seconds
        
        start_time = time.time()
        
        try:
            # Level 1: Memory cache
            if config.cache_level in ['memory', 'both']:
                self._set_memory_cache(cache_key, value, config, ttl)
            
            # Level 2: Redis cache
            if config.cache_level in ['redis', 'both'] and self._redis_client:
                try:
                    serialized_data = self._serialize_data(value, config.serialization, config.compression)
                    self._redis_client.setex(cache_key, ttl, serialized_data)
                    
                except Exception as e:
                    logger.warning(f"Redis set failed for {cache_key}: {e}")
            
            self._cache_stats['sets'] += 1
            perf_logger.log_execution_time('cache_set', time.time() - start_time)
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_key}: {e}")
    
    def _set_memory_cache(self, cache_key: str, value: Any, config: CacheConfig, ttl: int = None):
        """Set item in memory cache with size limits"""
        with self._lock:
            # Calculate expiry
            expiry = time.time() + (ttl or config.ttl_seconds)
            
            # Add to cache
            self._memory_cache[cache_key] = (value, expiry, config)
            
            # Enforce memory limits
            if len(self._memory_cache) > config.max_memory_items:
                # Remove oldest entries
                sorted_items = sorted(
                    self._memory_cache.items(),
                    key=lambda x: x[1][1]  # Sort by expiry time
                )
                
                # Remove 20% of oldest entries
                remove_count = max(1, len(sorted_items) // 5)
                for i in range(remove_count):
                    key_to_remove = sorted_items[i][0]
                    del self._memory_cache[key_to_remove]
            
            self._update_memory_size()
    
    def delete(self, namespace: str, key: str, **kwargs):
        """Delete item from all cache levels"""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        
        try:
            # Remove from memory
            with self._lock:
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    self._update_memory_size()
            
            # Remove from Redis
            if self._redis_client:
                self._redis_client.delete(cache_key)
            
            self._cache_stats['deletes'] += 1
            logger.debug(f"Cache delete: {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache delete error for {cache_key}: {e}")
    
    def clear_namespace(self, namespace: str):
        """Clear all items in a namespace"""
        try:
            pattern = f"trading:{namespace}:*"
            
            # Clear memory cache
            with self._lock:
                keys_to_remove = [k for k in self._memory_cache.keys() if k.startswith(pattern)]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                self._update_memory_size()
            
            # Clear Redis cache
            if self._redis_client:
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            
            logger.info(f"Cleared cache namespace: {namespace}")
            
        except Exception as e:
            logger.error(f"Cache clear error for namespace {namespace}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        redis_info = {}
        if self._redis_client:
            try:
                info = self._redis_client.info()
                redis_info = {
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            except Exception:
                pass
        
        return {
            **self._cache_stats,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_cache_size': len(self._memory_cache),
            'redis_info': redis_info,
            'timestamp': datetime.utcnow().isoformat()
        }

# Cache decorators
def cached(namespace: str, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """Decorator for automatic function result caching"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [str(arg) for arg in args] + [f"{k}:{v}" for k, v in kwargs.items()]
                cache_key = f"{func.__name__}:{':'.join(key_parts)}"
            
            # Try to get from cache
            result = cache_manager.get(namespace, cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache the result
            cache_manager.set(namespace, cache_key, result, ttl_override=ttl)
            
            perf_logger.log_execution_time(f'cached_function_{func.__name__}', execution_time)
            return result
        
        return wrapper
    return decorator

@contextmanager
def cache_warming_context(namespace: str):
    """Context manager for cache warming operations"""
    start_time = time.time()
    logger.info(f"Starting cache warming for {namespace}")
    
    try:
        yield cache_manager
        
    except Exception as e:
        logger.error(f"Cache warming failed for {namespace}: {e}")
        raise
    
    finally:
        duration = time.time() - start_time
        logger.info(f"Cache warming completed for {namespace} in {duration:.3f}s")

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions
def get_cached(namespace: str, key: str, **kwargs) -> Optional[Any]:
    """Get item from cache"""
    return cache_manager.get(namespace, key, **kwargs)

def set_cached(namespace: str, key: str, value: Any, ttl: Optional[int] = None, **kwargs):
    """Set item in cache"""
    cache_manager.set(namespace, key, value, ttl_override=ttl, **kwargs)

def delete_cached(namespace: str, key: str, **kwargs):
    """Delete item from cache"""
    cache_manager.delete(namespace, key, **kwargs)

def clear_cache_namespace(namespace: str):
    """Clear entire namespace"""
    cache_manager.clear_namespace(namespace)

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return cache_manager.get_stats()