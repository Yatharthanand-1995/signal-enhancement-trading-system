"""
API Performance Optimization Framework
Rate limiting, request throttling, response compression, and connection pooling.
"""
import time
import asyncio
import aiohttp
import requests
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import json
import gzip
import zlib
from collections import defaultdict, deque
import hashlib
import functools

# Use fallback imports for better compatibility
try:
    from .caching import cache_manager, cached
except ImportError:
    # Fallback cache manager
    class FallbackCacheManager:
        def get(self, *args, **kwargs):
            return None
        def set(self, *args, **kwargs):
            pass
    cache_manager = FallbackCacheManager()
    
    def cached(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from .error_handling import NetworkError, ErrorSeverity, handle_errors
    from .logging_setup import get_logger, perf_logger
except ImportError:
    # Fallback error handling
    import logging
    
    class NetworkError(Exception):
        def __init__(self, message, severity=None, original_error=None):
            super().__init__(message)
            self.severity = severity
            self.original_error = original_error
    
    class ErrorSeverity:
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
    
    def handle_errors(category=None, severity=None, recovery=True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in {func.__name__}: {e}")
                    if recovery:
                        return None
                    raise
            return wrapper
        return decorator
    
    def get_logger(name):
        return logging.getLogger(name)
    
    class PerfLogger:
        def log_execution_time(self, operation, time_taken):
            logging.info(f"{operation} took {time_taken:.3f}s")
    
    perf_logger = PerfLogger()

try:
    from config.enhanced_config import enhanced_config
except ImportError:
    # Fallback configuration
    enhanced_config = None

logger = get_logger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_second: int = 10
    burst_capacity: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    enable_priority_queue: bool = True
    max_queue_size: int = 1000

@dataclass
class CompressionConfig:
    """Response compression configuration"""
    enable_gzip: bool = True
    enable_deflate: bool = True
    min_size_bytes: int = 1024
    compression_level: int = 6

class TokenBucketLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens from bucket"""
        with self.lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Get time until tokens are available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate

class SlidingWindowLimiter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire permission for request"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we can make request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def time_until_available(self) -> float:
        """Get time until next request is allowed"""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            # Time until oldest request falls outside window
            oldest_request = self.requests[0]
            return max(0.0, self.window_size - (time.time() - oldest_request))

class PriorityRequestQueue:
    """Priority-based request queue"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues = {priority: deque() for priority in RequestPriority}
        self.total_size = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def put(self, item: Any, priority: RequestPriority = RequestPriority.MEDIUM) -> bool:
        """Add item to priority queue"""
        with self.condition:
            if self.total_size >= self.max_size:
                return False
            
            self.queues[priority].append(item)
            self.total_size += 1
            self.condition.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Get highest priority item from queue"""
        with self.condition:
            end_time = time.time() + timeout if timeout else None
            
            while self.total_size == 0:
                if timeout is not None:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        return None
                    self.condition.wait(remaining)
                else:
                    self.condition.wait()
            
            # Get from highest priority queue first
            for priority in sorted(RequestPriority, key=lambda x: x.value, reverse=True):
                if self.queues[priority]:
                    item = self.queues[priority].popleft()
                    self.total_size -= 1
                    return item
            
            return None
    
    def size(self) -> int:
        """Get total queue size"""
        with self.lock:
            return self.total_size
    
    def size_by_priority(self) -> Dict[RequestPriority, int]:
        """Get queue sizes by priority"""
        with self.lock:
            return {priority: len(queue) for priority, queue in self.queues.items()}

class ResponseCompressor:
    """Response compression utility"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def compress_response(self, data: Union[str, bytes], accept_encoding: str = None) -> tuple[bytes, str]:
        """Compress response data based on accept-encoding header"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Skip compression for small responses
        if len(data) < self.config.min_size_bytes:
            return data, 'identity'
        
        accept_encoding = accept_encoding or ''
        
        # Choose compression method
        if self.config.enable_gzip and 'gzip' in accept_encoding:
            compressed = gzip.compress(data, compresslevel=self.config.compression_level)
            return compressed, 'gzip'
        
        elif self.config.enable_deflate and 'deflate' in accept_encoding:
            compressed = zlib.compress(data, level=self.config.compression_level)
            return compressed, 'deflate'
        
        return data, 'identity'
    
    def decompress_response(self, data: bytes, encoding: str) -> bytes:
        """Decompress response data"""
        if encoding == 'gzip':
            return gzip.decompress(data)
        elif encoding == 'deflate':
            return zlib.decompress(data)
        else:
            return data

class ConnectionPoolManager:
    """HTTP connection pool manager"""
    
    def __init__(self):
        self.session_pools: Dict[str, requests.Session] = {}
        self.aio_sessions: Dict[str, aiohttp.ClientSession] = {}
        self.stats = {
            'active_sessions': 0,
            'total_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'connection_reuses': 0
        }
        self.lock = threading.Lock()
    
    def get_session(self, pool_name: str = 'default') -> requests.Session:
        """Get or create HTTP session with connection pooling"""
        with self.lock:
            if pool_name not in self.session_pools:
                session = requests.Session()
                
                # Configure connection pooling
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=3,
                    pool_block=False
                )
                
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                
                # Set timeout defaults
                session.timeout = 30
                
                self.session_pools[pool_name] = session
                self.stats['active_sessions'] += 1
                
                logger.debug(f"Created connection pool: {pool_name}")
            
            return self.session_pools[pool_name]
    
    @asynccontextmanager
    async def get_aio_session(self, pool_name: str = 'default'):
        """Get or create async HTTP session"""
        if pool_name not in self.aio_sessions:
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            self.aio_sessions[pool_name] = session
            logger.debug(f"Created async connection pool: {pool_name}")
        
        try:
            yield self.aio_sessions[pool_name]
        finally:
            pass  # Keep session alive for reuse
    
    def close_all_sessions(self):
        """Close all HTTP sessions"""
        with self.lock:
            for session in self.session_pools.values():
                session.close()
            self.session_pools.clear()
            
            # Close async sessions (should be done in async context)
            for session in self.aio_sessions.values():
                asyncio.create_task(session.close())
            self.aio_sessions.clear()
            
            self.stats['active_sessions'] = 0
            logger.info("All HTTP sessions closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.lock:
            return {
                **self.stats,
                'session_pools': len(self.session_pools),
                'aio_sessions': len(self.aio_sessions)
            }

class APIOptimizer:
    """Main API performance optimization coordinator"""
    
    def __init__(self, rate_limit_config: Optional[RateLimitConfig] = None):
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.rate_limiters: Dict[str, Union[TokenBucketLimiter, SlidingWindowLimiter]] = {}
        self.request_queue = PriorityRequestQueue(self.rate_limit_config.max_queue_size)
        self.compressor = ResponseCompressor(CompressionConfig())
        self.connection_manager = ConnectionPoolManager()
        
        self.stats = {
            'requests_processed': 0,
            'requests_throttled': 0,
            'requests_queued': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_saved_bytes': 0,
            'avg_processing_time': 0.0
        }
        
        self._initialize_rate_limiters()
        self._start_request_processor()
    
    def _initialize_rate_limiters(self):
        """Initialize rate limiters"""
        config = self.rate_limit_config
        
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.rate_limiters['primary'] = TokenBucketLimiter(
                capacity=config.burst_capacity,
                refill_rate=config.requests_per_second
            )
        else:  # SLIDING_WINDOW
            self.rate_limiters['primary'] = SlidingWindowLimiter(
                window_size=60,  # 1 minute window
                max_requests=config.requests_per_minute
            )
        
        logger.info(
            f"Rate limiter initialized: {config.strategy.value}",
            component='api_optimization',
            requests_per_minute=config.requests_per_minute
        )
    
    def _start_request_processor(self):
        """Start background request processor"""
        def process_requests():
            while True:
                try:
                    request_data = self.request_queue.get(timeout=1.0)
                    if request_data:
                        self._process_queued_request(request_data)
                except Exception as e:
                    logger.error("Request processor error")
                    time.sleep(1)
        
        processor_thread = threading.Thread(target=process_requests, daemon=True)
        processor_thread.start()
        logger.debug("Request processor started")
    
    def _process_queued_request(self, request_data: Dict[str, Any]):
        """Process a queued request"""
        try:
            # Extract request details
            callback = request_data.get('callback')
            if callback:
                callback()
                
            self.stats['requests_processed'] += 1
            
        except Exception as e:
            logger.error("Queued request processing failed")
    
    @handle_errors(category=ErrorSeverity.MEDIUM, recovery=True)
    def make_request(
        self,
        url: str,
        method: str = 'GET',
        priority: RequestPriority = RequestPriority.MEDIUM,
        use_cache: bool = True,
        cache_ttl: int = 300,
        **kwargs
    ) -> Optional[requests.Response]:
        """Make optimized HTTP request with rate limiting and caching"""
        start_time = time.time()
        
        try:
            # Generate cache key if caching is enabled
            cache_key = None
            if use_cache and method.upper() == 'GET':
                cache_data = {'url': url, 'params': kwargs.get('params', {})}
                cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
                
                # Check cache first
                cached_response = cache_manager.get('api_responses', cache_key)
                if cached_response:
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {url}")
                    return cached_response
                
                self.stats['cache_misses'] += 1
            
            # Check rate limiting
            rate_limiter = self.rate_limiters.get('primary')
            if rate_limiter and not rate_limiter.acquire():
                # Queue request if rate limited
                if priority in [RequestPriority.HIGH, RequestPriority.CRITICAL]:
                    # Wait briefly for high priority requests
                    time.sleep(0.1)
                    if not rate_limiter.acquire():
                        self._queue_request(url, method, priority, cache_key, cache_ttl, **kwargs)
                        return None
                else:
                    self._queue_request(url, method, priority, cache_key, cache_ttl, **kwargs)
                    return None
            
            # Make request using connection pool
            session = self.connection_manager.get_session('api_requests')
            
            # Add default headers
            headers = kwargs.get('headers', {})
            headers.setdefault('Accept-Encoding', 'gzip, deflate')
            headers.setdefault('User-Agent', 'TradingSystem/1.0')
            kwargs['headers'] = headers
            
            # Make the request
            response = session.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Cache successful GET responses
            if use_cache and method.upper() == 'GET' and cache_key and response.status_code == 200:
                cache_manager.set('api_responses', cache_key, response, ttl_override=cache_ttl)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_request_stats(processing_time, response.status_code)
            
            perf_logger.log_execution_time('api_request', processing_time)
            
            logger.debug(
                f"Request completed: {method} {url}",
                component='api_optimization',
                status_code=response.status_code,
                processing_time=processing_time
            )
            
            return response
            
        except requests.exceptions.RequestException as e:
            self.connection_manager.stats['failed_requests'] += 1
            logger.error(f"Request failed: {method} {url}")
            raise NetworkError(f"API request failed: {e}", severity=ErrorSeverity.HIGH, original_error=e)
    
    def _queue_request(self, url: str, method: str, priority: RequestPriority, cache_key: str, cache_ttl: int, **kwargs):
        """Queue request for later processing"""
        def request_callback():
            try:
                session = self.connection_manager.get_session('queued_requests')
                response = session.request(method, url, **kwargs)
                
                if cache_key and response.status_code == 200:
                    cache_manager.set('api_responses', cache_key, response, ttl_override=cache_ttl)
                    
            except Exception as e:
                logger.error(f"Queued request failed: {method} {url}")
        
        request_data = {
            'url': url,
            'method': method,
            'callback': request_callback,
            'timestamp': time.time()
        }
        
        if self.request_queue.put(request_data, priority):
            self.stats['requests_queued'] += 1
            logger.debug(f"Request queued: {method} {url}")
        else:
            self.stats['requests_throttled'] += 1
            logger.warning(f"Request dropped (queue full): {method} {url}")
    
    def _update_request_stats(self, processing_time: float, status_code: int):
        """Update request processing statistics"""
        self.stats['requests_processed'] += 1
        
        # Update rolling average processing time
        current_avg = self.stats['avg_processing_time']
        total_requests = self.stats['requests_processed']
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.stats['avg_processing_time'] = round(new_avg, 4)
    
    @cached('api_responses', ttl=300)
    def cached_request(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make cached API request (decorator-based caching)"""
        response = self.make_request(url, **kwargs)
        if response:
            try:
                return response.json()
            except:
                return {'raw_content': response.text}
        return None
    
    def batch_requests(self, requests: List[Dict[str, Any]], max_concurrent: int = 5) -> List[Optional[requests.Response]]:
        """Execute batch of requests with concurrency control"""
        results = []
        
        def make_single_request(request_config):
            try:
                return self.make_request(**request_config)
            except Exception as e:
                logger.error(f"Batch request failed")
                return None
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(make_single_request, req) for req in requests]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error("Batch request future failed")
                    results.append(None)
        
        logger.info(f"Batch requests completed: {len(results)} results")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive API optimization statistics"""
        queue_stats = self.request_queue.size_by_priority()
        
        return {
            'optimizer': self.stats,
            'rate_limiting': {
                'strategy': self.rate_limit_config.strategy.value,
                'requests_per_minute': self.rate_limit_config.requests_per_minute,
                'queue_size': self.request_queue.size(),
                'queue_by_priority': {p.name: size for p, size in queue_stats.items()}
            },
            'connection_pool': self.connection_manager.get_stats(),
            'cache_stats': cache_manager.get_stats(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.connection_manager.close_all_sessions()
        logger.info("API optimizer cleaned up")

# Global API optimizer instance
api_optimizer = APIOptimizer()

# Convenience functions
def make_optimized_request(
    url: str,
    method: str = 'GET',
    priority: RequestPriority = RequestPriority.MEDIUM,
    **kwargs
) -> Optional[requests.Response]:
    """Make optimized API request"""
    return api_optimizer.make_request(url, method, priority, **kwargs)

def batch_api_requests(requests: List[Dict[str, Any]], max_concurrent: int = 5) -> List[Optional[requests.Response]]:
    """Execute batch of API requests"""
    return api_optimizer.batch_requests(requests, max_concurrent)

def get_api_stats() -> Dict[str, Any]:
    """Get API optimization statistics"""
    return api_optimizer.get_stats()

# Cleanup function for graceful shutdown
def cleanup_api_resources():
    """Cleanup API optimization resources"""
    api_optimizer.cleanup()