"""
Shared Utilities
Common utility classes and functions used across microservices
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import sqlite3
import redis
from contextlib import asynccontextmanager
import structlog
import uuid
import hashlib
import jwt
from functools import wraps
import inspect

# Logger setup
class Logger:
    """Structured logger for microservices"""
    
    def __init__(self, service_name: str, level: str = "INFO"):
        self.service_name = service_name
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup Python logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, level.upper())
        )
        
        self.logger = structlog.get_logger(service_name)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

# Configuration management
class Config:
    """Configuration management for microservices"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config")
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from files"""
        config_dir = Path(self.config_path)
        
        # Load base config
        base_config_path = config_dir / "base.yaml"
        if base_config_path.exists():
            import yaml
            with open(base_config_path) as f:
                self._config.update(yaml.safe_load(f) or {})
        
        # Load environment-specific config
        environment = os.getenv("ENVIRONMENT", "development")
        env_config_path = config_dir / "environments" / f"{environment}.yaml"
        if env_config_path.exists():
            import yaml
            with open(env_config_path) as f:
                env_config = yaml.safe_load(f) or {}
                self._merge_config(self._config, env_config)
    
    def _merge_config(self, base: Dict, overlay: Dict):
        """Deep merge configuration dictionaries"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self._config.copy()

# Database management
class DatabaseManager:
    """Database connection and query management"""
    
    def __init__(self, 
                 db_type: str = "sqlite",
                 connection_string: Optional[str] = None):
        self.db_type = db_type
        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "sqlite:///data/service.db"
        )
        self._connection = None
    
    async def connect(self):
        """Establish database connection"""
        if self.db_type == "sqlite":
            # Extract path from sqlite URL
            db_path = self.connection_string.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(db_path)
            self._connection.row_factory = sqlite3.Row
        else:
            raise NotImplementedError(f"Database type {self.db_type} not implemented")
    
    async def disconnect(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute query and return results"""
        if not self._connection:
            await self.connect()
        
        cursor = self._connection.cursor()
        cursor.execute(query, params or ())
        
        if query.strip().upper().startswith('SELECT'):
            return [dict(row) for row in cursor.fetchall()]
        else:
            self._connection.commit()
            return []
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager"""
        if not self._connection:
            await self.connect()
        
        try:
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

# Cache management
class CacheManager:
    """Redis-based cache management"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
    
    async def connect(self):
        """Connect to Redis"""
        self._client = redis.from_url(self.redis_url, decode_responses=True)
        
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._client:
            await self.connect()
        
        try:
            value = self._client.get(key)
            return json.loads(value) if value else None
        except (redis.RedisError, json.JSONDecodeError):
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        if not self._client:
            await self.connect()
        
        try:
            self._client.setex(key, ttl, json.dumps(value, default=str))
        except redis.RedisError:
            pass  # Fail silently for cache operations
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self._client:
            await self.connect()
        
        try:
            self._client.delete(key)
        except redis.RedisError:
            pass
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._client:
            await self.connect()
        
        try:
            return bool(self._client.exists(key))
        except redis.RedisError:
            return False

# HTTP client for inter-service communication
class ServiceClient:
    """HTTP client for inter-service communication"""
    
    def __init__(self, 
                 service_registry: Optional['ServiceRegistry'] = None,
                 timeout: int = 30):
        self.service_registry = service_registry
        self.timeout = timeout
        self._session = None
    
    async def _get_session(self):
        """Get or create HTTP session"""
        if not self._session:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def call_service(self, 
                          service_name: str,
                          endpoint: str,
                          method: str = "GET",
                          data: Optional[Dict] = None,
                          headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Call another microservice"""
        
        # Discover service URL
        if self.service_registry:
            service_url = await self.service_registry.discover(service_name)
        else:
            service_url = os.getenv(f"{service_name.upper()}_URL", f"http://localhost:8000")
        
        if not service_url:
            raise ValueError(f"Service {service_name} not found")
        
        url = f"{service_url.rstrip('/')}/{endpoint.lstrip('/')}"
        session = await self._get_session()
        
        # Add request ID for tracing
        headers = headers or {}
        headers.setdefault('X-Request-ID', str(uuid.uuid4()))
        
        try:
            async with session.request(
                method.upper(),
                url,
                json=data,
                headers=headers
            ) as response:
                return await response.json()
                
        except Exception as e:
            raise RuntimeError(f"Service call failed: {e}")

# Retry decorator
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator for functions"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            raise last_exception
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            raise last_exception
            return sync_wrapper
    return decorator

# Rate limiting
class RateLimiter:
    """Simple rate limiter using sliding window"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if identifier in self._requests:
            self._requests[identifier] = [
                timestamp for timestamp in self._requests[identifier]
                if timestamp > window_start
            ]
        else:
            self._requests[identifier] = []
        
        # Check rate limit
        if len(self._requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self._requests[identifier].append(now)
        return True

# Validation utilities
def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    """Validate request data"""
    errors = {}
    
    for field in required_fields:
        if field not in data:
            errors[field] = "Field is required"
        elif data[field] is None:
            errors[field] = "Field cannot be null"
    
    if errors:
        raise ValueError(f"Validation errors: {errors}")
    
    return data

# Health check utilities
class HealthChecker:
    """Health check utilities"""
    
    @staticmethod
    async def check_database(db_manager: DatabaseManager) -> bool:
        """Check database connectivity"""
        try:
            await db_manager.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    @staticmethod
    async def check_cache(cache_manager: CacheManager) -> bool:
        """Check cache connectivity"""
        try:
            test_key = f"health_check_{uuid.uuid4()}"
            await cache_manager.set(test_key, "test", ttl=1)
            result = await cache_manager.get(test_key)
            await cache_manager.delete(test_key)
            return result == "test"
        except Exception:
            return False
    
    @staticmethod
    async def check_service(service_client: ServiceClient, service_name: str) -> bool:
        """Check service connectivity"""
        try:
            response = await service_client.call_service(
                service_name, "health", "GET"
            )
            return response.get("status") == "healthy"
        except Exception:
            return False

# Metrics collection utilities
class MetricsCollector:
    """Simple metrics collector"""
    
    def __init__(self):
        self._metrics = {}
        self._counters = {}
        self._histograms = {}
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment counter metric"""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric"""
        key = self._make_key(name, labels)
        self._metrics[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram metric"""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            "counters": self._counters.copy(),
            "gauges": self._metrics.copy(),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0
                }
                for k, v in self._histograms.items()
            }
        }
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key with labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

# Utility functions
def generate_correlation_id() -> str:
    """Generate correlation ID for request tracing"""
    return str(uuid.uuid4())

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def create_jwt_token(payload: Dict[str, Any], secret: str, expiry_hours: int = 24) -> str:
    """Create JWT token"""
    payload['exp'] = datetime.utcnow().timestamp() + (expiry_hours * 3600)
    return jwt.encode(payload, secret, algorithm='HS256')

def verify_jwt_token(token: str, secret: str) -> Dict[str, Any]:
    """Verify JWT token"""
    return jwt.decode(token, secret, algorithms=['HS256'])

def format_exception(exc: Exception) -> Dict[str, Any]:
    """Format exception for logging"""
    return {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_module": exc.__class__.__module__
    }