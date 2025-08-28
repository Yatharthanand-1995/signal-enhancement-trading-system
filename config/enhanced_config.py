"""
Enhanced Configuration with Security, Performance, and Reliability Features
"""
import os
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Enhanced security configuration"""
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', secrets.token_urlsafe(32)))
    password_min_length: int = 12
    max_login_attempts: int = 5
    session_timeout: int = 3600  # 1 hour
    jwt_algorithm: str = 'HS256'
    rate_limit_per_minute: int = int(os.getenv('REQUESTS_PER_MINUTE', 60))
    api_rate_limit: int = int(os.getenv('API_RATE_LIMIT', 100))
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        return len(api_key) >= 16 and api_key.isalnum()

@dataclass
class DatabaseConfig:
    """Enhanced database configuration with connection pooling"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', 5432))
    database: str = os.getenv('DB_NAME', 'trading_system')
    user: str = os.getenv('DB_USER', 'trading_user')
    password: str = os.getenv('DB_PASSWORD', 'secure_password')
    
    # Connection pooling
    pool_size: int = int(os.getenv('POOL_SIZE', 5))
    max_connections: int = int(os.getenv('MAX_CONNECTIONS', 20))
    pool_recycle: int = int(os.getenv('POOL_RECYCLE', 3600))
    
    # Security
    ssl_mode: str = 'prefer'
    connect_timeout: int = 30
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Enhanced Redis configuration"""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', 6379))
    db: int = int(os.getenv('REDIS_DB', 0))
    password: str = os.getenv('REDIS_PASSWORD', '')
    
    # Performance
    max_connections: int = 10
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    health_check_interval: int = 30

@dataclass
class LoggingConfig:
    """Enhanced logging configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: str = os.getenv('LOG_FILE', 'logs/trading_system.log')
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Structured logging
    enable_json_logs: bool = os.getenv('ENVIRONMENT', 'development') == 'production'
    log_sql_queries: bool = os.getenv('DEBUG', 'false').lower() == 'true'

@dataclass
class TradingConfig:
    """Enhanced trading strategy configuration with risk controls"""
    # Position sizing with enhanced risk controls
    max_position_size: float = 0.20  # Reduced from 0.25 for safety
    min_position_size: float = 0.02
    max_portfolio_exposure: float = 0.80  # Maximum total portfolio exposure
    
    # Risk management with multiple layers
    max_drawdown: float = 0.12  # Reduced from 0.15 for safety
    volatility_target: float = 0.12
    stop_loss_atr: float = 2.0
    max_daily_loss: float = 0.05  # 5% max daily loss
    
    # Signal thresholds with higher confidence requirements
    min_signal_strength: float = 0.65  # Increased from 0.6
    min_confidence: float = 0.70  # Increased from 0.65
    
    # Holding periods
    min_holding_days: int = 2
    max_holding_days: int = 15
    
    # Model parameters
    lstm_lookback: int = 30
    
    # Risk monitoring
    risk_check_interval: int = 300  # 5 minutes
    emergency_stop_loss: float = 0.08  # 8% emergency stop

@dataclass
class MLConfig:
    """Enhanced ML configuration with monitoring"""
    # LSTM parameters
    lstm_layers: int = 4
    lstm_units: list = field(default_factory=lambda: [250, 200, 150, 50])
    dropout_rate: float = 0.2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    
    # XGBoost parameters  
    n_estimators: int = 1000
    max_depth: int = 10
    xgb_learning_rate: float = 0.06
    subsample: float = 0.8
    
    # Model monitoring
    min_model_accuracy: float = 0.60
    model_retrain_threshold: float = 0.55
    prediction_confidence_threshold: float = 0.65
    
    # Feature engineering
    feature_lookback: int = 30
    rolling_window: int = 252

@dataclass
class MonitoringConfig:
    """System monitoring and alerting configuration"""
    health_check_interval: int = 60  # seconds
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None
    
    # Performance thresholds
    max_response_time: float = 2.0  # seconds
    max_memory_usage: float = 0.80  # 80%
    max_cpu_usage: float = 0.80  # 80%
    
    # Trading alerts
    max_consecutive_losses: int = 5
    unusual_volume_threshold: float = 2.0
    price_gap_threshold: float = 0.10  # 10%

class EnhancedConfig:
    """Enhanced configuration with security, monitoring, and performance optimizations"""
    
    def __init__(self):
        self.security = SecurityConfig()
        self.db = DatabaseConfig()
        self.redis = RedisConfig()
        self.logging = LoggingConfig()
        self.trading = TradingConfig()
        self.ml = MLConfig()
        self.monitoring = MonitoringConfig()
        
        # Environment
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # API configurations
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.polygon_key = os.getenv('POLYGON_KEY', '')
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate critical configuration settings"""
        errors = []
        
        # Check required API keys in production
        if self.environment == 'production':
            if not self.alpha_vantage_key or self.alpha_vantage_key == 'YOUR_ALPHA_VANTAGE_KEY_HERE':
                errors.append("ALPHA_VANTAGE_KEY not set for production")
            
            if not self.security.secret_key or len(self.security.secret_key) < 32:
                errors.append("SECRET_KEY too short for production")
        
        # Validate database connection
        if not self.db.password or self.db.password == 'CHANGE_THIS_PASSWORD_IN_PRODUCTION':
            logger.warning("Using default database password - change for production!")
        
        # Validate trading parameters
        if self.trading.max_position_size > 1.0 or self.trading.max_position_size < 0:
            errors.append("Invalid max_position_size - must be between 0 and 1")
        
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            if self.environment == 'production':
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'db': {
                'host': self.db.host,
                'port': self.db.port,
                'database': self.db.database,
                'user': self.db.user,
                'pool_size': self.db.pool_size,
                'max_connections': self.db.max_connections
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db
            },
            'trading': self.trading.__dict__,
            'ml': self.ml.__dict__,
            'monitoring': self.monitoring.__dict__,
            'has_api_keys': bool(self.alpha_vantage_key and self.alpha_vantage_key != 'YOUR_ALPHA_VANTAGE_KEY_HERE')
        }

# Global enhanced config instance
enhanced_config = EnhancedConfig()