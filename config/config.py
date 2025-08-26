"""
Configuration settings for the trading system
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', 5432))
    database: str = os.getenv('DB_NAME', 'trading_system')
    user: str = os.getenv('DB_USER', 'trading_user')
    password: str = os.getenv('DB_PASSWORD', 'secure_password')
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', 6379))
    db: int = int(os.getenv('REDIS_DB', 0))
    password: str = os.getenv('REDIS_PASSWORD', '')

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    # Position sizing
    max_position_size: float = 0.25  # 25% max allocation per position
    min_position_size: float = 0.02  # 2% min allocation per position
    
    # Risk management
    max_drawdown: float = 0.15  # 15% maximum drawdown
    volatility_target: float = 0.12  # 12% annualized volatility target
    stop_loss_atr: float = 2.0  # 2 ATR stop loss
    
    # Signal thresholds
    min_signal_strength: float = 0.6  # Minimum signal strength to trade
    min_confidence: float = 0.65  # Minimum confidence to trade
    
    # Holding periods
    min_holding_days: int = 2
    max_holding_days: int = 15
    
    # Model parameters
    lstm_lookback: int = 30  # 30 day lookback window
    technical_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.technical_weights is None:
            self.technical_weights = {
                'rsi': 0.25,
                'macd': 0.35,
                'bb': 0.25,
                'ma': 0.15
            }

@dataclass
class MLConfig:
    """Machine learning model configuration"""
    # LSTM parameters
    lstm_layers: int = 4
    lstm_units: list = None
    dropout_rate: float = 0.2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    
    # XGBoost parameters
    n_estimators: int = 1000
    max_depth: int = 10
    xgb_learning_rate: float = 0.06
    subsample: float = 0.8
    
    # Feature engineering
    feature_lookback: int = 30
    rolling_window: int = 252  # 1 year for normalization
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [250, 200, 150, 50]

@dataclass
class DataConfig:
    """Data management configuration"""
    # Data sources
    primary_source: str = 'yfinance'
    fallback_source: str = 'alpha_vantage'
    
    # Update schedules
    data_update_time: str = '06:00'  # UTC
    indicator_update_time: str = '07:00'  # UTC
    
    # Storage
    cache_ttl: int = 3600  # 1 hour cache TTL
    max_cache_size: int = 1000  # Max items in memory cache
    
    # Quality thresholds
    min_data_completeness: float = 0.95  # 95% data completeness required
    max_price_change: float = 0.20  # 20% max daily price change before flagging

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Walk-forward parameters
    training_window: int = 504  # 2 years
    testing_window: int = 63   # 3 months
    min_wf_efficiency: float = 0.6  # 60% walk-forward efficiency
    
    # Transaction costs
    commission_per_share: float = 0.005
    minimum_commission: float = 1.0
    slippage: float = 0.0005  # 0.05%
    market_impact_factor: float = 0.1
    
    # Performance thresholds
    min_sharpe_ratio: float = 1.5
    min_calmar_ratio: float = 2.0
    min_win_rate: float = 0.55

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    auto_refresh_interval: int = 30  # seconds
    max_display_signals: int = 100
    chart_height: int = 500
    heatmap_height: int = 400

class Config:
    """Main configuration class"""
    def __init__(self):
        self.db = DatabaseConfig()
        self.redis = RedisConfig()
        self.trading = TradingConfig()
        self.ml = MLConfig()
        self.data = DataConfig()
        self.backtest = BacktestConfig()
        self.dashboard = DashboardConfig()
        
        # Environment
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'true').lower() == 'true'
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'logs/trading_system.log')
        
        # API keys (to be set via environment variables)
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.polygon_key = os.getenv('POLYGON_KEY', '')
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'db': self.db.__dict__,
            'redis': self.redis.__dict__,
            'trading': self.trading.__dict__,
            'ml': self.ml.__dict__,
            'data': self.data.__dict__,
            'backtest': self.backtest.__dict__,
            'dashboard': self.dashboard.__dict__,
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'log_file': self.log_file
        }

# Global config instance
config = Config()