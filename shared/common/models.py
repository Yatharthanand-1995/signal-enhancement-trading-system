"""
Shared Data Models
Common Pydantic models used across microservices
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel as PydanticBaseModel, Field, validator
import uuid

class ServiceStatus(str, Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy" 
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class ResponseStatus(str, Enum):
    """Response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"

class BaseModel(PydanticBaseModel):
    """Base model with common fields and methods"""
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class ServiceResponse(BaseModel):
    """Standard service response format"""
    status: ResponseStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
class ErrorResponse(BaseModel):
    """Standard error response format"""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class HealthCheckResponse(BaseModel):
    """Health check response format"""
    service_name: str
    status: ServiceStatus
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dependencies: Dict[str, ServiceStatus] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None

class MetricsResponse(BaseModel):
    """Metrics response format"""
    service_name: str
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Trading-specific models
class Signal(BaseModel):
    """Trading signal model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    signal_type: str
    strength: float = Field(ge=0.0, le=1.0, description="Signal strength between 0 and 1")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level between 0 and 1")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('strength', 'confidence')
    def validate_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Value must be between 0.0 and 1.0')
        return v

class MarketData(BaseModel):
    """Market data model"""
    symbol: str
    timestamp: datetime
    price: float = Field(gt=0, description="Price must be positive")
    volume: int = Field(ge=0, description="Volume must be non-negative")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Portfolio(BaseModel):
    """Portfolio model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    positions: Dict[str, float] = Field(default_factory=dict)
    total_value: float = Field(ge=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RiskMetrics(BaseModel):
    """Risk metrics model"""
    portfolio_id: str
    var_95: float = Field(description="Value at Risk (95%)")
    cvar_95: float = Field(description="Conditional Value at Risk (95%)")
    sharpe_ratio: float
    volatility: float = Field(ge=0)
    max_drawdown: float = Field(le=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BacktestResult(BaseModel):
    """Backtest result model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float = Field(ge=0.0, le=1.0)
    total_trades: int = Field(ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MLModelInfo(BaseModel):
    """ML model information"""
    model_id: str
    model_name: str
    model_version: str
    model_type: str
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_trained: datetime
    features: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

class ConfigurationUpdate(BaseModel):
    """Configuration update model"""
    config_key: str
    config_value: Any
    environment: str = "development"
    updated_by: str
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ServiceMetrics(BaseModel):
    """Service metrics model"""
    service_name: str
    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU usage percentage")
    memory_usage: float = Field(ge=0.0, description="Memory usage in MB")
    request_count: int = Field(ge=0, description="Total requests processed")
    error_count: int = Field(ge=0, description="Total errors encountered")
    response_time_avg: float = Field(ge=0.0, description="Average response time in ms")
    uptime_seconds: int = Field(ge=0, description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AlertMessage(BaseModel):
    """Alert message model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service_name: str
    alert_type: str  # 'error', 'warning', 'info'
    message: str
    severity: str = Field(regex=r'^(low|medium|high|critical)$')
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

# Request/Response models for inter-service communication
class ServiceRequest(BaseModel):
    """Base service request model"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DataRequest(ServiceRequest):
    """Data service request model"""
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_type: str = "ohlcv"  # 'ohlcv', 'technical', 'fundamental'

class SignalRequest(ServiceRequest):
    """Signal service request model"""
    symbols: List[str]
    signal_types: List[str] = Field(default_factory=list)
    lookback_period: int = Field(default=30, ge=1)

class RiskAnalysisRequest(ServiceRequest):
    """Risk analysis request model"""
    portfolio_id: str
    analysis_type: str = "var"  # 'var', 'stress_test', 'scenario_analysis'
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)

class BacktestRequest(ServiceRequest):
    """Backtest request model"""
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    parameters: Dict[str, Any] = Field(default_factory=dict)

class MLPredictionRequest(ServiceRequest):
    """ML prediction request model"""
    model_id: str
    features: Dict[str, Any]
    prediction_horizon: int = Field(default=1, ge=1)

# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    
class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any]
    total_count: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=0)
    has_next: bool
    has_previous: bool

    @validator('total_pages', always=True)
    def calculate_total_pages(cls, v, values):
        total_count = values.get('total_count', 0)
        page_size = values.get('page_size', 50)
        return (total_count + page_size - 1) // page_size if total_count > 0 else 0

    @validator('has_next', always=True)
    def calculate_has_next(cls, v, values):
        page = values.get('page', 1)
        total_pages = values.get('total_pages', 0)
        return page < total_pages

    @validator('has_previous', always=True)
    def calculate_has_previous(cls, v, values):
        page = values.get('page', 1)
        return page > 1