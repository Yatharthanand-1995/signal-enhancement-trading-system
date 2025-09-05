"""
Service Interfaces
Abstract base classes and protocols for microservice interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio

from .models import (
    ServiceResponse, HealthCheckResponse, MetricsResponse,
    Signal, MarketData, Portfolio, RiskMetrics, BacktestResult,
    MLModelInfo, DataRequest, SignalRequest, RiskAnalysisRequest,
    BacktestRequest, MLPredictionRequest, PaginatedResponse
)

class ServiceInterface(ABC):
    """Base interface for all microservices"""
    
    def __init__(self, service_name: str, version: str = "1.0.0"):
        self.service_name = service_name
        self.version = version
        self._startup_time = datetime.utcnow()
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Perform health check and return status"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> MetricsResponse:
        """Get service metrics"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful service shutdown"""
        pass

class DataServiceInterface(ServiceInterface):
    """Interface for data management service"""
    
    @abstractmethod
    async def get_market_data(self, request: DataRequest) -> ServiceResponse:
        """Retrieve market data for specified symbols and time range"""
        pass
    
    @abstractmethod
    async def get_technical_indicators(self, 
                                     symbol: str,
                                     indicators: List[str],
                                     period: int = 30) -> ServiceResponse:
        """Calculate technical indicators"""
        pass
    
    @abstractmethod
    async def validate_data(self, data: List[MarketData]) -> ServiceResponse:
        """Validate market data quality"""
        pass
    
    @abstractmethod
    async def store_data(self, data: List[MarketData]) -> ServiceResponse:
        """Store market data"""
        pass
    
    @abstractmethod
    async def get_data_quality_metrics(self) -> ServiceResponse:
        """Get data quality metrics"""
        pass

class SignalServiceInterface(ServiceInterface):
    """Interface for signal generation service"""
    
    @abstractmethod
    async def generate_signals(self, request: SignalRequest) -> ServiceResponse:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    async def get_signal_history(self, 
                               symbol: str,
                               start_date: datetime,
                               end_date: datetime) -> ServiceResponse:
        """Get historical signals"""
        pass
    
    @abstractmethod
    async def evaluate_signal_performance(self, 
                                        signal_ids: List[str]) -> ServiceResponse:
        """Evaluate signal performance"""
        pass
    
    @abstractmethod
    async def get_signal_types(self) -> ServiceResponse:
        """Get available signal types"""
        pass

class MLServiceInterface(ServiceInterface):
    """Interface for machine learning service"""
    
    @abstractmethod
    async def train_model(self, 
                         model_type: str,
                         training_data: Dict[str, Any],
                         hyperparameters: Dict[str, Any]) -> ServiceResponse:
        """Train a machine learning model"""
        pass
    
    @abstractmethod
    async def predict(self, request: MLPredictionRequest) -> ServiceResponse:
        """Make predictions using trained model"""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_id: str) -> ServiceResponse:
        """Get model information and metadata"""
        pass
    
    @abstractmethod
    async def list_models(self) -> ServiceResponse:
        """List all available models"""
        pass
    
    @abstractmethod
    async def evaluate_model(self, 
                           model_id: str,
                           test_data: Dict[str, Any]) -> ServiceResponse:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    async def deploy_model(self, model_id: str) -> ServiceResponse:
        """Deploy model to production"""
        pass

class RiskServiceInterface(ServiceInterface):
    """Interface for risk management service"""
    
    @abstractmethod
    async def calculate_portfolio_risk(self, 
                                     request: RiskAnalysisRequest) -> ServiceResponse:
        """Calculate portfolio risk metrics"""
        pass
    
    @abstractmethod
    async def perform_stress_test(self, 
                                portfolio_id: str,
                                scenarios: List[Dict[str, Any]]) -> ServiceResponse:
        """Perform portfolio stress testing"""
        pass
    
    @abstractmethod
    async def get_risk_limits(self, portfolio_id: str) -> ServiceResponse:
        """Get risk limits for portfolio"""
        pass
    
    @abstractmethod
    async def check_risk_compliance(self, portfolio_id: str) -> ServiceResponse:
        """Check portfolio compliance with risk limits"""
        pass
    
    @abstractmethod
    async def calculate_var(self, 
                          portfolio_id: str,
                          confidence_level: float = 0.95,
                          holding_period: int = 1) -> ServiceResponse:
        """Calculate Value at Risk"""
        pass

class BacktestServiceInterface(ServiceInterface):
    """Interface for backtesting service"""
    
    @abstractmethod
    async def run_backtest(self, request: BacktestRequest) -> ServiceResponse:
        """Run backtesting simulation"""
        pass
    
    @abstractmethod
    async def get_backtest_results(self, backtest_id: str) -> ServiceResponse:
        """Get backtesting results"""
        pass
    
    @abstractmethod
    async def compare_strategies(self, 
                               strategy_ids: List[str]) -> ServiceResponse:
        """Compare multiple strategies"""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self, backtest_id: str) -> ServiceResponse:
        """Get detailed performance metrics"""
        pass
    
    @abstractmethod
    async def list_backtests(self, 
                           user_id: Optional[str] = None) -> ServiceResponse:
        """List backtests, optionally filtered by user"""
        pass

class AnalyticsServiceInterface(ServiceInterface):
    """Interface for analytics and reporting service"""
    
    @abstractmethod
    async def generate_performance_report(self, 
                                        portfolio_id: str,
                                        start_date: datetime,
                                        end_date: datetime) -> ServiceResponse:
        """Generate performance report"""
        pass
    
    @abstractmethod
    async def calculate_attribution(self, 
                                  portfolio_id: str,
                                  benchmark: str) -> ServiceResponse:
        """Calculate performance attribution"""
        pass
    
    @abstractmethod
    async def get_signal_effectiveness(self, 
                                     signal_type: str,
                                     period_days: int = 30) -> ServiceResponse:
        """Analyze signal effectiveness"""
        pass
    
    @abstractmethod
    async def create_dashboard_data(self, user_id: str) -> ServiceResponse:
        """Create data for user dashboard"""
        pass

class NotificationServiceInterface(ServiceInterface):
    """Interface for notification service"""
    
    @abstractmethod
    async def send_alert(self, 
                        alert_type: str,
                        message: str,
                        recipients: List[str],
                        severity: str = "medium") -> ServiceResponse:
        """Send alert notification"""
        pass
    
    @abstractmethod
    async def send_report(self, 
                         report_data: Dict[str, Any],
                         recipients: List[str],
                         report_type: str = "daily") -> ServiceResponse:
        """Send report via email/notification"""
        pass
    
    @abstractmethod
    async def get_notification_history(self, 
                                     user_id: str,
                                     limit: int = 50) -> ServiceResponse:
        """Get notification history for user"""
        pass
    
    @abstractmethod
    async def update_notification_preferences(self, 
                                            user_id: str,
                                            preferences: Dict[str, Any]) -> ServiceResponse:
        """Update user notification preferences"""
        pass

class GatewayServiceInterface(ServiceInterface):
    """Interface for API gateway service"""
    
    @abstractmethod
    async def route_request(self, 
                          service_name: str,
                          endpoint: str,
                          method: str,
                          payload: Dict[str, Any]) -> ServiceResponse:
        """Route request to appropriate service"""
        pass
    
    @abstractmethod
    async def authenticate_request(self, token: str) -> ServiceResponse:
        """Authenticate incoming request"""
        pass
    
    @abstractmethod
    async def get_service_status(self) -> ServiceResponse:
        """Get status of all registered services"""
        pass
    
    @abstractmethod
    async def register_service(self, 
                             service_name: str,
                             service_url: str,
                             health_check_url: str) -> ServiceResponse:
        """Register a new service with the gateway"""
        pass

class ConfigServiceInterface(ServiceInterface):
    """Interface for configuration service"""
    
    @abstractmethod
    async def get_config(self, 
                        key: str,
                        environment: str = "production") -> ServiceResponse:
        """Get configuration value"""
        pass
    
    @abstractmethod
    async def update_config(self, 
                          key: str,
                          value: Any,
                          environment: str = "production") -> ServiceResponse:
        """Update configuration value"""
        pass
    
    @abstractmethod
    async def get_feature_flag(self, flag_name: str) -> ServiceResponse:
        """Get feature flag value"""
        pass
    
    @abstractmethod
    async def toggle_feature_flag(self, flag_name: str, enabled: bool) -> ServiceResponse:
        """Toggle feature flag"""
        pass
    
    @abstractmethod
    async def get_all_configs(self, environment: str = "production") -> ServiceResponse:
        """Get all configurations for environment"""
        pass

# Protocol definitions for dependency injection
class ServiceRegistry(ABC):
    """Service registry for service discovery"""
    
    @abstractmethod
    async def register(self, service_name: str, service_url: str, metadata: Dict[str, Any] = None):
        """Register service"""
        pass
    
    @abstractmethod
    async def deregister(self, service_name: str):
        """Deregister service"""
        pass
    
    @abstractmethod
    async def discover(self, service_name: str) -> Optional[str]:
        """Discover service URL"""
        pass
    
    @abstractmethod
    async def list_services(self) -> Dict[str, str]:
        """List all registered services"""
        pass

class LoadBalancer(ABC):
    """Load balancer for service instances"""
    
    @abstractmethod
    async def get_instance(self, service_name: str) -> Optional[str]:
        """Get service instance using load balancing strategy"""
        pass
    
    @abstractmethod
    async def add_instance(self, service_name: str, instance_url: str):
        """Add service instance"""
        pass
    
    @abstractmethod
    async def remove_instance(self, service_name: str, instance_url: str):
        """Remove service instance"""
        pass
    
    @abstractmethod
    async def health_check_instances(self):
        """Perform health checks on all instances"""
        pass

class CircuitBreaker(ABC):
    """Circuit breaker for service fault tolerance"""
    
    @abstractmethod
    async def call(self, service_name: str, func, *args, **kwargs):
        """Make service call with circuit breaker protection"""
        pass
    
    @abstractmethod
    async def is_open(self, service_name: str) -> bool:
        """Check if circuit is open"""
        pass
    
    @abstractmethod
    async def reset(self, service_name: str):
        """Reset circuit breaker"""
        pass