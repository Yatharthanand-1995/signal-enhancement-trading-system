"""
Comprehensive Error Handling Framework for Trading System
Provides structured error handling, logging, and recovery mechanisms.
"""
import sys
import traceback
import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union, Type
from enum import Enum
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    DATABASE = "database"
    NETWORK = "network"
    DATA_PROCESSING = "data_processing"
    ML_MODEL = "ml_model"
    TRADING = "trading"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"

class TradingSystemError(Exception):
    """Base exception for all trading system errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.CONFIGURATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        self.timestamp = datetime.utcnow()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            'error_message': self.message,  # Renamed to avoid conflict with LogRecord.message
            'category': self.category.value,
            'severity': self.severity.value,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None
        }

class DatabaseError(TradingSystemError):
    """Database-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, **kwargs)

class NetworkError(TradingSystemError):
    """Network and API-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)

class DataProcessingError(TradingSystemError):
    """Data processing and validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA_PROCESSING, **kwargs)

class MLModelError(TradingSystemError):
    """Machine learning model errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.ML_MODEL, **kwargs)

class TradingError(TradingSystemError):
    """Trading logic and execution errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TRADING, **kwargs)

class ValidationError(TradingSystemError):
    """Input validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)

class ErrorHandler:
    """Centralized error handling with recovery strategies"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Register default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default error recovery strategies"""
        self.recovery_strategies = {
            ErrorCategory.DATABASE: self._database_recovery,
            ErrorCategory.NETWORK: self._network_recovery,
            ErrorCategory.EXTERNAL_API: self._api_recovery,
            ErrorCategory.ML_MODEL: self._model_recovery
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> bool:
        """
        Handle error with logging and optional recovery
        
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        # Convert to TradingSystemError if needed
        if not isinstance(error, TradingSystemError):
            trading_error = self._convert_to_trading_error(error)
        else:
            trading_error = error
        
        # Add context information
        if context:
            trading_error.details.update(context)
        
        # Log the error
        self._log_error(trading_error)
        
        # Update error counts
        self._update_error_counts(trading_error)
        
        # Attempt recovery if requested and strategy exists
        if attempt_recovery and trading_error.category in self.recovery_strategies:
            try:
                recovery_success = self.recovery_strategies[trading_error.category](trading_error)
                if recovery_success:
                    logger.info(f"Successfully recovered from {trading_error.category.value} error")
                return recovery_success
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {trading_error.category.value}: {recovery_error}")
        
        return False
    
    def _convert_to_trading_error(self, error: Exception) -> TradingSystemError:
        """Convert generic exception to TradingSystemError"""
        error_type = type(error).__name__
        
        # Map common exceptions to categories
        category_mapping = {
            'ConnectionError': ErrorCategory.DATABASE,
            'TimeoutError': ErrorCategory.NETWORK,
            'ValueError': ErrorCategory.VALIDATION,
            'KeyError': ErrorCategory.DATA_PROCESSING,
            'ImportError': ErrorCategory.CONFIGURATION,
            'ModuleNotFoundError': ErrorCategory.CONFIGURATION
        }
        
        category = category_mapping.get(error_type, ErrorCategory.CONFIGURATION)
        severity = ErrorSeverity.HIGH if error_type in ['ConnectionError', 'TimeoutError'] else ErrorSeverity.MEDIUM
        
        return TradingSystemError(
            message=str(error),
            category=category,
            severity=severity,
            original_error=error
        )
    
    def _log_error(self, error: TradingSystemError):
        """Log error with appropriate level based on severity"""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {error.message}", extra=error_dict)
        else:
            logger.info(f"LOW SEVERITY: {error.message}", extra=error_dict)
        
        # Log traceback for debugging
        if error.original_error:
            logger.debug("Original traceback:", exc_info=error.original_error)
    
    def _update_error_counts(self, error: TradingSystemError):
        """Update error counts for monitoring"""
        key = f"{error.category.value}_{error.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def _database_recovery(self, error: TradingSystemError) -> bool:
        """Recovery strategy for database errors"""
        # Implement database connection retry logic
        logger.info("Attempting database recovery...")
        return False  # Placeholder
    
    def _network_recovery(self, error: TradingSystemError) -> bool:
        """Recovery strategy for network errors"""
        # Implement network retry with exponential backoff
        logger.info("Attempting network recovery...")
        return False  # Placeholder
    
    def _api_recovery(self, error: TradingSystemError) -> bool:
        """Recovery strategy for API errors"""
        # Implement API retry with rate limiting
        logger.info("Attempting API recovery...")
        return False  # Placeholder
    
    def _model_recovery(self, error: TradingSystemError) -> bool:
        """Recovery strategy for ML model errors"""
        # Implement model fallback or reload
        logger.info("Attempting model recovery...")
        return False  # Placeholder
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error counts and patterns"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts.copy(),
            'critical_errors': sum(v for k, v in self.error_counts.items() if 'critical' in k),
            'high_errors': sum(v for k, v in self.error_counts.items() if 'high' in k)
        }

# Global error handler instance
error_handler = ErrorHandler()

def handle_errors(
    category: ErrorCategory = ErrorCategory.CONFIGURATION,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = True,
    recovery: bool = True
):
    """
    Decorator for automatic error handling
    
    Args:
        category: Error category for classification
        severity: Default severity level
        reraise: Whether to reraise the exception after handling
        recovery: Whether to attempt error recovery
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TradingSystemError as e:
                # Already a trading system error, just handle it
                recovery_success = error_handler.handle_error(e, attempt_recovery=recovery)
                if reraise and not recovery_success:
                    raise
                return None
            except Exception as e:
                # Convert and handle generic exception
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],  # Limit length
                    'kwargs': str(kwargs)[:100]
                }
                
                trading_error = TradingSystemError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    category=category,
                    severity=severity,
                    details=context,
                    original_error=e
                )
                
                recovery_success = error_handler.handle_error(trading_error, attempt_recovery=recovery)
                if reraise and not recovery_success:
                    raise trading_error
                return None
        
        return wrapper
    return decorator

@contextmanager
def error_context(
    operation_name: str,
    category: ErrorCategory = ErrorCategory.CONFIGURATION,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Context manager for error handling in code blocks"""
    try:
        yield
    except Exception as e:
        context = {'operation': operation_name}
        
        if isinstance(e, TradingSystemError):
            e.details.update(context)
            error_handler.handle_error(e)
        else:
            trading_error = TradingSystemError(
                message=f"Error in {operation_name}: {str(e)}",
                category=category,
                severity=severity,
                details=context,
                original_error=e
            )
            error_handler.handle_error(trading_error)
        
        raise

def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    max_retries: int = 3,
    **kwargs
) -> Any:
    """
    Safely execute a function with automatic retry and error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return if all attempts fail
        max_retries: Maximum number of retry attempts
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default_return if all attempts fail
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                # Exponential backoff
                import time
                time.sleep(2 ** attempt)
            else:
                logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                error_handler.handle_error(e, context={'function': func.__name__, 'attempts': max_retries + 1})
    
    return default_return