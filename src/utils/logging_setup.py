"""
Structured Logging System for Trading Platform
Provides comprehensive logging with structured data, rotation, and monitoring integration.
"""
import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback
from functools import wraps

from config.enhanced_config import enhanced_config

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields from record
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        if hasattr(record, 'trade_id'):
            log_entry['trade_id'] = record.trade_id
            
        if hasattr(record, 'symbol'):
            log_entry['symbol'] = record.symbol
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process']:
                if not key.startswith('_') and key != 'message':
                    log_entry[key] = value
        
        return json.dumps(log_entry)

class TradingSystemLogger:
    """Enhanced logger for the trading system"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with file and console handlers"""
        config = enhanced_config.logging
        
        # Set level
        level = getattr(logging, config.level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set formatters
        if config.enable_json_logs:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(config.format)
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        
        # Add console handler for development
        if enhanced_config.environment == 'development':
            self.logger.addHandler(console_handler)
        
        self.logger.propagate = False
    
    def info(self, message: str, **kwargs):
        """Log info message with extra context"""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra context"""
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra context"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        if exception:
            self.logger.error(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception"""
        if exception:
            self.logger.critical(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.critical(message, extra=kwargs)
    
    def log_trade(self, action: str, symbol: str, quantity: float, price: float, **kwargs):
        """Log trading action with structured data"""
        trade_data = {
            'component': 'trading',
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        self.info(f"Trade {action}: {symbol} {quantity}@{price}", **trade_data)
    
    def log_prediction(self, symbol: str, prediction: float, confidence: float, model: str, **kwargs):
        """Log ML prediction with structured data"""
        prediction_data = {
            'component': 'ml_prediction',
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'model': model,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        self.info(f"Prediction for {symbol}: {prediction} (confidence: {confidence})", **prediction_data)
    
    def log_performance(self, metric_name: str, value: float, period: str = 'daily', **kwargs):
        """Log performance metrics"""
        performance_data = {
            'component': 'performance',
            'metric': metric_name,
            'value': value,
            'period': period,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        self.info(f"Performance {metric_name}: {value} ({period})", **performance_data)
    
    def log_risk_event(self, event_type: str, severity: str, description: str, **kwargs):
        """Log risk management events"""
        risk_data = {
            'component': 'risk_management',
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        if severity.lower() in ['critical', 'high']:
            self.error(f"Risk Event [{severity}]: {description}", **risk_data)
        else:
            self.warning(f"Risk Event [{severity}]: {description}", **risk_data)

def get_logger(name: str) -> TradingSystemLogger:
    """Get or create a logger instance"""
    return TradingSystemLogger(name)

def log_function_call(logger_name: Optional[str] = None, log_args: bool = False, log_result: bool = False):
    """Decorator to log function calls"""
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            start_time = datetime.utcnow()
            
            log_data = {
                'component': 'function_call',
                'function': func_name,
                'start_time': start_time.isoformat()
            }
            
            if log_args:
                log_data['args'] = str(args)[:200]  # Limit length
                log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            logger.debug(f"Calling {func_name}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                result_data = {
                    'component': 'function_call',
                    'function': func_name,
                    'duration_seconds': duration,
                    'status': 'success'
                }
                
                if log_result and result is not None:
                    result_data['result'] = str(result)[:200]
                
                logger.debug(f"Completed {func_name} in {duration:.3f}s", **result_data)
                return result
                
            except Exception as e:
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                
                error_data = {
                    'component': 'function_call',
                    'function': func_name,
                    'duration_seconds': duration,
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                
                logger.error(f"Error in {func_name} after {duration:.3f}s: {e}", exception=e, **error_data)
                raise
        
        return wrapper
    return decorator

class PerformanceLogger:
    """Logger for performance metrics and monitoring"""
    
    def __init__(self):
        self.logger = get_logger('performance')
        self.metrics = {}
    
    def log_execution_time(self, operation: str, duration: float, **kwargs):
        """Log execution time for operations"""
        self.logger.info(
            f"Performance: {operation} took {duration:.3f}s",
            component='performance',
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(
            f"Memory usage: {component} using {memory_mb:.2f}MB",
            component='memory',
            memory_mb=memory_mb
        )
    
    def log_database_query(self, query_type: str, duration: float, rows: int = None):
        """Log database query performance"""
        log_data = {
            'component': 'database',
            'query_type': query_type,
            'duration_seconds': duration
        }
        
        if rows is not None:
            log_data['rows_affected'] = rows
        
        self.logger.info(f"DB Query: {query_type} in {duration:.3f}s", **log_data)

# Global performance logger
perf_logger = PerformanceLogger()

def setup_logging():
    """Setup global logging configuration"""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Only show warnings and errors from other libraries
    
    # Disable noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Create main application logger
    main_logger = get_logger('trading_system')
    main_logger.info("Logging system initialized", component='system')
    
    return main_logger

# Initialize logging when module is imported
main_logger = setup_logging()