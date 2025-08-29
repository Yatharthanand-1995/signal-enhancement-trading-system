"""
Safe ML Library Initialization
Provides robust initialization for TensorFlow and other ML libraries with proper error handling.
"""
import os
import sys
from contextlib import contextmanager
from typing import Optional, Dict, Any
import warnings

# Use relative imports for better compatibility
try:
    from .error_handling import MLModelError, ErrorSeverity, handle_errors
    from .logging_setup import get_logger
except ImportError:
    # Fallback to basic error handling if modules not available
    class MLModelError(Exception):
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
                    print(f"Error in {func.__name__}: {e}")
                    if recovery:
                        return None
                    raise
            return wrapper
        return decorator
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)

class MLLibraryManager:
    """Manages safe initialization and usage of ML libraries"""
    
    def __init__(self):
        self._tf_initialized = False
        self._tf_available = False
        self._xgb_available = False
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup minimal environment for ML libraries"""
        # Only essential TensorFlow environment variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        logger.info("ML environment configuration completed")
    
    def initialize_tensorflow(self) -> bool:
        """Minimal TensorFlow initialization with fallback"""
        if self._tf_initialized:
            return self._tf_available
            
        try:
            logger.info("Checking TensorFlow availability...")
            
            # Try to import TensorFlow - if this fails, it's not available
            import tensorflow
            
            # If we get here, TensorFlow is available
            self._tf_available = True
            self.tensorflow = tensorflow
            logger.info(f"TensorFlow {tensorflow.__version__} is available")
                
        except ImportError:
            logger.info("TensorFlow not installed - skipping")
            self._tf_available = False
        except Exception as e:
            logger.warning(f"TensorFlow check failed: {e}")
            self._tf_available = False
            
        self._tf_initialized = True
        return self._tf_available
    
    def _import_tensorflow_safe(self) -> bool:
        """Import TensorFlow in a separate process to avoid mutex conflicts"""
        def test_tf():
            try:
                import tensorflow as tf
                
                # Configure for CPU only
                tf.config.set_visible_devices([], 'GPU')
                
                # Simple test
                a = tf.constant([1.0, 2.0])
                b = tf.constant([3.0, 4.0])
                c = tf.add(a, b)
                
                # Verify result
                result = c.numpy()
                return len(result) == 2
                
            except Exception:
                return False
        
        try:
            # Use multiprocessing to isolate TensorFlow
            process = multiprocessing.Process(target=test_tf)
            process.start()
            process.join(timeout=30)  # 30 second timeout
            
            if process.exitcode == 0:
                # Now safely import in main process
                import tensorflow as tf
                tf.config.set_visible_devices([], 'GPU')
                tf.config.threading.set_inter_op_parallelism_threads(1)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def initialize_xgboost(self) -> bool:
        """Minimal XGBoost initialization with fallback"""
        try:
            logger.info("Checking XGBoost availability...")
            
            # Try to import XGBoost - if this fails, it's not available
            import xgboost
            
            # If we get here, XGBoost is available
            self._xgb_available = True
            self.xgboost = xgboost
            logger.info(f"XGBoost {xgboost.__version__} is available")
            return True
            
        except ImportError:
            logger.info("XGBoost not installed - skipping")
            self._xgb_available = False
            return False
        except Exception as e:
            logger.warning(f"XGBoost check failed: {e}")
            self._xgb_available = False
            return False
    
    @contextmanager
    def tensorflow_context(self):
        """Context manager for safe TensorFlow usage"""
        if not self._tf_available:
            raise MLModelError("TensorFlow not available", severity=ErrorSeverity.HIGH)
        
        try:
            import tensorflow as tf
            with tf.device('/CPU:0'):  # Force CPU usage
                yield tf
        except Exception as e:
            raise MLModelError(f"TensorFlow context error: {e}", severity=ErrorSeverity.HIGH, original_error=e)
    
    @contextmanager 
    def xgboost_context(self):
        """Context manager for safe XGBoost usage"""
        if not self._xgb_available:
            raise MLModelError("XGBoost not available", severity=ErrorSeverity.HIGH)
        
        try:
            import xgboost as xgb
            yield xgb
        except Exception as e:
            raise MLModelError(f"XGBoost context error: {e}", severity=ErrorSeverity.HIGH, original_error=e)
    
    def get_status(self) -> Dict[str, Any]:
        """Get ML library availability status"""
        return {
            'tensorflow_available': self._tf_available,
            'tensorflow_initialized': self._tf_initialized,
            'xgboost_available': self._xgb_available,
            'environment_configured': True
        }
    
    def safe_predict(self, model_type: str, model, data, **kwargs) -> Optional[Any]:
        """Safely run ML predictions with error handling"""
        try:
            if model_type.lower() == 'tensorflow':
                with self.tensorflow_context() as tf:
                    return model.predict(data, **kwargs)
            elif model_type.lower() == 'xgboost':
                with self.xgboost_context() as xgb:
                    return model.predict(data, **kwargs)
            else:
                raise MLModelError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_type}: {e}")
            raise MLModelError(f"Prediction failed: {e}", severity=ErrorSeverity.HIGH, original_error=e)

# Global ML manager instance
ml_manager = MLLibraryManager()

def initialize_ml_libraries() -> Dict[str, bool]:
    """Initialize all ML libraries and return status"""
    logger.info("Starting ML libraries initialization", component='ml')
    
    results = {
        'tensorflow': ml_manager.initialize_tensorflow(),
        'xgboost': ml_manager.initialize_xgboost()
    }
    
    success_count = sum(results.values())
    total_count = len(results)
    
    logger.info(
        f"ML initialization completed: {success_count}/{total_count} libraries available",
        component='ml',
        **results
    )
    
    return results

def get_ml_status() -> Dict[str, Any]:
    """Get current ML library status"""
    return ml_manager.get_status()