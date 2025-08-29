"""
Safe ML Library Initialization
Provides robust initialization for TensorFlow and other ML libraries with proper error handling.
"""
import os
import sys
import threading
import multiprocessing
from contextlib import contextmanager
from typing import Optional, Dict, Any
import warnings

from src.utils.error_handling import MLModelError, ErrorSeverity, handle_errors
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

class MLLibraryManager:
    """Manages safe initialization and usage of ML libraries"""
    
    def __init__(self):
        self._tf_initialized = False
        self._tf_lock = threading.Lock()
        self._tf_available = False
        self._xgb_available = False
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup optimal environment for ML libraries"""
        # TensorFlow environment variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
        
        # Threading configuration
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1' 
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        
        # Suppress protobuf warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
        
        logger.info("ML environment configuration completed", component='ml')
    
    @handle_errors(category=ErrorSeverity.HIGH, recovery=False)
    def initialize_tensorflow(self) -> bool:
        """Safely initialize TensorFlow in single-threaded mode"""
        if self._tf_initialized:
            return self._tf_available
            
        with self._tf_lock:
            if self._tf_initialized:
                return self._tf_available
                
            try:
                logger.info("Initializing TensorFlow in safe mode", component='ml')
                
                # Import in isolated process to avoid mutex conflicts
                result = self._import_tensorflow_safe()
                
                if result:
                    self._tf_available = True
                    logger.info("TensorFlow initialized successfully", component='ml')
                else:
                    logger.warning("TensorFlow initialization failed", component='ml')
                    
                self._tf_initialized = True
                return self._tf_available
                
            except Exception as e:
                logger.error("TensorFlow initialization error", exception=e, component='ml')
                self._tf_initialized = True
                self._tf_available = False
                return False
    
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
    
    @handle_errors(category=ErrorSeverity.MEDIUM, recovery=False)
    def initialize_xgboost(self) -> bool:
        """Safely initialize XGBoost"""
        try:
            import xgboost as xgb
            
            # Test basic functionality
            import numpy as np
            data = np.array([[1, 2], [3, 4]])
            labels = np.array([0, 1])
            
            dtrain = xgb.DMatrix(data, label=labels)
            param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
            model = xgb.train(param, dtrain, 1)
            
            self._xgb_available = True
            logger.info("XGBoost initialized successfully", component='ml')
            return True
            
        except Exception as e:
            logger.error("XGBoost initialization failed", exception=e, component='ml')
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
            logger.error(f"Prediction failed for {model_type}", exception=e, component='ml')
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