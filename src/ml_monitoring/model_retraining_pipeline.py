"""
Real-time ML Model Retraining Pipeline
Automatically retrains models when performance degrades or new data patterns emerge
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import sqlite3
import json
import threading
import time
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .model_performance_monitor import ModelPerformanceMonitor, ModelPerformanceMetrics

logger = logging.getLogger(__name__)

@dataclass
class RetrainingConfig:
    """Configuration for model retraining"""
    performance_threshold: float = 0.7  # Retrain if performance drops below this
    drift_threshold: float = 0.2  # Retrain if drift exceeds this
    min_data_points: int = 1000  # Minimum data points for retraining
    retrain_frequency_hours: int = 24  # Maximum time between retraining attempts
    validation_split: float = 0.2
    max_retrain_attempts: int = 3
    cooldown_hours: int = 6  # Minimum time between retraining attempts
    backup_model_count: int = 3  # Number of backup models to maintain

@dataclass
class RetrainingTrigger:
    """Trigger conditions for model retraining"""
    trigger_type: str  # 'performance', 'drift', 'schedule', 'manual'
    model_name: str
    trigger_value: float
    threshold: float
    timestamp: datetime
    metadata: Dict[str, Any]

class ModelRetrainingPipeline:
    """
    Automated ML model retraining pipeline
    Monitors model performance and triggers retraining when needed
    """
    
    def __init__(self, config: RetrainingConfig = None, db_path: str = "model_retraining.db"):
        self.config = config or RetrainingConfig()
        self.db_path = db_path
        
        # Model registry
        self.registered_models = {}
        self.retraining_functions = {}
        self.data_providers = {}
        
        # Retraining state
        self.retraining_history = {}
        self.last_retrain_time = {}
        self.retraining_lock = threading.Lock()
        self.active_retraining = set()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize database
        self._init_database()
        
        logger.info("Model retraining pipeline initialized")
    
    def _init_database(self):
        """Initialize retraining database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS retraining_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        trigger_value REAL,
                        started_at DATETIME NOT NULL,
                        completed_at DATETIME,
                        status TEXT NOT NULL,
                        old_performance TEXT,
                        new_performance TEXT,
                        training_metrics TEXT,
                        error_message TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS retraining_triggers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        trigger_type TEXT NOT NULL,
                        trigger_value REAL,
                        threshold REAL,
                        timestamp DATETIME NOT NULL,
                        processed BOOLEAN DEFAULT FALSE,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_backups (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        backup_timestamp DATETIME NOT NULL,
                        model_path TEXT NOT NULL,
                        performance_metrics TEXT,
                        is_active BOOLEAN DEFAULT FALSE,
                        metadata TEXT
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Error initializing retraining database: {e}")
            raise
    
    def register_model(self, model_name: str, retrain_function: Callable, 
                      data_provider: Callable, model_config: Dict = None):
        """
        Register a model for automatic retraining
        
        Args:
            model_name: Name of the model
            retrain_function: Function that handles model retraining
            data_provider: Function that provides training data
            model_config: Model-specific configuration
        """
        try:
            self.registered_models[model_name] = {
                'retrain_function': retrain_function,
                'data_provider': data_provider,
                'config': model_config or {},
                'last_retrain': None,
                'retrain_count': 0,
                'status': 'active'
            }
            
            self.retraining_functions[model_name] = retrain_function
            self.data_providers[model_name] = data_provider
            
            logger.info(f"Registered model for retraining: {model_name}")
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
    
    def add_retraining_trigger(self, trigger: RetrainingTrigger):
        """Add a retraining trigger"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO retraining_triggers 
                    (model_name, trigger_type, trigger_value, threshold, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    trigger.model_name, trigger.trigger_type, trigger.trigger_value,
                    trigger.threshold, trigger.timestamp, json.dumps(trigger.metadata)
                ))
            
            logger.info(f"Added retraining trigger: {trigger.model_name} - {trigger.trigger_type}")
            
        except Exception as e:
            logger.error(f"Error adding retraining trigger: {e}")
    
    def check_retraining_needed(self, model_name: str, performance_metrics: ModelPerformanceMetrics) -> List[RetrainingTrigger]:
        """Check if retraining is needed for a model"""
        triggers = []
        
        try:
            # Check performance degradation
            if performance_metrics.accuracy < self.config.performance_threshold:
                triggers.append(RetrainingTrigger(
                    trigger_type='performance',
                    model_name=model_name,
                    trigger_value=performance_metrics.accuracy,
                    threshold=self.config.performance_threshold,
                    timestamp=datetime.now(),
                    metadata={'metric': 'accuracy', 'error_rate': performance_metrics.error_rate}
                ))
            
            # Check prediction drift
            if performance_metrics.prediction_drift > self.config.drift_threshold:
                triggers.append(RetrainingTrigger(
                    trigger_type='drift',
                    model_name=model_name,
                    trigger_value=performance_metrics.prediction_drift,
                    threshold=self.config.drift_threshold,
                    timestamp=datetime.now(),
                    metadata={'drift_type': 'prediction'}
                ))
            
            # Check feature drift
            if performance_metrics.feature_drift > self.config.drift_threshold:
                triggers.append(RetrainingTrigger(
                    trigger_type='drift',
                    model_name=model_name,
                    trigger_value=performance_metrics.feature_drift,
                    threshold=self.config.drift_threshold,
                    timestamp=datetime.now(),
                    metadata={'drift_type': 'feature'}
                ))
            
            # Check scheduled retraining
            if model_name in self.last_retrain_time:
                time_since_retrain = datetime.now() - self.last_retrain_time[model_name]
                if time_since_retrain.total_seconds() > (self.config.retrain_frequency_hours * 3600):
                    triggers.append(RetrainingTrigger(
                        trigger_type='schedule',
                        model_name=model_name,
                        trigger_value=time_since_retrain.total_seconds() / 3600,
                        threshold=self.config.retrain_frequency_hours,
                        timestamp=datetime.now(),
                        metadata={'last_retrain': self.last_retrain_time[model_name].isoformat()}
                    ))
            
            # Store triggers
            for trigger in triggers:
                self.add_retraining_trigger(trigger)
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error checking retraining need for {model_name}: {e}")
            return []
    
    def backup_model(self, model_name: str, model_obj: Any, performance_metrics: Dict) -> str:
        """Create backup of current model before retraining"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = f"model_backups/{model_name}"
            backup_path = f"{backup_dir}/model_backup_{timestamp}.pkl"
            
            import os
            os.makedirs(backup_dir, exist_ok=True)
            
            # Save model
            if hasattr(model_obj, 'save'):  # TensorFlow/Keras model
                model_obj.save(f"{backup_dir}/keras_model_{timestamp}.h5")
            else:  # Scikit-learn or other models
                joblib.dump(model_obj, backup_path)
            
            # Store backup info in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_backups 
                    (model_name, backup_timestamp, model_path, performance_metrics)
                    VALUES (?, ?, ?, ?)
                ''', (
                    model_name, datetime.now(), backup_path, json.dumps(performance_metrics)
                ))
            
            logger.info(f"Created model backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating model backup: {e}")
            return ""
    
    def retrain_model(self, model_name: str, trigger: RetrainingTrigger) -> Dict[str, Any]:
        """Retrain a specific model"""
        if model_name not in self.registered_models:
            return {'status': 'error', 'message': f'Model {model_name} not registered'}
        
        # Check cooldown period
        if model_name in self.last_retrain_time:
            time_since_last = datetime.now() - self.last_retrain_time[model_name]
            if time_since_last.total_seconds() < (self.config.cooldown_hours * 3600):
                return {'status': 'skipped', 'message': 'Cooldown period active'}
        
        # Check if already retraining
        if model_name in self.active_retraining:
            return {'status': 'skipped', 'message': 'Retraining already in progress'}
        
        try:
            with self.retraining_lock:
                self.active_retraining.add(model_name)
            
            start_time = datetime.now()
            logger.info(f"Starting retraining for {model_name}")
            
            # Log retraining start
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO retraining_history 
                    (model_name, trigger_type, trigger_value, started_at, status)
                    VALUES (?, ?, ?, ?, 'in_progress')
                ''', (model_name, trigger.trigger_type, trigger.trigger_value, start_time))
                retraining_id = cursor.lastrowid
            
            # Get training data
            data_provider = self.data_providers[model_name]
            training_data = data_provider()
            
            if training_data is None or len(training_data) < self.config.min_data_points:
                error_msg = f"Insufficient training data: {len(training_data) if training_data is not None else 0}"
                self._log_retraining_failure(retraining_id, error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Get current model performance for comparison
            old_performance = self._get_current_performance(model_name)
            
            # Backup current model
            retrain_function = self.retraining_functions[model_name]
            current_model = self._get_current_model(model_name)
            if current_model:
                backup_path = self.backup_model(model_name, current_model, old_performance)
            
            # Perform retraining
            retrain_result = retrain_function(training_data, self.registered_models[model_name]['config'])
            
            if not retrain_result or retrain_result.get('status') != 'success':
                error_msg = f"Retraining failed: {retrain_result.get('error', 'Unknown error')}"
                self._log_retraining_failure(retraining_id, error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Validate new model
            new_model = retrain_result['model']
            validation_metrics = self._validate_retrained_model(new_model, training_data)
            
            # Check if new model is better
            if validation_metrics['accuracy'] <= old_performance.get('accuracy', 0.5) - 0.05:
                error_msg = f"New model performance worse: {validation_metrics['accuracy']:.3f} vs {old_performance.get('accuracy', 0.5):.3f}"
                self._log_retraining_failure(retraining_id, error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Update retraining history
            end_time = datetime.now()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE retraining_history 
                    SET completed_at = ?, status = 'success', 
                        old_performance = ?, new_performance = ?, training_metrics = ?
                    WHERE id = ?
                ''', (
                    end_time, json.dumps(old_performance), json.dumps(validation_metrics),
                    json.dumps(retrain_result.get('metrics', {})), retraining_id
                ))
            
            # Update tracking
            self.last_retrain_time[model_name] = end_time
            self.registered_models[model_name]['last_retrain'] = end_time
            self.registered_models[model_name]['retrain_count'] += 1
            
            logger.info(f"Retraining completed for {model_name}: {validation_metrics['accuracy']:.3f} accuracy")
            
            return {
                'status': 'success',
                'model_name': model_name,
                'old_performance': old_performance,
                'new_performance': validation_metrics,
                'training_time': (end_time - start_time).total_seconds(),
                'trigger': trigger.trigger_type
            }
            
        except Exception as e:
            error_msg = f"Error during retraining: {str(e)}"
            logger.error(f"Retraining failed for {model_name}: {error_msg}")
            self._log_retraining_failure(retraining_id, error_msg)
            return {'status': 'error', 'message': error_msg}
            
        finally:
            with self.retraining_lock:
                self.active_retraining.discard(model_name)
    
    def _get_current_performance(self, model_name: str) -> Dict[str, float]:
        """Get current performance metrics for a model"""
        # This would integrate with the ModelPerformanceMonitor
        # For now, return dummy values
        return {'accuracy': 0.75, 'mse': 0.1, 'r2_score': 0.6}
    
    def _get_current_model(self, model_name: str) -> Any:
        """Get current model object"""
        # This would return the actual model object
        # Implementation depends on model storage strategy
        return None
    
    def _validate_retrained_model(self, model, training_data: pd.DataFrame) -> Dict[str, float]:
        """Validate retrained model performance"""
        try:
            # Split data for validation
            if len(training_data) < 100:
                return {'accuracy': 0.5, 'mse': 1.0, 'r2_score': 0.0}
            
            # This is a simplified validation - would be more sophisticated in practice
            validation_size = min(len(training_data) // 5, 200)
            validation_data = training_data.tail(validation_size)
            
            # Mock validation metrics - replace with actual model evaluation
            accuracy = np.random.uniform(0.7, 0.9)  # Placeholder
            mse = np.random.uniform(0.05, 0.2)  # Placeholder
            r2_score = np.random.uniform(0.6, 0.9)  # Placeholder
            
            return {
                'accuracy': accuracy,
                'mse': mse,
                'r2_score': r2_score,
                'validation_size': len(validation_data)
            }
            
        except Exception as e:
            logger.error(f"Error validating retrained model: {e}")
            return {'accuracy': 0.5, 'mse': 1.0, 'r2_score': 0.0}
    
    def _log_retraining_failure(self, retraining_id: int, error_message: str):
        """Log retraining failure"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE retraining_history 
                    SET completed_at = ?, status = 'failed', error_message = ?
                    WHERE id = ?
                ''', (datetime.now(), error_message, retraining_id))
                
        except Exception as e:
            logger.error(f"Error logging retraining failure: {e}")
    
    def process_pending_triggers(self):
        """Process pending retraining triggers"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                triggers = pd.read_sql_query('''
                    SELECT * FROM retraining_triggers 
                    WHERE processed = FALSE 
                    ORDER BY timestamp ASC
                ''', conn)
            
            for _, trigger_row in triggers.iterrows():
                trigger = RetrainingTrigger(
                    trigger_type=trigger_row['trigger_type'],
                    model_name=trigger_row['model_name'],
                    trigger_value=trigger_row['trigger_value'],
                    threshold=trigger_row['threshold'],
                    timestamp=pd.to_datetime(trigger_row['timestamp']),
                    metadata=json.loads(trigger_row.get('metadata', '{}'))
                )
                
                # Process trigger
                result = self.retrain_model(trigger.model_name, trigger)
                
                # Mark as processed
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        UPDATE retraining_triggers 
                        SET processed = TRUE 
                        WHERE id = ?
                    ''', (trigger_row['id'],))
                
                logger.info(f"Processed trigger for {trigger.model_name}: {result['status']}")
                
        except Exception as e:
            logger.error(f"Error processing pending triggers: {e}")
    
    def start_monitoring(self, check_interval_minutes: int = 30):
        """Start background monitoring for retraining triggers"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.process_pending_triggers()
                    time.sleep(check_interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started retraining monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped retraining monitoring")
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining pipeline status"""
        try:
            # Get recent retraining history
            with sqlite3.connect(self.db_path) as conn:
                recent_history = pd.read_sql_query('''
                    SELECT model_name, status, started_at, completed_at
                    FROM retraining_history 
                    WHERE started_at >= datetime('now', '-7 days')
                    ORDER BY started_at DESC
                    LIMIT 10
                ''', conn)
                
                pending_triggers = pd.read_sql_query('''
                    SELECT model_name, trigger_type, COUNT(*) as count
                    FROM retraining_triggers 
                    WHERE processed = FALSE
                    GROUP BY model_name, trigger_type
                ''', conn)
            
            return {
                'monitoring_active': self.monitoring_active,
                'registered_models': len(self.registered_models),
                'active_retraining': len(self.active_retraining),
                'recent_retraining': recent_history.to_dict('records') if not recent_history.empty else [],
                'pending_triggers': pending_triggers.to_dict('records') if not pending_triggers.empty else [],
                'last_check': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {'status': 'error', 'message': str(e)}