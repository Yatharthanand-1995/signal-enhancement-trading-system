"""
ML Model Performance Monitoring System
Tracks model accuracy, drift, and performance degradation in real-time
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import pickle
import json
import sqlite3
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics"""
    model_name: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    prediction_drift: float
    feature_drift: float
    data_quality_score: float
    confidence_score: float
    prediction_count: int
    error_rate: float

@dataclass
class AlertConfig:
    """Configuration for model performance alerts"""
    accuracy_threshold: float = 0.85
    drift_threshold: float = 0.15
    error_rate_threshold: float = 0.10
    data_quality_threshold: float = 0.80
    window_size: int = 100
    alert_cooldown_minutes: int = 30

class ModelPerformanceMonitor:
    """
    Comprehensive ML model performance monitoring system
    Tracks accuracy, drift, and generates alerts for model degradation
    """
    
    def __init__(self, db_path: str = "ml_monitoring.db", alert_config: AlertConfig = None):
        self.db_path = db_path
        self.alert_config = alert_config or AlertConfig()
        
        # Performance tracking
        self.performance_history = {}
        self.prediction_cache = deque(maxlen=10000)
        self.feature_baseline = {}
        self.recent_alerts = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("Model performance monitor initialized")
    
    def _init_database(self):
        """Initialize monitoring database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        mse REAL,
                        mae REAL,
                        r2_score REAL,
                        prediction_drift REAL,
                        feature_drift REAL,
                        data_quality_score REAL,
                        confidence_score REAL,
                        prediction_count INTEGER,
                        error_rate REAL,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metrics TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        prediction REAL,
                        actual_value REAL,
                        confidence REAL,
                        features TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_model_time ON model_performance(model_name, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_model_time ON model_alerts(model_name, timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_predictions_model_time ON prediction_log(model_name, timestamp)')
                
        except Exception as e:
            logger.error(f"Error initializing monitoring database: {e}")
            raise
    
    def log_prediction(self, model_name: str, prediction: float, features: Dict[str, Any], 
                      confidence: float = None, actual_value: float = None, metadata: Dict = None):
        """Log a model prediction for monitoring"""
        try:
            timestamp = datetime.now()
            
            # Store in cache
            prediction_entry = {
                'model_name': model_name,
                'timestamp': timestamp,
                'prediction': prediction,
                'actual_value': actual_value,
                'confidence': confidence,
                'features': features,
                'metadata': metadata or {}
            }
            self.prediction_cache.append(prediction_entry)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO prediction_log 
                    (model_name, timestamp, prediction, actual_value, confidence, features, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_name, timestamp, prediction, actual_value, confidence,
                    json.dumps(features), json.dumps(metadata or {})
                ))
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def calculate_model_performance(self, model_name: str, window_hours: int = 24) -> Optional[ModelPerformanceMetrics]:
        """Calculate model performance metrics over a time window"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            
            # Get recent predictions with actual values
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT * FROM prediction_log 
                    WHERE model_name = ? AND timestamp >= ? AND actual_value IS NOT NULL
                    ORDER BY timestamp DESC
                ''', conn, params=(model_name, cutoff_time))
            
            if len(df) < 10:  # Need minimum predictions for meaningful metrics
                logger.warning(f"Insufficient data for {model_name}: {len(df)} predictions")
                return None
            
            predictions = df['prediction'].values
            actuals = df['actual_value'].values
            confidences = df['confidence'].fillna(0.5).values
            
            # Calculate regression metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            # Calculate classification-style metrics (assuming threshold at 0)
            pred_direction = (predictions > 0).astype(int)
            actual_direction = (actuals > 0).astype(int)
            
            accuracy = (pred_direction == actual_direction).mean()
            
            # Precision, recall, F1 for positive predictions
            true_positives = ((pred_direction == 1) & (actual_direction == 1)).sum()
            false_positives = ((pred_direction == 1) & (actual_direction == 0)).sum()
            false_negatives = ((pred_direction == 0) & (actual_direction == 1)).sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate drift metrics
            prediction_drift = self._calculate_prediction_drift(model_name, predictions)
            feature_drift = self._calculate_feature_drift(model_name, df)
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality(df)
            
            # Calculate confidence score
            confidence_score = confidences.mean()
            
            # Error rate
            error_rate = 1 - accuracy
            
            return ModelPerformanceMetrics(
                model_name=model_name,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mse=mse,
                mae=mae,
                r2_score=r2,
                prediction_drift=prediction_drift,
                feature_drift=feature_drift,
                data_quality_score=data_quality_score,
                confidence_score=confidence_score,
                prediction_count=len(df),
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance for {model_name}: {e}")
            return None
    
    def _calculate_prediction_drift(self, model_name: str, predictions: np.ndarray) -> float:
        """Calculate prediction drift compared to historical baseline"""
        try:
            # Get historical baseline
            if model_name not in self.performance_history:
                self.performance_history[model_name] = {'prediction_baseline': predictions.mean()}
                return 0.0
            
            baseline_mean = self.performance_history[model_name]['prediction_baseline']
            current_mean = predictions.mean()
            
            # Calculate relative drift
            if baseline_mean != 0:
                drift = abs(current_mean - baseline_mean) / abs(baseline_mean)
            else:
                drift = abs(current_mean)
            
            return float(drift)
            
        except Exception as e:
            logger.error(f"Error calculating prediction drift: {e}")
            return 0.0
    
    def _calculate_feature_drift(self, model_name: str, df: pd.DataFrame) -> float:
        """Calculate feature drift using feature statistics"""
        try:
            if df.empty:
                return 0.0
            
            # Parse features from JSON
            feature_dicts = df['features'].apply(json.loads).tolist()
            if not feature_dicts:
                return 0.0
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(feature_dicts)
            
            # Calculate current feature statistics
            current_stats = {}
            for col in feature_df.select_dtypes(include=[np.number]).columns:
                current_stats[col] = {
                    'mean': feature_df[col].mean(),
                    'std': feature_df[col].std()
                }
            
            # Compare with baseline
            if model_name not in self.feature_baseline:
                self.feature_baseline[model_name] = current_stats
                return 0.0
            
            baseline_stats = self.feature_baseline[model_name]
            
            # Calculate drift score
            total_drift = 0.0
            feature_count = 0
            
            for feature, current in current_stats.items():
                if feature in baseline_stats:
                    baseline = baseline_stats[feature]
                    
                    # Mean drift
                    if baseline['mean'] != 0:
                        mean_drift = abs(current['mean'] - baseline['mean']) / abs(baseline['mean'])
                    else:
                        mean_drift = abs(current['mean'])
                    
                    # Std drift
                    if baseline['std'] != 0:
                        std_drift = abs(current['std'] - baseline['std']) / baseline['std']
                    else:
                        std_drift = abs(current['std'])
                    
                    total_drift += (mean_drift + std_drift) / 2
                    feature_count += 1
            
            return total_drift / feature_count if feature_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating feature drift: {e}")
            return 0.0
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            if df.empty:
                return 0.0
            
            # Check for missing values
            missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
            
            # Check for outliers in predictions
            predictions = df['prediction']
            q75 = predictions.quantile(0.75)
            q25 = predictions.quantile(0.25)
            iqr = q75 - q25
            outlier_rate = ((predictions < (q25 - 1.5 * iqr)) | (predictions > (q75 + 1.5 * iqr))).mean()
            
            # Check confidence scores
            confidences = df['confidence'].fillna(0)
            low_confidence_rate = (confidences < 0.3).mean()
            
            # Combined quality score
            quality_score = 1.0 - (missing_rate * 0.4 + outlier_rate * 0.3 + low_confidence_rate * 0.3)
            return max(0.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return 0.5
    
    def store_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """Store performance metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance 
                    (model_name, timestamp, accuracy, precision_score, recall_score, f1_score,
                     mse, mae, r2_score, prediction_drift, feature_drift, data_quality_score,
                     confidence_score, prediction_count, error_rate, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.model_name, metrics.timestamp, metrics.accuracy, metrics.precision,
                    metrics.recall, metrics.f1_score, metrics.mse, metrics.mae, metrics.r2_score,
                    metrics.prediction_drift, metrics.feature_drift, metrics.data_quality_score,
                    metrics.confidence_score, metrics.prediction_count, metrics.error_rate,
                    json.dumps({'source': 'model_performance_monitor'})
                ))
            
            logger.info(f"Stored performance metrics for {metrics.model_name}")
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    def check_alerts(self, metrics: ModelPerformanceMetrics) -> List[Dict[str, Any]]:
        """Check for performance degradation and generate alerts"""
        alerts = []
        
        try:
            # Check accuracy threshold
            if metrics.accuracy < self.alert_config.accuracy_threshold:
                alerts.append({
                    'type': 'accuracy_degradation',
                    'severity': 'high' if metrics.accuracy < 0.7 else 'medium',
                    'message': f'Model accuracy dropped to {metrics.accuracy:.3f} (threshold: {self.alert_config.accuracy_threshold:.3f})',
                    'metric_value': metrics.accuracy,
                    'threshold': self.alert_config.accuracy_threshold
                })
            
            # Check prediction drift
            if metrics.prediction_drift > self.alert_config.drift_threshold:
                alerts.append({
                    'type': 'prediction_drift',
                    'severity': 'high' if metrics.prediction_drift > 0.3 else 'medium',
                    'message': f'High prediction drift detected: {metrics.prediction_drift:.3f}',
                    'metric_value': metrics.prediction_drift,
                    'threshold': self.alert_config.drift_threshold
                })
            
            # Check feature drift
            if metrics.feature_drift > self.alert_config.drift_threshold:
                alerts.append({
                    'type': 'feature_drift',
                    'severity': 'medium',
                    'message': f'Feature drift detected: {metrics.feature_drift:.3f}',
                    'metric_value': metrics.feature_drift,
                    'threshold': self.alert_config.drift_threshold
                })
            
            # Check error rate
            if metrics.error_rate > self.alert_config.error_rate_threshold:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'high',
                    'message': f'High error rate: {metrics.error_rate:.3f}',
                    'metric_value': metrics.error_rate,
                    'threshold': self.alert_config.error_rate_threshold
                })
            
            # Check data quality
            if metrics.data_quality_score < self.alert_config.data_quality_threshold:
                alerts.append({
                    'type': 'data_quality_issue',
                    'severity': 'medium',
                    'message': f'Data quality score low: {metrics.data_quality_score:.3f}',
                    'metric_value': metrics.data_quality_score,
                    'threshold': self.alert_config.data_quality_threshold
                })
            
            # Store alerts in database
            for alert in alerts:
                self._store_alert(metrics.model_name, alert, metrics)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
    
    def _store_alert(self, model_name: str, alert: Dict, metrics: ModelPerformanceMetrics):
        """Store alert in database"""
        try:
            # Check cooldown period
            alert_key = f"{model_name}_{alert['type']}"
            if alert_key in self.recent_alerts:
                last_alert = self.recent_alerts[alert_key]
                if (datetime.now() - last_alert).seconds < (self.alert_config.alert_cooldown_minutes * 60):
                    return  # Skip alert due to cooldown
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_alerts 
                    (model_name, timestamp, alert_type, severity, message, metrics)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_name, datetime.now(), alert['type'], alert['severity'],
                    alert['message'], json.dumps({
                        'accuracy': metrics.accuracy,
                        'prediction_drift': metrics.prediction_drift,
                        'feature_drift': metrics.feature_drift,
                        'error_rate': metrics.error_rate,
                        'data_quality_score': metrics.data_quality_score
                    })
                ))
            
            # Update cooldown tracking
            self.recent_alerts[alert_key] = datetime.now()
            
            logger.warning(f"ALERT [{alert['severity'].upper()}] {model_name}: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def get_model_health_status(self, model_name: str) -> Dict[str, Any]:
        """Get current health status for a model"""
        try:
            # Get recent performance
            metrics = self.calculate_model_performance(model_name, window_hours=1)
            if not metrics:
                return {'status': 'unknown', 'message': 'Insufficient data'}
            
            # Determine health status
            issues = []
            if metrics.accuracy < self.alert_config.accuracy_threshold:
                issues.append('low_accuracy')
            if metrics.prediction_drift > self.alert_config.drift_threshold:
                issues.append('prediction_drift')
            if metrics.feature_drift > self.alert_config.drift_threshold:
                issues.append('feature_drift')
            if metrics.error_rate > self.alert_config.error_rate_threshold:
                issues.append('high_error_rate')
            if metrics.data_quality_score < self.alert_config.data_quality_threshold:
                issues.append('data_quality')
            
            if not issues:
                status = 'healthy'
                message = 'Model performing within expected parameters'
            elif len(issues) == 1:
                status = 'warning'
                message = f'Minor issue detected: {issues[0]}'
            else:
                status = 'critical'
                message = f'Multiple issues detected: {", ".join(issues)}'
            
            return {
                'status': status,
                'message': message,
                'issues': issues,
                'metrics': {
                    'accuracy': metrics.accuracy,
                    'error_rate': metrics.error_rate,
                    'prediction_drift': metrics.prediction_drift,
                    'feature_drift': metrics.feature_drift,
                    'data_quality_score': metrics.data_quality_score,
                    'prediction_count': metrics.prediction_count
                },
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status for {model_name}: {e}")
            return {'status': 'error', 'message': f'Error: {str(e)}'}
    
    def get_performance_history(self, model_name: str, days: int = 7) -> pd.DataFrame:
        """Get performance history for a model"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                    SELECT * FROM model_performance 
                    WHERE model_name = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                ''', conn, params=(model_name, cutoff_time))
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def monitor_model(self, model_name: str) -> Dict[str, Any]:
        """Full monitoring cycle for a model"""
        try:
            # Calculate current performance
            metrics = self.calculate_model_performance(model_name)
            if not metrics:
                return {'status': 'no_data', 'message': 'Insufficient prediction data'}
            
            # Store metrics
            self.store_performance_metrics(metrics)
            
            # Check for alerts
            alerts = self.check_alerts(metrics)
            
            # Get health status
            health = self.get_model_health_status(model_name)
            
            return {
                'status': 'success',
                'model_name': model_name,
                'metrics': metrics,
                'alerts': alerts,
                'health': health,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring model {model_name}: {e}")
            return {'status': 'error', 'message': str(e)}