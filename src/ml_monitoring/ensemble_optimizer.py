"""
ML Ensemble Optimizer
Advanced ensemble management with dynamic weighting and model selection
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sqlite3
import json
from collections import defaultdict, deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Container for model predictions"""
    model_name: str
    prediction: float
    confidence: float
    timestamp: datetime
    features: Dict[str, Any]
    model_version: str

@dataclass
class EnsembleConfig:
    """Configuration for ensemble optimization"""
    min_models: int = 2
    max_models: int = 5
    performance_window_hours: int = 24
    rebalance_frequency_hours: int = 6
    min_confidence_threshold: float = 0.3
    weight_decay: float = 0.95
    performance_weight: float = 0.6
    diversity_weight: float = 0.4

class EnsembleOptimizer:
    """
    Advanced ensemble optimizer for ML model predictions
    Dynamically adjusts model weights based on performance and diversity
    """
    
    def __init__(self, config: EnsembleConfig = None, db_path: str = "ensemble_optimization.db"):
        self.config = config or EnsembleConfig()
        self.db_path = db_path
        
        # Model tracking
        self.active_models = {}
        self.model_weights = {}
        self.model_performance = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_cache = defaultdict(lambda: deque(maxlen=1000))
        
        # Ensemble optimization
        self.meta_learner = None
        self.weight_history = defaultdict(list)
        self.optimization_lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        logger.info("Ensemble optimizer initialized")
    
    def _init_database(self):
        """Initialize ensemble optimization database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS ensemble_weights (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        model_name TEXT NOT NULL,
                        weight REAL NOT NULL,
                        performance_score REAL,
                        confidence_score REAL,
                        diversity_score REAL,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS ensemble_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT,
                        ensemble_prediction REAL NOT NULL,
                        individual_predictions TEXT,
                        weights TEXT,
                        confidence REAL,
                        actual_value REAL,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT UNIQUE NOT NULL,
                        model_type TEXT,
                        version TEXT,
                        created_date DATETIME,
                        last_updated DATETIME,
                        status TEXT DEFAULT 'active',
                        performance_metrics TEXT,
                        metadata TEXT
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Error initializing ensemble database: {e}")
            raise
    
    def register_model(self, model_name: str, model_type: str, version: str, metadata: Dict = None):
        """Register a new model in the ensemble"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO model_registry 
                    (model_name, model_type, version, created_date, last_updated, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_name, model_type, version, datetime.now(), datetime.now(),
                    json.dumps(metadata or {})
                ))
            
            # Initialize with equal weights
            if model_name not in self.model_weights:
                self.model_weights[model_name] = 1.0 / max(len(self.active_models) + 1, 1)
                self.active_models[model_name] = {
                    'type': model_type,
                    'version': version,
                    'status': 'active',
                    'last_prediction': None
                }
            
            logger.info(f"Registered model: {model_name} (type: {model_type}, version: {version})")
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
    
    def add_prediction(self, prediction: ModelPrediction):
        """Add a prediction from an individual model"""
        try:
            # Store prediction
            self.prediction_cache[prediction.model_name].append({
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'timestamp': prediction.timestamp,
                'features': prediction.features
            })
            
            # Update model status
            if prediction.model_name in self.active_models:
                self.active_models[prediction.model_name]['last_prediction'] = prediction.timestamp
            
        except Exception as e:
            logger.error(f"Error adding prediction from {prediction.model_name}: {e}")
    
    def calculate_model_performance(self, model_name: str, window_hours: int = None) -> Dict[str, float]:
        """Calculate performance metrics for a single model"""
        try:
            window_hours = window_hours or self.config.performance_window_hours
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            
            # Get recent predictions
            predictions = [
                p for p in self.prediction_cache[model_name] 
                if p['timestamp'] >= cutoff_time
            ]
            
            if len(predictions) < 5:  # Need minimum predictions
                return {'accuracy': 0.5, 'consistency': 0.5, 'confidence': 0.5}
            
            pred_values = [p['prediction'] for p in predictions]
            confidences = [p['confidence'] for p in predictions]
            
            # Calculate metrics
            accuracy = self._calculate_accuracy_proxy(pred_values)
            consistency = 1.0 - np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8)
            avg_confidence = np.mean(confidences)
            
            return {
                'accuracy': float(np.clip(accuracy, 0.0, 1.0)),
                'consistency': float(np.clip(consistency, 0.0, 1.0)),
                'confidence': float(np.clip(avg_confidence, 0.0, 1.0))
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance for {model_name}: {e}")
            return {'accuracy': 0.5, 'consistency': 0.5, 'confidence': 0.5}
    
    def _calculate_accuracy_proxy(self, predictions: List[float]) -> float:
        """Calculate accuracy proxy when actual values aren't available"""
        try:
            # Use prediction stability and magnitude as proxies
            if len(predictions) < 2:
                return 0.5
            
            # Prediction stability (lower variance = higher stability)
            stability = 1.0 / (1.0 + np.var(predictions))
            
            # Magnitude reasonableness (predictions within reasonable ranges)
            reasonable_range = np.sum(np.abs(predictions) < 0.5) / len(predictions)
            
            # Trend consistency
            if len(predictions) > 5:
                recent = predictions[-5:]
                trend_consistency = 1.0 - np.std(np.diff(recent)) / (np.mean(np.abs(recent)) + 1e-8)
                trend_consistency = np.clip(trend_consistency, 0, 1)
            else:
                trend_consistency = 0.5
            
            # Combined accuracy proxy
            accuracy_proxy = (stability * 0.4 + reasonable_range * 0.3 + trend_consistency * 0.3)
            return float(np.clip(accuracy_proxy, 0.1, 0.9))
            
        except Exception as e:
            logger.error(f"Error calculating accuracy proxy: {e}")
            return 0.5
    
    def calculate_model_diversity(self, model_predictions: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate diversity scores for models"""
        try:
            diversity_scores = {}
            model_names = list(model_predictions.keys())
            
            for model_name in model_names:
                model_preds = np.array(model_predictions[model_name])
                other_models = [m for m in model_names if m != model_name]
                
                if not other_models:
                    diversity_scores[model_name] = 1.0
                    continue
                
                # Calculate correlation with other models
                correlations = []
                for other_model in other_models:
                    other_preds = np.array(model_predictions[other_model])
                    min_len = min(len(model_preds), len(other_preds))
                    
                    if min_len > 1:
                        corr = np.corrcoef(model_preds[-min_len:], other_preds[-min_len:])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                # Diversity = 1 - average correlation
                avg_correlation = np.mean(correlations) if correlations else 0.5
                diversity = 1.0 - avg_correlation
                diversity_scores[model_name] = float(np.clip(diversity, 0.1, 1.0))
            
            return diversity_scores
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return {name: 0.5 for name in model_predictions.keys()}
    
    def optimize_weights(self) -> Dict[str, float]:
        """Optimize ensemble weights based on performance and diversity"""
        try:
            with self.optimization_lock:
                # Get recent predictions for all active models
                model_predictions = {}
                cutoff_time = datetime.now() - timedelta(hours=self.config.performance_window_hours)
                
                for model_name in self.active_models:
                    predictions = [
                        p['prediction'] for p in self.prediction_cache[model_name] 
                        if p['timestamp'] >= cutoff_time
                    ]
                    
                    if len(predictions) >= 3:  # Minimum predictions required
                        model_predictions[model_name] = predictions[-50:]  # Last 50 predictions
                
                if len(model_predictions) < self.config.min_models:
                    logger.warning(f"Insufficient models for optimization: {len(model_predictions)}")
                    return self.model_weights
                
                # Calculate performance scores
                performance_scores = {}
                for model_name in model_predictions:
                    perf = self.calculate_model_performance(model_name)
                    performance_scores[model_name] = (
                        perf['accuracy'] * 0.5 + 
                        perf['consistency'] * 0.3 + 
                        perf['confidence'] * 0.2
                    )
                
                # Calculate diversity scores
                diversity_scores = self.calculate_model_diversity(model_predictions)
                
                # Optimize weights using constrained optimization
                model_names = list(model_predictions.keys())
                n_models = len(model_names)
                
                # Objective function: maximize weighted combination of performance and diversity
                def objective(weights):
                    total_score = 0.0
                    for i, model_name in enumerate(model_names):
                        perf_score = performance_scores[model_name]
                        div_score = diversity_scores[model_name]
                        combined_score = (
                            self.config.performance_weight * perf_score + 
                            self.config.diversity_weight * div_score
                        )
                        total_score += weights[i] * combined_score
                    return -total_score  # Minimize negative = maximize positive
                
                # Constraints: weights sum to 1, all weights >= 0
                constraints = [
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                ]
                bounds = [(0.05, 0.8) for _ in range(n_models)]  # Min 5%, max 80% weight
                
                # Initial guess: current weights or equal weights
                initial_weights = []
                for model_name in model_names:
                    weight = self.model_weights.get(model_name, 1.0 / n_models)
                    initial_weights.append(weight)
                initial_weights = np.array(initial_weights) / sum(initial_weights)
                
                # Optimize
                result = minimize(
                    objective, 
                    initial_weights, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100}
                )
                
                if result.success:
                    # Update weights
                    optimized_weights = {}
                    for i, model_name in enumerate(model_names):
                        optimized_weights[model_name] = float(result.x[i])
                    
                    # Apply weight decay to inactive models
                    for model_name in self.model_weights:
                        if model_name not in optimized_weights:
                            self.model_weights[model_name] *= self.config.weight_decay
                    
                    self.model_weights.update(optimized_weights)
                    
                    # Store weight update
                    self._store_weight_update(performance_scores, diversity_scores)
                    
                    logger.info(f"Optimized weights: {optimized_weights}")
                    return optimized_weights
                
                else:
                    logger.warning(f"Weight optimization failed: {result.message}")
                    return self.model_weights
                    
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
            return self.model_weights
    
    def _store_weight_update(self, performance_scores: Dict, diversity_scores: Dict):
        """Store weight update in database"""
        try:
            timestamp = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                for model_name, weight in self.model_weights.items():
                    perf_score = performance_scores.get(model_name, 0.0)
                    div_score = diversity_scores.get(model_name, 0.0)
                    
                    conn.execute('''
                        INSERT INTO ensemble_weights 
                        (timestamp, model_name, weight, performance_score, diversity_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (timestamp, model_name, weight, perf_score, div_score))
            
        except Exception as e:
            logger.error(f"Error storing weight update: {e}")
    
    def generate_ensemble_prediction(self, symbol: str, individual_predictions: Dict[str, ModelPrediction]) -> Tuple[float, float, Dict]:
        """Generate ensemble prediction from individual model predictions"""
        try:
            if not individual_predictions:
                return 0.0, 0.0, {'error': 'No individual predictions'}
            
            # Filter predictions by confidence threshold
            valid_predictions = {
                name: pred for name, pred in individual_predictions.items()
                if pred.confidence >= self.config.min_confidence_threshold
            }
            
            if not valid_predictions:
                # Use all predictions if none meet confidence threshold
                valid_predictions = individual_predictions
            
            # Calculate weighted ensemble prediction
            total_weight = 0.0
            weighted_prediction = 0.0
            weighted_confidence = 0.0
            
            prediction_details = {}
            
            for model_name, prediction in valid_predictions.items():
                weight = self.model_weights.get(model_name, 1.0 / len(valid_predictions))
                
                weighted_prediction += weight * prediction.prediction
                weighted_confidence += weight * prediction.confidence
                total_weight += weight
                
                prediction_details[model_name] = {
                    'prediction': prediction.prediction,
                    'confidence': prediction.confidence,
                    'weight': weight
                }
            
            # Normalize
            if total_weight > 0:
                ensemble_prediction = weighted_prediction / total_weight
                ensemble_confidence = weighted_confidence / total_weight
            else:
                ensemble_prediction = 0.0
                ensemble_confidence = 0.0
            
            # Calculate prediction uncertainty
            pred_values = [p.prediction for p in valid_predictions.values()]
            uncertainty = np.std(pred_values) if len(pred_values) > 1 else 0.0
            
            # Adjust confidence based on model agreement
            agreement_factor = max(0.5, 1.0 - uncertainty)
            final_confidence = ensemble_confidence * agreement_factor
            
            # Store ensemble prediction
            self._store_ensemble_prediction(symbol, ensemble_prediction, prediction_details, final_confidence)
            
            metadata = {
                'model_count': len(valid_predictions),
                'total_weight': total_weight,
                'uncertainty': uncertainty,
                'agreement_factor': agreement_factor,
                'individual_predictions': prediction_details
            }
            
            return float(ensemble_prediction), float(final_confidence), metadata
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {e}")
            return 0.0, 0.0, {'error': str(e)}
    
    def _store_ensemble_prediction(self, symbol: str, prediction: float, individual_preds: Dict, confidence: float):
        """Store ensemble prediction in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO ensemble_predictions 
                    (timestamp, symbol, ensemble_prediction, individual_predictions, weights, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(), symbol, prediction, 
                    json.dumps(individual_preds), 
                    json.dumps(dict(self.model_weights)), 
                    confidence
                ))
            
        except Exception as e:
            logger.error(f"Error storing ensemble prediction: {e}")
    
    def get_model_rankings(self) -> List[Dict]:
        """Get current model rankings based on performance"""
        try:
            rankings = []
            
            for model_name in self.active_models:
                perf = self.calculate_model_performance(model_name)
                weight = self.model_weights.get(model_name, 0.0)
                
                # Get recent prediction count
                cutoff_time = datetime.now() - timedelta(hours=24)
                recent_predictions = len([
                    p for p in self.prediction_cache[model_name] 
                    if p['timestamp'] >= cutoff_time
                ])
                
                rankings.append({
                    'model_name': model_name,
                    'weight': weight,
                    'performance_score': (perf['accuracy'] + perf['consistency'] + perf['confidence']) / 3,
                    'accuracy': perf['accuracy'],
                    'consistency': perf['consistency'],
                    'confidence': perf['confidence'],
                    'recent_predictions': recent_predictions,
                    'status': self.active_models[model_name]['status']
                })
            
            # Sort by performance score
            rankings.sort(key=lambda x: x['performance_score'], reverse=True)
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting model rankings: {e}")
            return []
    
    def rebalance_ensemble(self):
        """Periodic rebalancing of ensemble weights"""
        try:
            logger.info("Starting ensemble rebalancing...")
            
            # Optimize weights
            new_weights = self.optimize_weights()
            
            # Get model rankings
            rankings = self.get_model_rankings()
            
            # Remove underperforming models if we have too many
            if len(self.active_models) > self.config.max_models:
                # Sort by performance and remove worst performers
                sorted_models = sorted(rankings, key=lambda x: x['performance_score'])
                models_to_remove = sorted_models[:len(self.active_models) - self.config.max_models]
                
                for model_info in models_to_remove:
                    model_name = model_info['model_name']
                    if model_info['performance_score'] < 0.3:  # Only remove very poor performers
                        self.active_models[model_name]['status'] = 'inactive'
                        self.model_weights[model_name] = 0.0
                        logger.info(f"Deactivated underperforming model: {model_name}")
            
            logger.info("Ensemble rebalancing completed")
            return {
                'weights': new_weights,
                'rankings': rankings,
                'active_models': len([m for m in self.active_models.values() if m['status'] == 'active'])
            }
            
        except Exception as e:
            logger.error(f"Error during ensemble rebalancing: {e}")
            return {'error': str(e)}
    
    def get_ensemble_health(self) -> Dict[str, Any]:
        """Get current ensemble health status"""
        try:
            active_models = len([m for m in self.active_models.values() if m['status'] == 'active'])
            total_models = len(self.active_models)
            
            # Calculate average performance
            rankings = self.get_model_rankings()
            avg_performance = np.mean([r['performance_score'] for r in rankings]) if rankings else 0.0
            
            # Check for recent predictions
            recent_predictions = sum([
                len([p for p in self.prediction_cache[model_name] 
                    if p['timestamp'] >= datetime.now() - timedelta(hours=1)])
                for model_name in self.active_models
            ])
            
            # Determine health status
            if active_models < self.config.min_models:
                status = 'critical'
                message = f'Insufficient active models: {active_models}/{self.config.min_models}'
            elif avg_performance < 0.4:
                status = 'warning'
                message = f'Low average performance: {avg_performance:.3f}'
            elif recent_predictions == 0:
                status = 'warning'
                message = 'No recent predictions'
            else:
                status = 'healthy'
                message = 'Ensemble operating normally'
            
            return {
                'status': status,
                'message': message,
                'active_models': active_models,
                'total_models': total_models,
                'average_performance': avg_performance,
                'recent_predictions': recent_predictions,
                'top_models': rankings[:3],
                'last_rebalance': datetime.now()  # Would track actual rebalance time in production
            }
            
        except Exception as e:
            logger.error(f"Error getting ensemble health: {e}")
            return {'status': 'error', 'message': str(e)}