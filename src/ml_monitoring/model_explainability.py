"""
ML Model Explainability System
Provides interpretable explanations for model predictions using SHAP and LIME
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Optional SHAP import (would be installed in production)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available - using alternative explanations")

# Optional LIME import (would be installed in production)
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    logger.warning("LIME not available - using alternative explanations")

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportance:
    """Feature importance information"""
    feature_name: str
    importance_score: float
    importance_type: str  # 'shap', 'lime', 'permutation', 'builtin'
    confidence: float
    direction: str  # 'positive', 'negative', 'neutral'

@dataclass
class PredictionExplanation:
    """Explanation for a single prediction"""
    prediction_id: str
    model_name: str
    prediction_value: float
    base_value: float  # Expected value without features
    feature_contributions: List[FeatureImportance]
    explanation_method: str
    confidence_score: float
    created_at: datetime
    metadata: Dict[str, Any]

class ModelExplainabilitySystem:
    """
    Comprehensive model explainability system
    Provides interpretable explanations using multiple methods
    """
    
    def __init__(self, db_path: str = "model_explainability.db"):
        self.db_path = db_path
        
        # Explainers cache
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.surrogate_models = {}
        
        # Feature importance cache
        self.feature_importance_cache = {}
        self.global_importance = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("Model explainability system initialized")
    
    def _init_database(self):
        """Initialize explainability database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_explanations (
                        prediction_id TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        prediction_value REAL NOT NULL,
                        base_value REAL,
                        explanation_method TEXT NOT NULL,
                        confidence_score REAL,
                        feature_contributions TEXT,
                        created_at DATETIME NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS feature_importance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        feature_name TEXT NOT NULL,
                        importance_score REAL NOT NULL,
                        importance_type TEXT NOT NULL,
                        confidence REAL,
                        direction TEXT,
                        computed_at DATETIME NOT NULL,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS explanation_requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        request_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        requested_at DATETIME NOT NULL,
                        completed_at DATETIME,
                        result_summary TEXT,
                        error_message TEXT
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Error initializing explainability database: {e}")
            raise
    
    def setup_shap_explainer(self, model_name: str, model, training_data: pd.DataFrame):
        """Setup SHAP explainer for a model"""
        if not HAS_SHAP:
            logger.warning("SHAP not available - skipping SHAP explainer setup")
            return False
        
        try:
            # Select background data (sample for efficiency)
            background_size = min(100, len(training_data))
            background_data = training_data.sample(n=background_size, random_state=42)
            
            # Choose appropriate explainer based on model type
            model_type = type(model).__name__.lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type:
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            elif 'linear' in model_type:
                # Linear models
                explainer = shap.LinearExplainer(model, background_data)
            else:
                # General explainer (slower but works with any model)
                explainer = shap.Explainer(model, background_data)
            
            self.shap_explainers[model_name] = explainer
            
            logger.info(f"SHAP explainer setup for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer for {model_name}: {e}")
            return False
    
    def setup_lime_explainer(self, model_name: str, training_data: pd.DataFrame, feature_names: List[str]):
        """Setup LIME explainer for a model"""
        if not HAS_LIME:
            logger.warning("LIME not available - skipping LIME explainer setup")
            return False
        
        try:
            # Setup LIME tabular explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data.values,
                feature_names=feature_names,
                mode='regression',  # or 'classification' based on task
                discretize_continuous=True,
                random_state=42
            )
            
            self.lime_explainers[model_name] = explainer
            
            logger.info(f"LIME explainer setup for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up LIME explainer for {model_name}: {e}")
            return False
    
    def create_surrogate_model(self, model_name: str, black_box_model, 
                             training_data: pd.DataFrame, target_data: pd.Series):
        """Create interpretable surrogate model"""
        try:
            # Use RandomForest as surrogate (inherently interpretable)
            surrogate = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Generate predictions from black box model
            black_box_predictions = black_box_model.predict(training_data)
            
            # Train surrogate model to mimic black box
            surrogate.fit(training_data, black_box_predictions)
            
            # Calculate fidelity (how well surrogate mimics black box)
            surrogate_predictions = surrogate.predict(training_data)
            fidelity = np.corrcoef(black_box_predictions, surrogate_predictions)[0, 1]
            
            self.surrogate_models[model_name] = {
                'model': surrogate,
                'fidelity': fidelity,
                'feature_names': list(training_data.columns)
            }
            
            logger.info(f"Surrogate model created for {model_name} (fidelity: {fidelity:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error creating surrogate model for {model_name}: {e}")
            return False
    
    def explain_prediction_shap(self, model_name: str, instance: pd.Series) -> Optional[PredictionExplanation]:
        """Generate SHAP explanation for a prediction"""
        if not HAS_SHAP or model_name not in self.shap_explainers:
            return None
        
        try:
            explainer = self.shap_explainers[model_name]
            
            # Get SHAP values
            instance_array = instance.values.reshape(1, -1)
            shap_values = explainer.shap_values(instance_array)
            
            if isinstance(shap_values, list):  # Multi-output case
                shap_values = shap_values[0]
            
            # Get base value
            if hasattr(explainer, 'expected_value'):
                base_value = explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]
            else:
                base_value = 0.0
            
            # Create feature contributions
            feature_contributions = []
            for i, (feature_name, shap_value) in enumerate(zip(instance.index, shap_values[0])):
                direction = 'positive' if shap_value > 0 else 'negative' if shap_value < 0 else 'neutral'
                
                feature_contributions.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(abs(shap_value)),
                    importance_type='shap',
                    confidence=0.9,  # SHAP has high confidence
                    direction=direction
                ))
            
            # Sort by importance
            feature_contributions.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Calculate prediction value
            prediction_value = float(base_value + np.sum(shap_values))
            
            explanation = PredictionExplanation(
                prediction_id=f"shap_{model_name}_{int(datetime.now().timestamp())}",
                model_name=model_name,
                prediction_value=prediction_value,
                base_value=float(base_value),
                feature_contributions=feature_contributions,
                explanation_method='shap',
                confidence_score=0.9,
                created_at=datetime.now(),
                metadata={'shap_values_sum': float(np.sum(shap_values))}
            )
            
            # Store explanation
            self._store_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation for {model_name}: {e}")
            return None
    
    def explain_prediction_lime(self, model_name: str, model, instance: pd.Series) -> Optional[PredictionExplanation]:
        """Generate LIME explanation for a prediction"""
        if not HAS_LIME or model_name not in self.lime_explainers:
            return None
        
        try:
            explainer = self.lime_explainers[model_name]
            instance_array = instance.values
            
            # Generate explanation
            explanation_lime = explainer.explain_instance(
                instance_array, 
                model.predict,
                num_features=len(instance)
            )
            
            # Extract feature contributions
            feature_contributions = []
            for feature_idx, importance in explanation_lime.as_list():
                feature_name = instance.index[int(feature_idx)] if isinstance(feature_idx, (int, float)) else str(feature_idx)
                direction = 'positive' if importance > 0 else 'negative' if importance < 0 else 'neutral'
                
                feature_contributions.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(abs(importance)),
                    importance_type='lime',
                    confidence=0.7,  # LIME has moderate confidence
                    direction=direction
                ))
            
            # Sort by importance
            feature_contributions.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Get prediction value
            prediction_value = float(model.predict(instance_array.reshape(1, -1))[0])
            
            explanation = PredictionExplanation(
                prediction_id=f"lime_{model_name}_{int(datetime.now().timestamp())}",
                model_name=model_name,
                prediction_value=prediction_value,
                base_value=0.0,  # LIME doesn't provide base value
                feature_contributions=feature_contributions,
                explanation_method='lime',
                confidence_score=0.7,
                created_at=datetime.now(),
                metadata={'lime_score': float(explanation_lime.score)}
            )
            
            # Store explanation
            self._store_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation for {model_name}: {e}")
            return None
    
    def explain_prediction_surrogate(self, model_name: str, instance: pd.Series) -> Optional[PredictionExplanation]:
        """Generate explanation using surrogate model"""
        if model_name not in self.surrogate_models:
            return None
        
        try:
            surrogate_info = self.surrogate_models[model_name]
            surrogate_model = surrogate_info['model']
            
            # Get feature importance from surrogate
            feature_importance = surrogate_model.feature_importances_
            feature_names = surrogate_info['feature_names']
            
            # Calculate feature contributions
            instance_values = instance.values
            prediction_value = float(surrogate_model.predict(instance_values.reshape(1, -1))[0])
            
            feature_contributions = []
            for i, (feature_name, importance) in enumerate(zip(feature_names, feature_importance)):
                # Simple approximation of contribution
                contribution = importance * abs(instance_values[i])
                direction = 'positive' if instance_values[i] > 0 else 'negative'
                
                feature_contributions.append(FeatureImportance(
                    feature_name=feature_name,
                    importance_score=float(contribution),
                    importance_type='surrogate',
                    confidence=surrogate_info['fidelity'],
                    direction=direction
                ))
            
            # Sort by importance
            feature_contributions.sort(key=lambda x: x.importance_score, reverse=True)
            
            explanation = PredictionExplanation(
                prediction_id=f"surrogate_{model_name}_{int(datetime.now().timestamp())}",
                model_name=model_name,
                prediction_value=prediction_value,
                base_value=0.0,
                feature_contributions=feature_contributions,
                explanation_method='surrogate',
                confidence_score=surrogate_info['fidelity'],
                created_at=datetime.now(),
                metadata={'surrogate_fidelity': surrogate_info['fidelity']}
            )
            
            # Store explanation
            self._store_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating surrogate explanation for {model_name}: {e}")
            return None
    
    def explain_prediction(self, model_name: str, model, instance: pd.Series, 
                         methods: List[str] = None) -> Dict[str, PredictionExplanation]:
        """Generate comprehensive explanation using multiple methods"""
        if methods is None:
            methods = ['shap', 'lime', 'surrogate']
        
        explanations = {}
        
        # Try SHAP
        if 'shap' in methods:
            shap_explanation = self.explain_prediction_shap(model_name, instance)
            if shap_explanation:
                explanations['shap'] = shap_explanation
        
        # Try LIME
        if 'lime' in methods:
            lime_explanation = self.explain_prediction_lime(model_name, model, instance)
            if lime_explanation:
                explanations['lime'] = lime_explanation
        
        # Try surrogate
        if 'surrogate' in methods:
            surrogate_explanation = self.explain_prediction_surrogate(model_name, instance)
            if surrogate_explanation:
                explanations['surrogate'] = surrogate_explanation
        
        return explanations
    
    def get_global_feature_importance(self, model_name: str, model, 
                                    training_data: pd.DataFrame) -> List[FeatureImportance]:
        """Get global feature importance for the model"""
        try:
            feature_importance_list = []
            
            # Built-in feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                builtin_importance = model.feature_importances_
                for i, (feature_name, importance) in enumerate(zip(training_data.columns, builtin_importance)):
                    feature_importance_list.append(FeatureImportance(
                        feature_name=feature_name,
                        importance_score=float(importance),
                        importance_type='builtin',
                        confidence=0.8,
                        direction='neutral'
                    ))
            
            # Permutation importance (model-agnostic)
            try:
                # Use a subset for efficiency
                sample_size = min(500, len(training_data))
                sample_data = training_data.sample(n=sample_size, random_state=42)
                sample_predictions = model.predict(sample_data)
                
                perm_importance = permutation_importance(
                    model, sample_data, sample_predictions,
                    n_repeats=5, random_state=42
                )
                
                for i, (feature_name, importance, std) in enumerate(zip(
                    training_data.columns, 
                    perm_importance.importances_mean,
                    perm_importance.importances_std
                )):
                    confidence = max(0.1, 1.0 - (std / (abs(importance) + 1e-8)))
                    
                    feature_importance_list.append(FeatureImportance(
                        feature_name=feature_name,
                        importance_score=float(abs(importance)),
                        importance_type='permutation',
                        confidence=float(confidence),
                        direction='neutral'
                    ))
                    
            except Exception as e:
                logger.warning(f"Could not calculate permutation importance: {e}")
            
            # Sort by importance
            feature_importance_list.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Store in database
            self._store_feature_importance(model_name, feature_importance_list)
            
            return feature_importance_list
            
        except Exception as e:
            logger.error(f"Error calculating global feature importance for {model_name}: {e}")
            return []
    
    def _store_explanation(self, explanation: PredictionExplanation):
        """Store explanation in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO prediction_explanations 
                    (prediction_id, model_name, prediction_value, base_value, 
                     explanation_method, confidence_score, feature_contributions, 
                     created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    explanation.prediction_id, explanation.model_name,
                    explanation.prediction_value, explanation.base_value,
                    explanation.explanation_method, explanation.confidence_score,
                    json.dumps([{
                        'feature_name': fc.feature_name,
                        'importance_score': fc.importance_score,
                        'importance_type': fc.importance_type,
                        'confidence': fc.confidence,
                        'direction': fc.direction
                    } for fc in explanation.feature_contributions]),
                    explanation.created_at, json.dumps(explanation.metadata)
                ))
                
        except Exception as e:
            logger.error(f"Error storing explanation: {e}")
    
    def _store_feature_importance(self, model_name: str, feature_importance_list: List[FeatureImportance]):
        """Store feature importance in database"""
        try:
            timestamp = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                # Clear old importance for this model
                conn.execute('DELETE FROM feature_importance WHERE model_name = ?', (model_name,))
                
                # Insert new importance
                for fi in feature_importance_list:
                    conn.execute('''
                        INSERT INTO feature_importance 
                        (model_name, feature_name, importance_score, importance_type,
                         confidence, direction, computed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        model_name, fi.feature_name, fi.importance_score,
                        fi.importance_type, fi.confidence, fi.direction, timestamp
                    ))
                    
        except Exception as e:
            logger.error(f"Error storing feature importance: {e}")
    
    def generate_explanation_report(self, model_name: str, top_n: int = 10) -> Dict[str, Any]:
        """Generate comprehensive explanation report for a model"""
        try:
            # Get recent explanations
            with sqlite3.connect(self.db_path) as conn:
                explanations_df = pd.read_sql_query('''
                    SELECT * FROM prediction_explanations 
                    WHERE model_name = ? 
                    ORDER BY created_at DESC 
                    LIMIT 50
                ''', conn, params=(model_name,))
                
                feature_importance_df = pd.read_sql_query('''
                    SELECT * FROM feature_importance 
                    WHERE model_name = ?
                    ORDER BY importance_score DESC
                    LIMIT ?
                ''', conn, params=(model_name, top_n))
            
            report = {
                'model_name': model_name,
                'report_date': datetime.now(),
                'total_explanations': len(explanations_df),
                'explanation_methods': [],
                'top_features': [],
                'prediction_distribution': {},
                'confidence_distribution': {}
            }
            
            if not explanations_df.empty:
                # Explanation methods used
                report['explanation_methods'] = explanations_df['explanation_method'].value_counts().to_dict()
                
                # Prediction distribution
                report['prediction_distribution'] = {
                    'mean': float(explanations_df['prediction_value'].mean()),
                    'std': float(explanations_df['prediction_value'].std()),
                    'min': float(explanations_df['prediction_value'].min()),
                    'max': float(explanations_df['prediction_value'].max())
                }
                
                # Confidence distribution
                report['confidence_distribution'] = {
                    'mean': float(explanations_df['confidence_score'].mean()),
                    'std': float(explanations_df['confidence_score'].std())
                }
            
            if not feature_importance_df.empty:
                # Top features
                report['top_features'] = feature_importance_df.to_dict('records')
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report for {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    def get_feature_impact_summary(self, model_name: str, feature_name: str) -> Dict[str, Any]:
        """Get detailed impact summary for a specific feature"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all explanations mentioning this feature
                explanations_df = pd.read_sql_query('''
                    SELECT prediction_id, prediction_value, feature_contributions, created_at
                    FROM prediction_explanations 
                    WHERE model_name = ? AND feature_contributions LIKE ?
                    ORDER BY created_at DESC
                    LIMIT 100
                ''', conn, params=(model_name, f'%{feature_name}%'))
            
            if explanations_df.empty:
                return {'feature_name': feature_name, 'model_name': model_name, 'impact': 'No data'}
            
            feature_impacts = []
            for _, row in explanations_df.iterrows():
                contributions = json.loads(row['feature_contributions'])
                for contrib in contributions:
                    if contrib['feature_name'] == feature_name:
                        feature_impacts.append({
                            'importance_score': contrib['importance_score'],
                            'direction': contrib['direction'],
                            'prediction_value': row['prediction_value'],
                            'date': row['created_at']
                        })
            
            if not feature_impacts:
                return {'feature_name': feature_name, 'model_name': model_name, 'impact': 'No impacts found'}
            
            impacts_df = pd.DataFrame(feature_impacts)
            
            summary = {
                'feature_name': feature_name,
                'model_name': model_name,
                'total_appearances': len(feature_impacts),
                'average_importance': float(impacts_df['importance_score'].mean()),
                'max_importance': float(impacts_df['importance_score'].max()),
                'direction_distribution': impacts_df['direction'].value_counts().to_dict(),
                'correlation_with_prediction': float(impacts_df[['importance_score', 'prediction_value']].corr().iloc[0, 1]) if len(impacts_df) > 1 else 0.0,
                'recent_trend': self._calculate_feature_trend(impacts_df)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting feature impact for {feature_name}: {e}")
            return {'feature_name': feature_name, 'error': str(e)}
    
    def _calculate_feature_trend(self, impacts_df: pd.DataFrame) -> str:
        """Calculate trend for feature importance over time"""
        try:
            if len(impacts_df) < 5:
                return 'insufficient_data'
            
            # Sort by date and calculate trend
            impacts_df['date'] = pd.to_datetime(impacts_df['date'])
            impacts_df = impacts_df.sort_values('date')
            
            # Simple trend calculation
            recent_avg = impacts_df.tail(5)['importance_score'].mean()
            older_avg = impacts_df.head(5)['importance_score'].mean()
            
            if recent_avg > older_avg * 1.1:
                return 'increasing'
            elif recent_avg < older_avg * 0.9:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating feature trend: {e}")
            return 'unknown'