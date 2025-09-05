"""
ML Model Versioning System
Comprehensive model lifecycle management with versioning, rollback, and A/B testing
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import os
import shutil
import hashlib
import pickle
import joblib
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"
    FAILED = "failed"

class DeploymentStrategy(Enum):
    REPLACE = "replace"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    A_B_TEST = "a_b_test"

@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    model_name: str
    model_type: str
    status: ModelStatus
    created_at: datetime
    created_by: str
    file_path: str
    file_hash: str
    size_bytes: int
    performance_metrics: Dict[str, float]
    training_config: Dict[str, Any]
    dependencies: Dict[str, str]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    strategy: DeploymentStrategy
    traffic_split: float = 1.0  # Percentage of traffic for canary/A-B testing
    success_criteria: Dict[str, float] = field(default_factory=dict)
    rollback_conditions: Dict[str, float] = field(default_factory=dict)
    monitoring_duration_hours: int = 24
    auto_promote: bool = False

class ModelVersioningSystem:
    """
    Comprehensive ML model versioning and lifecycle management system
    """
    
    def __init__(self, storage_path: str = "model_storage", db_path: str = "model_versioning.db"):
        self.storage_path = Path(storage_path)
        self.db_path = db_path
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry = {}
        self.active_deployments = {}
        self.version_locks = {}
        
        # A/B testing and canary deployments
        self.traffic_router = {}
        self.deployment_metrics = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing models
        self._load_model_registry()
        
        logger.info(f"Model versioning system initialized with storage: {self.storage_path}")
    
    def _init_database(self):
        """Initialize versioning database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_versions (
                        model_id TEXT PRIMARY KEY,
                        version TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        created_by TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_hash TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        performance_metrics TEXT,
                        training_config TEXT,
                        dependencies TEXT,
                        tags TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS deployments (
                        deployment_id TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        environment TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        traffic_split REAL DEFAULT 1.0,
                        status TEXT NOT NULL,
                        deployed_at DATETIME NOT NULL,
                        deployed_by TEXT NOT NULL,
                        config TEXT,
                        rollback_model_id TEXT,
                        FOREIGN KEY (model_id) REFERENCES model_versions (model_id)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS deployment_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        deployment_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        recorded_at DATETIME NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_lineage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        child_model_id TEXT NOT NULL,
                        parent_model_id TEXT,
                        relationship_type TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (child_model_id) REFERENCES model_versions (model_id),
                        FOREIGN KEY (parent_model_id) REFERENCES model_versions (model_id)
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_versions_name_status ON model_versions(model_name, status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_deployments_env_status ON deployments(environment, status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_deployment_time ON deployment_metrics(deployment_id, recorded_at)')
                
        except Exception as e:
            logger.error(f"Error initializing versioning database: {e}")
            raise
    
    def _load_model_registry(self):
        """Load existing model registry from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('SELECT * FROM model_versions', conn)
            
            for _, row in df.iterrows():
                model_version = ModelVersion(
                    model_id=row['model_id'],
                    version=row['version'],
                    model_name=row['model_name'],
                    model_type=row['model_type'],
                    status=ModelStatus(row['status']),
                    created_at=pd.to_datetime(row['created_at']),
                    created_by=row['created_by'],
                    file_path=row['file_path'],
                    file_hash=row['file_hash'],
                    size_bytes=row['size_bytes'],
                    performance_metrics=json.loads(row['performance_metrics']),
                    training_config=json.loads(row['training_config']),
                    dependencies=json.loads(row['dependencies']),
                    tags=json.loads(row['tags']),
                    metadata=json.loads(row['metadata'])
                )
                
                self.model_registry[model_version.model_id] = model_version
            
            logger.info(f"Loaded {len(self.model_registry)} model versions from registry")
            
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
    
    def register_model(self, model_obj: Any, model_name: str, model_type: str,
                      performance_metrics: Dict[str, float], training_config: Dict[str, Any],
                      created_by: str, tags: List[str] = None, 
                      parent_model_id: str = None) -> str:
        """Register a new model version"""
        try:
            # Generate version
            version = self._generate_version(model_name)
            model_id = f"{model_name}_{version}"
            
            # Save model to storage
            file_path = self._save_model_to_storage(model_obj, model_id, model_type)
            
            # Calculate file hash and size
            file_hash = self._calculate_file_hash(file_path)
            size_bytes = os.path.getsize(file_path)
            
            # Get dependencies
            dependencies = self._extract_dependencies()
            
            # Create model version
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                model_name=model_name,
                model_type=model_type,
                status=ModelStatus.TRAINING,
                created_at=datetime.now(),
                created_by=created_by,
                file_path=str(file_path),
                file_hash=file_hash,
                size_bytes=size_bytes,
                performance_metrics=performance_metrics,
                training_config=training_config,
                dependencies=dependencies,
                tags=tags or [],
                metadata={}
            )
            
            # Store in database
            self._store_model_version(model_version)
            
            # Store lineage if parent provided
            if parent_model_id:
                self._store_model_lineage(model_id, parent_model_id, 'retrain')
            
            # Add to registry
            self.model_registry[model_id] = model_version
            
            logger.info(f"Registered model version: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for model"""
        try:
            # Get existing versions for this model
            existing_versions = [
                v.version for v in self.model_registry.values() 
                if v.model_name == model_name
            ]
            
            if not existing_versions:
                return "v1.0.0"
            
            # Parse versions and find highest
            max_version = [0, 0, 0]  # major, minor, patch
            for version in existing_versions:
                if version.startswith('v'):
                    version = version[1:]  # Remove 'v' prefix
                
                try:
                    parts = [int(x) for x in version.split('.')]
                    if len(parts) == 3:
                        if (parts[0] > max_version[0] or 
                            (parts[0] == max_version[0] and parts[1] > max_version[1]) or
                            (parts[0] == max_version[0] and parts[1] == max_version[1] and parts[2] > max_version[2])):
                            max_version = parts
                except ValueError:
                    continue
            
            # Increment patch version
            max_version[2] += 1
            return f"v{max_version[0]}.{max_version[1]}.{max_version[2]}"
            
        except Exception as e:
            logger.error(f"Error generating version: {e}")
            return f"v1.0.{int(datetime.now().timestamp())}"
    
    def _save_model_to_storage(self, model_obj: Any, model_id: str, model_type: str) -> Path:
        """Save model to storage"""
        try:
            model_dir = self.storage_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file format based on model type
            if 'tensorflow' in model_type.lower() or 'keras' in model_type.lower():
                model_path = model_dir / "model.h5"
                if hasattr(model_obj, 'save'):
                    model_obj.save(str(model_path))
                else:
                    raise ValueError("TensorFlow/Keras model doesn't have save method")
            
            elif 'pytorch' in model_type.lower():
                model_path = model_dir / "model.pt"
                import torch
                torch.save(model_obj.state_dict(), model_path)
            
            else:
                # Default to pickle/joblib for sklearn and other models
                model_path = model_dir / "model.pkl"
                joblib.dump(model_obj, model_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            metadata = {
                'model_id': model_id,
                'model_type': model_type,
                'saved_at': datetime.now().isoformat(),
                'file_format': model_path.suffix
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model to storage: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def _extract_dependencies(self) -> Dict[str, str]:
        """Extract current environment dependencies"""
        try:
            import pkg_resources
            
            dependencies = {}
            key_packages = [
                'tensorflow', 'torch', 'sklearn', 'scikit-learn', 
                'xgboost', 'lightgbm', 'pandas', 'numpy'
            ]
            
            installed_packages = {pkg.project_name.lower(): pkg.version 
                                for pkg in pkg_resources.working_set}
            
            for package in key_packages:
                if package in installed_packages:
                    dependencies[package] = installed_packages[package]
            
            return dependencies
            
        except Exception as e:
            logger.warning(f"Could not extract dependencies: {e}")
            return {}
    
    def _store_model_version(self, model_version: ModelVersion):
        """Store model version in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO model_versions 
                    (model_id, version, model_name, model_type, status, created_at,
                     created_by, file_path, file_hash, size_bytes, performance_metrics,
                     training_config, dependencies, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_version.model_id, model_version.version, model_version.model_name,
                    model_version.model_type, model_version.status.value, model_version.created_at,
                    model_version.created_by, model_version.file_path, model_version.file_hash,
                    model_version.size_bytes, json.dumps(model_version.performance_metrics),
                    json.dumps(model_version.training_config), json.dumps(model_version.dependencies),
                    json.dumps(model_version.tags), json.dumps(model_version.metadata)
                ))
                
        except Exception as e:
            logger.error(f"Error storing model version: {e}")
            raise
    
    def _store_model_lineage(self, child_model_id: str, parent_model_id: str, relationship_type: str):
        """Store model lineage relationship"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_lineage 
                    (child_model_id, parent_model_id, relationship_type, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (child_model_id, parent_model_id, relationship_type, datetime.now()))
                
        except Exception as e:
            logger.error(f"Error storing model lineage: {e}")
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        try:
            if model_id not in self.model_registry:
                logger.error(f"Model not found: {model_id}")
                return False
            
            # Update in memory
            self.model_registry[model_id].status = status
            
            # Update in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE model_versions 
                    SET status = ? 
                    WHERE model_id = ?
                ''', (status.value, model_id))
            
            logger.info(f"Updated model {model_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            return False
    
    def load_model(self, model_id: str) -> Any:
        """Load model from storage"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model not found: {model_id}")
            
            model_version = self.model_registry[model_id]
            file_path = Path(model_version.file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            # Load based on model type
            if model_version.model_type.lower() in ['tensorflow', 'keras']:
                import tensorflow as tf
                return tf.keras.models.load_model(str(file_path))
            
            elif model_version.model_type.lower() == 'pytorch':
                import torch
                model = torch.load(file_path)
                return model
            
            else:
                # Default to joblib
                return joblib.load(file_path)
                
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def deploy_model(self, model_id: str, environment: str, config: DeploymentConfig,
                    deployed_by: str) -> str:
        """Deploy model to specified environment"""
        try:
            if model_id not in self.model_registry:
                raise ValueError(f"Model not found: {model_id}")
            
            model_version = self.model_registry[model_id]
            
            if model_version.status != ModelStatus.STAGING:
                logger.warning(f"Deploying model not in staging status: {model_version.status}")
            
            deployment_id = f"{model_id}_{environment}_{int(datetime.now().timestamp())}"
            
            # Handle different deployment strategies
            if config.strategy == DeploymentStrategy.REPLACE:
                # Replace current production model
                self._replace_production_model(model_id, environment)
            
            elif config.strategy == DeploymentStrategy.CANARY:
                # Start canary deployment
                self._start_canary_deployment(model_id, environment, config.traffic_split)
            
            elif config.strategy == DeploymentStrategy.A_B_TEST:
                # Start A/B test
                self._start_ab_test(model_id, environment, config.traffic_split)
            
            # Store deployment record
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO deployments 
                    (deployment_id, model_id, environment, strategy, traffic_split,
                     status, deployed_at, deployed_by, config)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    deployment_id, model_id, environment, config.strategy.value,
                    config.traffic_split, 'active', datetime.now(), deployed_by,
                    json.dumps({
                        'success_criteria': config.success_criteria,
                        'rollback_conditions': config.rollback_conditions,
                        'monitoring_duration_hours': config.monitoring_duration_hours,
                        'auto_promote': config.auto_promote
                    })
                ))
            
            # Update model status
            self.update_model_status(model_id, ModelStatus.PRODUCTION)
            
            # Store deployment info
            self.active_deployments[deployment_id] = {
                'model_id': model_id,
                'environment': environment,
                'strategy': config.strategy,
                'deployed_at': datetime.now(),
                'config': config
            }
            
            logger.info(f"Deployed model {model_id} to {environment} using {config.strategy.value}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {e}")
            raise
    
    def _replace_production_model(self, model_id: str, environment: str):
        """Replace current production model"""
        # Implementation would depend on actual deployment infrastructure
        logger.info(f"Replacing production model in {environment} with {model_id}")
    
    def _start_canary_deployment(self, model_id: str, environment: str, traffic_split: float):
        """Start canary deployment"""
        # Implementation would route traffic between old and new models
        logger.info(f"Starting canary deployment for {model_id} with {traffic_split*100}% traffic")
    
    def _start_ab_test(self, model_id: str, environment: str, traffic_split: float):
        """Start A/B test deployment"""
        # Implementation would setup A/B testing infrastructure
        logger.info(f"Starting A/B test for {model_id} with {traffic_split*100}% traffic")
    
    def rollback_deployment(self, deployment_id: str, rollback_to_model_id: str = None) -> bool:
        """Rollback a deployment"""
        try:
            if deployment_id not in self.active_deployments:
                logger.error(f"Deployment not found: {deployment_id}")
                return False
            
            deployment_info = self.active_deployments[deployment_id]
            
            # Determine rollback target
            if not rollback_to_model_id:
                # Find previous production model
                rollback_to_model_id = self._find_previous_production_model(
                    deployment_info['environment']
                )
            
            if not rollback_to_model_id:
                logger.error("No rollback target found")
                return False
            
            # Perform rollback
            self._replace_production_model(rollback_to_model_id, deployment_info['environment'])
            
            # Update deployment status
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE deployments 
                    SET status = 'rolled_back', rollback_model_id = ?
                    WHERE deployment_id = ?
                ''', (rollback_to_model_id, deployment_id))
            
            # Update model statuses
            self.update_model_status(deployment_info['model_id'], ModelStatus.RETIRED)
            self.update_model_status(rollback_to_model_id, ModelStatus.PRODUCTION)
            
            # Remove from active deployments
            del self.active_deployments[deployment_id]
            
            logger.info(f"Rolled back deployment {deployment_id} to {rollback_to_model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back deployment {deployment_id}: {e}")
            return False
    
    def _find_previous_production_model(self, environment: str) -> Optional[str]:
        """Find previous production model for rollback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute('''
                    SELECT model_id FROM deployments 
                    WHERE environment = ? AND status = 'active'
                    ORDER BY deployed_at DESC 
                    LIMIT 1 OFFSET 1
                ''', (environment,)).fetchone()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error finding previous production model: {e}")
            return None
    
    def get_model_versions(self, model_name: str = None, status: ModelStatus = None) -> List[ModelVersion]:
        """Get model versions with optional filtering"""
        try:
            versions = list(self.model_registry.values())
            
            if model_name:
                versions = [v for v in versions if v.model_name == model_name]
            
            if status:
                versions = [v for v in versions if v.status == status]
            
            # Sort by creation time (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get model lineage (parents and children)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get parents
                parents = pd.read_sql_query('''
                    SELECT parent_model_id, relationship_type, created_at
                    FROM model_lineage 
                    WHERE child_model_id = ?
                ''', conn, params=(model_id,))
                
                # Get children
                children = pd.read_sql_query('''
                    SELECT child_model_id, relationship_type, created_at
                    FROM model_lineage 
                    WHERE parent_model_id = ?
                ''', conn, params=(model_id,))
            
            return {
                'model_id': model_id,
                'parents': parents.to_dict('records') if not parents.empty else [],
                'children': children.to_dict('records') if not children.empty else []
            }
            
        except Exception as e:
            logger.error(f"Error getting model lineage for {model_id}: {e}")
            return {'model_id': model_id, 'parents': [], 'children': []}
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status across all environments"""
        try:
            # Get active deployments
            with sqlite3.connect(self.db_path) as conn:
                deployments = pd.read_sql_query('''
                    SELECT d.*, mv.model_name, mv.version 
                    FROM deployments d
                    JOIN model_versions mv ON d.model_id = mv.model_id
                    WHERE d.status = 'active'
                    ORDER BY d.deployed_at DESC
                ''', conn)
            
            # Group by environment
            deployment_status = {}
            for _, deployment in deployments.iterrows():
                env = deployment['environment']
                if env not in deployment_status:
                    deployment_status[env] = []
                
                deployment_status[env].append({
                    'deployment_id': deployment['deployment_id'],
                    'model_name': deployment['model_name'],
                    'version': deployment['version'],
                    'strategy': deployment['strategy'],
                    'traffic_split': deployment['traffic_split'],
                    'deployed_at': deployment['deployed_at']
                })
            
            return {
                'deployments_by_environment': deployment_status,
                'total_active_deployments': len(deployments),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {'error': str(e)}
    
    def cleanup_old_versions(self, model_name: str, keep_count: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            versions = self.get_model_versions(model_name)
            
            # Filter out production and staging models
            cleanable_versions = [
                v for v in versions 
                if v.status not in [ModelStatus.PRODUCTION, ModelStatus.STAGING]
            ]
            
            if len(cleanable_versions) <= keep_count:
                return 0
            
            # Sort by creation time and select old versions to remove
            cleanable_versions.sort(key=lambda v: v.created_at)
            versions_to_remove = cleanable_versions[:-keep_count]
            
            removed_count = 0
            for version in versions_to_remove:
                try:
                    # Remove file
                    file_path = Path(version.file_path)
                    if file_path.exists():
                        model_dir = file_path.parent
                        shutil.rmtree(model_dir)
                    
                    # Remove from database
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('DELETE FROM model_versions WHERE model_id = ?', 
                                   (version.model_id,))
                    
                    # Remove from registry
                    del self.model_registry[version.model_id]
                    
                    removed_count += 1
                    logger.info(f"Cleaned up model version: {version.model_id}")
                    
                except Exception as e:
                    logger.error(f"Error removing model version {version.model_id}: {e}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions for {model_name}: {e}")
            return 0