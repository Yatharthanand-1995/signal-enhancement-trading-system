"""
Configuration Management System
Centralized configuration with environment inheritance, validation, and hot-reload capabilities
"""
import os
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import threading
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationMetadata:
    """Configuration metadata tracking"""
    environment: str
    version: str
    loaded_at: datetime
    sources: List[str]
    checksum: str
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration hot-reload"""
    
    def __init__(self, config_manager, watched_paths: List[Path]):
        self.config_manager = config_manager
        self.watched_paths = watched_paths
        self.last_reload = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml', '.json']:
            # Debounce rapid file changes
            now = time.time()
            if file_path in self.last_reload:
                if now - self.last_reload[file_path] < 1.0:  # 1 second debounce
                    return
                    
            self.last_reload[file_path] = now
            logger.info(f"Configuration file changed: {file_path}")
            
            # Trigger reload
            try:
                self.config_manager._reload_configuration()
            except Exception as e:
                logger.error(f"Failed to reload configuration after file change: {e}")

class ConfigurationManager:
    """
    Advanced configuration management system with environment inheritance,
    validation, hot-reload, and feature flags
    """
    
    def __init__(self, config_root: str = "config", environment: str = None):
        self.config_root = Path(config_root)
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        
        # Configuration state
        self.config = {}
        self.metadata = None
        self.feature_flags = {}
        self.secrets = {}
        
        # Hot-reload state
        self.hot_reload_enabled = False
        self.file_observer = None
        self.reload_callbacks = []
        
        # Thread safety
        self.config_lock = threading.RLock()
        
        # Validation schema cache
        self.schemas = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info(f"Configuration manager initialized for environment: {self.environment}")
    
    def _load_configuration(self):
        """Load configuration with environment inheritance"""
        try:
            with self.config_lock:
                config_sources = []
                merged_config = {}
                
                # 1. Load base configuration
                base_config_path = self.config_root / "base.yaml"
                if base_config_path.exists():
                    base_config = self._load_yaml_file(base_config_path)
                    merged_config.update(base_config)
                    config_sources.append(str(base_config_path))
                    logger.debug(f"Loaded base configuration: {base_config_path}")
                
                # 2. Load environment-specific configuration
                env_config_path = self.config_root / "environments" / f"{self.environment}.yaml"
                if env_config_path.exists():
                    env_config = self._load_yaml_file(env_config_path)
                    merged_config = self._deep_merge(merged_config, env_config)
                    config_sources.append(str(env_config_path))
                    logger.debug(f"Loaded environment configuration: {env_config_path}")
                
                # 3. Load feature-specific configurations
                features_dir = self.config_root / "features"
                if features_dir.exists():
                    for feature_file in features_dir.glob("*.yaml"):
                        feature_config = self._load_yaml_file(feature_file)
                        
                        # Feature flags go to separate namespace
                        if feature_file.name == "feature_flags.yaml":
                            self.feature_flags.update(feature_config)
                        else:
                            # Other feature configs merge into main config
                            merged_config = self._deep_merge(merged_config, feature_config)
                        
                        config_sources.append(str(feature_file))
                        logger.debug(f"Loaded feature configuration: {feature_file}")
                
                # 4. Load secrets (environment variables override)
                self._load_secrets()
                
                # 5. Apply environment variable overrides
                merged_config = self._apply_environment_overrides(merged_config)
                
                # 6. Validate configuration
                validation_errors = self._validate_configuration(merged_config)
                
                # 7. Calculate checksum
                config_str = json.dumps(merged_config, sort_keys=True)
                checksum = hashlib.sha256(config_str.encode()).hexdigest()[:16]
                
                # 8. Update state
                self.config = merged_config
                self.metadata = ConfigurationMetadata(
                    environment=self.environment,
                    version=self._get_config_version(),
                    loaded_at=datetime.now(),
                    sources=config_sources,
                    checksum=checksum,
                    valid=len(validation_errors) == 0,
                    validation_errors=validation_errors
                )
                
                if validation_errors:
                    logger.warning(f"Configuration validation errors: {validation_errors}")
                else:
                    logger.info(f"Configuration loaded successfully (checksum: {checksum})")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse YAML file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries with override precedence"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _load_secrets(self):
        """Load secrets with secure handling"""
        try:
            secrets_dir = self.config_root / "secrets"
            if not secrets_dir.exists():
                return
            
            for secret_file in secrets_dir.glob("*.yaml"):
                if secret_file.name.endswith('.template'):
                    continue  # Skip template files
                
                try:
                    secrets = self._load_yaml_file(secret_file)
                    self.secrets.update(secrets)
                    logger.debug(f"Loaded secrets from: {secret_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load secrets from {secret_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading secrets: {e}")
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides with prefix CONFIG_"""
        result = config.copy()
        
        for key, value in os.environ.items():
            if key.startswith('CONFIG_'):
                config_key = key[7:]  # Remove CONFIG_ prefix
                
                # Convert nested keys (CONFIG_DATABASE__HOST -> database.host)
                keys = config_key.lower().split('__')
                
                # Navigate to nested structure
                current = result
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Parse value (try JSON first, then string)
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                
                current[keys[-1]] = parsed_value
                logger.debug(f"Applied environment override: {config_key} = {parsed_value}")
        
        return result
    
    def _validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schemas"""
        errors = []
        
        try:
            # Load validation schemas
            schema_dir = self.config_root / "schemas"
            if not schema_dir.exists():
                return errors  # No validation if no schemas
            
            for schema_file in schema_dir.glob("*.json"):
                try:
                    with open(schema_file, 'r') as f:
                        schema = json.load(f)
                        
                    # Validate relevant config section
                    section_name = schema_file.stem
                    if section_name in config:
                        validate(instance=config[section_name], schema=schema)
                        
                except ValidationError as e:
                    errors.append(f"Validation error in {section_name}: {e.message}")
                except Exception as e:
                    logger.warning(f"Failed to validate schema {schema_file}: {e}")
        
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
        
        return errors
    
    def _get_config_version(self) -> str:
        """Get configuration version from git or timestamp"""
        try:
            # Try git commit hash first
            import subprocess
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True, cwd=self.config_root.parent)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback to timestamp
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """Get configuration value with dot notation support"""
        with self.config_lock:
            try:
                # Support nested keys with dot notation (e.g., "database.host")
                keys = key.split('.')
                current = self.config
                
                for k in keys:
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        if required:
                            raise KeyError(f"Required configuration key not found: {key}")
                        return default
                
                return current
                
            except Exception as e:
                if required:
                    raise KeyError(f"Error accessing configuration key '{key}': {e}")
                return default
    
    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        with self.config_lock:
            return self.feature_flags.get(flag_name, default)
    
    def set_feature_flag(self, flag_name: str, value: bool):
        """Set feature flag value (runtime only)"""
        with self.config_lock:
            self.feature_flags[flag_name] = value
            logger.info(f"Feature flag updated: {flag_name} = {value}")
            
            # Notify callbacks
            self._notify_reload_callbacks(f"feature_flag:{flag_name}")
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret value with dot notation support"""
        with self.config_lock:
            keys = key.split('.')
            current = self.secrets
            
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            
            return current
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration (excluding secrets)"""
        with self.config_lock:
            return self.config.copy()
    
    def get_metadata(self) -> ConfigurationMetadata:
        """Get configuration metadata"""
        with self.config_lock:
            return self.metadata
    
    def enable_hot_reload(self, callback: callable = None):
        """Enable hot-reload with optional callback"""
        try:
            if self.hot_reload_enabled:
                return
            
            if callback:
                self.reload_callbacks.append(callback)
            
            # Setup file watcher
            watched_paths = [
                self.config_root / "base.yaml",
                self.config_root / "environments",
                self.config_root / "features"
            ]
            
            existing_paths = [p for p in watched_paths if p.exists()]
            
            if existing_paths:
                self.file_observer = Observer()
                file_watcher = ConfigFileWatcher(self, existing_paths)
                
                for path in existing_paths:
                    self.file_observer.schedule(file_watcher, str(path), recursive=True)
                
                self.file_observer.start()
                self.hot_reload_enabled = True
                
                logger.info("Hot-reload enabled for configuration files")
            
        except Exception as e:
            logger.error(f"Failed to enable hot-reload: {e}")
    
    def disable_hot_reload(self):
        """Disable hot-reload"""
        try:
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join()
                self.file_observer = None
            
            self.hot_reload_enabled = False
            logger.info("Hot-reload disabled")
            
        except Exception as e:
            logger.error(f"Failed to disable hot-reload: {e}")
    
    def _reload_configuration(self):
        """Reload configuration from files"""
        try:
            logger.info("Reloading configuration...")
            old_checksum = self.metadata.checksum if self.metadata else None
            
            self._load_configuration()
            
            new_checksum = self.metadata.checksum
            if old_checksum != new_checksum:
                logger.info(f"Configuration reloaded (checksum: {old_checksum} -> {new_checksum})")
                self._notify_reload_callbacks("configuration_reload")
            else:
                logger.debug("Configuration unchanged after reload")
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def _notify_reload_callbacks(self, trigger: str):
        """Notify registered callbacks about configuration changes"""
        for callback in self.reload_callbacks:
            try:
                callback(trigger, self)
            except Exception as e:
                logger.error(f"Configuration reload callback error: {e}")
    
    def validate_current_config(self) -> Dict[str, Any]:
        """Validate current configuration and return validation report"""
        with self.config_lock:
            validation_errors = self._validate_configuration(self.config)
            
            return {
                'valid': len(validation_errors) == 0,
                'errors': validation_errors,
                'environment': self.environment,
                'version': self.metadata.version if self.metadata else 'unknown',
                'checksum': self.metadata.checksum if self.metadata else 'unknown',
                'sources': self.metadata.sources if self.metadata else [],
                'loaded_at': self.metadata.loaded_at.isoformat() if self.metadata else None
            }
    
    def export_config(self, include_secrets: bool = False, format: str = 'yaml') -> str:
        """Export configuration to string format"""
        with self.config_lock:
            export_data = {
                'metadata': {
                    'environment': self.environment,
                    'version': self.metadata.version if self.metadata else 'unknown',
                    'exported_at': datetime.now().isoformat(),
                    'checksum': self.metadata.checksum if self.metadata else 'unknown'
                },
                'configuration': self.config.copy(),
                'feature_flags': self.feature_flags.copy()
            }
            
            if include_secrets:
                export_data['secrets'] = self.secrets.copy()
            
            if format.lower() == 'json':
                return json.dumps(export_data, indent=2, default=str)
            else:  # yaml
                return yaml.dump(export_data, default_flow_style=False, sort_keys=False)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.disable_hot_reload()
        except:
            pass

# Global configuration instance
_config_manager = None

def get_config_manager(config_root: str = "config", environment: str = None) -> ConfigurationManager:
    """Get or create global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_root, environment)
    
    return _config_manager

def get_config(key: str, default: Any = None, required: bool = False) -> Any:
    """Convenient function to get configuration value"""
    return get_config_manager().get(key, default, required)

def get_feature_flag(flag_name: str, default: bool = False) -> bool:
    """Convenient function to get feature flag"""
    return get_config_manager().get_feature_flag(flag_name, default)

def get_secret(key: str, default: Any = None) -> Any:
    """Convenient function to get secret value"""
    return get_config_manager().get_secret(key, default)