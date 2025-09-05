"""
Advanced ML Alerting System
Intelligent alerting with escalation, deduplication, and anomaly detection
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import threading
import time
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Configuration for alert rules"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'anomaly'
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 30
    escalation_minutes: int = 60
    tags: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    model_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class NotificationChannel:
    """Base class for notification channels"""
    
    def __init__(self, channel_id: str, config: Dict[str, Any]):
        self.channel_id = channel_id
        self.config = config
    
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        raise NotImplementedError

class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    def send_notification(self, alert: Alert) -> bool:
        try:
            smtp_server = self.config.get('smtp_server', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('username')
            password = self.config.get('password')
            from_email = self.config.get('from_email')
            to_emails = self.config.get('to_emails', [])
            
            if not to_emails:
                logger.warning("No recipient emails configured")
                return False
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] ML Alert: {alert.model_name}"
            
            body = f"""
Alert Details:
- Model: {alert.model_name}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value}
- Threshold: {alert.threshold}
- Severity: {alert.severity.value.upper()}
- Created: {alert.created_at}
- Message: {alert.message}

Alert ID: {alert.alert_id}
Rule ID: {alert.rule_id}
            """.strip()
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel (webhook-based)"""
    
    def send_notification(self, alert: Alert) -> bool:
        try:
            import requests
            
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return False
            
            # Color coding by severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.LOW: "#ffb347", 
                AlertSeverity.MEDIUM: "#ff8c00",
                AlertSeverity.HIGH: "#ff4500",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": f"ML Alert: {alert.model_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": f"{alert.current_value:.4f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Alert ID", "value": alert.alert_id, "short": False}
                    ],
                    "timestamp": int(alert.created_at.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

class AnomalyDetector:
    """Anomaly detection for alert triggering"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.models = {}
        self.scalers = {}
        self.training_data = defaultdict(lambda: deque(maxlen=1000))
    
    def fit(self, metric_name: str, data: np.ndarray):
        """Fit anomaly detection model for a metric"""
        try:
            if len(data) < 50:  # Need minimum data points
                logger.warning(f"Insufficient data for anomaly detection: {len(data)}")
                return
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.reshape(-1, 1))
            
            # Fit Isolation Forest
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42
            )
            model.fit(scaled_data)
            
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            
            logger.info(f"Anomaly detection model fitted for {metric_name}")
            
        except Exception as e:
            logger.error(f"Error fitting anomaly model for {metric_name}: {e}")
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous"""
        try:
            if metric_name not in self.models:
                return False, 0.0
            
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]
            
            # Scale and predict
            scaled_value = scaler.transform([[value]])
            anomaly_score = model.decision_function(scaled_value)[0]
            is_anomaly = model.predict(scaled_value)[0] == -1
            
            return is_anomaly, float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Error detecting anomaly for {metric_name}: {e}")
            return False, 0.0
    
    def update_training_data(self, metric_name: str, value: float):
        """Update training data and retrain if needed"""
        try:
            self.training_data[metric_name].append(value)
            
            # Retrain periodically
            if len(self.training_data[metric_name]) % 100 == 0:
                data = np.array(list(self.training_data[metric_name]))
                self.fit(metric_name, data)
                
        except Exception as e:
            logger.error(f"Error updating training data: {e}")

class AdvancedAlertingSystem:
    """
    Advanced alerting system with intelligent deduplication,
    escalation, and anomaly detection
    """
    
    def __init__(self, db_path: str = "ml_alerting.db"):
        self.db_path = db_path
        
        # Alert management
        self.alert_rules = {}
        self.active_alerts = {}
        self.notification_channels = {}
        self.alert_history = deque(maxlen=10000)
        
        # Deduplication and rate limiting
        self.alert_cooldowns = {}
        self.alert_counts = defaultdict(int)
        self.suppression_rules = {}
        
        # Anomaly detection
        self.anomaly_detector = AnomalyDetector()
        
        # Background processing
        self.processing_active = False
        self.processor_thread = None
        
        # Initialize database
        self._init_database()
        
        # Setup default notification channels
        self._setup_default_channels()
        
        logger.info("Advanced alerting system initialized")
    
    def _init_database(self):
        """Initialize alerting database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alert_rules (
                        rule_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        metric_name TEXT NOT NULL,
                        condition TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        severity TEXT NOT NULL,
                        enabled BOOLEAN DEFAULT TRUE,
                        cooldown_minutes INTEGER DEFAULT 30,
                        escalation_minutes INTEGER DEFAULT 60,
                        tags TEXT,
                        notification_channels TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        rule_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold REAL NOT NULL,
                        severity TEXT NOT NULL,
                        status TEXT NOT NULL,
                        message TEXT,
                        metadata TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        resolved_at DATETIME,
                        acknowledged_by TEXT,
                        FOREIGN KEY (rule_id) REFERENCES alert_rules (rule_id)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS notification_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL,
                        channel_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        sent_at DATETIME NOT NULL,
                        error_message TEXT,
                        FOREIGN KEY (alert_id) REFERENCES alerts (alert_id)
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_model_status ON alerts(model_name, status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
                
        except Exception as e:
            logger.error(f"Error initializing alerting database: {e}")
            raise
    
    def _setup_default_channels(self):
        """Setup default notification channels"""
        # Email channel (would be configured via environment variables)
        self.notification_channels['email'] = EmailNotificationChannel('email', {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'from_email': 'alerts@trading-system.com',
            'to_emails': ['admin@trading-system.com']
        })
        
        # Slack channel (would be configured via environment variables)
        self.notification_channels['slack'] = SlackNotificationChannel('slack', {
            'webhook_url': None  # Would be set from config
        })
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO alert_rules 
                    (rule_id, name, description, metric_name, condition, threshold, severity,
                     enabled, cooldown_minutes, escalation_minutes, tags, notification_channels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule.rule_id, rule.name, rule.description, rule.metric_name,
                    rule.condition, rule.threshold, rule.severity.value, rule.enabled,
                    rule.cooldown_minutes, rule.escalation_minutes,
                    json.dumps(rule.tags), json.dumps(rule.notification_channels)
                ))
            
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.rule_id}")
            
        except Exception as e:
            logger.error(f"Error adding alert rule: {e}")
    
    def check_metric(self, model_name: str, metric_name: str, value: float, metadata: Dict = None) -> List[Alert]:
        """Check if metric value triggers any alerts"""
        triggered_alerts = []
        
        try:
            # Update anomaly detection training data
            self.anomaly_detector.update_training_data(f"{model_name}_{metric_name}", value)
            
            # Check all relevant rules
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                if rule.metric_name != metric_name:
                    continue
                
                # Check cooldown
                cooldown_key = f"{model_name}_{rule_id}"
                if cooldown_key in self.alert_cooldowns:
                    if datetime.now() - self.alert_cooldowns[cooldown_key] < timedelta(minutes=rule.cooldown_minutes):
                        continue
                
                # Evaluate condition
                triggered = False
                
                if rule.condition == 'gt' and value > rule.threshold:
                    triggered = True
                elif rule.condition == 'lt' and value < rule.threshold:
                    triggered = True
                elif rule.condition == 'eq' and abs(value - rule.threshold) < 1e-6:
                    triggered = True
                elif rule.condition == 'anomaly':
                    is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(
                        f"{model_name}_{metric_name}", value
                    )
                    if is_anomaly:
                        triggered = True
                        metadata = metadata or {}
                        metadata['anomaly_score'] = anomaly_score
                
                if triggered:
                    alert = self._create_alert(rule, model_name, value, metadata)
                    triggered_alerts.append(alert)
                    
                    # Set cooldown
                    self.alert_cooldowns[cooldown_key] = datetime.now()
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Error checking metric {metric_name} for {model_name}: {e}")
            return []
    
    def _create_alert(self, rule: AlertRule, model_name: str, current_value: float, metadata: Dict = None) -> Alert:
        """Create new alert"""
        alert_id = f"{model_name}_{rule.rule_id}_{int(datetime.now().timestamp())}"
        
        message = f"{rule.name}: {rule.metric_name} = {current_value:.4f} ({rule.condition} {rule.threshold:.4f})"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            model_name=model_name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            message=message,
            metadata=metadata or {}
        )
        
        # Store in database
        self._store_alert(alert)
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Send notifications
        self._send_notifications(alert, rule)
        
        logger.warning(f"ALERT [{alert.severity.value.upper()}] {model_name}: {message}")
        
        return alert
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts 
                    (alert_id, rule_id, model_name, metric_name, current_value, threshold,
                     severity, status, message, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id, alert.rule_id, alert.model_name, alert.metric_name,
                    alert.current_value, alert.threshold, alert.severity.value,
                    alert.status.value, alert.message, json.dumps(alert.metadata),
                    alert.created_at, alert.updated_at
                ))
                
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert"""
        for channel_id in rule.notification_channels:
            if channel_id not in self.notification_channels:
                logger.warning(f"Notification channel not found: {channel_id}")
                continue
            
            try:
                channel = self.notification_channels[channel_id]
                success = channel.send_notification(alert)
                
                # Log notification attempt
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO notification_log 
                        (alert_id, channel_id, status, sent_at, error_message)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        alert.alert_id, channel_id, 
                        'success' if success else 'failed',
                        datetime.now(), 
                        None if success else 'Send failed'
                    ))
                    
            except Exception as e:
                logger.error(f"Error sending notification via {channel_id}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET status = ?, acknowledged_by = ?, updated_at = ?
                    WHERE alert_id = ?
                ''', (alert.status.value, acknowledged_by, alert.updated_at, alert_id))
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Resolve an active alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.updated_at = datetime.now()
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET status = ?, resolved_at = ?, updated_at = ?
                    WHERE alert_id = ?
                ''', (alert.status.value, alert.resolved_at, alert.updated_at, alert_id))
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self, model_name: str = None, severity: AlertSeverity = None) -> List[Alert]:
        """Get currently active alerts"""
        try:
            alerts = list(self.active_alerts.values())
            
            # Filter by model name
            if model_name:
                alerts = [a for a in alerts if a.model_name == model_name]
            
            # Filter by severity
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            # Sort by severity and creation time
            severity_order = {
                AlertSeverity.CRITICAL: 5,
                AlertSeverity.HIGH: 4,
                AlertSeverity.MEDIUM: 3,
                AlertSeverity.LOW: 2,
                AlertSeverity.INFO: 1
            }
            
            alerts.sort(key=lambda a: (severity_order.get(a.severity, 0), a.created_at), reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status"""
        try:
            active_alerts = list(self.active_alerts.values())
            
            # Count by severity
            severity_counts = {severity.value: 0 for severity in AlertSeverity}
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
            
            # Count by model
            model_counts = defaultdict(int)
            for alert in active_alerts:
                model_counts[alert.model_name] += 1
            
            # Recent alert stats
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_alerts = [a for a in active_alerts if a.created_at >= recent_cutoff]
            
            return {
                'total_active_alerts': len(active_alerts),
                'severity_distribution': severity_counts,
                'alerts_by_model': dict(model_counts),
                'recent_alerts_24h': len(recent_alerts),
                'alert_rules_count': len(self.alert_rules),
                'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule.enabled),
                'notification_channels': list(self.notification_channels.keys()),
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {'error': str(e)}
    
    def setup_default_ml_rules(self):
        """Setup default ML monitoring alert rules"""
        default_rules = [
            AlertRule(
                rule_id='accuracy_degradation_critical',
                name='Critical Accuracy Degradation',
                description='Model accuracy dropped below critical threshold',
                metric_name='accuracy',
                condition='lt',
                threshold=0.6,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=60,
                escalation_minutes=30,
                notification_channels=['email', 'slack']
            ),
            AlertRule(
                rule_id='accuracy_degradation_high',
                name='High Accuracy Degradation',
                description='Model accuracy dropped below high threshold',
                metric_name='accuracy',
                condition='lt',
                threshold=0.75,
                severity=AlertSeverity.HIGH,
                cooldown_minutes=120,
                notification_channels=['slack']
            ),
            AlertRule(
                rule_id='high_error_rate',
                name='High Error Rate',
                description='Model error rate exceeded threshold',
                metric_name='error_rate',
                condition='gt',
                threshold=0.15,
                severity=AlertSeverity.HIGH,
                cooldown_minutes=60,
                notification_channels=['email', 'slack']
            ),
            AlertRule(
                rule_id='prediction_drift_high',
                name='High Prediction Drift',
                description='Prediction drift detected',
                metric_name='prediction_drift',
                condition='gt',
                threshold=0.2,
                severity=AlertSeverity.MEDIUM,
                cooldown_minutes=180,
                notification_channels=['slack']
            ),
            AlertRule(
                rule_id='feature_drift_medium',
                name='Feature Drift Detection',
                description='Feature drift detected',
                metric_name='feature_drift',
                condition='gt',
                threshold=0.15,
                severity=AlertSeverity.MEDIUM,
                cooldown_minutes=240,
                notification_channels=['slack']
            ),
            AlertRule(
                rule_id='data_quality_low',
                name='Low Data Quality',
                description='Data quality score below threshold',
                metric_name='data_quality_score',
                condition='lt',
                threshold=0.7,
                severity=AlertSeverity.LOW,
                cooldown_minutes=360,
                notification_channels=['slack']
            ),
            AlertRule(
                rule_id='accuracy_anomaly',
                name='Accuracy Anomaly Detection',
                description='Anomalous accuracy pattern detected',
                metric_name='accuracy',
                condition='anomaly',
                threshold=0.0,  # Not used for anomaly detection
                severity=AlertSeverity.MEDIUM,
                cooldown_minutes=180,
                notification_channels=['slack']
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
        
        logger.info(f"Setup {len(default_rules)} default ML monitoring rules")