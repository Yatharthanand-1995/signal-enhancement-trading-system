#!/usr/bin/env python3
"""
Real-time Performance Dashboard
Flask web application showing system performance, trading metrics, and health status.
"""
import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
from typing import Dict, Any, List
import psutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.caching import get_cache_stats
from src.utils.database import db_manager
from src.utils.api_optimization import get_api_stats
from src.utils.realtime_processing import get_realtime_stats
from src.utils.logging_setup import get_logger
from config.enhanced_config import enhanced_config

logger = get_logger(__name__)

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
app.config['SECRET_KEY'] = enhanced_config.security.secret_key
socketio = SocketIO(app, cors_allowed_origins="*")

class PerformanceDashboard:
    """Main dashboard controller"""
    
    def __init__(self):
        self.metrics_history = {
            'timestamps': [],
            'system_cpu': [],
            'system_memory': [],
            'database_connections': [],
            'cache_hit_rate': [],
            'api_response_time': [],
            'event_processing_rate': [],
            'signal_confidence': [],
            'active_signals': [],
            'composite_score': [],
            'signal_strength': []
        }
        self.max_history_points = 100
        self._start_metrics_collector()
    
    def _start_metrics_collector(self):
        """Start background metrics collection"""
        def collect_metrics():
            while True:
                try:
                    metrics = self.collect_current_metrics()
                    self._update_history(metrics)
                    
                    # Emit to connected clients
                    socketio.emit('metrics_update', {
                        'metrics': metrics,
                        'history': self.get_metrics_history()
                    })
                    
                    time.sleep(5)  # Collect every 5 seconds
                    
                except Exception as e:
                    logger.error("Metrics collection error", exception=e, component='dashboard')
                    time.sleep(10)
        
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
        logger.info("Metrics collector started", component='dashboard')
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Database metrics
            try:
                db_stats = db_manager.get_performance_stats()
                db_health = db_manager.get_health_status()
            except Exception as e:
                logger.warning(f"Database metrics unavailable: {e}", component='dashboard')
                db_stats = {'total_connections': 0, 'query_count': 0, 'average_query_time': 0}
                db_health = {'healthy': False, 'response_time': 0}
            
            # Cache metrics
            try:
                cache_stats = get_cache_stats()
            except Exception as e:
                logger.warning(f"Cache metrics unavailable: {e}", component='dashboard')
                cache_stats = {'hit_rate_percent': 0, 'memory_cache_size': 0}
            
            # API metrics
            try:
                api_stats = get_api_stats()
            except Exception as e:
                logger.warning(f"API metrics unavailable: {e}", component='dashboard')
                api_stats = {'optimizer': {'avg_processing_time': 0, 'requests_processed': 0}}
            
            # Real-time processing metrics
            try:
                realtime_stats = get_realtime_stats()
            except Exception as e:
                logger.warning(f"Real-time metrics unavailable: {e}", component='dashboard')
                realtime_stats = {'stream_processor': {'messages_per_second': 0, 'active_streams': 0}}
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': round(memory.used / (1024**3), 2),
                    'memory_total_gb': round(memory.total / (1024**3), 2),
                    'disk_percent': disk.percent,
                    'disk_used_gb': round(disk.used / (1024**3), 2),
                    'disk_total_gb': round(disk.total / (1024**3), 2)
                },
                'database': {
                    'healthy': db_health.get('healthy', False),
                    'response_time': db_health.get('response_time', 0),
                    'connections': db_stats.get('total_connections', 0),
                    'query_count': db_stats.get('query_count', 0),
                    'avg_query_time': db_stats.get('average_query_time', 0)
                },
                'cache': {
                    'hit_rate': cache_stats.get('hit_rate_percent', 0),
                    'memory_size': cache_stats.get('memory_cache_size', 0),
                    'redis_connected': cache_stats.get('redis_connected', False),
                    'total_hits': cache_stats.get('hits', 0),
                    'total_misses': cache_stats.get('misses', 0)
                },
                'api': {
                    'avg_response_time': api_stats.get('optimizer', {}).get('avg_processing_time', 0),
                    'requests_processed': api_stats.get('optimizer', {}).get('requests_processed', 0),
                    'requests_queued': api_stats.get('optimizer', {}).get('requests_queued', 0),
                    'cache_hits': api_stats.get('optimizer', {}).get('cache_hits', 0)
                },
                'realtime': {
                    'messages_per_second': realtime_stats.get('stream_processor', {}).get('messages_per_second', 0),
                    'active_streams': realtime_stats.get('stream_processor', {}).get('active_streams', 0),
                    'buffer_utilization': realtime_stats.get('stream_processor', {}).get('buffer_utilization', 0),
                    'websocket_connections': realtime_stats.get('websocket_manager', {}).get('active_connections', 0)
                },
                'trading': self._get_trading_metrics()
            }
            
        except Exception as e:
            logger.error("Error collecting metrics", exception=e, component='dashboard')
            return self._get_default_metrics()
    
    def _get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading-specific metrics including signal analysis"""
        try:
            # Get basic trading metrics
            basic_metrics = self._get_basic_trading_metrics()
            
            # Get signal metrics
            signal_metrics = self._get_signal_metrics()
            
            # Combine all metrics
            return {**basic_metrics, **signal_metrics}
            
        except Exception as e:
            logger.warning(f"Trading metrics unavailable: {e}", component='dashboard')
            return self._get_default_trading_metrics()
    
    def _get_basic_trading_metrics(self) -> Dict[str, Any]:
        """Get basic trading metrics"""
        return {
            'active_positions': 0,
            'daily_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'risk_utilization': 0.0
        }
    
    def _get_signal_metrics(self) -> Dict[str, Any]:
        """Get comprehensive signal metrics including confidence, raw values, and final scores"""
        try:
            # Initialize signal metrics structure
            signal_data = {
                'signals': {
                    'total_active_signals': 0,
                    'avg_confidence': 0.0,
                    'confidence_distribution': {
                        'high_confidence': 0,  # >= 0.8
                        'medium_confidence': 0,  # 0.5 - 0.8
                        'low_confidence': 0  # < 0.5
                    },
                    'raw_signals': {
                        'rsi_avg': 0.0,
                        'macd_signal_count': 0,
                        'bollinger_squeeze': 0,
                        'volume_spike_count': 0
                    },
                    'final_scores': {
                        'avg_composite_score': 0.0,
                        'score_distribution': {
                            'strong_buy': 0,
                            'buy': 0, 
                            'neutral': 0,
                            'sell': 0,
                            'strong_sell': 0
                        },
                        'weighted_signal_strength': 0.0
                    }
                }
            }
            
            # Try to get real signal data
            signal_data['signals'] = self._collect_live_signal_data()
            
            return signal_data
            
        except Exception as e:
            logger.warning(f"Signal metrics collection failed: {e}", component='dashboard')
            return {'signals': self._get_default_signal_metrics()}
    
    def _collect_live_signal_data(self) -> Dict[str, Any]:
        """Collect live signal data from signal processing systems"""
        try:
            # Try to connect to ensemble signal scorer
            from src.strategy.ensemble_signal_scoring import EnsembleSignalScorer
            
            # This would be injected or configured properly in production
            scorer = EnsembleSignalScorer()
            
            # Generate mock data for demonstration (replace with real data in production)
            import random
            import numpy as np
            
            # Simulate signal confidence data
            confidence_values = [random.uniform(0.3, 0.95) for _ in range(10)]
            avg_confidence = np.mean(confidence_values) if confidence_values else 0.0
            
            # Count confidence distribution
            high_conf = sum(1 for c in confidence_values if c >= 0.8)
            med_conf = sum(1 for c in confidence_values if 0.5 <= c < 0.8)
            low_conf = sum(1 for c in confidence_values if c < 0.5)
            
            # Simulate raw signal values
            raw_signals = {
                'rsi_avg': random.uniform(30, 70),
                'macd_signal_count': random.randint(1, 5),
                'bollinger_squeeze': random.randint(0, 3),
                'volume_spike_count': random.randint(0, 4)
            }
            
            # Simulate final composite scores
            composite_scores = [random.uniform(-1, 1) for _ in range(10)]
            avg_composite = np.mean(composite_scores) if composite_scores else 0.0
            
            # Score distribution
            strong_buy = sum(1 for s in composite_scores if s >= 0.6)
            buy = sum(1 for s in composite_scores if 0.2 <= s < 0.6)
            neutral = sum(1 for s in composite_scores if -0.2 <= s < 0.2)
            sell = sum(1 for s in composite_scores if -0.6 <= s < -0.2)
            strong_sell = sum(1 for s in composite_scores if s < -0.6)
            
            return {
                'total_active_signals': len(confidence_values),
                'avg_confidence': round(avg_confidence, 3),
                'confidence_distribution': {
                    'high_confidence': high_conf,
                    'medium_confidence': med_conf,
                    'low_confidence': low_conf
                },
                'raw_signals': {
                    'rsi_avg': round(raw_signals['rsi_avg'], 2),
                    'macd_signal_count': raw_signals['macd_signal_count'],
                    'bollinger_squeeze': raw_signals['bollinger_squeeze'],
                    'volume_spike_count': raw_signals['volume_spike_count']
                },
                'final_scores': {
                    'avg_composite_score': round(avg_composite, 3),
                    'score_distribution': {
                        'strong_buy': strong_buy,
                        'buy': buy,
                        'neutral': neutral,
                        'sell': sell,
                        'strong_sell': strong_sell
                    },
                    'weighted_signal_strength': round(abs(avg_composite), 3)
                }
            }
            
        except ImportError:
            logger.warning("Signal scoring module not available", component='dashboard')
            return self._get_default_signal_metrics()
        except Exception as e:
            logger.error(f"Error collecting live signal data: {e}", component='dashboard')
            return self._get_default_signal_metrics()
    
    def _get_default_signal_metrics(self) -> Dict[str, Any]:
        """Return default signal metrics when collection fails"""
        return {
            'total_active_signals': 0,
            'avg_confidence': 0.0,
            'confidence_distribution': {
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            },
            'raw_signals': {
                'rsi_avg': 0.0,
                'macd_signal_count': 0,
                'bollinger_squeeze': 0,
                'volume_spike_count': 0
            },
            'final_scores': {
                'avg_composite_score': 0.0,
                'score_distribution': {
                    'strong_buy': 0,
                    'buy': 0,
                    'neutral': 0,
                    'sell': 0,
                    'strong_sell': 0
                },
                'weighted_signal_strength': 0.0
            }
        }
    
    def _get_default_trading_metrics(self) -> Dict[str, Any]:
        """Return default trading metrics when collection fails"""
        return {
            'active_positions': 0,
            'daily_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'risk_utilization': 0.0,
            'signals': self._get_default_signal_metrics()
        }
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when collection fails"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {'cpu_percent': 0, 'memory_percent': 0, 'disk_percent': 0},
            'database': {'healthy': False, 'response_time': 0, 'connections': 0},
            'cache': {'hit_rate': 0, 'memory_size': 0, 'redis_connected': False},
            'api': {'avg_response_time': 0, 'requests_processed': 0},
            'realtime': {'messages_per_second': 0, 'active_streams': 0},
            'trading': {'active_positions': 0, 'daily_pnl': 0.0}
        }
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update metrics history for charts"""
        timestamp = datetime.fromisoformat(metrics['timestamp'].replace('Z', '+00:00'))
        
        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['system_cpu'].append(metrics['system']['cpu_percent'])
        self.metrics_history['system_memory'].append(metrics['system']['memory_percent'])
        self.metrics_history['database_connections'].append(metrics['database']['connections'])
        self.metrics_history['cache_hit_rate'].append(metrics['cache']['hit_rate'])
        self.metrics_history['api_response_time'].append(metrics['api']['avg_response_time'] * 1000)  # Convert to ms
        self.metrics_history['event_processing_rate'].append(metrics['realtime']['messages_per_second'])
        
        # Add signal metrics to history
        signal_data = metrics.get('trading', {}).get('signals', {})
        self.metrics_history['signal_confidence'].append(signal_data.get('avg_confidence', 0.0))
        self.metrics_history['active_signals'].append(signal_data.get('total_active_signals', 0))
        self.metrics_history['composite_score'].append(signal_data.get('final_scores', {}).get('avg_composite_score', 0.0))
        self.metrics_history['signal_strength'].append(signal_data.get('final_scores', {}).get('weighted_signal_strength', 0.0))
        
        # Keep only recent history
        if len(self.metrics_history['timestamps']) > self.max_history_points:
            for key in self.metrics_history:
                self.metrics_history[key] = self.metrics_history[key][-self.max_history_points:]
    
    def get_metrics_history(self) -> Dict[str, List]:
        """Get metrics history for charts"""
        return {
            'timestamps': [t.isoformat() for t in self.metrics_history['timestamps']],
            'system_cpu': self.metrics_history['system_cpu'],
            'system_memory': self.metrics_history['system_memory'],
            'database_connections': self.metrics_history['database_connections'],
            'cache_hit_rate': self.metrics_history['cache_hit_rate'],
            'api_response_time': self.metrics_history['api_response_time'],
            'event_processing_rate': self.metrics_history['event_processing_rate'],
            'signal_confidence': self.metrics_history['signal_confidence'],
            'active_signals': self.metrics_history['active_signals'],
            'composite_score': self.metrics_history['composite_score'],
            'signal_strength': self.metrics_history['signal_strength']
        }
    
    def generate_performance_charts(self) -> Dict[str, Any]:
        """Generate Plotly charts for performance data"""
        history = self.get_metrics_history()
        
        if not history['timestamps']:
            return {}
        
        charts = {}
        
        # System Performance Chart
        charts['system_performance'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['system_cpu'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'CPU %',
                    'line': {'color': '#FF6B6B'}
                },
                {
                    'x': history['timestamps'],
                    'y': history['system_memory'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Memory %',
                    'line': {'color': '#4ECDC4'}
                }
            ],
            'layout': {
                'title': 'System Performance',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Percentage'},
                'height': 300
            }
        }
        
        # Database Performance Chart
        charts['database_performance'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['database_connections'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Connections',
                    'line': {'color': '#45B7D1'}
                }
            ],
            'layout': {
                'title': 'Database Connections',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Count'},
                'height': 300
            }
        }
        
        # Cache Performance Chart
        charts['cache_performance'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['cache_hit_rate'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Hit Rate %',
                    'line': {'color': '#96CEB4'}
                }
            ],
            'layout': {
                'title': 'Cache Hit Rate',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Percentage'},
                'height': 300
            }
        }
        
        # API Performance Chart
        charts['api_performance'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['api_response_time'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Response Time (ms)',
                    'line': {'color': '#FECA57'}
                }
            ],
            'layout': {
                'title': 'API Response Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Milliseconds'},
                'height': 300
            }
        }
        
        # Signal Confidence Chart
        charts['signal_confidence'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['signal_confidence'],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Avg Confidence',
                    'line': {'color': '#9B59B6'},
                    'marker': {'size': 6}
                }
            ],
            'layout': {
                'title': 'Signal Confidence Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Confidence (0-1)', 'range': [0, 1]},
                'height': 300
            }
        }
        
        # Active Signals Chart
        charts['active_signals'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['active_signals'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Active Signals',
                    'line': {'color': '#E67E22'},
                    'fill': 'tozeroy'
                }
            ],
            'layout': {
                'title': 'Active Signal Count',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Number of Signals'},
                'height': 300
            }
        }
        
        # Composite Score Chart
        charts['composite_score'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['composite_score'],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Composite Score',
                    'line': {'color': '#27AE60'},
                    'marker': {'size': 4}
                }
            ],
            'layout': {
                'title': 'Signal Composite Score',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Score (-1 to 1)', 'range': [-1, 1]},
                'height': 300,
                'shapes': [
                    # Add horizontal line at 0
                    {
                        'type': 'line',
                        'xref': 'paper',
                        'x0': 0,
                        'x1': 1,
                        'yref': 'y',
                        'y0': 0,
                        'y1': 0,
                        'line': {'color': 'rgba(255,255,255,0.3)', 'width': 1, 'dash': 'dot'}
                    }
                ]
            }
        }
        
        # Signal Strength Chart  
        charts['signal_strength'] = {
            'data': [
                {
                    'x': history['timestamps'],
                    'y': history['signal_strength'],
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Signal Strength',
                    'line': {'color': '#E74C3C'},
                    'fill': 'tozeroy'
                }
            ],
            'layout': {
                'title': 'Weighted Signal Strength',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Strength (0-1)', 'range': [0, 1]},
                'height': 300
            }
        }
        
        return charts

# Initialize dashboard
dashboard = PerformanceDashboard()

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """API endpoint for current metrics"""
    metrics = dashboard.collect_current_metrics()
    return jsonify(metrics)

@app.route('/api/charts')
def get_charts():
    """API endpoint for performance charts"""
    charts = dashboard.generate_performance_charts()
    return jsonify(charts)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        metrics = dashboard.collect_current_metrics()
        
        # Determine overall health
        health_score = 0
        total_checks = 0
        
        # System health
        if metrics['system']['cpu_percent'] < 80:
            health_score += 1
        if metrics['system']['memory_percent'] < 85:
            health_score += 1
        total_checks += 2
        
        # Database health
        if metrics['database']['healthy']:
            health_score += 1
        total_checks += 1
        
        # Cache health
        if metrics['cache']['redis_connected']:
            health_score += 1
        total_checks += 1
        
        health_percentage = (health_score / total_checks) * 100 if total_checks > 0 else 0
        
        return jsonify({
            'status': 'healthy' if health_percentage > 75 else 'warning' if health_percentage > 50 else 'critical',
            'health_score': health_percentage,
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'system': 'ok' if metrics['system']['cpu_percent'] < 80 else 'high_cpu',
                'database': 'ok' if metrics['database']['healthy'] else 'unavailable',
                'cache': 'ok' if metrics['cache']['redis_connected'] else 'disconnected'
            }
        })
        
    except Exception as e:
        logger.error("Health check failed", exception=e, component='dashboard')
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Dashboard client connected", component='dashboard')
    
    # Send initial data
    metrics = dashboard.collect_current_metrics()
    history = dashboard.get_metrics_history()
    charts = dashboard.generate_performance_charts()
    
    emit('initial_data', {
        'metrics': metrics,
        'history': history,
        'charts': charts
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Dashboard client disconnected", component='dashboard')

@socketio.on('request_refresh')
def handle_refresh_request():
    """Handle manual refresh request"""
    metrics = dashboard.collect_current_metrics()
    charts = dashboard.generate_performance_charts()
    
    emit('refresh_data', {
        'metrics': metrics,
        'charts': charts
    })

def create_dashboard_template():
    """Create the HTML template for the dashboard"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading System Performance Dashboard</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #4ECDC4;
            margin: 0;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 10px;
        }
        .status-healthy { background-color: #2ECC71; }
        .status-warning { background-color: #F39C12; }
        .status-critical { background-color: #E74C3C; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #2c2c2c;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .metric-title {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-unit {
            font-size: 14px;
            color: #888;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .chart-container {
            background-color: #2c2c2c;
            border-radius: 8px;
            padding: 20px;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #4ECDC4;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .refresh-btn:hover {
            background-color: #45B7D1;
        }
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #2c2c2c;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>Trading System Performance Dashboard</h1>
            <span id="health-status">System Status: <span class="status-indicator status-healthy" id="status-indicator"></span></span>
            <div style="margin-top: 10px; color: #888;" id="last-update">Last Update: --</div>
        </div>

        <button class="refresh-btn" onclick="requestRefresh()">Refresh</button>

        <div class="metrics-grid" id="metrics-grid">
            <!-- Metrics will be populated by JavaScript -->
        </div>

        <div class="charts-grid" id="charts-grid">
            <!-- Charts will be populated by JavaScript -->
        </div>

        <div class="connection-status" id="connection-status">
            <span class="status-indicator status-critical" id="connection-indicator"></span>
            Connecting...
        </div>
    </div>

    <script>
        const socket = io();
        let isConnected = false;

        socket.on('connect', function() {
            isConnected = true;
            updateConnectionStatus('Connected', 'healthy');
        });

        socket.on('disconnect', function() {
            isConnected = false;
            updateConnectionStatus('Disconnected', 'critical');
        });

        socket.on('initial_data', function(data) {
            updateMetrics(data.metrics);
            updateCharts(data.charts);
        });

        socket.on('metrics_update', function(data) {
            updateMetrics(data.metrics);
        });

        socket.on('refresh_data', function(data) {
            updateMetrics(data.metrics);
            updateCharts(data.charts);
        });

        function updateConnectionStatus(status, level) {
            document.getElementById('connection-status').innerHTML = 
                `<span class="status-indicator status-${level}" id="connection-indicator"></span> ${status}`;
        }

        function updateMetrics(metrics) {
            const metricsGrid = document.getElementById('metrics-grid');
            const lastUpdate = document.getElementById('last-update');
            
            lastUpdate.textContent = `Last Update: ${new Date(metrics.timestamp).toLocaleString()}`;
            
            // Update health status
            const healthStatus = metrics.database.healthy && metrics.cache.redis_connected ? 'healthy' : 'warning';
            document.getElementById('status-indicator').className = `status-indicator status-${healthStatus}`;
            
            const metricsHtml = `
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value" style="color: ${metrics.system.cpu_percent > 80 ? '#E74C3C' : '#4ECDC4'}">${metrics.system.cpu_percent.toFixed(1)}</div>
                    <div class="metric-unit">%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value" style="color: ${metrics.system.memory_percent > 85 ? '#E74C3C' : '#4ECDC4'}">${metrics.system.memory_percent.toFixed(1)}</div>
                    <div class="metric-unit">%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Database Health</div>
                    <div class="metric-value" style="color: ${metrics.database.healthy ? '#2ECC71' : '#E74C3C'}">${metrics.database.healthy ? 'Healthy' : 'Down'}</div>
                    <div class="metric-unit">${metrics.database.response_time.toFixed(3)}s response</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Cache Hit Rate</div>
                    <div class="metric-value" style="color: ${metrics.cache.hit_rate > 80 ? '#2ECC71' : '#F39C12'}">${metrics.cache.hit_rate.toFixed(1)}</div>
                    <div class="metric-unit">%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">API Response Time</div>
                    <div class="metric-value">${(metrics.api.avg_response_time * 1000).toFixed(1)}</div>
                    <div class="metric-unit">ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Real-time Events</div>
                    <div class="metric-value">${metrics.realtime.messages_per_second.toFixed(1)}</div>
                    <div class="metric-unit">events/sec</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Active Streams</div>
                    <div class="metric-value">${metrics.realtime.active_streams}</div>
                    <div class="metric-unit">streams</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">WebSocket Clients</div>
                    <div class="metric-value">${metrics.realtime.websocket_connections}</div>
                    <div class="metric-unit">connections</div>
                </div>
            `;
            
            metricsGrid.innerHTML = metricsHtml;
        }

        function updateCharts(charts) {
            const chartsGrid = document.getElementById('charts-grid');
            chartsGrid.innerHTML = '';
            
            Object.keys(charts).forEach(chartId => {
                const chartDiv = document.createElement('div');
                chartDiv.className = 'chart-container';
                chartDiv.id = chartId;
                chartsGrid.appendChild(chartDiv);
                
                const chart = charts[chartId];
                chart.layout.plot_bgcolor = '#2c2c2c';
                chart.layout.paper_bgcolor = '#2c2c2c';
                chart.layout.font = {color: '#ffffff'};
                
                Plotly.newPlot(chartId, chart.data, chart.layout, {responsive: true});
            });
        }

        function requestRefresh() {
            if (isConnected) {
                socket.emit('request_refresh');
            }
        }

        // Auto-refresh charts every 30 seconds
        setInterval(() => {
            if (isConnected) {
                socket.emit('request_refresh');
            }
        }, 30000);
    </script>
</body>
</html>'''
    
    template_path = os.path.join(template_dir, 'dashboard.html')
    with open(template_path, 'w') as f:
        f.write(template_content)
    
    logger.info(f"Dashboard template created at {template_path}", component='dashboard')

def run_dashboard(host='localhost', port=5000, debug=False):
    """Run the performance dashboard"""
    try:
        # Create template
        create_dashboard_template()
        
        logger.info(f"Starting Performance Dashboard on http://{host}:{port}", component='dashboard')
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
        
    except Exception as e:
        logger.error("Failed to start dashboard", exception=e, component='dashboard')
        raise

if __name__ == '__main__':
    run_dashboard(debug=True)