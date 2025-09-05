"""
Monitoring and Observability
Comprehensive monitoring, metrics, tracing, and health checking for microservices
"""

import time
import asyncio
import psutil
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import traceback
from contextlib import asynccontextmanager
import uuid

# Prometheus-style metrics (without requiring prometheus_client)
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    help_text: str = ""

class MetricsCollector:
    """Advanced metrics collector with Prometheus-style metrics"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._summaries: Dict[str, List[float]] = {}
        self._metrics_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Standard system metrics
        self._setup_system_metrics()
    
    def _setup_system_metrics(self):
        """Setup standard system metrics"""
        self.register_gauge("system_cpu_usage_percent", "CPU usage percentage")
        self.register_gauge("system_memory_usage_bytes", "Memory usage in bytes")
        self.register_gauge("system_disk_usage_percent", "Disk usage percentage")
        self.register_counter("service_requests_total", "Total service requests")
        self.register_counter("service_errors_total", "Total service errors")
        self.register_histogram("service_request_duration_seconds", "Request duration")
    
    def register_counter(self, name: str, help_text: str = ""):
        """Register a counter metric"""
        key = self._make_key(name)
        self._counters[key] = 0.0
        self._metrics_metadata[key] = {
            "type": MetricType.COUNTER,
            "help": help_text
        }
    
    def register_gauge(self, name: str, help_text: str = ""):
        """Register a gauge metric"""
        key = self._make_key(name)
        self._gauges[key] = 0.0
        self._metrics_metadata[key] = {
            "type": MetricType.GAUGE,
            "help": help_text
        }
    
    def register_histogram(self, name: str, help_text: str = ""):
        """Register a histogram metric"""
        key = self._make_key(name)
        self._histograms[key] = []
        self._metrics_metadata[key] = {
            "type": MetricType.HISTOGRAM,
            "help": help_text
        }
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment counter metric"""
        key = self._make_key(name, labels)
        if key not in self._counters:
            self._counters[key] = 0.0
        self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric value"""
        key = self._make_key(name, labels)
        self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add observation to histogram"""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
    
    def observe_summary(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add observation to summary"""
        key = self._make_key(name, labels)
        if key not in self._summaries:
            self._summaries[key] = []
        self._summaries[key].append(value)
    
    def collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.set_gauge("system_cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("system_memory_usage_bytes", memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.set_gauge("system_disk_usage_percent", disk_percent)
            
        except Exception as e:
            logging.warning(f"Failed to collect system metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        self.collect_system_metrics()
        
        metrics = {
            "service_name": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": {
                name: self._calculate_histogram_stats(values)
                for name, values in self._histograms.items()
            },
            "summaries": {
                name: self._calculate_summary_stats(values)
                for name, values in self._summaries.items()
            }
        }
        
        return metrics
    
    def _calculate_histogram_stats(self, values: List[float]) -> Dict[str, Any]:
        """Calculate histogram statistics"""
        if not values:
            return {"count": 0, "sum": 0, "buckets": {}}
        
        sorted_values = sorted(values)
        count = len(values)
        total = sum(values)
        
        # Calculate percentiles
        percentiles = {}
        for p in [50, 90, 95, 99]:
            idx = int((p / 100) * (count - 1))
            percentiles[f"p{p}"] = sorted_values[idx]
        
        return {
            "count": count,
            "sum": total,
            "avg": total / count,
            "min": min(values),
            "max": max(values),
            "percentiles": percentiles
        }
    
    def _calculate_summary_stats(self, values: List[float]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        return self._calculate_histogram_stats(values)
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key with labels"""
        if not labels:
            return name
        
        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_parts)}}}"
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        metrics = self.get_metrics()
        
        # Export counters
        for name, value in metrics["counters"].items():
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Export gauges
        for name, value in metrics["gauges"].items():
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        return '\n'.join(lines)

class TracingManager:
    """Distributed tracing manager"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._active_spans: Dict[str, 'Span'] = {}
    
    def create_span(self, operation_name: str, parent_span_id: Optional[str] = None) -> 'Span':
        """Create a new tracing span"""
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            operation_name=operation_name,
            service_name=self.service_name,
            parent_span_id=parent_span_id,
            start_time=time.time()
        )
        
        self._active_spans[span.span_id] = span
        return span
    
    def finish_span(self, span_id: str, status: str = "ok", error: Optional[str] = None):
        """Finish tracing span"""
        if span_id in self._active_spans:
            span = self._active_spans[span_id]
            span.finish(status, error)
            del self._active_spans[span_id]
            
            # In production, this would be sent to a tracing backend
            self._log_span(span)
    
    def _log_span(self, span: 'Span'):
        """Log span data (in production, send to tracing backend)"""
        span_data = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "operation_name": span.operation_name,
            "service_name": span.service_name,
            "parent_span_id": span.parent_span_id,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "status": span.status,
            "error": span.error,
            "tags": span.tags,
            "logs": span.logs
        }
        
        logging.info(f"TRACE: {json.dumps(span_data)}")

@dataclass
class Span:
    """Tracing span"""
    span_id: str
    trace_id: str
    operation_name: str
    service_name: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    status: str = "in_progress"
    error: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def add_tag(self, key: str, value: Any):
        """Add tag to span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add log to span"""
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        })
    
    def finish(self, status: str = "ok", error: Optional[str] = None):
        """Finish span"""
        self.end_time = time.time()
        self.status = status
        if error:
            self.error = error

class HealthChecker:
    """Health checking for services and dependencies"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._health_checks: Dict[str, Callable] = {}
        self._startup_time = datetime.utcnow()
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self._health_checks[name] = check_func
    
    async def perform_health_checks(self) -> Dict[str, Any]:
        """Perform all registered health checks"""
        results = {
            "service_name": self.service_name,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self._startup_time).total_seconds(),
            "checks": {}
        }
        
        overall_healthy = True
        
        for name, check_func in self._health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()
                
                duration = time.time() - start_time
                
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "duration_ms": round(duration * 1000, 2),
                    "details": check_result if isinstance(check_result, dict) else None
                }
                
                if not check_result:
                    overall_healthy = False
                    
            except Exception as e:
                results["checks"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "duration_ms": 0
                }
                overall_healthy = False
        
        results["status"] = "healthy" if overall_healthy else "unhealthy"
        return results

class AlertManager:
    """Alert management and notification"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._alert_handlers: Dict[str, List[Callable]] = {}
    
    def register_alert_handler(self, alert_type: str, handler: Callable):
        """Register alert handler"""
        if alert_type not in self._alert_handlers:
            self._alert_handlers[alert_type] = []
        self._alert_handlers[alert_type].append(handler)
    
    async def send_alert(self, 
                        alert_type: str,
                        message: str,
                        severity: str = "medium",
                        metadata: Dict[str, Any] = None):
        """Send alert"""
        alert_data = {
            "service_name": self.service_name,
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Log the alert
        logging.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")
        
        # Call registered handlers
        if alert_type in self._alert_handlers:
            for handler in self._alert_handlers[alert_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert_data)
                    else:
                        handler(alert_data)
                except Exception as e:
                    logging.error(f"Alert handler failed: {e}")

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._request_times: List[float] = []
        self._error_count = 0
        self._request_count = 0
    
    @asynccontextmanager
    async def monitor_request(self, operation_name: str):
        """Context manager to monitor request performance"""
        start_time = time.time()
        self._request_count += 1
        
        try:
            yield
        except Exception as e:
            self._error_count += 1
            logging.error(f"Request failed in {operation_name}: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self._request_times.append(duration)
            
            # Keep only last 1000 requests
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self._request_times:
            return {
                "total_requests": self._request_count,
                "error_count": self._error_count,
                "error_rate": 0.0,
                "avg_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0
            }
        
        sorted_times = sorted(self._request_times)
        count = len(sorted_times)
        
        return {
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "avg_response_time": sum(sorted_times) / count,
            "p95_response_time": sorted_times[int(0.95 * count)],
            "p99_response_time": sorted_times[int(0.99 * count)],
            "min_response_time": min(sorted_times),
            "max_response_time": max(sorted_times)
        }

class MonitoringSetup:
    """Complete monitoring setup for a microservice"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics = MetricsCollector(service_name)
        self.tracing = TracingManager(service_name)
        self.health = HealthChecker(service_name)
        self.alerts = AlertManager(service_name)
        self.performance = PerformanceMonitor(service_name)
        
        # Setup basic health checks
        self._setup_basic_health_checks()
    
    def _setup_basic_health_checks(self):
        """Setup basic health checks"""
        def check_memory_usage():
            """Check if memory usage is under 90%"""
            memory = psutil.virtual_memory()
            return memory.percent < 90
        
        def check_disk_space():
            """Check if disk usage is under 90%"""
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) < 0.9
        
        self.health.register_health_check("memory", check_memory_usage)
        self.health.register_health_check("disk", check_disk_space)
    
    async def setup_periodic_tasks(self):
        """Setup periodic monitoring tasks"""
        async def collect_metrics():
            while True:
                self.metrics.collect_system_metrics()
                await asyncio.sleep(60)  # Collect every minute
        
        async def health_check_task():
            while True:
                health_result = await self.health.perform_health_checks()
                if health_result["status"] != "healthy":
                    await self.alerts.send_alert(
                        "health_check_failed",
                        f"Service health check failed: {health_result}",
                        "high"
                    )
                await asyncio.sleep(30)  # Check every 30 seconds
        
        # Start background tasks
        asyncio.create_task(collect_metrics())
        asyncio.create_task(health_check_task())
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data"""
        return {
            "service_name": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics.get_metrics(),
            "performance": self.performance.get_performance_stats(),
            "health": None  # Would need to await health checks
        }