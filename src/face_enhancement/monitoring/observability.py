"""Observability & Monitoring.

Provides:
- Structured logging
- Metrics collection (Prometheus-compatible)
- Distributed tracing
- Health checks
- Performance monitoring
- Error tracking
- Custom dashboards
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
import threading
from collections import defaultdict
from loguru import logger
import json


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Similar to histogram


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data point."""

    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    help_text: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": float(self.value),
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "help": self.help_text,
        }


@dataclass
class Span:
    """Distributed tracing span."""

    span_id: str
    trace_id: str
    parent_id: Optional[str]
    operation_name: str

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)

    error: bool = False
    error_message: Optional[str] = None

    def finish(self) -> None:
        """Finish span."""
        self.end_time = datetime.now()
        delta = self.end_time - self.start_time
        self.duration_ms = delta.total_seconds() * 1000

    def add_tag(self, key: str, value: Any) -> None:
        """Add tag to span."""
        self.tags[key] = value

    def log_event(self, event: str, **kwargs) -> None:
        """Log event in span."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            **kwargs,
        }
        self.logs.append(log_entry)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "operation": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "error": self.error,
            "error_message": self.error_message,
        }


class MetricsCollector:
    """Collect and export metrics (Prometheus-compatible)."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Metric] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        self._lock = threading.Lock()

        logger.info("Metrics collector initialized")

    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict] = None) -> None:
        """
        Increment counter metric.

        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
        """
        key = self._make_key(name, labels)

        with self._lock:
            self.counters[key] += value

    def gauge(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """
        Set gauge metric.

        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
        """
        key = self._make_key(name, labels)

        with self._lock:
            self.gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """
        Record histogram value.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        key = self._make_key(name, labels)

        with self._lock:
            self.histograms[key].append(value)

    def get_metrics(self) -> List[Metric]:
        """Get all metrics."""
        metrics = []

        with self._lock:
            # Counters
            for key, value in self.counters.items():
                name, labels = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    type=MetricType.COUNTER,
                    value=value,
                    labels=labels,
                ))

            # Gauges
            for key, value in self.gauges.items():
                name, labels = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    type=MetricType.GAUGE,
                    value=value,
                    labels=labels,
                ))

            # Histograms (export percentiles)
            for key, values in self.histograms.items():
                if not values:
                    continue

                name, labels = self._parse_key(key)

                sorted_values = sorted(values)

                # Export percentiles
                for p in [50, 95, 99]:
                    idx = int(len(sorted_values) * p / 100)
                    percentile_value = sorted_values[min(idx, len(sorted_values) - 1)]

                    percentile_labels = labels.copy()
                    percentile_labels["quantile"] = str(p / 100)

                    metrics.append(Metric(
                        name=f"{name}_percentile",
                        type=MetricType.HISTOGRAM,
                        value=percentile_value,
                        labels=percentile_labels,
                    ))

        return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.get_metrics()

        for metric in metrics:
            # Help text
            if metric.help_text:
                lines.append(f"# HELP {metric.name} {metric.help_text}")

            # Type
            lines.append(f"# TYPE {metric.name} {metric.type.value}")

            # Metric
            label_str = ""
            if metric.labels:
                label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                label_str = "{" + ",".join(label_pairs) + "}"

            lines.append(f"{metric.name}{label_str} {metric.value}")

        return "\n".join(lines)

    def _make_key(self, name: str, labels: Optional[Dict] = None) -> str:
        """Create unique key for metric."""
        if labels:
            label_str = json.dumps(labels, sort_keys=True)
            return f"{name}:{label_str}"
        return name

    def _parse_key(self, key: str) -> Tuple[str, Dict]:
        """Parse metric key."""
        if ':' in key:
            name, label_str = key.split(':', 1)
            labels = json.loads(label_str)
            return name, labels
        return key, {}


class DistributedTracer:
    """Distributed tracing for request tracking."""

    def __init__(self):
        """Initialize distributed tracer."""
        self.spans: Dict[str, Span] = {}
        self._lock = threading.Lock()

        logger.info("Distributed tracer initialized")

    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Span:
        """
        Start new span.

        Args:
            operation_name: Name of operation
            trace_id: Trace ID (generated if not provided)
            parent_id: Parent span ID

        Returns:
            Started span
        """
        import uuid

        if trace_id is None:
            trace_id = str(uuid.uuid4())

        span_id = str(uuid.uuid4())

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            operation_name=operation_name,
        )

        with self._lock:
            self.spans[span_id] = span

        return span

    def finish_span(self, span: Span) -> None:
        """Finish span and record."""
        span.finish()
        logger.debug(
            f"Span finished: {span.operation_name} "
            f"({span.duration_ms:.2f}ms)"
        )

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for trace."""
        with self._lock:
            spans = [
                span for span in self.spans.values()
                if span.trace_id == trace_id
            ]

        # Sort by start time
        spans.sort(key=lambda s: s.start_time)

        return spans


class HealthChecker:
    """Health check system."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        logger.info("Health checker initialized")

    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register health check.

        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def check_health(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Health status
        """
        results = {}
        all_healthy = True

        for name, check_func in self.checks.items():
            try:
                healthy = check_func()
                results[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "healthy": healthy,
                }

                if not healthy:
                    all_healthy = False

            except Exception as e:
                results[name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e),
                }
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.now().isoformat(),
        }


class PerformanceMonitor:
    """Monitor performance metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize performance monitor.

        Args:
            metrics_collector: Metrics collector
        """
        self.metrics = metrics_collector
        logger.info("Performance monitor initialized")

    def measure(self, operation: str, labels: Optional[Dict] = None):
        """
        Context manager for measuring operation time.

        Usage:
            with monitor.measure("process_image"):
                process_image()
        """
        return self.OperationTimer(self.metrics, operation, labels)

    class OperationTimer:
        """Context manager for timing operations."""

        def __init__(self, metrics: MetricsCollector, operation: str, labels: Optional[Dict]):
            self.metrics = metrics
            self.operation = operation
            self.labels = labels or {}
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration_ms = (time.time() - self.start_time) * 1000

            # Record duration
            self.metrics.histogram(
                f"{self.operation}_duration_ms",
                duration_ms,
                self.labels,
            )

            # Record success/failure
            labels = self.labels.copy()
            labels["status"] = "error" if exc_type else "success"

            self.metrics.counter(
                f"{self.operation}_total",
                1.0,
                labels,
            )

            return False  # Don't suppress exceptions


class StructuredLogger:
    """Structured logging with context."""

    def __init__(self):
        """Initialize structured logger."""
        # Configure loguru
        logger.add(
            "logs/app.log",
            rotation="500 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} | {message}",
            serialize=True,  # JSON output
        )

        logger.add(
            "logs/error.log",
            rotation="100 MB",
            retention="90 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} | {message}",
            serialize=True,
        )

        logger.info("Structured logger initialized")

    def log(
        self,
        level: LogLevel,
        message: str,
        **context
    ) -> None:
        """
        Log with context.

        Args:
            level: Log level
            message: Log message
            **context: Additional context fields
        """
        log_func = getattr(logger, level.value)

        # Bind context
        context_logger = logger.bind(**context)

        # Log
        log_func_with_context = getattr(context_logger, level.value)
        log_func_with_context(message)


class MonitoringDashboard:
    """Monitoring dashboard data provider."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        health_checker: HealthChecker,
    ):
        """
        Initialize monitoring dashboard.

        Args:
            metrics_collector: Metrics collector
            health_checker: Health checker
        """
        self.metrics = metrics_collector
        self.health = health_checker

        logger.info("Monitoring dashboard initialized")

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
        return {
            "health": self.health.check_health(),
            "metrics": [m.to_dict() for m in self.metrics.get_metrics()],
            "timestamp": datetime.now().isoformat(),
        }

    def get_system_overview(self) -> Dict:
        """Get system overview."""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now().isoformat(),
        }


# Global instances
metrics_collector = MetricsCollector()
distributed_tracer = DistributedTracer()
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor(metrics_collector)
structured_logger = StructuredLogger()
