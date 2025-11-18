"""Deployment tracking and monitoring for production models.

Tracks:
- Model inference latency
- Error rates
- Input/output distributions
- Model drift detection
- Resource utilization
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from loguru import logger


@dataclass
class InferenceRecord:
    """Record of a single inference."""

    timestamp: datetime
    model_name: str
    model_version: str
    latency_ms: float
    input_shape: tuple
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class DeploymentTracker:
    """
    Track and monitor deployed models in production.

    Provides:
    - Inference latency tracking
    - Error rate monitoring
    - Input distribution tracking
    - Drift detection
    - Alerting on anomalies
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_threshold_latency: float = 1000.0,  # ms
        alert_threshold_error_rate: float = 0.05,  # 5%
    ):
        """
        Initialize deployment tracker.

        Args:
            window_size: Number of recent inferences to track
            alert_threshold_latency: Latency threshold for alerts (ms)
            alert_threshold_error_rate: Error rate threshold for alerts
        """
        self.window_size = window_size
        self.alert_threshold_latency = alert_threshold_latency
        self.alert_threshold_error_rate = alert_threshold_error_rate

        # Track recent inferences
        self.recent_inferences: Dict[str, deque] = {}

        # Baseline statistics for drift detection
        self.baselines: Dict[str, Dict] = {}

        logger.info("Deployment tracker initialized")

    def record_inference(
        self,
        model_name: str,
        model_version: str,
        latency_ms: float,
        input_shape: tuple,
        success: bool = True,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record a model inference.

        Args:
            model_name: Model name
            model_version: Model version
            latency_ms: Inference latency in milliseconds
            input_shape: Input tensor shape
            success: Whether inference succeeded
            error_message: Error message if failed
            metrics: Optional quality metrics
        """
        model_key = f"{model_name}:{model_version}"

        if model_key not in self.recent_inferences:
            self.recent_inferences[model_key] = deque(maxlen=self.window_size)

        record = InferenceRecord(
            timestamp=datetime.now(),
            model_name=model_name,
            model_version=model_version,
            latency_ms=latency_ms,
            input_shape=input_shape,
            success=success,
            error_message=error_message,
            metrics=metrics or {},
        )

        self.recent_inferences[model_key].append(record)

        # Check for alerts
        self._check_alerts(model_key)

    def get_statistics(
        self,
        model_name: str,
        model_version: str,
        time_window: Optional[timedelta] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for a deployed model.

        Args:
            model_name: Model name
            model_version: Model version
            time_window: Optional time window for stats

        Returns:
            Statistics dictionary
        """
        model_key = f"{model_name}:{model_version}"

        if model_key not in self.recent_inferences:
            return {}

        inferences = list(self.recent_inferences[model_key])

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            inferences = [i for i in inferences if i.timestamp >= cutoff]

        if not inferences:
            return {}

        # Calculate statistics
        latencies = [i.latency_ms for i in inferences]
        successes = [i.success for i in inferences]

        stats = {
            "total_inferences": len(inferences),
            "success_rate": sum(successes) / len(successes),
            "error_rate": 1 - sum(successes) / len(successes),
            "latency": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
            },
            "time_range": {
                "start": min(i.timestamp for i in inferences).isoformat(),
                "end": max(i.timestamp for i in inferences).isoformat(),
            },
        }

        # Aggregate metrics if available
        if any(i.metrics for i in inferences):
            metrics_by_key = {}
            for inf in inferences:
                if inf.metrics:
                    for key, value in inf.metrics.items():
                        if key not in metrics_by_key:
                            metrics_by_key[key] = []
                        metrics_by_key[key].append(value)

            stats["metrics"] = {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
                for key, values in metrics_by_key.items()
            }

        return stats

    def set_baseline(
        self,
        model_name: str,
        model_version: str,
        baseline_stats: Optional[Dict] = None,
    ) -> None:
        """
        Set baseline statistics for drift detection.

        Args:
            model_name: Model name
            model_version: Model version
            baseline_stats: Optional baseline stats (if None, uses current stats)
        """
        model_key = f"{model_name}:{model_version}"

        if baseline_stats is None:
            # Use current statistics as baseline
            baseline_stats = self.get_statistics(model_name, model_version)

        self.baselines[model_key] = baseline_stats
        logger.info(f"Set baseline for {model_key}")

    def detect_drift(
        self,
        model_name: str,
        model_version: str,
        drift_threshold: float = 0.2,  # 20% change
    ) -> Dict[str, Any]:
        """
        Detect model drift compared to baseline.

        Args:
            model_name: Model name
            model_version: Model version
            drift_threshold: Threshold for drift detection (fraction)

        Returns:
            Drift detection results
        """
        model_key = f"{model_name}:{model_version}"

        if model_key not in self.baselines:
            return {"error": "No baseline set for this model"}

        baseline = self.baselines[model_key]
        current = self.get_statistics(model_name, model_version)

        if not current:
            return {"error": "No current statistics available"}

        # Compare latency
        latency_drift = abs(
            current["latency"]["mean"] - baseline["latency"]["mean"]
        ) / baseline["latency"]["mean"]

        # Compare error rate
        error_rate_drift = abs(
            current["error_rate"] - baseline["error_rate"]
        )

        # Compare metrics if available
        metrics_drift = {}
        if "metrics" in current and "metrics" in baseline:
            for key in current["metrics"]:
                if key in baseline["metrics"]:
                    curr_mean = current["metrics"][key]["mean"]
                    base_mean = baseline["metrics"][key]["mean"]
                    if base_mean != 0:
                        drift = abs(curr_mean - base_mean) / abs(base_mean)
                        metrics_drift[key] = drift

        # Determine if drift detected
        drift_detected = (
            latency_drift > drift_threshold
            or error_rate_drift > drift_threshold
            or any(d > drift_threshold for d in metrics_drift.values())
        )

        results = {
            "drift_detected": drift_detected,
            "drift_threshold": drift_threshold,
            "latency_drift": latency_drift,
            "error_rate_drift": error_rate_drift,
            "metrics_drift": metrics_drift,
            "baseline": baseline,
            "current": current,
        }

        if drift_detected:
            logger.warning(f"Drift detected for {model_key}: {results}")

        return results

    def get_error_logs(
        self,
        model_name: str,
        model_version: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Get recent error logs for a model.

        Args:
            model_name: Model name
            model_version: Model version
            limit: Maximum number of errors to return

        Returns:
            List of error records
        """
        model_key = f"{model_name}:{model_version}"

        if model_key not in self.recent_inferences:
            return []

        errors = [
            {
                "timestamp": i.timestamp.isoformat(),
                "error_message": i.error_message,
                "input_shape": i.input_shape,
                "latency_ms": i.latency_ms,
            }
            for i in self.recent_inferences[model_key]
            if not i.success
        ]

        return errors[-limit:]

    def get_input_distribution(
        self,
        model_name: str,
        model_version: str,
    ) -> Dict[str, Any]:
        """
        Get input shape distribution for a model.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            Distribution statistics
        """
        model_key = f"{model_name}:{model_version}"

        if model_key not in self.recent_inferences:
            return {}

        shapes = [i.input_shape for i in self.recent_inferences[model_key]]

        # Count shape occurrences
        shape_counts = {}
        for shape in shapes:
            shape_str = str(shape)
            shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1

        return {
            "total_samples": len(shapes),
            "unique_shapes": len(shape_counts),
            "shape_distribution": shape_counts,
        }

    def _check_alerts(self, model_key: str) -> None:
        """
        Check for alerts based on recent inferences.

        Args:
            model_key: Model key (name:version)
        """
        inferences = list(self.recent_inferences[model_key])

        # Only check if we have enough samples
        if len(inferences) < 10:
            return

        # Check latency
        recent_latencies = [i.latency_ms for i in inferences[-10:]]
        avg_latency = np.mean(recent_latencies)

        if avg_latency > self.alert_threshold_latency:
            logger.warning(
                f"High latency alert for {model_key}: {avg_latency:.2f}ms "
                f"(threshold: {self.alert_threshold_latency}ms)"
            )

        # Check error rate
        recent_successes = [i.success for i in inferences[-10:]]
        error_rate = 1 - sum(recent_successes) / len(recent_successes)

        if error_rate > self.alert_threshold_error_rate:
            logger.warning(
                f"High error rate alert for {model_key}: {error_rate:.2%} "
                f"(threshold: {self.alert_threshold_error_rate:.2%})"
            )

    def export_report(
        self,
        model_name: str,
        model_version: str,
        output_path: str,
    ) -> None:
        """
        Export monitoring report to file.

        Args:
            model_name: Model name
            model_version: Model version
            output_path: Output file path
        """
        import json

        stats = self.get_statistics(model_name, model_version)
        errors = self.get_error_logs(model_name, model_version)
        distribution = self.get_input_distribution(model_name, model_version)

        report = {
            "model": {"name": model_name, "version": model_version},
            "statistics": stats,
            "recent_errors": errors,
            "input_distribution": distribution,
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported monitoring report to {output_path}")


class ResourceMonitor:
    """Monitor resource utilization for model serving."""

    def __init__(self):
        """Initialize resource monitor."""
        self.gpu_available = self._check_gpu()
        logger.info(f"Resource monitor initialized (GPU: {self.gpu_available})")

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.

        Returns:
            GPU statistics or empty dict if not available
        """
        if not self.gpu_available:
            return {}

        try:
            import torch

            gpu_stats = {}

            for i in range(torch.cuda.device_count()):
                gpu_stats[f"gpu_{i}"] = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_mb": torch.cuda.memory_allocated(i) / 1024**2,
                    "memory_reserved_mb": torch.cuda.memory_reserved(i) / 1024**2,
                    "max_memory_allocated_mb": torch.cuda.max_memory_allocated(i)
                    / 1024**2,
                }

            return gpu_stats

        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return {}

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get system memory statistics.

        Returns:
            Memory statistics in MB
        """
        try:
            import psutil

            memory = psutil.virtual_memory()

            return {
                "total_mb": memory.total / 1024**2,
                "available_mb": memory.available / 1024**2,
                "used_mb": memory.used / 1024**2,
                "percent": memory.percent,
            }

        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {}

    def get_cpu_stats(self) -> Dict[str, float]:
        """
        Get CPU statistics.

        Returns:
            CPU statistics
        """
        try:
            import psutil

            return {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
            }

        except ImportError:
            logger.warning("psutil not available for CPU monitoring")
            return {}

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get all resource statistics.

        Returns:
            Combined resource statistics
        """
        return {
            "gpu": self.get_gpu_stats(),
            "memory": self.get_memory_stats(),
            "cpu": self.get_cpu_stats(),
            "timestamp": datetime.now().isoformat(),
        }
