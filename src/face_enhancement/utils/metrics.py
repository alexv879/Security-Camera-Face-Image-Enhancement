"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time


# Metrics
enhancement_requests = Counter(
    'face_enhancement_requests_total',
    'Total number of enhancement requests',
    ['model', 'status']
)

enhancement_duration = Histogram(
    'face_enhancement_duration_seconds',
    'Time spent processing enhancement',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

faces_detected = Histogram(
    'face_enhancement_faces_detected',
    'Number of faces detected per image',
    buckets=[0, 1, 2, 3, 5, 10, 20]
)

image_size = Histogram(
    'face_enhancement_image_size_pixels',
    'Size of processed images in pixels',
    buckets=[100000, 500000, 1000000, 2000000, 5000000, 10000000]
)

active_requests = Gauge(
    'face_enhancement_active_requests',
    'Number of requests currently being processed'
)

pipeline_info = Info(
    'face_enhancement_pipeline',
    'Information about the enhancement pipeline'
)


def track_request(model: str = "unknown"):
    """Decorator to track request metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                enhancement_duration.labels(model=model).observe(duration)
                enhancement_requests.labels(model=model, status=status).inc()
                active_requests.dec()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                enhancement_duration.labels(model=model).observe(duration)
                enhancement_requests.labels(model=model, status=status).inc()
                active_requests.dec()

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def record_faces_detected(count: int) -> None:
    """Record number of faces detected."""
    faces_detected.observe(count)


def record_image_size(width: int, height: int) -> None:
    """Record image size in pixels."""
    pixels = width * height
    image_size.observe(pixels)


def set_pipeline_info(config: dict) -> None:
    """Set pipeline configuration info."""
    pipeline_info.info(config)
