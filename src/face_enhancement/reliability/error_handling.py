"""Reliability Patterns.

Provides:
- Error handling and recovery
- Retry logic with exponential backoff
- Circuit breaker pattern
- Graceful degradation
- Fallback mechanisms
- Timeout management
"""

from typing import Optional, Callable, Any, Type, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import time
import threading
from functools import wraps
from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"  # Fixed delay
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True  # Add randomness

    # Retry conditions
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    retry_on_result: Optional[Callable[[Any], bool]] = None


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before half-open
    expected_exception: Type[Exception] = Exception


class RetryableError(Exception):
    """Error that should trigger retry."""
    pass


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


class TimeoutError(Exception):
    """Operation timed out."""
    pass


def retry(config: Optional[RetryConfig] = None):
    """
    Retry decorator with exponential backoff.

    Usage:
        @retry(RetryConfig(max_attempts=3))
        def flaky_function():
            # might fail
            pass
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # Check if should retry based on result
                    if config.retry_on_result and config.retry_on_result(result):
                        logger.warning(
                            f"Retry condition met for result, "
                            f"attempt {attempt + 1}/{config.max_attempts}"
                        )
                        raise RetryableError("Result triggered retry")

                    # Success
                    if attempt > 0:
                        logger.info(
                            f"Success after {attempt + 1} attempts: {func.__name__}"
                        )

                    return result

                except config.retry_on_exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        # Calculate delay
                        if config.strategy == RetryStrategy.EXPONENTIAL:
                            delay = min(
                                config.initial_delay * (config.backoff_factor ** attempt),
                                config.max_delay
                            )
                        elif config.strategy == RetryStrategy.LINEAR:
                            delay = min(
                                config.initial_delay * (attempt + 1),
                                config.max_delay
                            )
                        else:  # FIXED
                            delay = config.initial_delay

                        # Add jitter
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random())

                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed: {func.__name__}"
                        )

            # All attempts exhausted
            raise last_exception

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascade failures by failing fast when errors exceed threshold.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

        self._lock = threading.Lock()

        logger.info(f"Circuit breaker initialized: {self.config}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError if circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                # Check if should transition to half-open
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. "
                        f"Retry after {self.config.timeout}s"
                    )

        # Try to call function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1

                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED (recovered)")

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                # Failed while testing, back to open
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker: HALF_OPEN -> OPEN (still failing)")

            elif self.failure_count >= self.config.failure_threshold:
                # Too many failures, open circuit
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker: CLOSED -> OPEN "
                    f"({self.failure_count} failures)"
                )

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            logger.info("Circuit breaker manually reset")


def circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """
    Circuit breaker decorator.

    Usage:
        @circuit_breaker(CircuitBreakerConfig(failure_threshold=5))
        def unreliable_function():
            # might fail
            pass
    """
    breaker = CircuitBreaker(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        # Expose breaker for manual control
        wrapper.circuit_breaker = breaker

        return wrapper

    return decorator


def with_timeout(seconds: float):
    """
    Timeout decorator.

    Usage:
        @with_timeout(30.0)
        def slow_function():
            # might take too long
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError("Function timed out")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                logger.error(f"Function timed out after {seconds}s: {func.__name__}")
                raise TimeoutError(f"Function timed out after {seconds}s")

            if isinstance(result[0], Exception):
                raise result[0]

            return result[0]

        return wrapper

    return decorator


class Fallback:
    """Fallback mechanism for graceful degradation."""

    @staticmethod
    def with_fallback(
        primary: Callable,
        fallback: Callable,
        fallback_on_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ) -> Any:
        """
        Try primary function, fall back to fallback on error.

        Args:
            primary: Primary function to try
            fallback: Fallback function
            fallback_on_exceptions: Exceptions that trigger fallback

        Returns:
            Result from primary or fallback
        """
        try:
            return primary()
        except fallback_on_exceptions as e:
            logger.warning(
                f"Primary function failed: {e}. Using fallback."
            )
            return fallback()


class BulkheadPool:
    """Bulkhead pattern - isolate resources."""

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize bulkhead pool.

        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent

        logger.info(f"Bulkhead pool initialized: max_concurrent={max_concurrent}")

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with bulkhead protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        acquired = self.semaphore.acquire(blocking=True, timeout=30.0)

        if not acquired:
            raise TimeoutError("Could not acquire bulkhead slot")

        try:
            return func(*args, **kwargs)
        finally:
            self.semaphore.release()


class ErrorRecovery:
    """Error recovery strategies."""

    @staticmethod
    def safe_execute(
        func: Callable,
        default: Any = None,
        log_errors: bool = True,
    ) -> Any:
        """
        Execute function safely with error handling.

        Args:
            func: Function to execute
            default: Default value on error
            log_errors: Whether to log errors

        Returns:
            Function result or default
        """
        try:
            return func()
        except Exception as e:
            if log_errors:
                logger.error(f"Error in safe_execute: {e}")
            return default

    @staticmethod
    def with_recovery(
        func: Callable,
        recovery_func: Callable[[Exception], None],
    ) -> Any:
        """
        Execute with recovery function on error.

        Args:
            func: Function to execute
            recovery_func: Recovery function (receives exception)

        Returns:
            Function result
        """
        try:
            return func()
        except Exception as e:
            logger.warning(f"Error occurred, running recovery: {e}")
            recovery_func(e)
            raise


# Global bulkhead pools
default_bulkhead = BulkheadPool(max_concurrent=100)
cpu_intensive_bulkhead = BulkheadPool(max_concurrent=4)
io_intensive_bulkhead = BulkheadPool(max_concurrent=50)
