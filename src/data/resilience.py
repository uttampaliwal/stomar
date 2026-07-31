"""Resilience utilities: retry with exponential backoff and circuit breaker.

Usage:
    from src.data.resilience import retry_with_backoff, CircuitBreaker

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def fetch_data():
        ...

    yf_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    data = yf_breaker.call(yf_download, ticker)
"""

import time
import logging
import threading
import functools
from typing import Callable

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter for third-party API fairness.

    Guarantees at most `rate` calls per second per key, enforced with a
    short sleep inside acquire(). Safe to share across threads.

    Usage:
        limiter = RateLimiter(rate=3, burst=5)   # 3 req/s, burst of 5
        with limiter.acquire("yfinance"):
            data = yf.download(ticker)
    """

    def __init__(self, rate: float = 3.0, burst: int = 5, name: str = "rate_limiter"):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.burst = max(1, burst)
        self.name = name
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str = "default"):
        """Context manager: blocks until a token is available for *key*."""
        return _RateLimiterGuard(self, key)

    def _wait_for_token(self, key: str):
        with self._lock:
            now = time.monotonic()
            refill = (now - self._last.get(key, now)) * self.rate
            tokens = min(self.burst, self._tokens.get(key, self.burst) + refill)
            if tokens >= 1.0:
                self._tokens[key] = tokens - 1.0
                self._last[key] = now
                return
            # Must wait until one token accrues
            wait = (1.0 - tokens) / self.rate
        time.sleep(wait)
        with self._lock:
            self._tokens[key] = 0.0
            self._last[key] = time.monotonic()


class _RateLimiterGuard:
    """Context manager returned by RateLimiter.acquire()."""

    def __init__(self, limiter: RateLimiter, key: str):
        self._limiter = limiter
        self._key = key

    def __enter__(self):
        self._limiter._wait_for_token(self._key)
        return self

    def __exit__(self, *exc):
        return False


def throttle(rate: float = 3.0, burst: int = 5, key: str = "default"):
    """Decorator: apply a token-bucket rate limit to a function call.

    Usage:
        @throttle(rate=2, burst=4, key="nse")
        def fetch_quote(symbol):
            ...
    """
    limiter = RateLimiter(rate=rate, burst=burst, name=key)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with limiter.acquire(key):
                return func(*args, **kwargs)
        return wrapper

    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable | None = None,
):
    """Decorator: retry a function with exponential backoff + jitter.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retry).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Cap on delay between retries.
        exceptions: Tuple of exception types to retry on.
        on_retry: Optional callback(attempt, exception, delay) called before each retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1 + max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break
                    import random
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__qualname__, attempt + 1, 1 + max_retries, exc, delay,
                    )
                    if on_retry:
                        on_retry(attempt + 1, exc, delay)
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


class CircuitBreaker:
    """Track failures per service; open the circuit after a threshold.

    States:
        CLOSED  — requests pass through normally.
        OPEN    — requests are rejected immediately without calling the function.
        HALF_OPEN — one probe request is allowed through after a cooldown.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        try:
            data = breaker.call(yf.download, ticker)
        except CircuitOpenError:
            # breaker is open — fall back to cache
            data = get_cached(ticker)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    class CircuitOpenError(Exception):
        """Raised when a circuit breaker is open and rejects a request."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_allowed = True
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                self._half_open_allowed = True
        return self._state

    def _on_success(self):
        self._failure_count = 0
        if self._state != self.CLOSED:
            logger.info("Circuit breaker '%s' closed (recovered)", self.name)
        self._state = self.CLOSED

    def _on_failure(self, exc: Exception):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            logger.warning(
                "Circuit breaker '%s' re-opened (half-open probe failed: %s)",
                self.name, exc,
            )
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                "Circuit breaker '%s' opened after %d failures",
                self.name, self._failure_count,
            )

    def call(self, func: Callable, *args, **kwargs):
        """Call *func* through the circuit breaker.

        Raises CircuitOpenError if the circuit is open.
        """
        with self._lock:
            current_state = self.state
            if current_state == self.OPEN:
                raise self.CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open "
                    f"(retry after {self.recovery_timeout}s)"
                )
            if current_state == self.HALF_OPEN and not self._half_open_allowed:
                raise self.CircuitOpenError(
                    f"Circuit breaker '{self.name}' is half-open (probe pending)"
                )
            self._half_open_allowed = False
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._on_success()
            return result
        except Exception as exc:
            with self._lock:
                self._on_failure(exc)
            raise


# ---------------------------------------------------------------------------
# Module-level circuit breakers for shared services
# ---------------------------------------------------------------------------

yf_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=120,
    name="yfinance",
)

nse_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=90,
    name="nse_api",
)
