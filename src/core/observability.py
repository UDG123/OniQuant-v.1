"""OpenTelemetry-compatible observability — latency tracking & auto-kill.

Provides three instrumentation layers consumed by the FastAPI gateway,
Binance WebSocket consumer, and Celery workers:

1. **FeedLatencyTracker** — compares the exchange-originated timestamp
   (``E`` field in Binance payloads, epoch ms) with the local
   high-precision monotonic clock to measure feed ingestion lag.

2. **PipelineLatencyTracker** — records wall-clock duration from Stage 1
   (Pydantic validation) through Stage 8 (Telegram broadcast) of the
   webhook pipeline, plus per-stage breakdowns.

3. **LatencyCircuitBreaker** — auto-kill rule: if the P99 feed latency
   exceeds 200 ms for 3+ consecutive ticks, the global volatility lock
   is set to ``LATENCY_HALT`` via Redis.

All metrics are exported via ``prometheus_client`` and served at
``/api/v1/metrics`` by the FastAPI gateway.

Usage::

    from src.core.observability import (
        feed_tracker,
        pipeline_tracker,
        circuit_breaker,
        get_metrics_app,
    )
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    make_wsgi_app,
)

logger = logging.getLogger("oniquant.observability")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_P99_THRESHOLD_MS: float = float(os.getenv("LATENCY_P99_THRESHOLD_MS", "200"))
_CONSECUTIVE_BREACHES: int = int(os.getenv("LATENCY_BREACH_COUNT", "3"))
_P99_WINDOW: int = int(os.getenv("LATENCY_P99_WINDOW", "100"))

# ---------------------------------------------------------------------------
# Prometheus registry (dedicated so we don't collide with default)
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Metric definitions — Feed Latency
# ---------------------------------------------------------------------------

FEED_LATENCY_HISTOGRAM = Histogram(
    "oniquant_feed_latency_ms",
    "Latency between exchange event timestamp and local receipt (ms)",
    labelnames=["symbol", "source"],
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
    registry=REGISTRY,
)

FEED_LATENCY_P99_GAUGE = Gauge(
    "oniquant_feed_latency_p99_ms",
    "Rolling P99 feed latency over the last N ticks (ms)",
    labelnames=["symbol"],
    registry=REGISTRY,
)

FEED_TICKS_TOTAL = Counter(
    "oniquant_feed_ticks_total",
    "Total book-ticker messages processed",
    labelnames=["symbol"],
    registry=REGISTRY,
)

FEED_TICKS_NO_TIMESTAMP = Counter(
    "oniquant_feed_ticks_no_exchange_ts",
    "Ticks received without an exchange-originated timestamp",
    labelnames=["symbol"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Metric definitions — Pipeline Latency
# ---------------------------------------------------------------------------

PIPELINE_LATENCY_HISTOGRAM = Histogram(
    "oniquant_pipeline_latency_ms",
    "Total webhook pipeline latency from Stage 1 to Stage 8 (ms)",
    labelnames=["desk_id", "action"],
    buckets=(5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000),
    registry=REGISTRY,
)

PIPELINE_STAGE_HISTOGRAM = Histogram(
    "oniquant_pipeline_stage_latency_ms",
    "Per-stage latency within the webhook pipeline (ms)",
    labelnames=["stage"],
    buckets=(0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500),
    registry=REGISTRY,
)

PIPELINE_REQUESTS_TOTAL = Counter(
    "oniquant_pipeline_requests_total",
    "Total webhook requests processed",
    labelnames=["desk_id", "outcome"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Metric definitions — Network Round-Trip
# ---------------------------------------------------------------------------

NETWORK_RTT_HISTOGRAM = Histogram(
    "oniquant_network_rtt_ms",
    "Network round-trip latency to external services",
    labelnames=["service"],
    buckets=(5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Metric definitions — Circuit Breaker
# ---------------------------------------------------------------------------

CIRCUIT_BREAKER_STATE = Gauge(
    "oniquant_circuit_breaker_active",
    "1 if the latency circuit breaker has tripped, 0 otherwise",
    registry=REGISTRY,
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "oniquant_circuit_breaker_trips_total",
    "Total number of times the latency circuit breaker has tripped",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Metric definitions — Service info
# ---------------------------------------------------------------------------

SERVICE_INFO = Info(
    "oniquant",
    "OniQuant service metadata",
    registry=REGISTRY,
)
SERVICE_INFO.info({
    "version": os.getenv("ONIQUANT_VERSION", "0.1.0"),
    "environment": os.getenv("RAILWAY_ENVIRONMENT", "development"),
})


# ---------------------------------------------------------------------------
# FeedLatencyTracker
# ---------------------------------------------------------------------------


class FeedLatencyTracker:
    """Measures exchange-to-local ingestion latency per symbol.

    Binance ``bookTicker`` payloads include an ``E`` field (event time,
    epoch milliseconds) on futures and newer spot streams.  The tracker
    compares this against the local high-precision clock to estimate
    feed lag.

    Maintains a per-symbol rolling window for P99 computation.
    """

    def __init__(self, p99_window: int = _P99_WINDOW) -> None:
        self._windows: dict[str, deque[float]] = {}
        self._p99_window = p99_window

    def _get_window(self, symbol: str) -> deque[float]:
        if symbol not in self._windows:
            self._windows[symbol] = deque(maxlen=self._p99_window)
        return self._windows[symbol]

    def record_tick(
        self,
        symbol: str,
        exchange_epoch_ms: int | None,
        local_epoch_ms: float | None = None,
        source: str = "binance",
    ) -> float | None:
        """Record a single tick's feed latency.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. "BTCUSDT").
        exchange_epoch_ms : int | None
            Exchange-originated event timestamp in epoch milliseconds
            (the ``E`` field from Binance payloads).  ``None`` when the
            field is absent.
        local_epoch_ms : float | None
            Local receipt time in epoch ms.  Defaults to
            ``time.time() * 1000``.
        source : str
            Feed source label for the histogram (default "binance").

        Returns
        -------
        float | None
            Computed latency in ms, or ``None`` if the exchange timestamp
            was unavailable.
        """
        sym = symbol.upper()
        FEED_TICKS_TOTAL.labels(symbol=sym).inc()

        if exchange_epoch_ms is None:
            FEED_TICKS_NO_TIMESTAMP.labels(symbol=sym).inc()
            return None

        if local_epoch_ms is None:
            local_epoch_ms = time.time() * 1000.0

        latency_ms = local_epoch_ms - float(exchange_epoch_ms)

        # Clamp negative values (clock skew) to zero.
        latency_ms = max(latency_ms, 0.0)

        FEED_LATENCY_HISTOGRAM.labels(symbol=sym, source=source).observe(latency_ms)

        window = self._get_window(sym)
        window.append(latency_ms)

        p99 = self._compute_p99(window)
        FEED_LATENCY_P99_GAUGE.labels(symbol=sym).set(p99)

        return latency_ms

    def get_p99(self, symbol: str) -> float:
        """Return the current rolling P99 for *symbol*."""
        window = self._windows.get(symbol.upper())
        if not window:
            return 0.0
        return self._compute_p99(window)

    @staticmethod
    def _compute_p99(window: deque[float]) -> float:
        if len(window) < 2:
            return window[0] if window else 0.0
        sorted_vals = sorted(window)
        idx = int(len(sorted_vals) * 0.99)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]


# ---------------------------------------------------------------------------
# PipelineLatencyTracker
# ---------------------------------------------------------------------------


class PipelineLatencyTracker:
    """Tracks wall-clock latency across webhook pipeline stages.

    Usage in the webhook handler::

        tracker = pipeline_tracker.start("signal-abc", desk_id=1, action="BUY")
        # ... Stage 2 ...
        tracker.mark("dedup")
        # ... Stage 3 ...
        tracker.mark("volatility_lock")
        # ... etc ...
        tracker.finish("executed")
    """

    class _Span:
        """Lightweight span tracking stage timestamps."""

        __slots__ = (
            "signal_id", "desk_id", "action",
            "_start_ns", "_last_ns", "_stages",
        )

        def __init__(
            self, signal_id: str, desk_id: int | str, action: str,
        ) -> None:
            self.signal_id = signal_id
            self.desk_id = str(desk_id)
            self.action = action
            self._start_ns = time.monotonic_ns()
            self._last_ns = self._start_ns
            self._stages: list[tuple[str, float]] = []

        def mark(self, stage_name: str) -> None:
            """Record a stage boundary."""
            now = time.monotonic_ns()
            stage_ms = (now - self._last_ns) / 1_000_000
            self._stages.append((stage_name, stage_ms))
            PIPELINE_STAGE_HISTOGRAM.labels(stage=stage_name).observe(stage_ms)
            self._last_ns = now

        def finish(self, outcome: str = "completed") -> float:
            """Finalise the span and record total pipeline latency.

            Returns total latency in ms.
            """
            total_ns = time.monotonic_ns() - self._start_ns
            total_ms = total_ns / 1_000_000

            PIPELINE_LATENCY_HISTOGRAM.labels(
                desk_id=self.desk_id, action=self.action,
            ).observe(total_ms)

            PIPELINE_REQUESTS_TOTAL.labels(
                desk_id=self.desk_id, outcome=outcome,
            ).inc()

            logger.debug(
                "Pipeline %s: %.1fms total | stages: %s",
                self.signal_id,
                total_ms,
                " → ".join(f"{s}={m:.1f}ms" for s, m in self._stages),
            )

            return total_ms

    def start(
        self, signal_id: str, desk_id: int | str = 0, action: str = "UNKNOWN",
    ) -> _Span:
        """Begin tracking a new pipeline invocation."""
        return self._Span(signal_id, desk_id, action)


# ---------------------------------------------------------------------------
# Network RTT helper
# ---------------------------------------------------------------------------


@contextmanager
def track_network_rtt(service: str) -> Generator[None, None, None]:
    """Context manager that records network round-trip time.

    Usage::

        with track_network_rtt("claude_api"):
            response = await client.post(...)
    """
    start = time.monotonic_ns()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic_ns() - start) / 1_000_000
        NETWORK_RTT_HISTOGRAM.labels(service=service).observe(elapsed_ms)


# ---------------------------------------------------------------------------
# LatencyCircuitBreaker
# ---------------------------------------------------------------------------


class LatencyCircuitBreaker:
    """Auto-kill rule: trip when P99 > threshold for N consecutive ticks.

    When tripped, sets ``global:volatility_lock`` to ``"LATENCY_HALT"``
    via Redis, blocking all new order execution until the condition
    clears or an operator lifts the halt.
    """

    def __init__(
        self,
        threshold_ms: float = _P99_THRESHOLD_MS,
        consecutive_breaches: int = _CONSECUTIVE_BREACHES,
    ) -> None:
        self._threshold_ms = threshold_ms
        self._required = consecutive_breaches
        self._breach_count = 0
        self._tripped = False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    async def evaluate(
        self,
        p99_ms: float,
        redis_client: Any,
    ) -> bool:
        """Evaluate the latest P99 reading against the threshold.

        Parameters
        ----------
        p99_ms : float
            Current P99 feed latency in milliseconds.
        redis_client : redis.asyncio.Redis
            Async Redis client for setting the volatility lock.

        Returns
        -------
        bool
            ``True`` if the circuit breaker **just tripped** on this call.
        """
        if p99_ms > self._threshold_ms:
            self._breach_count += 1
            logger.warning(
                "Latency breach %d/%d: P99=%.1fms > %.1fms threshold",
                self._breach_count,
                self._required,
                p99_ms,
                self._threshold_ms,
            )
        else:
            if self._breach_count > 0:
                logger.info(
                    "Latency breach count reset (P99=%.1fms < %.1fms)",
                    p99_ms,
                    self._threshold_ms,
                )
            self._breach_count = 0
            return False

        if self._breach_count >= self._required and not self._tripped:
            self._tripped = True
            CIRCUIT_BREAKER_STATE.set(1)
            CIRCUIT_BREAKER_TRIPS.inc()

            await redis_client.set(
                "global:volatility_lock",
                "LATENCY_HALT",
                ex=86_400,
            )

            logger.critical(
                "LATENCY CIRCUIT BREAKER TRIPPED — P99=%.1fms exceeded "
                "%.1fms for %d consecutive ticks. "
                "global:volatility_lock set to LATENCY_HALT.",
                p99_ms,
                self._threshold_ms,
                self._required,
            )
            return True

        return False

    def reset(self) -> None:
        """Manually reset the circuit breaker state."""
        self._breach_count = 0
        self._tripped = False
        CIRCUIT_BREAKER_STATE.set(0)
        logger.info("Latency circuit breaker manually reset")


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

feed_tracker = FeedLatencyTracker()
pipeline_tracker = PipelineLatencyTracker()
circuit_breaker = LatencyCircuitBreaker()


def get_metrics_app():
    """Return a WSGI app that serves Prometheus metrics at /."""
    return make_wsgi_app(REGISTRY)


def generate_metrics_text() -> bytes:
    """Generate Prometheus text-format metrics for direct HTTP responses."""
    return generate_latest(REGISTRY)
