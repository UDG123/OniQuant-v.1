"""Sentinel forensic-audit service — causality & slippage verification.

Periodically queries ``trade_log`` and ``reasoning_logs`` to enforce two
post-trade integrity invariants:

1. **Causality Check** — the ClaudeCTO approval timestamp
   (``reasoning_logs.created_at`` for APPROVED rows) must be *strictly*
   earlier than the trade execution timestamp (``trade_log.opened_at``).
   A violation indicates clock skew, race conditions, or data-pipeline
   corruption.

2. **Slippage Audit** — the absolute delta between the signal price
   embedded in the reasoning payload and the recorded fill price
   (``trade_log.entry_price``) must not exceed 3× the ATR captured at
   entry (``ml_training_data.atr_at_entry``).  Breaches are flagged for
   manual review.

Every trade that fails either check produces a ``ForensicTrace`` JSON
artifact written to the ``audit_artifacts/`` directory and logged at
WARNING level.

Usage — standalone::

    auditor = SentinelAudit(postgres_url="postgresql://…")
    traces = auditor.run_audit(lookback_hours=24)

Usage — Celery beat (add to an existing worker's schedule)::

    from src.core.sentinel_audit import SentinelAudit
    auditor = SentinelAudit()
    auditor.run_audit()
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
).replace("+asyncpg", "")

_DEFAULT_REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_SLIPPAGE_ATR_MULTIPLIER: float = float(
    os.getenv("SENTINEL_SLIPPAGE_ATR_MULT", "3.0")
)
_DEFAULT_LOOKBACK_HOURS: int = int(os.getenv("SENTINEL_LOOKBACK_HOURS", "24"))
_ARTIFACT_DIR: str = os.getenv("SENTINEL_ARTIFACT_DIR", "audit_artifacts")

logger = logging.getLogger("oniquant.core.sentinel_audit")

# ---------------------------------------------------------------------------
# ForensicTrace — immutable evidence container
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ForensicTrace:
    """JSON-serialisable artifact emitted for every failed sanity check."""

    trade_id: int
    signal_id: str
    symbol: str
    desk_id: int
    side: str
    checks_failed: list[str]

    # Causality fields
    approved_at: str | None = None
    execution_timestamp: str | None = None
    causality_delta_ms: int | None = None

    # Slippage fields
    signal_price: float | None = None
    fill_price: float | None = None
    slippage_abs: float | None = None
    atr_at_entry: float | None = None
    slippage_atr_ratio: float | None = None

    # Context
    audited_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    severity: str = "WARNING"

    def to_json(self) -> str:
        """Produce a deterministic, pretty-printed JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

# Lateral join: for each closed trade, find the most recent APPROVED
# reasoning_log on the same symbol + desk that precedes (or violates)
# the trade's opened_at.  Also pull ml_training_data for ATR + the
# signal_price extracted from the hydrated_payload JSON.
_AUDIT_QUERY = text("""
    SELECT
        tl.id              AS trade_id,
        tl.desk_id,
        tl.symbol,
        tl.side,
        tl.entry_price     AS fill_price,
        tl.opened_at       AS execution_ts,
        rl.signal_id,
        rl.created_at      AS approved_at,
        rl.hydrated_payload,
        ml.atr_at_entry
    FROM trade_log tl
    LEFT JOIN LATERAL (
        SELECT r.signal_id, r.created_at, r.hydrated_payload
        FROM reasoning_logs r
        WHERE r.symbol   = tl.symbol
          AND r.desk_id  = tl.desk_id
          AND r.status   = 'APPROVED'
          AND r.created_at <= tl.opened_at + INTERVAL '5 minutes'
        ORDER BY r.created_at DESC
        LIMIT 1
    ) rl ON TRUE
    LEFT JOIN LATERAL (
        SELECT m.atr_at_entry
        FROM ml_training_data m
        WHERE m.symbol  = tl.symbol
          AND m.desk_id = tl.desk_id
        ORDER BY ABS(EXTRACT(EPOCH FROM m.closed_at - tl.closed_at))
        LIMIT 1
    ) ml ON TRUE
    WHERE tl.status    = 'SIM_CLOSED'
      AND tl.closed_at >= NOW() - MAKE_INTERVAL(hours => :lookback_hours)
    ORDER BY tl.closed_at DESC
""")


# ---------------------------------------------------------------------------
# SentinelAudit service
# ---------------------------------------------------------------------------


class SentinelAudit:
    """Forensic audit engine for post-trade integrity verification.

    Parameters
    ----------
    postgres_url : str | None
        PostgreSQL connection string.  Falls back to ``$POSTGRES_URL``.
    redis_url : str | None
        Stored for downstream consumers; not directly queried by the
        auditor.  Falls back to ``$REDIS_URL``.
    artifact_dir : str | Path
        Directory where ``ForensicTrace`` JSON artefacts are written.
    """

    def __init__(
        self,
        postgres_url: str | None = None,
        redis_url: str | None = None,
        artifact_dir: str | Path = _ARTIFACT_DIR,
    ) -> None:
        pg_url = (postgres_url or _DEFAULT_POSTGRES_URL).replace("+asyncpg", "")
        self._engine = create_engine(pg_url, pool_pre_ping=True, pool_size=3)
        self._session_factory = sessionmaker(bind=self._engine)
        self._redis_url = redis_url or _DEFAULT_REDIS_URL
        self._artifact_dir = Path(artifact_dir)

    # ----- public API -------------------------------------------------------

    def run_audit(
        self,
        lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
    ) -> list[ForensicTrace]:
        """Execute causality + slippage checks and return failed traces.

        Parameters
        ----------
        lookback_hours : int
            Only audit trades whose ``closed_at`` falls within this many
            hours from now.  Defaults to 24.

        Returns
        -------
        list[ForensicTrace]
            One trace per trade that violated at least one invariant.
        """
        traces: list[ForensicTrace] = []

        with self._session_factory() as session:
            rows = session.execute(
                _AUDIT_QUERY,
                {"lookback_hours": lookback_hours},
            ).fetchall()

        logger.info(
            "Sentinel audit: %d closed trades in the last %d h",
            len(rows),
            lookback_hours,
        )

        for row in rows:
            failed = self._audit_single_trade(row)
            if failed is not None:
                traces.append(failed)
                self._export_artifact(failed)

        if traces:
            logger.warning(
                "Sentinel audit complete: %d / %d trades FAILED checks",
                len(traces),
                len(rows),
            )
        else:
            logger.info("Sentinel audit complete: all %d trades PASSED", len(rows))

        return traces

    # ----- per-trade auditing -----------------------------------------------

    def _audit_single_trade(self, row: Any) -> ForensicTrace | None:
        """Run both checks against a single trade row.

        Returns ``None`` when all checks pass.
        """
        checks_failed: list[str] = []

        # --- Causality Check ------------------------------------------------
        approved_at: datetime | None = row.approved_at
        execution_ts: datetime | None = row.execution_ts
        causality_delta_ms: int | None = None

        if approved_at is not None and execution_ts is not None:
            delta = execution_ts - approved_at
            causality_delta_ms = int(delta.total_seconds() * 1000)
            if causality_delta_ms <= 0:
                checks_failed.append("CAUSALITY_VIOLATION")
                logger.warning(
                    "Causality violation: trade %d approved_at=%s >= execution_ts=%s "
                    "(delta=%d ms)",
                    row.trade_id,
                    approved_at.isoformat(),
                    execution_ts.isoformat(),
                    causality_delta_ms,
                )
        elif approved_at is None:
            checks_failed.append("MISSING_APPROVAL_RECORD")
            logger.warning(
                "No APPROVED reasoning_log found for trade %d (%s %s)",
                row.trade_id,
                row.symbol,
                row.side,
            )

        # --- Slippage Audit -------------------------------------------------
        fill_price: float = float(row.fill_price)
        signal_price: float | None = self._extract_signal_price(
            row.hydrated_payload,
        )
        atr: float | None = (
            float(row.atr_at_entry) if row.atr_at_entry is not None else None
        )
        slippage_abs: float | None = None
        slippage_atr_ratio: float | None = None

        if signal_price is not None and atr is not None and atr > 0:
            slippage_abs = abs(fill_price - signal_price)
            slippage_atr_ratio = slippage_abs / atr
            if slippage_atr_ratio > _SLIPPAGE_ATR_MULTIPLIER:
                checks_failed.append("EXCESSIVE_SLIPPAGE")
                logger.warning(
                    "Slippage breach: trade %d signal=%.8f fill=%.8f "
                    "slip=%.8f ATR=%.8f ratio=%.2f (limit=%.1f)",
                    row.trade_id,
                    signal_price,
                    fill_price,
                    slippage_abs,
                    atr,
                    slippage_atr_ratio,
                    _SLIPPAGE_ATR_MULTIPLIER,
                )
        elif signal_price is None:
            logger.debug(
                "Could not extract signal_price from payload for trade %d",
                row.trade_id,
            )

        if not checks_failed:
            return None

        return ForensicTrace(
            trade_id=row.trade_id,
            signal_id=row.signal_id or "unknown",
            symbol=row.symbol,
            desk_id=row.desk_id,
            side=row.side,
            checks_failed=checks_failed,
            approved_at=(
                approved_at.isoformat() if approved_at is not None else None
            ),
            execution_timestamp=(
                execution_ts.isoformat() if execution_ts is not None else None
            ),
            causality_delta_ms=causality_delta_ms,
            signal_price=signal_price,
            fill_price=fill_price,
            slippage_abs=slippage_abs,
            atr_at_entry=atr,
            slippage_atr_ratio=slippage_atr_ratio,
            severity=(
                "CRITICAL"
                if "CAUSALITY_VIOLATION" in checks_failed
                else "WARNING"
            ),
        )

    # ----- signal price extraction ------------------------------------------

    @staticmethod
    def _extract_signal_price(hydrated_payload: str | None) -> float | None:
        """Best-effort extraction of ``signal_price`` from the JSON payload.

        The hydrated payload written by ``claude_cto_dispatcher`` contains the
        original webhook body under ``webhook.price`` or ``webhook.close``.
        """
        if not hydrated_payload:
            return None
        try:
            payload = json.loads(hydrated_payload)
        except (json.JSONDecodeError, TypeError):
            return None

        webhook: dict = payload.get("webhook", {})
        for key in ("price", "close", "signal_price", "entry_price"):
            val = webhook.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    # ----- artifact export --------------------------------------------------

    def _export_artifact(self, trace: ForensicTrace) -> Path:
        """Write a ForensicTrace JSON file and return its path."""
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"forensic_{trace.trade_id}_{trace.symbol}_"
            f"{trace.audited_at.replace(':', '-')}.json"
        )
        path = self._artifact_dir / filename
        path.write_text(trace.to_json(), encoding="utf-8")
        logger.info("Forensic artifact exported → %s", path)
        return path
