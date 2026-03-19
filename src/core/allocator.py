"""Dynamic Fractional Kelly Criterion capital allocator.

Periodically fetches the 7-day trailing win rate (*p*) and average
payout ratio (*b* = avg_win / avg_loss) for each of the five trading
desks from the ``trade_log`` PostgreSQL table, computes the optimal
Kelly fraction, and tempers it with a configurable fractional factor
(default 0.25) to protect against parameter mis-estimation.

Kelly formula::

    f* = (p · b − (1 − p)) / b

where:
    p = win probability  (7-day trailing win rate)
    b = payout ratio     (avg winning pnl / avg losing pnl)

The 0.25 × f* fractional dampener converts the theoretical optimum
into a conservative real-world allocation that limits drawdown risk
from sampling noise in the trailing window.

After computing the ``SizeRecommendation`` for each desk, the service
writes it directly into the ``desk:{id}:state`` Redis hash so the
execution engine (``src.services.execution``) reads the ``kelly_size``
field at order-placement time and sets the trade ``quantity``.

Usage — standalone::

    allocator = KellyAllocator(postgres_url="postgresql://…")
    recs = allocator.compute_all_desks()
    for r in recs:
        print(r.to_dict())

Usage — single desk before order placement::

    rec = allocator.compute(desk_id=3, account_equity=100_000.0)
    signal["quantity"] = rec.position_size

Usage — Celery beat (periodic refresh + desk_state update)::

    from src.core.allocator import KellyAllocator
    allocator = KellyAllocator()
    allocator.refresh_and_cache()  # writes to Redis desk state + kelly keys
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
).replace("+asyncpg", "")

_DEFAULT_REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Trailing window for performance stats.
_TRAILING_DAYS: int = int(os.getenv("KELLY_TRAILING_DAYS", "7"))

# Fractional Kelly factor — 0.25 = quarter-Kelly (conservative default).
_FRACTIONAL_KELLY: float = float(os.getenv("KELLY_FRACTION", "0.25"))

# Hard caps to prevent degenerate allocations.
_MAX_KELLY_FRACTION: float = float(os.getenv("KELLY_MAX_FRACTION", "0.15"))
_MIN_KELLY_FRACTION: float = 0.0
_MIN_TRADES_REQUIRED: int = int(os.getenv("KELLY_MIN_TRADES", "10"))

# Default account equity when Redis desk state is unavailable.
_DEFAULT_ACCOUNT_EQUITY: float = float(
    os.getenv("KELLY_DEFAULT_EQUITY", "100000.0")
)

# All five desk IDs.
_DESK_IDS: list[int] = [1, 2, 3, 4, 5]

logger = logging.getLogger("oniquant.core.allocator")

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DeskStats:
    """7-day trailing performance statistics for a single desk.

    Attributes
    ----------
    win_rate : float
        *p* — fraction of winning trades in the trailing window.
    payout_ratio : float
        *b* — avg_win_pct / avg_loss_pct.  The average dollar won per
        dollar risked.  Also called the "reward-to-risk ratio".
    edge : float
        *p · b − (1 − p)* — the expected value per unit bet.
        Positive edge means the desk has a statistical advantage.
    """

    desk_id: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float         # p
    avg_win_pct: float
    avg_loss_pct: float
    payout_ratio: float     # b = avg_win / avg_loss
    edge: float             # p·b − q
    period_days: int


@dataclass(slots=True, frozen=True)
class SizeRecommendation:
    """Position-sizing recommendation consumed by execution.py.

    The ``position_size`` field maps directly to the ``quantity`` key
    in the signal dict passed to ``execute_dynamic_limit_order()``.
    The full recommendation is also written into the ``desk:{id}:state``
    Redis hash under the ``kelly_size`` key so the execution engine can
    read it at order time.
    """

    desk_id: int
    raw_kelly_fraction: float    # f* = (p·b − q) / b
    fractional_kelly: float      # 0.25 × f*
    capped_fraction: float       # min(fractional, hard_cap)
    account_equity: float
    position_size: float         # equity × capped_fraction
    risk_budget_pct: float       # capped_fraction × 100
    stats: DeskStats
    computed_at: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict for Redis caching and API responses."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# SQL — 7-day trailing desk performance
# ---------------------------------------------------------------------------

_DESK_STATS_QUERY = text("""
    SELECT
        desk_id,
        COUNT(*)                                              AS total_trades,
        SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END)        AS winning_trades,
        SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END)       AS losing_trades,
        AVG(CASE WHEN pnl_pct > 0 THEN pnl_pct END)         AS avg_win_pct,
        AVG(CASE WHEN pnl_pct <= 0 THEN ABS(pnl_pct) END)   AS avg_loss_pct
    FROM trade_log
    WHERE status = 'SIM_CLOSED'
      AND closed_at >= NOW() - MAKE_INTERVAL(days => :trailing_days)
      AND (:desk_id IS NULL OR desk_id = :desk_id)
    GROUP BY desk_id
    ORDER BY desk_id
""")

# ---------------------------------------------------------------------------
# Kelly computation (vectorised for multi-desk batch)
# ---------------------------------------------------------------------------


def _kelly_fraction(
    win_rates: np.ndarray,
    payout_ratios: np.ndarray,
) -> np.ndarray:
    """Vectorised full-Kelly computation.

    f* = (p · b − q) / b

    where p = win probability, b = payout ratio (avg_win / avg_loss),
    q = 1 − p.

    Returns 0.0 for any desk where the formula yields a non-positive
    fraction (i.e. negative edge — do not bet).
    """
    q = 1.0 - win_rates
    raw = (win_rates * payout_ratios - q) / np.where(
        payout_ratios > 0, payout_ratios, 1.0,
    )
    return np.maximum(raw, 0.0)


# ---------------------------------------------------------------------------
# KellyAllocator
# ---------------------------------------------------------------------------


class KellyAllocator:
    """Dynamic Fractional Kelly capital management service.

    Fetches the trailing win rate and profit factor per desk from
    PostgreSQL, computes the optimal investment fraction, and applies
    a fractional Kelly dampener to produce conservative position-sizing
    recommendations.

    Parameters
    ----------
    postgres_url : str | None
        Synchronous SQLAlchemy URL.  Falls back to ``$POSTGRES_URL``.
    trailing_days : int
        Look-back window for performance statistics (default 7).
    kelly_fraction : float
        Fractional Kelly multiplier (default 0.25 = quarter-Kelly).
    max_fraction : float
        Hard cap on the final allocation fraction (default 0.15).
    min_trades : int
        Minimum closed trades required to produce a recommendation.
        Desks below this threshold receive a zero allocation.
    """

    def __init__(
        self,
        postgres_url: str | None = None,
        trailing_days: int = _TRAILING_DAYS,
        kelly_fraction: float = _FRACTIONAL_KELLY,
        max_fraction: float = _MAX_KELLY_FRACTION,
        min_trades: int = _MIN_TRADES_REQUIRED,
    ) -> None:
        pg_url = (postgres_url or _DEFAULT_POSTGRES_URL).replace("+asyncpg", "")
        self._engine = create_engine(pg_url, pool_pre_ping=True, pool_size=3)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False,
        )
        self._trailing_days = trailing_days
        self._kelly_fraction = kelly_fraction
        self._max_fraction = max_fraction
        self._min_trades = min_trades

    # ----- data loading -----------------------------------------------------

    def _fetch_desk_stats(
        self,
        desk_id: int | None = None,
    ) -> dict[int, DeskStats]:
        """Query 7-day trailing stats from ``trade_log``.

        Returns a dict keyed by desk_id.  Desks with no closed trades
        in the window are absent from the result.
        """
        with self._session_factory() as session:
            rows = session.execute(
                _DESK_STATS_QUERY,
                {
                    "trailing_days": self._trailing_days,
                    "desk_id": desk_id,
                },
            ).fetchall()

        stats: dict[int, DeskStats] = {}
        for row in rows:
            did = int(row.desk_id)
            total = int(row.total_trades)
            wins = int(row.winning_trades or 0)
            losses = int(row.losing_trades or 0)

            p = wins / total if total > 0 else 0.0   # win rate
            avg_win = float(row.avg_win_pct or 0.0)
            avg_loss = float(row.avg_loss_pct or 0.0)

            # b = avg_win / avg_loss (payout ratio).
            b = avg_win / avg_loss if avg_loss > 0 else 0.0

            # Edge = p·b − (1−p).  Positive → statistical advantage.
            q = 1.0 - p
            edge = p * b - q

            stats[did] = DeskStats(
                desk_id=did,
                total_trades=total,
                winning_trades=wins,
                losing_trades=losses,
                win_rate=round(p, 6),
                avg_win_pct=round(avg_win, 8),
                avg_loss_pct=round(avg_loss, 8),
                payout_ratio=round(b, 6),
                edge=round(edge, 6),
                period_days=self._trailing_days,
            )

        return stats

    # ----- single-desk computation ------------------------------------------

    def compute(
        self,
        desk_id: int,
        account_equity: float = _DEFAULT_ACCOUNT_EQUITY,
    ) -> SizeRecommendation:
        """Compute a position-sizing recommendation for a single desk.

        Parameters
        ----------
        desk_id : int
            Desk identifier (1–5).
        account_equity : float
            Current account equity in base currency.

        Returns
        -------
        SizeRecommendation
            Ready-to-use sizing object.  ``position_size`` is the
            monetary amount to allocate to the next trade.
        """
        stats_map = self._fetch_desk_stats(desk_id=desk_id)
        stats = stats_map.get(desk_id)

        now = datetime.now(timezone.utc).isoformat()

        # Insufficient data — return zero allocation.
        if stats is None or stats.total_trades < self._min_trades:
            empty_stats = stats or DeskStats(
                desk_id=desk_id,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                profit_factor=0.0,
                period_days=self._trailing_days,
            )
            return SizeRecommendation(
                desk_id=desk_id,
                raw_kelly_fraction=0.0,
                fractional_kelly=0.0,
                capped_fraction=0.0,
                account_equity=account_equity,
                position_size=0.0,
                risk_budget_pct=0.0,
                stats=empty_stats,
                computed_at=now,
            )

        return self._build_recommendation(stats, account_equity, now)

    # ----- batch computation ------------------------------------------------

    def compute_all_desks(
        self,
        account_equities: dict[int, float] | None = None,
    ) -> list[SizeRecommendation]:
        """Compute recommendations for all five desks in one DB round-trip.

        Parameters
        ----------
        account_equities : dict[int, float] | None
            Per-desk equity overrides.  Defaults to
            ``_DEFAULT_ACCOUNT_EQUITY`` for every desk.

        Returns
        -------
        list[SizeRecommendation]
            One recommendation per desk (desks 1–5), ordered by desk_id.
        """
        equities = account_equities or {}
        stats_map = self._fetch_desk_stats(desk_id=None)
        now = datetime.now(timezone.utc).isoformat()

        recommendations: list[SizeRecommendation] = []

        # Vectorised Kelly over all desks with sufficient data.
        eligible_desks: list[int] = []
        win_rates_list: list[float] = []
        payout_ratios_list: list[float] = []

        for did in _DESK_IDS:
            stats = stats_map.get(did)
            if stats is not None and stats.total_trades >= self._min_trades:
                eligible_desks.append(did)
                win_rates_list.append(stats.win_rate)
                payout_ratios_list.append(stats.payout_ratio)

        # Compute Kelly fractions in one vectorised call.
        kelly_map: dict[int, float] = {}
        if eligible_desks:
            raw_fractions = _kelly_fraction(
                np.array(win_rates_list, dtype=np.float64),
                np.array(payout_ratios_list, dtype=np.float64),
            )
            for i, did in enumerate(eligible_desks):
                kelly_map[did] = float(raw_fractions[i])

        # Build recommendations.
        for did in _DESK_IDS:
            stats = stats_map.get(did)
            equity = equities.get(did, _DEFAULT_ACCOUNT_EQUITY)

            if stats is None or stats.total_trades < self._min_trades:
                empty_stats = stats or DeskStats(
                    desk_id=did,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    avg_win_pct=0.0,
                    avg_loss_pct=0.0,
                    payout_ratio=0.0,
                    edge=0.0,
                    period_days=self._trailing_days,
                )
                recommendations.append(SizeRecommendation(
                    desk_id=did,
                    raw_kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    capped_fraction=0.0,
                    account_equity=equity,
                    position_size=0.0,
                    risk_budget_pct=0.0,
                    stats=empty_stats,
                    computed_at=now,
                ))
                continue

            rec = self._build_recommendation(
                stats, equity, now, raw_kelly_override=kelly_map.get(did),
            )
            recommendations.append(rec)

        logger.info(
            "Kelly allocator: %d/%d desks eligible | allocations: %s",
            len(eligible_desks),
            len(_DESK_IDS),
            {r.desk_id: f"{r.capped_fraction:.4f}" for r in recommendations},
        )

        return recommendations

    # ----- Redis desk_state update + caching ---------------------------------

    def refresh_and_cache(self) -> list[SizeRecommendation]:
        """Compute all desks, update desk_state, and cache recommendations.

        For each desk this method:

        1. Writes the full ``SizeRecommendation`` JSON to
           ``kelly:{desk_id}:recommendation`` (24-hour TTL) for
           dashboards and API consumers.
        2. Updates the ``desk:{desk_id}:state`` hash with the fields
           the execution engine reads at order time:

           * ``kelly_size`` — the position size in base currency
           * ``kelly_fraction`` — the capped fraction (0–1)
           * ``kelly_edge`` — the desk's estimated edge (p·b − q)
           * ``kelly_updated_at`` — ISO timestamp of this computation

        The execution engine's ``signal.get("quantity", 1.0)`` can be
        replaced with a desk_state lookup for ``kelly_size`` to deploy
        the Kelly-optimal capital per trade.

        Requires the ``redis`` package (sync client).
        """
        import redis

        recs = self.compute_all_desks()

        r = redis.from_url(
            os.getenv("REDIS_URL", _DEFAULT_REDIS_URL),
            decode_responses=False,
        )

        pipe = r.pipeline(transaction=False)
        for rec in recs:
            # --- Full recommendation JSON (for dashboards / API) ----------
            kelly_key = f"kelly:{rec.desk_id}:recommendation"
            pipe.setex(kelly_key, 86_400, rec.to_json())

            # --- desk_state fields (for execution engine) -----------------
            state_key = f"desk:{rec.desk_id}:state"
            pipe.hset(state_key, mapping={
                b"kelly_size": str(rec.position_size).encode(),
                b"kelly_fraction": str(rec.capped_fraction).encode(),
                b"kelly_edge": str(rec.stats.edge).encode(),
                b"kelly_payout_ratio": str(rec.stats.payout_ratio).encode(),
                b"kelly_win_rate": str(rec.stats.win_rate).encode(),
                b"kelly_raw_f": str(rec.raw_kelly_fraction).encode(),
                b"kelly_updated_at": rec.computed_at.encode(),
            })

        pipe.execute()

        logger.info(
            "Kelly allocator: %d desks updated in Redis "
            "(desk_state + kelly cache) | sizes: %s",
            len(recs),
            {r.desk_id: f"${r.position_size:,.0f}" for r in recs},
        )
        return recs

    # ----- internal ---------------------------------------------------------

    def _build_recommendation(
        self,
        stats: DeskStats,
        account_equity: float,
        computed_at: str,
        raw_kelly_override: float | None = None,
    ) -> SizeRecommendation:
        """Build a SizeRecommendation from desk stats + Kelly math.

        f* = (p · b − (1 − p)) / b

        where p = win_rate, b = payout_ratio (avg_win / avg_loss).
        """
        p = stats.win_rate
        b = stats.payout_ratio
        q = 1.0 - p

        # f* = (p·b − q) / b
        if raw_kelly_override is not None:
            raw_kelly = raw_kelly_override
        else:
            raw_kelly = ((p * b - q) / b) if b > 0 else 0.0
            raw_kelly = max(raw_kelly, 0.0)

        # Apply fractional Kelly dampener.
        fractional = raw_kelly * self._kelly_fraction

        # Hard cap.
        capped = min(fractional, self._max_fraction)
        capped = max(capped, _MIN_KELLY_FRACTION)

        # Position size in base currency.
        position_size = round(account_equity * capped, 2)

        return SizeRecommendation(
            desk_id=stats.desk_id,
            raw_kelly_fraction=round(raw_kelly, 6),
            fractional_kelly=round(fractional, 6),
            capped_fraction=round(capped, 6),
            account_equity=account_equity,
            position_size=position_size,
            risk_budget_pct=round(capped * 100, 4),
            stats=stats,
            computed_at=computed_at,
        )
