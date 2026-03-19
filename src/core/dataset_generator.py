"""Triple Barrier Method dataset labelling service.

Processes historical OHLCV data from the ``ml_training_data`` table and
assigns categorical labels based on three dynamic exit criteria:

* **Upper barrier (Profit-Take)** — ``entry_price × (1 + pt_mult × σ)``.
  Price touches or exceeds this level → **Label 2** (Buy).
* **Lower barrier (Stop-Loss)** — ``entry_price × (1 − sl_mult × σ)``.
  Price touches or falls below → **Label 0** (Sell).
* **Vertical barrier (Time-Exhaustion)** — neither price barrier is hit
  within ``max_holding`` forward bars → **Label 1** (Neutral).

Barriers are **dynamic**: ``σ`` is the rolling daily volatility (standard
deviation of log-returns) at each observation, so barrier widths
automatically widen during volatile regimes and tighten during calm ones.

Implementation strategy
-----------------------
The core scan is **fully vectorised** via a chunked 2-D NumPy broadcast.
For each chunk of observations, a ``(chunk × max_holding)`` matrix of
future close-prices is built.  Barrier breaches are detected with
element-wise comparisons across the entire matrix — no Python-level row
iteration.  Chunks keep peak memory bounded regardless of dataset size.

A pure-loop fallback (``_apply_triple_barrier_loop``) exists for
environments where memory is extremely constrained.

Usage::

    from src.core.dataset_generator import TripleBarrierLabeler

    labeler = TripleBarrierLabeler(
        pt_multiplier=2.0,
        sl_multiplier=2.0,
        max_holding_period=50,
    )
    result = labeler.generate(symbol="BTCUSDT")
    df = result.dataframe
    print(df[["symbol", "close_price", "daily_volatility", "upper_barrier",
              "lower_barrier", "label"]].head(30))

CLI::

    python -m src.core.dataset_generator --symbol BTCUSDT --max-hold 50
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
).replace("+asyncpg", "")

_DEFAULT_VOL_WINDOW: int = int(os.getenv("TB_VOL_WINDOW", "20"))
_DEFAULT_PT_MULTIPLIER: float = float(os.getenv("TB_PT_MULTIPLIER", "2.0"))
_DEFAULT_SL_MULTIPLIER: float = float(os.getenv("TB_SL_MULTIPLIER", "2.0"))
_DEFAULT_MAX_HOLDING: int = int(os.getenv("TB_MAX_HOLDING", "50"))
_DEFAULT_LOOKBACK: int = int(os.getenv("TB_LOOKBACK", "10000"))

# Chunk size for the 2-D vectorised scan.  Each chunk allocates
# chunk_size × max_holding × 8 bytes.  At 4096 × 50 this is ~1.6 MB —
# comfortably in L2/L3 cache on modern CPUs.
_CHUNK_SIZE: int = int(os.getenv("TB_CHUNK_SIZE", "4096"))

logger = logging.getLogger("oniquant.core.dataset_generator")

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LabelingResult:
    """Summary returned after a labelling run."""

    total_observations: int
    labeled_buy: int
    labeled_sell: int
    labeled_neutral: int
    skipped_insufficient_vol: int
    vol_window: int
    pt_multiplier: float
    sl_multiplier: float
    max_holding_period: int
    dataframe: pd.DataFrame


# ---------------------------------------------------------------------------
# Core vectorised barrier scan — chunked 2-D broadcast
# ---------------------------------------------------------------------------


def _apply_triple_barrier_chunked(
    prices: np.ndarray,
    volatilities: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    chunk_size: int = _CHUNK_SIZE,
) -> np.ndarray:
    """Chunked, fully-vectorised Triple Barrier scan.

    Processes observations in blocks of ``chunk_size`` rows.  Within each
    block a ``(block × max_holding)`` matrix of future prices is built
    via fancy indexing into a NaN-padded price array, then barrier
    breaches are detected with pure NumPy element-wise ops.

    Peak memory: ``chunk_size × max_holding × 8`` bytes per block
    (the padded price vector is shared and read-only).

    Parameters
    ----------
    prices : np.ndarray
        1-D float64 close-price series, chronologically ordered.
    volatilities : np.ndarray
        1-D float64 rolling daily volatility at each observation.
    pt_mult : float
        Profit-take barrier width as a multiple of volatility.
    sl_mult : float
        Stop-loss barrier width as a multiple of volatility.
    max_holding : int
        Vertical barrier — maximum forward bars to scan.
    chunk_size : int
        Rows per vectorised chunk (default 4096).

    Returns
    -------
    np.ndarray
        Integer-coded labels: 2 (Buy/PT), 0 (Sell/SL), 1 (Neutral).
        ``NaN`` where volatility is unavailable or non-positive.
    """
    n = len(prices)
    labels = np.full(n, np.nan, dtype=np.float64)

    # Pad the price array with NaN so out-of-bounds look-ahead returns NaN
    # instead of raising IndexError.  NaN comparisons (>= / <=) correctly
    # evaluate to False, so padded entries never trigger a barrier.
    padded = np.empty(n + max_holding, dtype=np.float64)
    padded[:n] = prices
    padded[n:] = np.nan

    # Shared offset vector: [1, 2, …, max_holding].
    offsets = np.arange(1, max_holding + 1)  # (H,)

    # Pre-compute validity mask once — avoids repeated NaN checks per chunk.
    valid = (~np.isnan(volatilities)) & (volatilities > 0)

    never = max_holding  # Sentinel: "no barrier touched within window".

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_len = end - start

        # Row indices for this chunk.
        rows = np.arange(start, end)  # (C,)

        # Build the future-price matrix: future[i, j] = padded[row_i + j + 1].
        idx = rows[:, np.newaxis] + offsets[np.newaxis, :]  # (C, H)
        future = padded[idx]  # (C, H)

        # Dynamic barriers per observation.
        p = prices[start:end, np.newaxis]                  # (C, 1)
        v = volatilities[start:end, np.newaxis]             # (C, 1)
        upper = p * (1.0 + pt_mult * v)                    # (C, H) broadcast
        lower = p * (1.0 - sl_mult * v)                    # (C, H) broadcast

        # Boolean breach masks.
        pt_hit = future >= upper  # (C, H)
        sl_hit = future <= lower  # (C, H)

        # First-touch index per barrier.  argmax on a bool array returns the
        # index of the first True; if no True exists, any() is False and we
        # use `never` as the sentinel.
        pt_any = pt_hit.any(axis=1)   # (C,)
        sl_any = sl_hit.any(axis=1)   # (C,)

        pt_first = np.where(pt_any, pt_hit.argmax(axis=1), never)  # (C,)
        sl_first = np.where(sl_any, sl_hit.argmax(axis=1), never)  # (C,)

        # Slice the valid mask for this chunk.
        v_chunk = valid[start:end]  # (C,)

        # Default: Neutral (time-exhaustion / no barrier hit).
        chunk_labels = np.full(chunk_len, np.nan, dtype=np.float64)
        chunk_labels[v_chunk] = 1

        # Buy: PT hit first (ties broken in favour of PT).
        buy = v_chunk & (pt_first <= sl_first) & (pt_first < never)
        chunk_labels[buy] = 2

        # Sell: SL hit strictly first.
        sell = v_chunk & (sl_first < pt_first) & (sl_first < never)
        chunk_labels[sell] = 0

        labels[start:end] = chunk_labels

    return labels


# ---------------------------------------------------------------------------
# Fallback: pure-loop barrier scan (no 2-D allocation)
# ---------------------------------------------------------------------------


def _apply_triple_barrier_loop(
    prices: np.ndarray,
    volatilities: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
) -> np.ndarray:
    """Scalar-loop Triple Barrier scan — minimal memory footprint.

    Used when even the chunked approach would exceed available memory
    (e.g. max_holding > 10 000) or for correctness cross-validation.
    """
    n = len(prices)
    labels = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        vol_i = volatilities[i]
        if np.isnan(vol_i) or vol_i <= 0:
            continue

        entry = prices[i]
        upper = entry * (1.0 + pt_mult * vol_i)
        lower = entry * (1.0 - sl_mult * vol_i)

        # Scan forward up to max_holding bars.
        end = min(i + max_holding + 1, n)
        label = 1  # Default: Neutral (vertical barrier).

        for j in range(i + 1, end):
            if prices[j] >= upper:
                label = 2  # Buy — profit-take hit first.
                break
            if prices[j] <= lower:
                label = 0  # Sell — stop-loss hit first.
                break

        labels[i] = label

    return labels


# ---------------------------------------------------------------------------
# Dispatcher: choose the best scan strategy
# ---------------------------------------------------------------------------


def apply_triple_barrier(
    prices: np.ndarray,
    volatilities: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    chunk_size: int = _CHUNK_SIZE,
) -> np.ndarray:
    """Apply the Triple Barrier Method to a chronological price series.

    Automatically selects the chunked-vectorised implementation unless
    the max_holding window is extreme (> 10 000), in which case the
    scalar loop is used to avoid excessive allocation.

    Parameters
    ----------
    prices : np.ndarray
        1-D float64 close-price series.
    volatilities : np.ndarray
        1-D float64 rolling daily volatility per observation.
    pt_mult : float
        Profit-take barrier as a multiple of volatility.
    sl_mult : float
        Stop-loss barrier as a multiple of volatility.
    max_holding : int
        Vertical barrier — max forward bars.
    chunk_size : int
        Block size for the vectorised path.

    Returns
    -------
    np.ndarray
        Labels: 2 (Buy/PT), 0 (Sell/SL), 1 (Neutral), NaN (skipped).
    """
    if max_holding > 10_000:
        logger.info(
            "max_holding=%d exceeds chunked threshold — using loop scan", max_holding,
        )
        return _apply_triple_barrier_loop(
            prices, volatilities, pt_mult, sl_mult, max_holding,
        )

    return _apply_triple_barrier_chunked(
        prices, volatilities, pt_mult, sl_mult, max_holding, chunk_size,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_QUERY = text("""
    SELECT
        id,
        desk_id,
        symbol,
        side,
        entry_price,
        close_price,
        pnl_pct,
        atr_at_entry,
        volume_ratio,
        rsi_at_entry,
        adx_at_entry,
        cci_at_entry,
        ema_distance,
        cvd_zscore,
        lorentzian_score,
        consensus_score,
        closed_at
    FROM ml_training_data
    WHERE (:symbol IS NULL OR symbol = :symbol)
    ORDER BY closed_at ASC
    LIMIT :limit
""")


# ---------------------------------------------------------------------------
# TripleBarrierLabeler
# ---------------------------------------------------------------------------


class TripleBarrierLabeler:
    """Dataset labelling service using the Triple Barrier Method.

    Barriers are dynamically scaled by the rolling daily volatility
    (standard deviation of log-returns over ``vol_window`` observations)
    of each asset's close-price series.  This ensures barrier widths
    automatically adapt to regime changes — wider during volatile
    periods, narrower during calm ones.

    Parameters
    ----------
    postgres_url : str | None
        Synchronous SQLAlchemy connection URL.
    vol_window : int
        Rolling window for daily volatility estimation (default 20).
    pt_multiplier : float
        Profit-take barrier as a multiple of σ (default 2.0).
    sl_multiplier : float
        Stop-loss barrier as a multiple of σ (default 2.0).
    max_holding_period : int
        Vertical barrier — max forward bars before time-exhaustion
        labels the observation as Neutral (default 50).
    lookback : int
        Maximum observations to load from the database.
    """

    def __init__(
        self,
        postgres_url: str | None = None,
        vol_window: int = _DEFAULT_VOL_WINDOW,
        pt_multiplier: float = _DEFAULT_PT_MULTIPLIER,
        sl_multiplier: float = _DEFAULT_SL_MULTIPLIER,
        max_holding_period: int = _DEFAULT_MAX_HOLDING,
        lookback: int = _DEFAULT_LOOKBACK,
    ) -> None:
        pg_url = (postgres_url or _DEFAULT_POSTGRES_URL).replace("+asyncpg", "")
        self._engine = create_engine(pg_url, pool_pre_ping=True, pool_size=3)
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False,
        )
        self._vol_window = vol_window
        self._pt_mult = pt_multiplier
        self._sl_mult = sl_multiplier
        self._max_holding = max_holding_period
        self._lookback = lookback

    # ----- data loading -----------------------------------------------------

    def _load_data(self, symbol: str | None = None) -> pd.DataFrame:
        """Fetch historical observations from ``ml_training_data``."""
        with self._session_factory() as session:
            result = session.execute(
                _QUERY,
                {"symbol": symbol, "limit": self._lookback},
            )
            rows = result.fetchall()
            columns = list(result.keys())

        if not rows:
            logger.warning("No data returned from ml_training_data")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)

        for col in ("entry_price", "close_price", "pnl_pct", "atr_at_entry"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)
        df.sort_values("closed_at", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    # ----- volatility estimation --------------------------------------------

    @staticmethod
    def _compute_rolling_volatility(
        close_prices: pd.Series,
        window: int,
    ) -> pd.Series:
        """Rolling daily volatility from log-returns.

        σ_t = std(log(P_t / P_{t-1})) over the last ``window`` bars.

        The first ``window`` observations are NaN (insufficient history).
        """
        log_returns = np.log(close_prices / close_prices.shift(1))
        return log_returns.rolling(window=window, min_periods=window).std()

    # ----- public API -------------------------------------------------------

    def generate(
        self,
        symbol: str | None = None,
    ) -> LabelingResult:
        """Run the Triple Barrier labelling pipeline.

        Parameters
        ----------
        symbol : str | None
            Optional symbol filter (e.g. ``"BTCUSDT"``).  When ``None``,
            processes all symbols (per-symbol volatility is computed
            within each group).

        Returns
        -------
        LabelingResult
            Labelled DataFrame with audit columns:

            * ``daily_volatility`` — rolling σ used to scale barriers
            * ``upper_barrier`` — dynamic PT level for this observation
            * ``lower_barrier`` — dynamic SL level for this observation
            * ``barrier_width_pct`` — (upper − lower) / close as a %
            * ``label`` — 2 (Buy), 0 (Sell), 1 (Neutral), NaN (skipped)
        """
        df = self._load_data(symbol)

        if df.empty:
            return LabelingResult(
                total_observations=0,
                labeled_buy=0,
                labeled_sell=0,
                labeled_neutral=0,
                skipped_insufficient_vol=0,
                vol_window=self._vol_window,
                pt_multiplier=self._pt_mult,
                sl_multiplier=self._sl_mult,
                max_holding_period=self._max_holding,
                dataframe=df,
            )

        # Initialise output columns.
        df["daily_volatility"] = np.nan
        df["upper_barrier"] = np.nan
        df["lower_barrier"] = np.nan
        df["barrier_width_pct"] = np.nan
        df["label"] = np.nan

        symbols = df["symbol"].unique()
        logger.info(
            "Triple Barrier labelling: %d observations across %d symbol(s) "
            "(vol_window=%d, pt=%.2fσ, sl=%.2fσ, max_hold=%d bars)",
            len(df), len(symbols),
            self._vol_window, self._pt_mult, self._sl_mult, self._max_holding,
        )

        for sym in symbols:
            mask = df["symbol"] == sym
            group = df.loc[mask]

            if len(group) < self._vol_window + 1:
                logger.debug(
                    "Symbol %s: %d obs (need %d) — skipping",
                    sym, len(group), self._vol_window + 1,
                )
                continue

            # --- Rolling daily volatility per symbol --------------------------
            vol = self._compute_rolling_volatility(
                group["close_price"], self._vol_window,
            )
            df.loc[mask, "daily_volatility"] = vol.values

            prices = group["close_price"].values.astype(np.float64)
            vols = vol.values.astype(np.float64)

            # --- Compute barrier levels for auditability ----------------------
            upper = prices * (1.0 + self._pt_mult * vols)
            lower = prices * (1.0 - self._sl_mult * vols)
            width_pct = np.where(
                prices > 0,
                ((upper - lower) / prices) * 100.0,
                np.nan,
            )

            df.loc[mask, "upper_barrier"] = upper
            df.loc[mask, "lower_barrier"] = lower
            df.loc[mask, "barrier_width_pct"] = width_pct

            # --- Vectorised barrier scan --------------------------------------
            labels = apply_triple_barrier(
                prices=prices,
                volatilities=vols,
                pt_mult=self._pt_mult,
                sl_mult=self._sl_mult,
                max_holding=self._max_holding,
            )
            df.loc[mask, "label"] = labels

        # --- Cast labels to nullable int for clean output ---------------------
        valid_labels = df["label"].notna()
        df.loc[valid_labels, "label"] = df.loc[valid_labels, "label"].astype(int)

        total = len(df)
        n_buy = int((df["label"] == 2).sum())
        n_sell = int((df["label"] == 0).sum())
        n_neutral = int((df["label"] == 1).sum())
        n_skipped = int(df["label"].isna().sum())

        logger.info(
            "Labelling complete: %d total | "
            "Buy=%d (%.1f%%) | Sell=%d (%.1f%%) | Neutral=%d (%.1f%%) | "
            "Skipped=%d",
            total,
            n_buy, 100 * n_buy / max(total, 1),
            n_sell, 100 * n_sell / max(total, 1),
            n_neutral, 100 * n_neutral / max(total, 1),
            n_skipped,
        )

        return LabelingResult(
            total_observations=total,
            labeled_buy=n_buy,
            labeled_sell=n_sell,
            labeled_neutral=n_neutral,
            skipped_insufficient_vol=n_skipped,
            vol_window=self._vol_window,
            pt_multiplier=self._pt_mult,
            sl_multiplier=self._sl_mult,
            max_holding_period=self._max_holding,
            dataframe=df,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Triple Barrier Method dataset labeller",
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--vol-window", type=int, default=_DEFAULT_VOL_WINDOW)
    parser.add_argument("--pt-mult", type=float, default=_DEFAULT_PT_MULTIPLIER)
    parser.add_argument("--sl-mult", type=float, default=_DEFAULT_SL_MULTIPLIER)
    parser.add_argument("--max-hold", type=int, default=_DEFAULT_MAX_HOLDING)
    parser.add_argument("--lookback", type=int, default=_DEFAULT_LOOKBACK)
    args = parser.parse_args()

    labeler = TripleBarrierLabeler(
        vol_window=args.vol_window,
        pt_multiplier=args.pt_mult,
        sl_multiplier=args.sl_mult,
        max_holding_period=args.max_hold,
        lookback=args.lookback,
    )

    result = labeler.generate(symbol=args.symbol)

    print(f"\nConfig: vol_window={result.vol_window} "
          f"pt={result.pt_multiplier}σ sl={result.sl_multiplier}σ "
          f"max_hold={result.max_holding_period} bars")
    print(f"\nTotal observations: {result.total_observations}")
    print(f"  Buy     (label=2): {result.labeled_buy}")
    print(f"  Sell    (label=0): {result.labeled_sell}")
    print(f"  Neutral (label=1): {result.labeled_neutral}")
    print(f"  Skipped (NaN vol): {result.skipped_insufficient_vol}")

    if not result.dataframe.empty:
        cols = ["symbol", "close_price", "daily_volatility",
                "upper_barrier", "lower_barrier", "barrier_width_pct", "label"]
        print(f"\nSample:\n{result.dataframe[cols].head(30)}")
