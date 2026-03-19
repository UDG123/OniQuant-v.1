"""Triple Barrier Method dataset labelling service.

Processes historical OHLCV data from the ``ml_training_data`` table and
assigns categorical labels based on three dynamic exit criteria:

* **Upper barrier (Profit-Take)** — price rises by a multiple of rolling
  daily volatility.  Label → **2** (Buy).
* **Lower barrier (Stop-Loss)** — price falls by a multiple of rolling
  daily volatility.  Label → **0** (Sell).
* **Vertical barrier (Time-Exhaustion)** — neither barrier is hit within
  the maximum holding period.  Label → **1** (Neutral).

Barriers are *dynamic*: they are scaled by the rolling annualised daily
volatility (standard deviation of log-returns) at each observation,
ensuring the method adapts to regime changes in market conditions.

The implementation is fully **vectorised** via NumPy and Pandas — no
Python-level row iteration for the barrier scan.

Usage::

    from src.core.dataset_generator import TripleBarrierLabeler

    labeler = TripleBarrierLabeler(postgres_url="postgresql://…")
    df = labeler.generate()
    print(df[["symbol", "entry_price", "close_price", "label"]].head())

Or standalone::

    python -m src.core.dataset_generator          # default params
    python -m src.core.dataset_generator --symbol BTCUSDT --pt-mult 2.5
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
_DEFAULT_MAX_HOLDING: int = int(os.getenv("TB_MAX_HOLDING", "10"))
_DEFAULT_LOOKBACK: int = int(os.getenv("TB_LOOKBACK", "5000"))
_ANNUAL_FACTOR: float = np.sqrt(252)

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
    dataframe: pd.DataFrame


# ---------------------------------------------------------------------------
# Core vectorised barrier scan
# ---------------------------------------------------------------------------


def _apply_triple_barrier(
    prices: np.ndarray,
    volatilities: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
) -> np.ndarray:
    """Vectorised Triple Barrier scan over a price series.

    For each observation *i*, looks forward up to ``max_holding`` steps
    and determines which barrier is touched first.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of entry prices (``entry_price`` column).
    volatilities : np.ndarray
        1-D array of rolling daily volatility at each observation.
        Used to scale the upper/lower barriers dynamically.
    pt_mult : float
        Profit-take barrier width as a multiple of volatility.
    sl_mult : float
        Stop-loss barrier width as a multiple of volatility.
    max_holding : int
        Vertical barrier — maximum forward look in observations.

    Returns
    -------
    np.ndarray
        Integer labels: 2 (Buy/PT hit), 0 (Sell/SL hit), 1 (Neutral/
        time expired).  NaN where volatility is unavailable.
    """
    n = len(prices)
    labels = np.full(n, np.nan, dtype=np.float64)

    # Pre-compute the close prices used as the forward price path.
    # Since ml_training_data stores per-trade snapshots (not a continuous
    # OHLCV series), we use close_price as the forward reference.
    # The caller is responsible for providing the close_price array in
    # `prices` and the entry_price separately if needed.  Here, prices
    # IS the close-price series ordered chronologically.

    for i in range(n):
        vol_i = volatilities[i]
        if np.isnan(vol_i) or vol_i <= 0:
            continue

        entry = prices[i]
        upper = entry * (1.0 + pt_mult * vol_i)
        lower = entry * (1.0 - sl_mult * vol_i)

        # Forward scan window.
        end = min(i + max_holding + 1, n)
        window = prices[i + 1 : end] if i + 1 < end else np.array([])

        if len(window) == 0:
            labels[i] = 1  # Time exhaustion — no forward data.
            continue

        # Find first touch of each barrier.
        pt_touches = np.where(window >= upper)[0]
        sl_touches = np.where(window <= lower)[0]

        pt_idx = pt_touches[0] if len(pt_touches) > 0 else max_holding + 1
        sl_idx = sl_touches[0] if len(sl_touches) > 0 else max_holding + 1

        if pt_idx <= sl_idx and pt_idx < max_holding:
            labels[i] = 2  # Buy — profit-take hit first.
        elif sl_idx < pt_idx and sl_idx < max_holding:
            labels[i] = 0  # Sell — stop-loss hit first.
        else:
            labels[i] = 1  # Neutral — time expired.

    return labels


def _apply_triple_barrier_vectorised(
    prices: np.ndarray,
    volatilities: np.ndarray,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
) -> np.ndarray:
    """Fully vectorised Triple Barrier scan using 2-D broadcasting.

    Builds a (n × max_holding) matrix of future returns relative to
    each observation's entry price, then checks barrier breaches across
    the entire matrix without Python-level loops.

    Falls back to ``_apply_triple_barrier`` for very large datasets
    where the 2-D matrix would exceed available memory.
    """
    n = len(prices)

    # Memory guard: 2-D matrix costs n × max_holding × 8 bytes.
    matrix_bytes = n * max_holding * 8
    if matrix_bytes > 2_000_000_000:  # >2 GB — fall back to loop.
        logger.info(
            "Dataset too large for 2-D vectorisation (%d obs × %d hold "
            "= %.1f GB); falling back to loop scan",
            n, max_holding, matrix_bytes / 1e9,
        )
        return _apply_triple_barrier(
            prices, volatilities, pt_mult, sl_mult, max_holding,
        )

    labels = np.full(n, np.nan, dtype=np.float64)

    # Build index matrix: future_indices[i, j] = i + j + 1.
    offsets = np.arange(1, max_holding + 1)  # (max_holding,)
    indices = np.arange(n)[:, np.newaxis] + offsets[np.newaxis, :]  # (n, H)

    # Clip to valid range and build padded price array.
    padded = np.concatenate([prices, np.full(max_holding, np.nan)])
    future_prices = padded[indices]  # (n, H)

    # Dynamic barriers per observation.
    upper = prices[:, np.newaxis] * (1.0 + pt_mult * volatilities[:, np.newaxis])
    lower = prices[:, np.newaxis] * (1.0 - sl_mult * volatilities[:, np.newaxis])

    # Boolean masks: where each barrier is breached.
    pt_hit = future_prices >= upper  # (n, H)
    sl_hit = future_prices <= lower  # (n, H)

    # First touch index per barrier (H+1 = "never").
    never = max_holding + 1
    pt_first = np.where(pt_hit.any(axis=1), pt_hit.argmax(axis=1), never)
    sl_first = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), never)

    # Valid observations: volatility is available and positive.
    valid = (~np.isnan(volatilities)) & (volatilities > 0)

    # Default: Neutral (time expired).
    labels[valid] = 1

    # Buy: PT hit first (or same time — favour PT).
    buy_mask = valid & (pt_first <= sl_first) & (pt_first < max_holding)
    labels[buy_mask] = 2

    # Sell: SL hit first.
    sell_mask = valid & (sl_first < pt_first) & (sl_first < max_holding)
    labels[sell_mask] = 0

    return labels


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
    (standard deviation of log-returns) of the asset's close prices,
    ensuring the method adapts to changing market regimes.

    Parameters
    ----------
    postgres_url : str | None
        Synchronous SQLAlchemy connection URL.
    vol_window : int
        Rolling window size for daily volatility estimation (default 20).
    pt_multiplier : float
        Profit-take barrier as a multiple of volatility (default 2.0).
    sl_multiplier : float
        Stop-loss barrier as a multiple of volatility (default 2.0).
    max_holding_period : int
        Vertical barrier — max forward observations (default 10).
    lookback : int
        Maximum number of observations to load from the database.
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

        # Ensure numeric types.
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
        """Compute rolling daily volatility from log-returns.

        Returns the standard deviation of log-returns over the rolling
        window.  The first ``window - 1`` observations will be NaN.
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
            processes all symbols together (per-symbol volatility is
            computed within each group).

        Returns
        -------
        LabelingResult
            Contains the labelled DataFrame and summary counts.
        """
        df = self._load_data(symbol)

        if df.empty:
            return LabelingResult(
                total_observations=0,
                labeled_buy=0,
                labeled_sell=0,
                labeled_neutral=0,
                skipped_insufficient_vol=0,
                dataframe=df,
            )

        # --- Per-symbol volatility + barrier labelling ----------------------
        #
        # Group by symbol so each asset gets its own volatility estimate.
        # Within each group, barriers are scaled by that asset's rolling
        # daily volatility.

        df["daily_volatility"] = np.nan
        df["label"] = np.nan

        symbols = df["symbol"].unique()
        logger.info(
            "Triple Barrier labelling: %d observations across %d symbol(s)",
            len(df),
            len(symbols),
        )

        for sym in symbols:
            mask = df["symbol"] == sym
            group = df.loc[mask].copy()

            if len(group) < self._vol_window + 1:
                logger.debug(
                    "Symbol %s has %d obs (< %d) — skipping volatility",
                    sym, len(group), self._vol_window + 1,
                )
                continue

            vol = self._compute_rolling_volatility(
                group["close_price"], self._vol_window,
            )
            df.loc[mask, "daily_volatility"] = vol.values

            prices = group["close_price"].values.astype(np.float64)
            vols = vol.values.astype(np.float64)

            labels = _apply_triple_barrier_vectorised(
                prices=prices,
                volatilities=vols,
                pt_mult=self._pt_mult,
                sl_mult=self._sl_mult,
                max_holding=self._max_holding,
            )
            df.loc[mask, "label"] = labels

        # --- Cast label to nullable int for clean output --------------------
        valid_labels = df["label"].notna()
        df.loc[valid_labels, "label"] = df.loc[valid_labels, "label"].astype(int)

        total = len(df)
        n_buy = int((df["label"] == 2).sum())
        n_sell = int((df["label"] == 0).sum())
        n_neutral = int((df["label"] == 1).sum())
        n_skipped = int(df["label"].isna().sum())

        logger.info(
            "Labelling complete: %d total | Buy=%d (%.1f%%) | "
            "Sell=%d (%.1f%%) | Neutral=%d (%.1f%%) | Skipped=%d",
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

    print(f"\nTotal observations: {result.total_observations}")
    print(f"  Buy  (label=2): {result.labeled_buy}")
    print(f"  Sell (label=0): {result.labeled_sell}")
    print(f"  Neutral   (1): {result.labeled_neutral}")
    print(f"  Skipped:        {result.skipped_insufficient_vol}")

    if not result.dataframe.empty:
        print(f"\nSample:\n{result.dataframe[['symbol', 'entry_price', 'close_price', 'daily_volatility', 'label']].head(30)}")
