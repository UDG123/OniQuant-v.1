"""Desk 1 scalping strategy — Order Flow Imbalance micro-structure detection.

Detects micro-structural momentum using Order Flow Imbalance (OFI) on
1-minute to 5-minute data.  A LONG signal fires when cumulative OFI
diverges positively against a flat or dipping price; SHORT for the
inverse.  All computation is fully vectorised via NumPy / Pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ScalpingResult:
    signal: str              # "LONG", "SHORT", or "NEUTRAL"
    ofi_score: float         # normalised cumulative OFI (-1 … +1)
    ofi_raw: float           # raw cumulative OFI over the window
    price_delta_pct: float   # price change % over the same window
    dynamic_stop: float      # price level for a ≤1 % account-risk stop
    is_valid_setup: bool     # True when OFI/price divergence is actionable


class Desk1ScalpingStrategy:
    """Vectorised Order-Flow Imbalance scalper for Desk 1.

    Parameters
    ----------
    ofi_window : int
        Rolling window for cumulative OFI aggregation (default 14 bars).
    ofi_threshold : float
        Minimum |normalised OFI| to consider the imbalance significant
        (default 0.60 — top 40 % of the distribution).
    price_flat_pct : float
        Maximum absolute price change (%) over the window for the price
        to be considered "flat / dipping" relative to a strong OFI reading
        (default 0.15 %).
    max_risk_pct : float
        Maximum risk per trade as a fraction of account equity
        (default 0.01 = 1 %).
    atr_period : int
        ATR look-back for dynamic stop-loss sizing (default 14).
    atr_stop_mult : float
        ATR multiplier for the raw stop distance (default 1.5).
    """

    def __init__(
        self,
        ofi_window: int = 14,
        ofi_threshold: float = 0.60,
        price_flat_pct: float = 0.15,
        max_risk_pct: float = 0.01,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
    ) -> None:
        self.ofi_window = ofi_window
        self.ofi_threshold = ofi_threshold
        self.price_flat_pct = price_flat_pct
        self.max_risk_pct = max_risk_pct
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    # ------------------------------------------------------------------
    # Core vectorised OFI engine
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ofi(df: pd.DataFrame) -> pd.Series:
        """Compute per-bar Order Flow Imbalance from bid/ask volume changes.

        OFI_t = Δbid_volume_t − Δask_volume_t

        If explicit ``bid_volume`` / ``ask_volume`` columns are absent the
        method falls back to a tick-rule proxy:
            bid_volume ≈ volume × (close − low) / (high − low)
            ask_volume ≈ volume × (high − close) / (high − low)
        """
        if {"bid_volume", "ask_volume"}.issubset(df.columns):
            delta_bid = df["bid_volume"].diff()
            delta_ask = df["ask_volume"].diff()
        else:
            hl_range = (df["high"] - df["low"]).replace(0, np.nan)
            bid_proxy = df["volume"] * (df["close"] - df["low"]) / hl_range
            ask_proxy = df["volume"] * (df["high"] - df["close"]) / hl_range
            delta_bid = bid_proxy.diff()
            delta_ask = ask_proxy.diff()

        return (delta_bid - delta_ask).fillna(0.0)

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """True Range → exponential moving average (Wilder smoothing)."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=self.atr_period, min_periods=self.atr_period, adjust=False).mean()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        account_equity: float = 100_000.0,
    ) -> ScalpingResult:
        """Evaluate the latest bar for an OFI-divergence scalp setup.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (columns: open, high, low, close, volume).
            Optionally includes ``bid_volume`` and ``ask_volume``.
            Must contain at least ``ofi_window + atr_period`` rows.
        account_equity : float
            Current account equity for stop-loss sizing (default 100 000).

        Returns
        -------
        ScalpingResult
        """
        min_rows = self.ofi_window + self.atr_period
        if len(df) < min_rows:
            return ScalpingResult(
                signal="NEUTRAL",
                ofi_score=0.0,
                ofi_raw=0.0,
                price_delta_pct=0.0,
                dynamic_stop=0.0,
                is_valid_setup=False,
            )

        # --- OFI computation (vectorised) ---------------------------------
        ofi_per_bar: pd.Series = self._compute_ofi(df)
        cum_ofi: pd.Series = ofi_per_bar.rolling(self.ofi_window).sum()

        # Normalise to [-1, 1] using the rolling window's own extremes
        roll_max = cum_ofi.rolling(self.ofi_window * 4, min_periods=self.ofi_window).max()
        roll_min = cum_ofi.rolling(self.ofi_window * 4, min_periods=self.ofi_window).min()
        denom = (roll_max - roll_min).replace(0, np.nan)
        ofi_norm: pd.Series = (2.0 * (cum_ofi - roll_min) / denom - 1.0).fillna(0.0)

        latest_ofi_norm = float(ofi_norm.iat[-1])
        latest_ofi_raw = float(cum_ofi.iat[-1])

        # --- Price change over the same window ----------------------------
        close = df["close"]
        price_start = close.iloc[-self.ofi_window]
        price_end = close.iat[-1]
        price_delta_pct = ((price_end - price_start) / price_start) * 100.0

        # --- ATR for dynamic stop-loss ------------------------------------
        atr_series = self._compute_atr(df)
        atr_val = float(atr_series.iat[-1])
        stop_distance = atr_val * self.atr_stop_mult

        # Cap stop distance so risk never exceeds max_risk_pct of equity
        max_dollar_risk = account_equity * self.max_risk_pct
        if stop_distance > 0 and price_end > 0:
            # position_size * stop_distance ≤ max_dollar_risk
            # ⇒ stop_distance ≤ max_dollar_risk  (for a 1-unit position)
            stop_distance = min(stop_distance, max_dollar_risk)

        # --- Signal logic: OFI / price divergence -------------------------
        strong_bid = latest_ofi_norm >= self.ofi_threshold
        strong_ask = latest_ofi_norm <= -self.ofi_threshold
        price_flat_or_dip = price_delta_pct <= self.price_flat_pct
        price_flat_or_rise = price_delta_pct >= -self.price_flat_pct

        if strong_bid and price_flat_or_dip:
            signal = "LONG"
            dynamic_stop = round(price_end - stop_distance, 8)
            is_valid = True
        elif strong_ask and price_flat_or_rise:
            signal = "SHORT"
            dynamic_stop = round(price_end + stop_distance, 8)
            is_valid = True
        else:
            signal = "NEUTRAL"
            dynamic_stop = 0.0
            is_valid = False

        return ScalpingResult(
            signal=signal,
            ofi_score=round(latest_ofi_norm, 4),
            ofi_raw=round(latest_ofi_raw, 4),
            price_delta_pct=round(price_delta_pct, 4),
            dynamic_stop=dynamic_stop,
            is_valid_setup=is_valid,
        )
