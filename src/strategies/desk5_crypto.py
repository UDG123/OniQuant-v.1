"""Desk 5 crypto strategy — CVD z-score momentum with volatility regime filter.

Cumulative Volume Delta (CVD) measures net aggressive buying/selling.
When the CVD z-score spikes while the ATR regime is expanding, this
strategy confirms directional momentum suitable for crypto positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CryptoResult:
    signal: str              # "LONG", "SHORT", or "NEUTRAL"
    cvd_zscore: float        # z-score of cumulative volume delta
    volatility_regime: str   # "expanding", "contracting", or "normal"
    rsi: float
    atr: float
    dynamic_stop: float
    is_valid_setup: bool


class Desk5CryptoStrategy:
    """CVD-momentum strategy for crypto Desk 5.

    Parameters
    ----------
    cvd_window : int
        Rolling window for CVD z-score calculation (default 20).
    cvd_threshold : float
        Minimum |z-score| for a momentum signal (default 2.0).
    atr_period : int
        ATR look-back (default 14).
    atr_regime_window : int
        Window for ATR regime detection — expanding vs contracting (default 20).
    atr_regime_mult : float
        ATR must exceed its rolling mean by this factor to qualify as
        "expanding" (default 1.25).
    rsi_period : int
        RSI look-back (default 14).
    atr_stop_mult : float
        ATR multiplier for stop distance (default 2.0 — wider for crypto).
    max_risk_pct : float
        Maximum per-trade risk as fraction of equity (default 0.01).
    """

    def __init__(
        self,
        cvd_window: int = 20,
        cvd_threshold: float = 2.0,
        atr_period: int = 14,
        atr_regime_window: int = 20,
        atr_regime_mult: float = 1.25,
        rsi_period: int = 14,
        atr_stop_mult: float = 2.0,
        max_risk_pct: float = 0.01,
    ) -> None:
        self.cvd_window = cvd_window
        self.cvd_threshold = cvd_threshold
        self.atr_period = atr_period
        self.atr_regime_window = atr_regime_window
        self.atr_regime_mult = atr_regime_mult
        self.rsi_period = rsi_period
        self.atr_stop_mult = atr_stop_mult
        self.max_risk_pct = max_risk_pct

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cvd(df: pd.DataFrame) -> pd.Series:
        """Tick-rule proxy for Cumulative Volume Delta.

        delta_t = volume_t × (2 × (close − low) / (high − low) − 1)
        CVD = cumsum(delta_t)
        """
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        delta = df["volume"] * (2.0 * (df["close"] - df["low"]) / hl_range - 1.0)
        return delta.fillna(0.0).cumsum()

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=self.atr_period, min_periods=self.atr_period, adjust=False).mean()

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> np.ndarray:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        alpha = 1.0 / period
        avg_gain = pd.Series(gain).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        return 100.0 - 100.0 / (1.0 + rs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        account_equity: float = 100_000.0,
    ) -> CryptoResult:
        """Evaluate the latest bar for a CVD-momentum crypto entry.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (columns: open, high, low, close, volume).
        account_equity : float
            Current account equity for stop-loss sizing.

        Returns
        -------
        CryptoResult
        """
        min_rows = max(self.cvd_window, self.atr_period, self.rsi_period) + self.atr_regime_window
        if len(df) < min_rows:
            return CryptoResult(
                signal="NEUTRAL", cvd_zscore=0.0, volatility_regime="normal",
                rsi=50.0, atr=0.0, dynamic_stop=0.0, is_valid_setup=False,
            )

        # --- CVD z-score --------------------------------------------------
        cvd = self._compute_cvd(df)
        cvd_mean = cvd.rolling(self.cvd_window).mean()
        cvd_std = cvd.rolling(self.cvd_window).std().replace(0, np.nan)
        cvd_z = ((cvd - cvd_mean) / cvd_std).fillna(0.0)
        z_val = float(cvd_z.iat[-1])

        # --- ATR + volatility regime --------------------------------------
        atr_series = self._compute_atr(df)
        atr_val = float(atr_series.iat[-1])
        atr_roll_mean = float(atr_series.rolling(self.atr_regime_window).mean().iat[-1])

        if atr_val > atr_roll_mean * self.atr_regime_mult:
            vol_regime = "expanding"
        elif atr_val < atr_roll_mean / self.atr_regime_mult:
            vol_regime = "contracting"
        else:
            vol_regime = "normal"

        # --- RSI ----------------------------------------------------------
        close_arr = df["close"].values.astype(np.float64)
        rsi_arr = self._rsi(close_arr, self.rsi_period)
        rsi_val = float(rsi_arr[-1])

        # --- Stop-loss (capped at 1 % equity risk) ------------------------
        price = close_arr[-1]
        stop_dist = min(atr_val * self.atr_stop_mult, account_equity * self.max_risk_pct)

        # --- Signal: CVD momentum + expanding volatility ------------------
        expanding = vol_regime == "expanding"

        if z_val >= self.cvd_threshold and expanding:
            signal = "LONG"
            dynamic_stop = round(price - stop_dist, 8)
            valid = True
        elif z_val <= -self.cvd_threshold and expanding:
            signal = "SHORT"
            dynamic_stop = round(price + stop_dist, 8)
            valid = True
        else:
            signal = "NEUTRAL"
            dynamic_stop = 0.0
            valid = False

        return CryptoResult(
            signal=signal,
            cvd_zscore=round(z_val, 4),
            volatility_regime=vol_regime,
            rsi=round(rsi_val, 4),
            atr=round(atr_val, 8),
            dynamic_stop=dynamic_stop,
            is_valid_setup=valid,
        )
