"""Desk 2 FX strategy — Kalman-filtered mean-reversion on major pairs.

Uses a scalar Kalman filter to extract a smoothed price estimate and its
prediction-error z-score.  Entries trigger when the z-score breaches a
configurable band while RSI confirms the counter-trend exhaustion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FXResult:
    signal: str              # "LONG", "SHORT", or "NEUTRAL"
    kalman_z_score: float    # z-score of price vs Kalman estimate
    kalman_price: float      # filtered price estimate
    rsi: float
    atr: float
    dynamic_stop: float
    is_valid_setup: bool


class Desk2FXStrategy:
    """Kalman-filter mean-reversion strategy for FX Desk 2.

    Parameters
    ----------
    z_entry : float
        z-score magnitude required to enter (default 2.0).
    rsi_period : int
        RSI look-back (default 14).
    rsi_oversold : float
        RSI level confirming exhaustion for LONG (default 35).
    rsi_overbought : float
        RSI level confirming exhaustion for SHORT (default 65).
    atr_period : int
        ATR period for dynamic stop (default 14).
    atr_stop_mult : float
        ATR multiplier for stop distance (default 1.5).
    max_risk_pct : float
        Maximum per-trade risk as fraction of equity (default 0.01).
    process_var : float
        Kalman process noise variance Q (default 1e-5).
    measurement_var : float
        Kalman measurement noise variance R (default 1e-3).
    """

    def __init__(
        self,
        z_entry: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        max_risk_pct: float = 0.01,
        process_var: float = 1e-5,
        measurement_var: float = 1e-3,
    ) -> None:
        self.z_entry = z_entry
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_risk_pct = max_risk_pct
        self.Q = process_var
        self.R = measurement_var

    # ------------------------------------------------------------------
    # Kalman filter (scalar, vectorised via NumPy pre-allocation)
    # ------------------------------------------------------------------

    def _kalman_filter(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run a scalar Kalman filter over *prices*.

        Returns
        -------
        x_hat : np.ndarray   – filtered estimates
        residuals : np.ndarray – measurement residuals (innovation)
        """
        n = len(prices)
        x_hat = np.empty(n)
        P = np.empty(n)
        residuals = np.empty(n)

        # Initialise
        x_hat[0] = prices[0]
        P[0] = 1.0
        residuals[0] = 0.0

        Q, R = self.Q, self.R
        for t in range(1, n):
            # Predict
            x_pred = x_hat[t - 1]
            P_pred = P[t - 1] + Q
            # Update
            innovation = prices[t] - x_pred
            S = P_pred + R
            K = P_pred / S
            x_hat[t] = x_pred + K * innovation
            P[t] = (1 - K) * P_pred
            residuals[t] = innovation

        return x_hat, residuals

    @staticmethod
    def _rsi(close: np.ndarray, period: int) -> np.ndarray:
        """Wilder-smoothed RSI computed over a NumPy array."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        alpha = 1.0 / period
        avg_gain = pd.Series(gain).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        return 100.0 - 100.0 / (1.0 + rs)

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
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
    ) -> FXResult:
        """Evaluate most recent bar for a Kalman mean-reversion entry.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (columns: open, high, low, close, volume).
        account_equity : float
            Current account equity for stop-loss sizing.

        Returns
        -------
        FXResult
        """
        min_rows = max(self.atr_period, self.rsi_period) + 30
        if len(df) < min_rows:
            return FXResult(
                signal="NEUTRAL", kalman_z_score=0.0, kalman_price=0.0,
                rsi=50.0, atr=0.0, dynamic_stop=0.0, is_valid_setup=False,
            )

        close = df["close"].values.astype(np.float64)

        # --- Kalman filter ------------------------------------------------
        x_hat, residuals = self._kalman_filter(close)
        std = np.std(residuals[self.rsi_period:]) or 1e-12
        z_scores = residuals / std
        z = float(z_scores[-1])

        # --- RSI ----------------------------------------------------------
        rsi_arr = self._rsi(close, self.rsi_period)
        rsi_val = float(rsi_arr[-1])

        # --- ATR / stop ---------------------------------------------------
        atr_series = self._compute_atr(df)
        atr_val = float(atr_series.iat[-1])
        stop_dist = min(atr_val * self.atr_stop_mult, account_equity * self.max_risk_pct)

        price = close[-1]
        kalman_price = float(x_hat[-1])

        # --- Signal -------------------------------------------------------
        if z <= -self.z_entry and rsi_val <= self.rsi_oversold:
            signal = "LONG"
            dynamic_stop = round(price - stop_dist, 8)
            valid = True
        elif z >= self.z_entry and rsi_val >= self.rsi_overbought:
            signal = "SHORT"
            dynamic_stop = round(price + stop_dist, 8)
            valid = True
        else:
            signal = "NEUTRAL"
            dynamic_stop = 0.0
            valid = False

        return FXResult(
            signal=signal,
            kalman_z_score=round(z, 4),
            kalman_price=round(kalman_price, 8),
            rsi=round(rsi_val, 4),
            atr=round(atr_val, 8),
            dynamic_stop=dynamic_stop,
            is_valid_setup=valid,
        )
