from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta


@dataclass(slots=True)
class BreakoutResult:
    is_valid_breakout: bool
    risk_reward_ratio: float
    atr: float
    trend_direction: str


class Desk4GoldStrategy:
    """Vectorized XAU/USD breakout strategy for Desk 4.

    Parameters
    ----------
    atr_period : int
        ATR look-back window (default 14).
    ema_period : int
        EMA period for higher-timeframe trend alignment (default 50).
    volume_factor : float
        Multiplier above rolling-average volume required for confirmation (default 1.5).
    volume_window : int
        Rolling window length for the volume average (default 20).
    rr_multiplier : float
        Target distance as a multiple of ATR for reward calculation (default 2.0).
    """

    def __init__(
        self,
        atr_period: int = 14,
        ema_period: int = 50,
        volume_factor: float = 1.5,
        volume_window: int = 20,
        rr_multiplier: float = 2.0,
    ):
        self.atr_period = atr_period
        self.ema_period = ema_period
        self.volume_factor = volume_factor
        self.volume_window = volume_window
        self.rr_multiplier = rr_multiplier

    def evaluate(
        self,
        df: pd.DataFrame,
        resistance: float,
        support: float,
    ) -> BreakoutResult:
        """Evaluate the most recent bar for a volatility breakout.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns: open, high, low, close, volume.
        resistance : float
            Current resistance level.
        support : float
            Current support level.

        Returns
        -------
        BreakoutResult
        """
        # --- Vectorized indicator computation ---
        atr_series: pd.Series = ta.atr(
            df["high"], df["low"], df["close"], length=self.atr_period
        )
        ema_series: pd.Series = ta.ema(df["close"], length=self.ema_period)
        vol_avg: pd.Series = df["volume"].rolling(self.volume_window).mean()

        # Latest values
        close = df["close"].iat[-1]
        atr = atr_series.iat[-1]
        ema = ema_series.iat[-1]
        volume = df["volume"].iat[-1]
        avg_volume = vol_avg.iat[-1]

        # --- Trend alignment ---
        trend_direction = "bullish" if close > ema else "bearish"

        # --- Volume confirmation ---
        volume_confirmed = volume > avg_volume * self.volume_factor

        # --- Breakout detection ---
        bullish_breakout = close > resistance and trend_direction == "bullish"
        bearish_breakout = close < support and trend_direction == "bearish"
        is_valid_breakout = (bullish_breakout or bearish_breakout) and volume_confirmed

        # --- Risk / Reward ---
        risk = atr
        reward = atr * self.rr_multiplier
        rr_ratio = round(reward / risk, 4) if risk > 0 else 0.0

        return BreakoutResult(
            is_valid_breakout=is_valid_breakout,
            risk_reward_ratio=rr_ratio,
            atr=round(atr, 4),
            trend_direction=trend_direction,
        )
