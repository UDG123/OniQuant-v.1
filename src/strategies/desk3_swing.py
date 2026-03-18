"""Desk 3 swing strategy — Lorentzian distance classification for trend evaluation.

Uses a Lorentzian metric over normalised indicator features (RSI, ADX,
CCI, price-vs-EMA ratio) to classify the current bar against a sliding
look-back window.  A majority of nearest-neighbour labels above the
configurable threshold confirms a valid swing setup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass(slots=True)
class SwingResult:
    is_valid_setup: bool
    lorentzian_score: float
    trend_direction: str
    rsi: float
    adx: float


class Desk3SwingStrategy:
    """Lorentzian distance classifier for swing-trade validation on Desk 3.

    Parameters
    ----------
    neighbours : int
        Number of nearest neighbours for the classification vote (default 8).
    lookback : int
        Historical window to search for neighbours (default 200).
    threshold : float
        Fraction of neighbours that must agree for a valid setup (default 0.6).
    rsi_period : int
        RSI look-back (default 14).
    adx_period : int
        ADX look-back (default 14).
    cci_period : int
        CCI look-back (default 20).
    ema_period : int
        EMA period for trend alignment (default 50).
    """

    def __init__(
        self,
        neighbours: int = 8,
        lookback: int = 200,
        threshold: float = 0.6,
        rsi_period: int = 14,
        adx_period: int = 14,
        cci_period: int = 20,
        ema_period: int = 50,
    ):
        self.neighbours = neighbours
        self.lookback = lookback
        self.threshold = threshold
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.cci_period = cci_period
        self.ema_period = ema_period

    @staticmethod
    def _lorentzian_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Lorentzian distance: sum(log(1 + |a_i - b_i|))."""
        return float(np.sum(np.log1p(np.abs(a - b))))

    def evaluate(self, df: pd.DataFrame) -> SwingResult:
        """Classify the most recent bar using Lorentzian nearest neighbours.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns: open, high, low, close, volume.
            Must contain at least ``lookback + max(indicator periods)`` rows.

        Returns
        -------
        SwingResult
        """
        # --- Indicator computation ----------------------------------------
        rsi: pd.Series = ta.rsi(df["close"], length=self.rsi_period)
        adx: pd.Series = ta.adx(
            df["high"], df["low"], df["close"], length=self.adx_period
        )[f"ADX_{self.adx_period}"]
        cci: pd.Series = ta.cci(
            df["high"], df["low"], df["close"], length=self.cci_period
        )
        ema: pd.Series = ta.ema(df["close"], length=self.ema_period)

        ratio = df["close"] / ema

        # --- Build feature matrix (normalised) ----------------------------
        features = pd.DataFrame(
            {"rsi": rsi, "adx": adx, "cci": cci, "ratio": ratio}
        ).dropna()

        if len(features) < self.lookback + 1:
            return SwingResult(
                is_valid_setup=False,
                lorentzian_score=0.0,
                trend_direction="neutral",
                rsi=rsi.iat[-1] if not rsi.empty else 0.0,
                adx=adx.iat[-1] if not adx.empty else 0.0,
            )

        mean = features.mean()
        std = features.std().replace(0, 1)
        norm = ((features - mean) / std).values

        current = norm[-1]
        window = norm[-(self.lookback + 1) : -1]

        # --- Label each historical bar: 1 = next close up, 0 = down ------
        close_vals = df["close"].iloc[features.index].values
        labels = (np.roll(close_vals, -1) > close_vals).astype(int)
        labels = labels[-(self.lookback + 1) : -1]

        # --- k-NN via Lorentzian distance ---------------------------------
        distances = np.array(
            [self._lorentzian_distance(current, row) for row in window]
        )
        nearest_idx = np.argpartition(distances, self.neighbours)[
            : self.neighbours
        ]
        nearest_labels = labels[nearest_idx]
        bullish_ratio = float(nearest_labels.mean())

        lorentzian_score = round(bullish_ratio, 4)
        is_valid = bullish_ratio >= self.threshold

        latest_close = df["close"].iat[-1]
        latest_ema = ema.iat[-1]
        trend_direction = "bullish" if latest_close > latest_ema else "bearish"

        return SwingResult(
            is_valid_setup=is_valid,
            lorentzian_score=lorentzian_score,
            trend_direction=trend_direction,
            rsi=round(float(rsi.iat[-1]), 4),
            adx=round(float(adx.iat[-1]), 4),
        )
