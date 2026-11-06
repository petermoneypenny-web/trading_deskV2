import math
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("trading_desk_v2.indicators")


def _sanitize_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Ensures floats are JSON-compliant and replaces NaN/Inf with default or None."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculates intraday Volume-Weighted Average Price (VWAP).
    Typical Price = (High + Low + Close) / 3
    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    """
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]
    cum_tp_vol = tp_vol.cumsum()
    cum_vol = df["volume"].cumsum()

    # Avoid zero division
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.ffill().bfill()


def calculate_emas(df: pd.DataFrame, fast_period: int = 9, slow_period: int = 21) -> Dict[str, pd.Series]:
    """Calculates fast and slow Exponential Moving Averages."""
    if df.empty or len(df) < slow_period:
        return {"ema_fast": pd.Series(dtype=float), "ema_slow": pd.Series(dtype=float)}

    ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
    return {"ema_fast": ema_fast, "ema_slow": ema_slow}


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Wilder's Relative Strength Index (RSI)."""
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype=float)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) for volatility and stop sizing."""
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.ffill().bfill()


def calculate_orb(df: pd.DataFrame, bars_count: int = 15) -> Dict[str, Optional[float]]:
    """
    Calculates Opening Range Breakout (ORB) High and Low based on first N bars.
    """
    if df.empty or len(df) < bars_count:
        return {"orb_high": None, "orb_low": None}

    opening_slice = df.iloc[:bars_count]
    high_val = _sanitize_float(opening_slice["high"].max())
    low_val = _sanitize_float(opening_slice["low"].min())
    return {
        "orb_high": high_val,
        "orb_low": low_val
    }


def evaluate_intraday_technical_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Combines indicators into a sanitized, JSON-safe technical snapshot.
    """
    if df.empty or len(df) < 25:
        return {
            "status": "INSUFFICIENT_DATA",
            "bars_analyzed": len(df),
            "signal": "NEUTRAL",
            "bias_score": 0,
            "metrics": {}
        }

    vwap_series = calculate_vwap(df)
    emas = calculate_emas(df, fast_period=9, slow_period=21)
    rsi_series = calculate_rsi(df, period=14)
    atr_series = calculate_atr(df, period=14)
    orb = calculate_orb(df, bars_count=15)

    last_close = _sanitize_float(df["close"].iloc[-1], default=0.0)
    current_vwap = _sanitize_float(vwap_series.iloc[-1] if not vwap_series.empty else last_close, default=last_close)
    current_ema_fast = _sanitize_float(emas["ema_fast"].iloc[-1] if not emas["ema_fast"].empty else last_close, default=last_close)
    current_ema_slow = _sanitize_float(emas["ema_slow"].iloc[-1] if not emas["ema_slow"].empty else last_close, default=last_close)
    current_rsi = _sanitize_float(rsi_series.iloc[-1] if not rsi_series.empty else 50.0, default=50.0)
    current_atr = _sanitize_float(atr_series.iloc[-1] if not atr_series.empty else 0.5, default=0.5)

    # Calculate distance from VWAP safely
    dist_from_vwap_pct = None
    if current_vwap and current_vwap > 0 and last_close:
        dist_from_vwap_pct = _sanitize_float(((last_close - current_vwap) / current_vwap) * 100.0)

    # Determine Setup Signals
    signal = "NEUTRAL"
    bias_score = 0

    if last_close and current_vwap and current_ema_fast and current_ema_slow:
        if last_close > current_vwap and current_ema_fast > current_ema_slow:
            bias_score += 1
        elif last_close < current_vwap and current_ema_fast < current_ema_slow:
            bias_score -= 1

    if orb["orb_high"] and last_close and last_close > orb["orb_high"]:
        bias_score += 1
    elif orb["orb_low"] and last_close and last_close < orb["orb_low"]:
        bias_score -= 1

    if current_rsi is not None:
        if current_rsi > 55:
            bias_score += 1
        elif current_rsi < 45:
            bias_score -= 1

    if bias_score >= 2:
        signal = "BULLISH_MOMENTUM"
    elif bias_score <= -2:
        signal = "BEARISH_MOMENTUM"
    elif current_rsi is not None and current_rsi < 30 and current_ema_fast and last_close and last_close > current_ema_fast:
        signal = "OVERSOLD_REVERSAL_LONG"
    elif current_rsi is not None and current_rsi > 70 and current_ema_fast and last_close and last_close < current_ema_fast:
        signal = "OVERBOUGHT_REVERSAL_SHORT"

    return {
        "status": "READY",
        "bars_analyzed": len(df),
        "signal": signal,
        "bias_score": bias_score,
        "metrics": {
            "last_price": last_close,
            "vwap": current_vwap,
            "ema_9": current_ema_fast,
            "ema_21": current_ema_slow,
            "rsi_14": current_rsi,
            "atr_14": current_atr,
            "orb_high": orb["orb_high"],
            "orb_low": orb["orb_low"],
            "dist_from_vwap_pct": dist_from_vwap_pct
        }
    }