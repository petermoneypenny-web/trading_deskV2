import math
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("trading_desk_v2.indicators")


def _sanitize_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]
    cum_tp_vol = tp_vol.cumsum()
    cum_vol = df["volume"].cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.ffill().bfill()


def calculate_emas(df: pd.DataFrame, fast_period: int = 9, slow_period: int = 21) -> Dict[str, pd.Series]:
    if df.empty or len(df) < slow_period:
        return {"ema_fast": pd.Series(dtype=float), "ema_slow": pd.Series(dtype=float)}

    ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
    return {"ema_fast": ema_fast, "ema_slow": ema_slow}


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    if df.empty or len(df) < bars_count:
        return {"orb_high": None, "orb_low": None}

    opening_slice = df.iloc[:bars_count]
    return {
        "orb_high": _sanitize_float(opening_slice["high"].max()),
        "orb_low": _sanitize_float(opening_slice["low"].min())
    }


def evaluate_intraday_technical_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
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

    dist_from_vwap_pct = 0.0
    if current_vwap and current_vwap > 0 and last_close:
        dist_from_vwap_pct = _sanitize_float(((last_close - current_vwap) / current_vwap) * 100.0, default=0.0)

    signal = "NEUTRAL"
    bias_score = 0

    # Trend alignment check
    is_bull_trend = last_close > current_vwap and current_ema_fast >= current_ema_slow
    is_bear_trend = last_close < current_vwap and current_ema_fast <= current_ema_slow

    # PULLBACK FILTER: Never buy if extended more than 0.75% above VWAP; never short if extended > 0.75% below VWAP.
    # High-probability momentum entry: Price is in trend, but within 0.10% - 0.70% of VWAP or EMA ribbon
    is_bull_pullback = is_bull_trend and (0.05 <= dist_from_vwap_pct <= 0.70)
    is_bear_pullback = is_bear_trend and (-0.70 <= dist_from_vwap_pct <= -0.05)

    if is_bull_pullback:
        if 48 <= current_rsi <= 68:  # Healthy momentum, not overbought
            signal = "BULLISH_MOMENTUM"
            bias_score = 2
    elif is_bear_pullback:
        if 32 <= current_rsi <= 52:  # Healthy breakdown, not oversold
            signal = "BEARISH_MOMENTUM"
            bias_score = -2

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