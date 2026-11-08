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


def _session_slice(df: pd.DataFrame, session_start) -> pd.DataFrame:
    """
    Restricts a DataFrame to bars at or after ``session_start``.

    This is the fix for the "rolling ORB / rolling VWAP" bug. Previously every
    indicator was computed over whatever window the caller happened to pass
    (a 120-bar rolling slice, or the tail of a 5-day download), which meant the
    "Opening Range Breakout" was really "the high/low of the last 15 minutes"
    and "VWAP" was a rolling 120-minute average that hugs price. Both recompute
    every bar and both fire constantly, which is why 44% of ticker-bars were
    producing a directional signal.

    With ``session_start`` supplied, VWAP is anchored to the 09:30 ET open and
    the ORB is the genuine first-15-minutes range.
    """
    if session_start is None or df.empty:
        return df
    try:
        idx = df.index
        if getattr(session_start, "tzinfo", None) is None and getattr(idx, "tz", None) is not None:
            session_start = session_start.replace(tzinfo=idx.tz)
        mask = idx >= session_start
        sliced = df.loc[mask]
        return sliced if not sliced.empty else df
    except (TypeError, ValueError):
        return df


def calculate_vwap(df: pd.DataFrame, session_start=None) -> pd.Series:
    """
    Calculates intraday Volume-Weighted Average Price (VWAP).
    Typical Price = (High + Low + Close) / 3
    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

    When ``session_start`` is given the cumulative sums are anchored to the
    session open, which is what "intraday VWAP" is supposed to mean.
    """
    df = _session_slice(df, session_start)
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
    """
    Calculates Wilder's Relative Strength Index (RSI).

    FIX: the old implementation did ``rsi.fillna(50.0)``. When there are zero
    down-bars ``avg_loss`` is 0, ``rs`` is NaN and RSI collapses to 50 (neutral),
    throwing away the strongest possible momentum reading. It is now 100.
    """
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype=float)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Zero down-bars => RSI 100, zero up-bars => RSI 0. Never "neutral".
    rsi = rsi.mask(avg_gain.eq(0) & avg_loss.eq(0), 50.0)
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    rsi = rsi.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates Average True Range (ATR) for volatility and stop sizing.

    FIX: Wilder's ATR seeds with the *mean* of the first ``period`` true ranges.
    The old code used ``ewm(alpha=1/period, adjust=False)`` which seeds from the
    first bar's true range alone, badly biasing the early values, and then
    ``ffill().bfill()`` papered over the NaN warm-up window.
    """
    if df.empty or len(df) < 2:
        return pd.Series(dtype=float)

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Reindex onto df's index so the Wilder recursion stays aligned with the
    # bars even if dropna() would have shifted the series.
    true_range = true_range.reindex(df.index)
    if true_range.isna().all():
        return pd.Series(dtype=float)
    true_range = true_range.ffill().bfill()

    if len(true_range) < period:
        return true_range.copy()

    seed = float(true_range.iloc[:period].mean())
    values = [seed]
    for tr_val in true_range.iloc[period:]:
        values.append((values[-1] * (period - 1) + float(tr_val)) / period)

    core = pd.Series(values, index=true_range.index[period - 1 :])
    # Backfill the warm-up window with the seed so early bars are usable.
    if len(core) < len(df):
        pad_len = len(df) - len(core)
        pad = pd.Series([seed] * pad_len, index=df.index[:pad_len])
        core = pd.concat([pad, core])
    return core.reindex(df.index).ffill().bfill()


def calculate_orb(df: pd.DataFrame, bars_count: int = 15, session_start=None) -> Dict[str, Optional[float]]:
    """
    Calculates Opening Range Breakout (ORB) High and Low based on the first N
    bars *of the session* (09:30 ET onwards), not the first N bars of whatever
    rolling window the caller passed in.
    """
    df = _session_slice(df, session_start)
    if df.empty or len(df) < bars_count:
        return {"orb_high": None, "orb_low": None}

    opening_slice = df.iloc[:bars_count]
    return {
        "orb_high": _sanitize_float(opening_slice["high"].max()),
        "orb_low": _sanitize_float(opening_slice["low"].min())
    }


def calculate_dollar_volume(df: pd.DataFrame, lookback: int = 15) -> float:
    """Dollar volume traded over the last ``lookback`` bars (liquidity filter)."""
    if df.empty or "volume" not in df or "close" not in df:
        return 0.0
    window = df.tail(lookback)
    return float((window["close"].astype(float) * window["volume"].astype(float)).sum())


def score_setup(analysis: Dict[str, Any]) -> float:
    """
    Ranks competing setups so the engine can take the best 1-2 instead of every
    ticker that happens to vote bullish.

    Higher is better. Components are normalised to roughly 0-1 each:
      * proximity to VWAP (modest pullbacks close to value score higher)
      * RSI in the healthy band rather than at the edge
      * relative volume confirmation
      * ATR regime sanity (penalise dead names and exploding volatility)
    """
    if not analysis or analysis.get("status") != "READY":
        return 0.0

    m = analysis.get("metrics", {})
    signal = analysis.get("signal", "NEUTRAL")
    if signal in ("NEUTRAL", None):
        return 0.0

    dist = m.get("dist_from_vwap_pct")
    rsi = m.get("rsi_14")
    rel_vol = m.get("relative_volume")
    atr_pct = m.get("atr_pct")

    score = 0.0

    if dist is not None:
        # Sweet spot is a modest pullback, ~0.05-0.35% from VWAP.
        score += max(0.0, 1.0 - abs(abs(dist) - 0.20) / 0.50)

    if rsi is not None:
        # Bullish band 48-68, bearish 32-52; peak in the middle of each.
        centre = 58.0 if "BULLISH" in signal else 42.0
        score += max(0.0, 1.0 - abs(rsi - centre) / 12.0)

    if rel_vol is not None:
        score += min(1.0, max(0.0, (rel_vol - 0.5) / 1.0))

    if atr_pct is not None:
        # Penalise both dead names and exploding volatility.
        if atr_pct < 0.03:
            score -= 0.5
        elif atr_pct > 0.60:
            score -= 0.5

    return round(score, 4)


def evaluate_intraday_technical_snapshot(df: pd.DataFrame, session_start=None) -> Dict[str, Any]:
    """
    Combines indicators into a sanitized, JSON-safe technical snapshot.

    ``session_start`` anchors VWAP and the opening range to the trading session.
    """
    if df.empty or len(df) < 25:
        return {
            "status": "INSUFFICIENT_DATA",
            "bars_analyzed": len(df),
            "signal": "NEUTRAL",
            "bias_score": 0,
            "metrics": {}
        }

    vwap_series = calculate_vwap(df, session_start=session_start)
    emas = calculate_emas(df, fast_period=9, slow_period=21)
    rsi_series = calculate_rsi(df, period=14)
    atr_series = calculate_atr(df, period=14)
    orb = calculate_orb(df, bars_count=15, session_start=session_start)

    last_close = _sanitize_float(df["close"].iloc[-1], default=0.0)
    current_vwap = _sanitize_float(vwap_series.iloc[-1] if not vwap_series.empty else last_close, default=last_close)
    current_ema_fast = _sanitize_float(emas["ema_fast"].iloc[-1] if not emas["ema_fast"].empty else last_close, default=last_close)
    current_ema_slow = _sanitize_float(emas["ema_slow"].iloc[-1] if not emas["ema_slow"].empty else last_close, default=last_close)
    current_rsi = _sanitize_float(rsi_series.iloc[-1] if not rsi_series.empty else 50.0, default=50.0)
    current_atr = _sanitize_float(atr_series.iloc[-1] if not atr_series.empty else 0.5, default=0.5)

    # Calculate distance from VWAP safely
    dist_from_vwap_pct = 0.0
    if current_vwap and current_vwap > 0 and last_close:
        dist_from_vwap_pct = _sanitize_float(((last_close - current_vwap) / current_vwap) * 100.0, default=0.0)

    # Volatility and liquidity context
    atr_pct = None
    if current_atr and last_close:
        atr_pct = _sanitize_float((current_atr / last_close) * 100.0, default=0.0)

    dollar_volume_15m = calculate_dollar_volume(df, lookback=15)
    avg_volume_15m = float(df["volume"].astype(float).tail(15).mean()) if len(df) >= 15 else 0.0
    session_avg_volume = float(df["volume"].astype(float).mean()) if len(df) else 0.0
    relative_volume = None
    if session_avg_volume > 0:
        relative_volume = _sanitize_float(avg_volume_15m / session_avg_volume, default=0.0)

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

    snapshot = {
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
            "dist_from_vwap_pct": dist_from_vwap_pct,
            "atr_pct": atr_pct,
            "dollar_volume_15m": _sanitize_float(dollar_volume_15m, default=0.0),
            "relative_volume": relative_volume,
        }
    }
    snapshot["setup_score"] = score_setup(snapshot)
    return snapshot
