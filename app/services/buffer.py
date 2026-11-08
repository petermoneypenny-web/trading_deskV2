from collections import deque
from typing import Dict, Optional, List, Any
from datetime import datetime, time
import pandas as pd
import logging

logger = logging.getLogger("trading_desk_v2.buffer")

# A full US regular session is 390 one-minute bars. The buffer used to hold only
# 120, which is what forced VWAP and the opening range to be computed on a
# rolling window instead of being anchored to the session open.
MAX_SESSION_1M_BARS = 400

# RTH open/close in US/Eastern. Bars outside this window (pre/post market) are
# dropped so they cannot pollute the session VWAP or the opening range.
RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


def _is_rth(timestamp_str: str) -> bool:
    """True if a bar's timestamp falls inside 09:30-16:00 US/Eastern."""
    try:
        ts = pd.to_datetime(timestamp_str, utc=True)
        local = ts.tz_convert("America/New_York").time()
        return RTH_OPEN <= local < RTH_CLOSE
    except Exception:
        return True  # fail open: better to keep a bar than to silently drop data


class CandleBufferManager:
    """
    Maintains a rolling sliding-window buffer of intraday candles in memory.
    Prevents repetitive SQLite queries and disk latency during rapid signal evaluation.

    The 1m buffer now holds a full trading session (400 bars) rather than 120, so
    VWAP and the opening range can be anchored to 09:30 ET. Pre-market bars are
    filtered out on load.
    """

    def __init__(self, max_1m_bars: int = MAX_SESSION_1M_BARS, max_5m_bars: int = 60):
        self.max_1m_bars = max_1m_bars
        self.max_5m_bars = max_5m_bars
        # Map: ticker -> list of bar dictionaries
        self._buffers_1m: Dict[str, deque] = {}
        self._buffers_5m: Dict[str, deque] = {}
        # First RTH bar seen in the current session, per ticker.
        self._session_start: Dict[str, Any] = {}

    def _ensure_ticker(self, ticker: str):
        if ticker not in self._buffers_1m:
            self._buffers_1m[ticker] = deque(maxlen=self.max_1m_bars)
        if ticker not in self._buffers_5m:
            self._buffers_5m[ticker] = deque(maxlen=self.max_5m_bars)

    def load_initial_history(self, ticker: str, df: pd.DataFrame, interval: str = "1m"):
        """Populates the in-memory deque from a newly downloaded DataFrame."""
        self._ensure_ticker(ticker)
        target_deque = self._buffers_1m[ticker] if interval == "1m" else self._buffers_5m[ticker]
        target_deque.clear()

        dropped = 0
        for idx, row in df.iterrows():
            timestamp_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
            if interval == "1m" and not _is_rth(timestamp_str):
                dropped += 1
                continue
            bar = {
                "timestamp": timestamp_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            }
            target_deque.append(bar)

        if dropped:
            logger.debug(f"{ticker}: dropped {dropped} non-RTH bars from buffer load")

        # Anchor point for session-scoped indicators.
        if target_deque:
            self._session_start[ticker] = target_deque[0]["timestamp"]

    def update_bar(self, ticker: str, bar: Dict[str, Any], interval: str = "1m"):
        """Appends or updates the latest bar in the buffer."""
        self._ensure_ticker(ticker)
        target_deque = self._buffers_1m[ticker] if interval == "1m" else self._buffers_5m[ticker]

        if target_deque and target_deque[-1]["timestamp"] == bar["timestamp"]:
            # In-progress bar update (same minute/interval)
            target_deque[-1] = bar
        else:
            if interval == "1m" and not _is_rth(bar["timestamp"]):
                return
            if not target_deque:
                self._session_start[ticker] = bar["timestamp"]
            target_deque.append(bar)

    def get_session_start(self, ticker: str, interval: str = "1m"):
        """Timestamp of the first bar of the current session, or None."""
        return self._session_start.get(ticker)

    def reset_session(self, ticker: Optional[str] = None):
        """Clears buffers at the start of a new trading session."""
        targets = [ticker] if ticker else list(self._buffers_1m)
        for t in targets:
            if t in self._buffers_1m:
                self._buffers_1m[t].clear()
            if t in self._buffers_5m:
                self._buffers_5m[t].clear()
            self._session_start.pop(t, None)

    def get_dataframe(self, ticker: str, interval: str = "1m") -> pd.DataFrame:
        """Returns the rolling buffer as a standardized pandas DataFrame."""
        self._ensure_ticker(ticker)
        target_deque = self._buffers_1m[ticker] if interval == "1m" else self._buffers_5m[ticker]
        if not target_deque:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(list(target_deque))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_latest_bar(self, ticker: str, interval: str = "1m") -> Optional[Dict[str, Any]]:
        self._ensure_ticker(ticker)
        target_deque = self._buffers_1m[ticker] if interval == "1m" else self._buffers_5m[ticker]
        return target_deque[-1] if target_deque else None


buffer_manager = CandleBufferManager()
