from collections import deque
from typing import Dict, Optional, List, Any
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger("trading_desk_v2.buffer")


class CandleBufferManager:
    """
    Maintains a rolling sliding-window buffer of intraday candles in memory.
    Prevents repetitive SQLite queries and disk latency during rapid signal evaluation.
    """
    def __init__(self, max_1m_bars: int = 120, max_5m_bars: int = 60):
        self.max_1m_bars = max_1m_bars
        self.max_5m_bars = max_5m_bars
        # Map: ticker -> list of bar dictionaries
        self._buffers_1m: Dict[str, deque] = {}
        self._buffers_5m: Dict[str, deque] = {}

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

        for idx, row in df.iterrows():
            timestamp_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
            bar = {
                "timestamp": timestamp_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            }
            target_deque.append(bar)

    def update_bar(self, ticker: str, bar: Dict[str, Any], interval: str = "1m"):
        """Appends or updates the latest bar in the buffer."""
        self._ensure_ticker(ticker)
        target_deque = self._buffers_1m[ticker] if interval == "1m" else self._buffers_5m[ticker]

        if target_deque and target_deque[-1]["timestamp"] == bar["timestamp"]:
            # In-progress bar update (same minute/interval)
            target_deque[-1] = bar
        else:
            target_deque.append(bar)

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