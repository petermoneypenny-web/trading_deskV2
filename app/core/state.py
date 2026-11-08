from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class MarketSessionState:
    is_active: bool = False
    session_phase: str = "CLOSED"  # PRE_MARKET, RTH_OPEN, RTH_NORMAL, RTH_MIDDAY, RTH_EOD_CLOSING, POST_MARKET, CLOSED
    active_tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA"])
    latest_quotes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    technical_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    agent_evaluations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    open_positions: Dict[str, Any] = field(default_factory=dict)
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    last_loop_timestamp: Optional[datetime] = None
    last_fetch_error: Optional[str] = None
    macro_bias: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL (derived from SPY / QQQ)
    daily_trade_counts: Dict[str, int] = field(default_factory=dict)  # {ticker: count_today}

    # --- added ------------------------------------------------------------
    daily_trade_count: int = 0        # total trades taken today (vs per-ticker counts)
    session_date: Optional[str] = None  # YYYY-MM-DD of the session being traded
    entries_blocked: Dict[str, int] = field(default_factory=dict)  # reason -> count
    last_agent_latency_ms: Optional[float] = None


system_state = MarketSessionState()
