from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class MarketSessionState:
    is_active: bool = False
    session_phase: str = "CLOSED"  # PRE_MARKET, RTH_OPEN, RTH_NORMAL, POST_MARKET, CLOSED
    active_tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA"])
    latest_quotes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    technical_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    open_positions: Dict[str, Any] = field(default_factory=dict)
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0
    last_loop_timestamp: Optional[datetime] = None
    last_fetch_error: Optional[str] = None


system_state = MarketSessionState()