"""
Central risk configuration.

`rulebook.json` used to be dead configuration: nothing in the codebase read it,
so the limits it declared (`max_total_open_positions`, `min_risk_reward_ratio`,
`min_dollar_volume`, ...) existed only on paper. This module loads it once and
exposes it as a typed, validated object that every other service consumes.

Load order (first match wins):
    1. RISK_CONFIG_PATH env var
    2. <repo root>/rulebook.json
    3. built-in DEFAULTS below

Every value is also overridable by an environment variable of the same name,
which is what lets you tune the engine inside the container without a rebuild.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Dict, List, Optional

logger = None  # populated lazily to avoid a logging/config import cycle


# --------------------------------------------------------------------------
# Correlation clusters
# --------------------------------------------------------------------------
# SPY/QQQ/NVDA/AAPL are 0.7-0.95 correlated intraday, so opening all four at
# once is one directional bet at 4x size, not a diversified book. Positions are
# grouped into clusters and the engine caps how many may be open per cluster.
TICKER_CLUSTERS: Dict[str, List[str]] = {
    "INDEX": ["SPY", "QQQ", "IWM", "DIA"],
    "SEMIS": ["NVDA", "TSM", "INTC", "MU", "AMD", "AVGO", "QCOM"],
    "MEGA_TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"],
}

_TICKER_TO_CLUSTER: Dict[str, str] = {
    t: cluster for cluster, members in TICKER_CLUSTERS.items() for t in members
}


def cluster_for(ticker: str) -> str:
    """Cluster a ticker belongs to. Unknown tickers form their own cluster."""
    return _TICKER_TO_CLUSTER.get(ticker.upper(), f"SOLO:{ticker.upper()}")


# --------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------
@dataclass
class RiskConfig:
    # --- inherited from the original rulebook.json -------------------------
    min_risk_reward_ratio: float = 2.0
    max_risk_score: int = 6
    min_stock_price: float = 10.0
    min_dollar_volume: float = 25_000_000.0
    max_portfolio_risk_pct: float = 3.0
    max_open_trades_per_ticker: int = 1
    # Distinct from the above: how many *entries* per ticker per day. The
    # original engine had MAX_TRADES_PER_TICKER_DAILY = 3 but only ever read
    # the simultaneous-open cap, so a ticker could be entered once and then
    # locked out for the rest of the day.
    max_trades_per_ticker_daily: int = 3
    max_total_open_positions: int = 4
    atr_stop_multiplier: float = 2.5
    max_stop_distance_pct: float = 0.03           # 3.0% of price ceiling

    # --- stop / target geometry -------------------------------------------
    # The old engine floored the stop at 0.55% of price. On a 1-minute chart
    # that is inside the noise band, which is what produced the 50% stop-out
    # rate. The floor is raised and the stop is now derived from ATR.
    min_stop_pct: float = 0.008          # 0.80% of price
    target_rr: float = 2.5               # target = stop_distance * target_rr

    # --- position sizing ---------------------------------------------------
    max_risk_per_trade: float = 50.0
    # NOTE: this must be >= max_risk_per_trade / min_stop_pct, otherwise the
    # position-value cap binds before the risk budget and the engine silently
    # under-risks every trade (which is exactly what happened before: a $2500
    # cap with a 0.55% stop pinned real risk at $13.75, 26% of the $50 budget).
    # reconcile() enforces that relationship.
    max_position_value: float = 6250.0
    max_daily_loss: float = 120.0        # rescaled to be actually reachable
    max_trades_per_day: int = 6
    max_positions_per_cluster: int = 1
    max_total_open_positions: int = 3
    equity: float = 25_000.0

    # --- trade management --------------------------------------------------
    max_hold_minutes: int = 45           # time stop
    breakeven_trigger_r: float = 1.0     # move stop to entry after +1R
    cooldown_minutes: int = 30

    # --- transaction costs -------------------------------------------------
    commission_per_share: float = 0.005
    slippage_pct: float = 0.0002         # 2 bps each way

    # --- liquidity ---------------------------------------------------------
    dollar_volume_lookback_bars: int = 15
    min_relative_volume: float = 0.8

    # --- derived (not loaded from file) ------------------------------------
    _clusters: Dict[str, int] = field(default_factory=dict, repr=False)

    # ----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskConfig":
        known = {f.name for f in fields(cls) if not f.name.startswith("_")}
        unknown = sorted(set(data) - known)
        if unknown:
            _log().warning(f"Ignoring unknown rulebook keys: {unknown}")
        merged = {k: v for k, v in data.items() if k in known}
        cfg = cls(**merged)
        cfg.validate()
        cfg.reconcile()
        return cfg

    @classmethod
    def load(cls, path: Optional[str] = None) -> "RiskConfig":
        path = path or os.environ.get("RISK_CONFIG_PATH") or _default_path()
        data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
            except Exception as exc:  # pragma: no cover - defensive
                _log().error(f"Failed to read {path}: {exc}. Using defaults.")
        else:
            _log().warning(f"No rulebook at {path}; using built-in defaults.")

        # environment overrides, e.g. RISK_MAX_DAILY_LOSS=90
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            env = os.environ.get(f"RISK_{f.name.upper()}")
            if env is None:
                continue
            try:
                data[f.name] = f.type(env) if isinstance(f.type, type) else env
            except (TypeError, ValueError):
                _log().warning(f"Ignoring malformed RISK_{f.name.upper()}={env!r}")

        return cls.from_dict(data)

    # ----------------------------------------------------------------------
    def validate(self) -> None:
        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be > 0")
        if self.max_stop_distance_pct < self.min_stop_pct:
            raise ValueError("max_stop_distance_pct must be >= min_stop_pct")
        if self.target_rr <= 1.0:
            raise ValueError("target_rr must be > 1.0 for a positive-expectancy shape")
        if self.min_risk_reward_ratio > self.target_rr:
            raise ValueError(
                "min_risk_reward_ratio cannot exceed target_rr; no trade could pass"
            )
        if self.max_risk_per_trade <= 0 or self.max_position_value <= 0:
            raise ValueError("risk and position caps must be > 0")
        if self.max_daily_loss > self.max_risk_per_trade * self.max_trades_per_day:
            raise ValueError(
                "max_daily_loss exceeds the worst case the per-trade caps allow; "
                "the circuit breaker could never fire"
            )
        if self.max_positions_per_cluster < 1:
            raise ValueError("max_positions_per_cluster must be >= 1")

    def reconcile(self) -> "RiskConfig":
        """
        Makes the position-value cap consistent with the risk budget.

        max_position_value must be at least max_risk_per_trade / min_stop_pct
        for the risk budget to be the binding constraint. If it is lower the
        capital cap wins on every trade and the engine silently under-risks --
        the original bug. We raise the cap and say so rather than accept a
        configuration that cannot express its own risk budget.
        """
        required = self.max_risk_per_trade / max(self.min_stop_pct, 1e-9)
        if self.max_position_value < required:
            _log().warning(
                f"max_position_value=${self.max_position_value:,.0f} is below the "
                f"${required:,.0f} needed to express a ${self.max_risk_per_trade:.0f} "
                f"risk budget at a {self.min_stop_pct * 100:.2f}% minimum stop. "
                f"Raising it to ${required:,.0f} so risk (not capital) binds."
            )
            self.max_position_value = round(required, 2)

        worst_case = self.max_risk_per_trade * self.max_trades_per_day
        if self.max_daily_loss > worst_case:
            _log().warning(
                f"max_daily_loss=${self.max_daily_loss:,.0f} exceeds the worst case "
                f"(${worst_case:,.0f}) the per-trade caps allow; lowering it so the "
                "circuit breaker can actually fire."
            )
            self.max_daily_loss = round(worst_case * 0.6, 2)
        return self

    # ----------------------------------------------------------------------
    def cluster_cap(self, cluster: str) -> int:
        return self.max_positions_per_cluster

    def max_cluster_exposure_usd(self) -> float:
        """Dollar ceiling for a single correlation cluster."""
        return min(
            self.max_position_value * self.max_positions_per_cluster,
            self.equity * 0.12,
        )

    def round_trip_cost(self, shares: float, price: float) -> float:
        """Commission + slippage for a full round trip, in dollars."""
        commission = self.commission_per_share * shares * 2.0
        slippage = self.slippage_pct * price * shares * 2.0
        return commission + slippage

    def breakeven_stop(self, action: str, entry_price: float) -> float:
        """Stop price that leaves the trade flat after costs."""
        buffer = self.commission_per_share + self.slippage_pct * entry_price
        if action == "BUY":
            return round(entry_price + buffer, 4)
        return round(entry_price - buffer, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


def _default_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "rulebook.json")


def _log():
    global logger
    if logger is None:
        import logging

        logger = logging.getLogger("trading_desk_v2.risk_config")
    return logger


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------
_CONFIG: Optional[RiskConfig] = None


def get_risk_config(reload: bool = False) -> RiskConfig:
    """Process-wide RiskConfig singleton."""
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = RiskConfig.load()
    return _CONFIG


def set_risk_config(cfg: RiskConfig) -> None:
    global _CONFIG
    _CONFIG = cfg
