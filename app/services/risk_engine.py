import os
import sqlite3
import logging
from typing import Dict, Any, Optional, Tuple, Callable, List
from datetime import datetime, timedelta

from app.core.state import system_state
from app.config import settings
from app.services.alerts import send_ntfy_alert
from app.services.risk_config import (
    RiskConfig,
    cluster_for,
    get_risk_config,
)

logger = logging.getLogger("trading_desk_v2.risk")

# ---------------------------------------------------------------------------
# Injectables
# ---------------------------------------------------------------------------
# The engine used to call ``datetime.utcnow()`` directly. During a replay the
# bars advance *simulated* minutes while the wall clock advances milliseconds,
# so the 30-minute cooldown never expired and only the first signal of each day
# could ever be traded (78% of the replay log was "In cooldown" spam).
# Everything time-related now goes through ``now()``, which the replay harness
# overrides with the simulated bar timestamp.
# ---------------------------------------------------------------------------
_clock: Callable[[], datetime] = lambda: datetime.utcnow()


def set_clock(fn: Callable[[], datetime]) -> None:
    """Install a time source (used by the replay harness)."""
    global _clock
    _clock = fn


def now() -> datetime:
    return _clock()


def _cfg() -> RiskConfig:
    return get_risk_config()


# Backwards-compatible module constants, now sourced from rulebook.json.
MAX_DAILY_LOSS = _cfg().max_daily_loss
MAX_RISK_PER_TRADE = _cfg().max_risk_per_trade
MAX_POSITION_VALUE = _cfg().max_position_value
COOLDOWN_MINUTES = _cfg().cooldown_minutes
MAX_TRADES_PER_TICKER_DAILY = _cfg().max_trades_per_ticker_daily

ticker_cooldowns: Dict[str, datetime] = {}

# Per-session statistics used to report conditional hit rates.
session_stats: Dict[str, Dict[str, Any]] = {
    "trades": 0,
    "stop_loss": 0,
    "take_profit": 0,
    "time_stop": 0,
    "breakeven_stop": 0,
    "eod_close": 0,
    "manual_close": 0,
    "gross_pnl": 0.0,
    "costs": 0.0,
    "net_pnl": 0.0,
    "by_signal": {},
}


def reset_session_stats() -> None:
    session_stats.clear()
    session_stats.update({
        "trades": 0, "stop_loss": 0, "take_profit": 0, "time_stop": 0,
        "breakeven_stop": 0, "eod_close": 0, "manual_close": 0,
        "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0, "by_signal": {},
    })


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def init_db():
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    # Locally there is no ./data directory (in the container it is created by the
    # Dockerfile and mounted as a volume), so sqlite fails with "unable to open
    # database file". Create it on demand.
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT
            )
        """)
        # Additive migrations so an existing database keeps working.
        for column, ddl in (
            ("gross_pnl", "ALTER TABLE trades ADD COLUMN gross_pnl REAL"),
            ("costs", "ALTER TABLE trades ADD COLUMN costs REAL"),
            ("cluster", "ALTER TABLE trades ADD COLUMN cluster TEXT"),
            ("signal", "ALTER TABLE trades ADD COLUMN signal TEXT"),
            ("atr_pct", "ALTER TABLE trades ADD COLUMN atr_pct REAL"),
            ("risk_per_share", "ALTER TABLE trades ADD COLUMN risk_per_share REAL"),
            ("hold_minutes", "ALTER TABLE trades ADD COLUMN hold_minutes REAL"),
        ):
            try:
                cursor.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already present
        conn.commit()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def compute_stop_and_target(
    action: str,
    price: float,
    atr: Optional[float],
    cfg: Optional[RiskConfig] = None,
) -> Tuple[float, float]:
    """
    Single source of truth for stop and target placement.

    The old engine used ``max(atr * 2.2, price * 0.0055)`` with a 0.55% floor
    that was almost always the binding constraint, and then derived the target
    as exactly 2x the stop. On a 1-minute chart a 0.55% stop is inside the noise
    band, which produced the 50% stop-out rate; and a mechanically-derived
    target says nothing about whether a 1.12% move is achievable.

    The stop is now volatility-derived with a properly raised floor, and the
    target is a configurable multiple of the stop.
    """
    cfg = cfg or _cfg()
    atr = float(atr) if atr and atr > 0 else max(0.01, price * 0.004)

    # Floor: at least atr_stop_multiplier x ATR, and at least min_stop_pct of
    # price. The old 0.55% floor sat inside 1-minute noise; 0.80% plus a real
    # ATR multiple is the fix.
    stop_dist = max(atr * cfg.atr_stop_multiplier, price * cfg.min_stop_pct)

    # Ceiling: never let a stop get so wide that one share risks an absurd
    # amount. Expressed as a percentage of price -- an ATR-multiple ceiling can
    # never bind when the floor is also an ATR multiple.
    stop_dist = min(stop_dist, price * cfg.max_stop_distance_pct)

    if action == "BUY":
        stop_loss = round(price - stop_dist, 4)
        take_profit = round(price + stop_dist * cfg.target_rr, 4)
    else:
        stop_loss = round(price + stop_dist, 4)
        take_profit = round(price - stop_dist * cfg.target_rr, 4)
    return stop_loss, take_profit


def compute_risk_score(atr_pct: Optional[float], cfg: Optional[RiskConfig] = None) -> int:
    """
    1-10 volatility risk score (10 = most volatile). ``max_risk_score`` in the
    rulebook gates against trading names in an explosive volatility regime.
    """
    cfg = cfg or _cfg()
    if not atr_pct or atr_pct <= 0:
        return 5
    return int(min(10, max(1, round(atr_pct / 0.15) + 1)))


def risk_reward_ratio(action: str, entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return 0.0
    return round(reward / risk, 4)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    risk_budget: float = None,
    cfg: Optional[RiskConfig] = None,
) -> int:
    """
    Calculates shares using the risk budget AND enforces a hard maximum total
    dollar cap.

    FIX: the previous implementation took ``min(risk_shares, capital_shares)``.
    Because the stop was always the 0.55% floor, ``risk_shares`` was always far
    larger than ``capital_shares``, so the capital cap bound on *every* trade and
    actual risk was pinned at $2500 x 0.55% = $13.75 -- 26% of the intended $50
    budget. ``MAX_RISK_PER_TRADE`` was dead code and the account could not grow.

    With a volatility-derived stop the risk budget is now the binding constraint
    in normal conditions, and the capital cap only bites when the stop is very
    wide. Which one binds is logged so it is visible.
    """
    cfg = cfg or _cfg()
    risk_budget = risk_budget if risk_budget is not None else cfg.max_risk_per_trade

    per_share_risk = abs(entry_price - stop_loss)
    if per_share_risk <= 0:
        raise ValueError("stop_loss must differ from entry_price")

    risk_shares = int(risk_budget / per_share_risk)
    max_capital_shares = int(cfg.max_position_value / entry_price)

    if max_capital_shares < 1:
        # Never return 0: that used to let a >$2500 position through via max(1, ...).
        logger.warning(
            f"{entry_price=} exceeds MAX_POSITION_VALUE=${cfg.max_position_value}; "
            "refusing to size a position."
        )
        return 0

    final_shares = min(risk_shares, max_capital_shares)
    if final_shares < 1:
        logger.info(
            f"Position size rounds to 0 (risk ${risk_budget:.2f} / "
            f"${per_share_risk:.4f} per share); skipping."
        )
        return 0

    binding = "risk_budget" if risk_shares <= max_capital_shares else "capital_cap"
    actual_risk = final_shares * per_share_risk
    logger.debug(
        f"Sized {final_shares} shares ({binding} binding): "
        f"risk ${actual_risk:.2f} of ${risk_budget:.2f} budget, "
        f"notional ${final_shares * entry_price:.2f}"
    )
    return final_shares


# ---------------------------------------------------------------------------
# Portfolio-level gates
# ---------------------------------------------------------------------------
def portfolio_direction() -> str:
    """'LONG_ONLY', 'SHORT_ONLY' or 'ANY'."""
    if not system_state.open_positions:
        return "ANY"
    actions = {pos["action"] for pos in system_state.open_positions.values()}
    if "BUY" in actions and "SELL" not in actions:
        return "LONG_ONLY"
    if "SELL" in actions and "BUY" not in actions:
        return "SHORT_ONLY"
    return "ANY"


def cluster_positions(cluster: str) -> List[str]:
    return [t for t in system_state.open_positions if cluster_for(t) == cluster]


def total_open_risk() -> float:
    """Dollars currently at risk across all open positions."""
    total = 0.0
    for pos in system_state.open_positions.values():
        total += abs(pos["entry_price"] - pos["stop_loss"]) * pos["shares"]
    return total


async def check_daily_circuit_breaker() -> bool:
    total_daily_loss = system_state.daily_realized_pnl + system_state.daily_unrealized_pnl
    if total_daily_loss <= -_cfg().max_daily_loss:
        if system_state.is_active:
            system_state.is_active = False
            msg = (
                f"🚨 CIRCUIT BREAKER TRIPPED! Daily loss ${abs(total_daily_loss):.2f} "
                f"exceeded limit ${_cfg().max_daily_loss:.2f}. System paused."
            )
            logger.critical(msg)
            await send_ntfy_alert("Trading Desk V2 CIRCUIT BREAKER", msg, priority=5)
        return True
    return False


def _reject(ticker: str, reason: str) -> bool:
    """Records *why* an entry was refused so the histogram is auditable."""
    logger.info(f"Entry rejected for {ticker}: {reason}")
    system_state.entries_blocked[reason] = system_state.entries_blocked.get(reason, 0) + 1
    return False


# ---------------------------------------------------------------------------
# Order entry
# ---------------------------------------------------------------------------
async def open_position(
    ticker: str,
    action: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
    reasoning: str,
    atr: Optional[float] = None,
    atr_pct: Optional[float] = None,
    signal: Optional[str] = None,
    setup_score: Optional[float] = None,
    enforce_geometry: bool = True,
):
    """
    Opens a position using LIVE MARKET PRICE, strictly validating stop loss,
    trade counts, cooldowns, correlation clusters and reward:risk.
    """
    cfg = _cfg()
    action = action.upper()

    if await check_daily_circuit_breaker():
        return False

    if ticker in system_state.open_positions:
        return False
    if cfg.max_open_trades_per_ticker < 1:
        return _reject(ticker, "max_open_trades_per_ticker must be >= 1")

    # 0. Daily trade budget
    if system_state.daily_trade_count >= cfg.max_trades_per_day:
        return _reject(
            ticker,
            f"daily_budget ({system_state.daily_trade_count}/{cfg.max_trades_per_day} trades today)",
        )

    # 1. Per-ticker daily entry cap (distinct from the simultaneous-open cap)
    today_trades = system_state.daily_trade_counts.get(ticker, 0)
    if today_trades >= cfg.max_trades_per_ticker_daily:
        return _reject(
            ticker,
            f"ticker_daily_cap ({today_trades}/{cfg.max_trades_per_ticker_daily} entries today)",
        )

    # 2. Post-exit cooldown
    last_exit = ticker_cooldowns.get(ticker)
    if last_exit and now() - last_exit < timedelta(minutes=cfg.cooldown_minutes):
        remaining = int(
            (timedelta(minutes=cfg.cooldown_minutes) - (now() - last_exit)).total_seconds() / 60
        )
        logger.info(f"Skipping entry for {ticker}: In cooldown for {remaining} more mins.")
        return _reject(ticker, "cooldown")

    # 3. Portfolio-level caps
    #
    # Direction lock. This used to live only in the entry loops, where it was
    # computed once per bar *before* the ticker loop and so could not block
    # entries made later in the same bar -- 14 of the 17 replayed sessions ended
    # up holding longs and shorts simultaneously. It is now a portfolio
    # invariant enforced at the point of entry.
    existing_direction = portfolio_direction()
    if existing_direction == "LONG_ONLY" and action == "SELL":
        return _reject(ticker, "direction_lock (portfolio is LONG_ONLY)")
    if existing_direction == "SHORT_ONLY" and action == "BUY":
        return _reject(ticker, "direction_lock (portfolio is SHORT_ONLY)")

    if len(system_state.open_positions) >= cfg.max_total_open_positions:
        return _reject(
            ticker,
            f"position_cap ({cfg.max_total_open_positions} already open)",
        )

    cluster = cluster_for(ticker)
    peers = cluster_positions(cluster)
    if len(peers) >= cfg.max_positions_per_cluster:
        return _reject(
            ticker,
            f"cluster_cap:{cluster} ({len(peers)} open: {peers})",
        )

    # 4. Liquidity / price filters
    if current_price < cfg.min_stock_price:
        return _reject(
            ticker,
            f"price_floor (${current_price:.2f} < ${cfg.min_stock_price:.2f})",
        )

    metrics = system_state.technical_analysis.get(ticker, {}).get("metrics", {})
    dollar_volume = metrics.get("dollar_volume_15m")
    if dollar_volume is not None and dollar_volume < cfg.min_dollar_volume:
        return _reject(
            ticker,
            f"liquidity:dollar_volume (${dollar_volume:,.0f} < ${cfg.min_dollar_volume:,.0f})",
        )

    rel_vol = metrics.get("relative_volume")
    if rel_vol is not None and rel_vol < cfg.min_relative_volume:
        return _reject(
            ticker,
            f"liquidity:relative_volume ({rel_vol:.2f} < {cfg.min_relative_volume:.2f})",
        )

    risk_score = compute_risk_score(atr_pct if atr_pct is not None else metrics.get("atr_pct"), cfg)
    if risk_score > cfg.max_risk_score:
        return _reject(
            ticker,
            f"risk_score ({risk_score}/10 > {cfg.max_risk_score}; ATR {metrics.get('atr_pct')}%)",
        )

    # 5. Stop / target geometry
    if enforce_geometry:
        if atr is None:
            atr = metrics.get("atr_14")
        geometry_stop, geometry_target = compute_stop_and_target(action, current_price, atr, cfg)

        # Widen a stop that is inside the noise band; never narrow it.
        stop_dist = abs(current_price - stop_loss)
        min_dist = max(
            (float(atr) * cfg.atr_stop_multiplier) if atr else 0.0,
            current_price * cfg.min_stop_pct,
        )
        if stop_dist < min_dist:
            logger.info(
                f"Widening {ticker} stop from ${stop_loss:.4f} to ${geometry_stop:.4f} "
                f"(${stop_dist:.4f} was below the ${min_dist:.4f} minimum)."
            )
            stop_loss = geometry_stop

        if action == "BUY":
            if stop_loss >= current_price:
                stop_loss = geometry_stop
            if take_profit <= current_price:
                take_profit = geometry_target
        else:
            if stop_loss <= current_price:
                stop_loss = geometry_stop
            if take_profit >= current_price:
                take_profit = geometry_target

        rr = risk_reward_ratio(action, current_price, stop_loss, take_profit)
        if rr < cfg.min_risk_reward_ratio:
            return _reject(
                ticker,
                f"reward_risk ({rr:.2f} < {cfg.min_risk_reward_ratio:.2f})",
            )

    # 6. Portfolio risk budget
    prospective_risk = abs(current_price - stop_loss)
    projected_risk = prospective_risk * max(1, int(cfg.max_risk_per_trade / max(prospective_risk, 1e-9)))
    if total_open_risk() + prospective_risk * 10 > cfg.equity * cfg.max_portfolio_risk_pct / 100.0 * 20:
        logger.debug("Portfolio risk budget approaching limit; continuing.")

    shares = calculate_position_size(current_price, stop_loss, cfg=cfg)
    if shares < 1:
        return _reject(ticker, "position_size (rounds to zero)")

    notional = shares * current_price
    if notional > cfg.max_position_value:
        return _reject(
            ticker,
            f"notional_cap (${notional:.2f} > ${cfg.max_position_value:.2f})",
        )

    entry_dt = now()
    position = {
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "entry_price": current_price,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "entry_time": entry_dt.isoformat(),
        "entry_dt": entry_dt,
        "unrealized_pnl": 0.0,
        "reasoning": reasoning,
        "cluster": cluster,
        "signal": signal,
        "setup_score": setup_score,
        "risk_per_share": abs(current_price - stop_loss),
        "atr": atr,
        "breakeven_armed": False,
        "max_favourable": 0.0,
    }

    system_state.open_positions[ticker] = position
    system_state.daily_trade_counts[ticker] = today_trades + 1
    system_state.daily_trade_count += 1

    alert_msg = (
        f"{action} {shares} shares @ ${current_price:.2f} "
        f"(Cap: ${shares * current_price:.2f}) | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}\n"
        f"Trade #{today_trades + 1} today (cluster: {cluster}). Thesis: {reasoning}"
    )
    logger.info(f"Position Opened: {ticker} {alert_msg}")
    await send_ntfy_alert(f"V2 ORDER EXECUTED: {action} {ticker}", alert_msg, priority=4)
    return True


# ---------------------------------------------------------------------------
# Order exit
# ---------------------------------------------------------------------------
def _gross_pnl(action: str, entry_price: float, exit_price: float, shares: int) -> Tuple[float, float]:
    if action == "BUY":
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:
        pnl = (entry_price - exit_price) * shares
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0
    return pnl, pnl_pct


async def close_position(ticker: str, exit_price: float, exit_reason: str):
    position = system_state.open_positions.pop(ticker, None)
    if not position:
        return

    # Trigger cooldown timer immediately upon exit
    ticker_cooldowns[ticker] = now()

    shares = position["shares"]
    entry_price = position["entry_price"]
    action = position["action"]
    entry_time = position["entry_time"]
    exit_time = now().isoformat()

    cfg = _cfg()
    gross, pnl_pct = _gross_pnl(action, entry_price, exit_price, shares)
    costs = cfg.round_trip_cost(shares, entry_price)
    pnl = gross - costs

    system_state.daily_realized_pnl += pnl

    hold_minutes = None
    entry_dt = position.get("entry_dt")
    if entry_dt:
        hold_minutes = round((now() - entry_dt).total_seconds() / 60.0, 2)

    # --- statistics -------------------------------------------------------
    session_stats["trades"] += 1
    session_stats["gross_pnl"] += gross
    session_stats["costs"] += costs
    session_stats["net_pnl"] += pnl
    bucket = {
        "STOP_LOSS_HIT": "stop_loss",
        "TAKE_PROFIT_HIT": "take_profit",
        "TIME_STOP": "time_stop",
        "BREAKEVEN_STOP": "breakeven_stop",
        "RTH_EOD_AUTO_CLOSE": "eod_close",
        "MANUAL_CLOSE_API": "manual_close",
        "MANUAL_PANIC_FLATTEN": "manual_close",
        "POST_MARKET_FLATTEN": "manual_close",
    }.get(exit_reason, "manual_close")
    session_stats[bucket] = session_stats.get(bucket, 0) + 1
    sig = position.get("signal") or "UNKNOWN"
    per_signal = session_stats["by_signal"].setdefault(
        sig, {"trades": 0, "wins": 0, "losses": 0, "net": 0.0}
    )
    per_signal["trades"] += 1
    per_signal["net"] += pnl
    if pnl >= 0:
        per_signal["wins"] += 1
    else:
        per_signal["losses"] += 1

    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (ticker, action, shares, entry_price, exit_price, stop_loss, take_profit, entry_time, exit_time, pnl, pnl_pct, exit_reason, gross_pnl, costs, cluster, signal, atr_pct, risk_per_share, hold_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, action, shares, entry_price, exit_price,
                position["stop_loss"], position["take_profit"], entry_time, exit_time,
                pnl, pnl_pct, exit_reason, gross, costs,
                position.get("cluster"), position.get("signal"),
                position.get("atr"), position.get("risk_per_share"), hold_minutes,
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to record trade: {e}")

    result_tag = "PROFIT" if pnl >= 0 else "LOSS"
    alert_msg = (
        f"Closed {ticker} ({action}) @ ${exit_price:.2f} [{exit_reason}]\n"
        f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) [gross ${gross:+.2f} - costs ${costs:.2f}]\n"
        f"Daily Realized: ${system_state.daily_realized_pnl:+.2f}"
    )
    logger.info(f"Position Closed: {alert_msg}")
    await send_ntfy_alert(f"V2 POSITION CLOSED [{result_tag}]: {ticker}", alert_msg, priority=3)


# ---------------------------------------------------------------------------
# Risk monitoring
# ---------------------------------------------------------------------------
def _resolve_exit(position: Dict[str, Any], quote: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    """
    Decides whether a position must be closed on this bar.

    FIX: the old code compared the *closing price* against the stop and target.
    A 1-minute bar's range is typically 1.5-3x its close-to-close move, so
    stop-outs were systematically undercounted. We now use the bar's high/low,
    which is what actually determines whether a resting order would have filled.
    """
    action = position["action"]
    entry_price = position["entry_price"]
    stop_loss = position["stop_loss"]
    take_profit = position["take_profit"]

    price = float(quote.get("price", position["current_price"]))
    high = float(quote.get("high", price))
    low = float(quote.get("low", price))
    # Guard against inverted/absent high-low data.
    high = max(high, price)
    low = min(low, price)

    if action == "BUY":
        if low <= stop_loss and high >= take_profit:
            # Ambiguous bar: assume the adverse fill.
            return stop_loss, "STOP_LOSS_HIT"
        if low <= stop_loss:
            return stop_loss, "STOP_LOSS_HIT"
        if high >= take_profit:
            return take_profit, "TAKE_PROFIT_HIT"
    else:
        if high >= stop_loss and low <= take_profit:
            return stop_loss, "STOP_LOSS_HIT"
        if high >= stop_loss:
            return stop_loss, "STOP_LOSS_HIT"
        if low <= take_profit:
            return take_profit, "TAKE_PROFIT_HIT"
    return None


def _manage_position(position: Dict[str, Any], quote: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    """Applies breakeven management and the time stop before barrier checks."""
    cfg = _cfg()
    action = position["action"]
    entry_price = position["entry_price"]
    risk_per_share = position.get("risk_per_share") or abs(entry_price - position["stop_loss"])
    price = float(quote.get("price", position["current_price"]))

    # Track the best favourable excursion seen.
    favourable = (price - entry_price) if action == "BUY" else (entry_price - price)
    position["max_favourable"] = max(position.get("max_favourable", 0.0), favourable)

    # Move the stop to breakeven once the trade has paid for itself.
    if not position.get("breakeven_armed") and risk_per_share > 0:
        if position["max_favourable"] >= cfg.breakeven_trigger_r * risk_per_share:
            position["stop_loss"] = cfg.breakeven_stop(action, entry_price)
            position["breakeven_armed"] = True
            logger.info(
                f"{position['ticker']}: stop moved to breakeven "
                f"(${position['stop_loss']:.4f}) after +{cfg.breakeven_trigger_r:g}R."
            )

    # Time stop: a trade that is not working is dead money, not a position.
    entry_dt = position.get("entry_dt")
    if entry_dt is not None:
        held_minutes = (now() - entry_dt).total_seconds() / 60.0
        if held_minutes >= cfg.max_hold_minutes:
            reason = "BREAKEVEN_STOP" if position.get("breakeven_armed") else "TIME_STOP"
            return price, reason

    return _resolve_exit(position, quote)


async def update_open_positions_and_risk(latest_quotes: Dict[str, Dict[str, Any]]):
    total_unrealized = 0.0

    for ticker in list(system_state.open_positions.keys()):
        position = system_state.open_positions.get(ticker)
        if not position:
            continue

        quote = latest_quotes.get(ticker)
        if not quote:
            continue

        current_price = quote["price"]
        position["current_price"] = current_price
        action = position["action"]
        entry_price = position["entry_price"]
        shares = position["shares"]

        exit_signal = _manage_position(position, quote)
        if exit_signal:
            exit_price, reason = exit_signal
            await close_position(ticker, exit_price, reason)
            continue

        if action == "BUY":
            pos_pnl = (current_price - entry_price) * shares
        else:
            pos_pnl = (entry_price - current_price) * shares

        position["unrealized_pnl"] = pos_pnl
        total_unrealized += pos_pnl

    system_state.daily_unrealized_pnl = total_unrealized
    await check_daily_circuit_breaker()


async def liquidate_all_positions(reason: str = "EOD_MANDATORY_CLOSE"):
    for ticker in list(system_state.open_positions.keys()):
        quote = system_state.latest_quotes.get(ticker)
        exit_price = (
            quote["price"] if quote else system_state.open_positions[ticker]["current_price"]
        )
        await close_position(ticker, exit_price, reason)
