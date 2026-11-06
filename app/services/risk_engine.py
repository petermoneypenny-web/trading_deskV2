import sqlite3
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.state import system_state
from app.config import settings
from app.services.stream import send_ntfy_alert

logger = logging.getLogger("trading_desk_v2.risk")

# Risk Parameters
MAX_DAILY_LOSS = 250.0        # Hard circuit breaker: Max allowable daily loss in USD
MAX_RISK_PER_TRADE = 50.0     # Maximum dollars risked on a single trade entry
DEFAULT_ACCOUNT_EQUITY = 10000.0


def init_db():
    """Initializes SQLite database tables for trades and performance tracking."""
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
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
        conn.commit()


def calculate_position_size(entry_price: float, stop_loss: float, risk_budget: float = MAX_RISK_PER_TRADE) -> int:
    """
    Calculates number of shares based on fixed dollar risk and stop loss distance.
    """
    per_share_risk = abs(entry_price - stop_loss)
    if per_share_risk <= 0.001:
        return 1
    shares = int(risk_budget / per_share_risk)
    return max(1, shares)


async def check_daily_circuit_breaker() -> bool:
    """
    Checks if total daily losses exceed the maximum daily threshold.
    Trips circuit breaker and pauses the trading engine if violated.
    """
    total_daily_loss = system_state.daily_realized_pnl + system_state.daily_unrealized_pnl
    if total_daily_loss <= -MAX_DAILY_LOSS:
        if system_state.is_active:
            system_state.is_active = False
            msg = f"🚨 CIRCUIT BREAKER TRIPPED! Daily loss ${abs(total_daily_loss):.2f} exceeded limit ${MAX_DAILY_LOSS:.2f}. System paused."
            logger.critical(msg)
            await send_ntfy_alert("Trading Desk V2 CIRCUIT BREAKER", msg, priority=5)
        return True
    return False


async def open_position(ticker: str, action: str, entry_price: float, stop_loss: float, take_profit: float, reasoning: str):
    """
    Evaluates risk rules and opens a new intraday paper position.
    """
    if await check_daily_circuit_breaker():
        logger.warning(f"Rejecting entry for {ticker}: Daily circuit breaker active.")
        return False

    if ticker in system_state.open_positions:
        logger.info(f"Position already active for {ticker}. Ignoring duplicate signal.")
        return False

    shares = calculate_position_size(entry_price, stop_loss)
    position = {
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "entry_price": entry_price,
        "current_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "entry_time": datetime.utcnow().isoformat(),
        "unrealized_pnl": 0.0,
        "reasoning": reasoning
    }

    system_state.open_positions[ticker] = position
    alert_msg = f"{action} {shares} shares @ ${entry_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}\n{reasoning}"
    logger.info(f"Position Opened: {ticker} {alert_msg}")
    await send_ntfy_alert(f"V2 ORDER EXECUTED: {action} {ticker}", alert_msg, priority=4)
    return True


async def close_position(ticker: str, exit_price: float, exit_reason: str):
    """
    Closes an active intraday position and logs the trade to SQLite.
    """
    position = system_state.open_positions.pop(ticker, None)
    if not position:
        return

    shares = position["shares"]
    entry_price = position["entry_price"]
    action = position["action"]
    entry_time = position["entry_time"]
    exit_time = datetime.utcnow().isoformat()

    if action == "BUY":
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:  # SELL / SHORT
        pnl = (entry_price - exit_price) * shares
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

    system_state.daily_realized_pnl += pnl

    # Save to SQLite database
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (ticker, action, shares, entry_price, exit_price, stop_loss, take_profit, entry_time, exit_time, pnl, pnl_pct, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticker, action, shares, entry_price, exit_price, position["stop_loss"], position["take_profit"], entry_time, exit_time, pnl, pnl_pct, exit_reason))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to record trade to database: {e}")

    result_tag = "PROFIT" if pnl >= 0 else "LOSS"
    alert_msg = f"Closed {ticker} ({action}) @ ${exit_price:.2f} [{exit_reason}]\nP&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)\nDaily PnL: ${system_state.daily_realized_pnl:+.2f}"
    logger.info(f"Position Closed: {alert_msg}")
    await send_ntfy_alert(f"V2 POSITION CLOSED [{result_tag}]: {ticker}", alert_msg, priority=3)


async def update_open_positions_and_risk(latest_quotes: Dict[str, Dict[str, Any]]):
    """
    Checks stop loss, take profit targets, and updates unrealized P&L for every active position.
    """
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
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]

        # Calculate live unrealized PnL
        if action == "BUY":
            pos_pnl = (current_price - entry_price) * shares
            # Check Stop Loss
            if current_price <= stop_loss:
                await close_position(ticker, current_price, "STOP_LOSS_HIT")
                continue
            # Check Take Profit
            elif current_price >= take_profit:
                await close_position(ticker, current_price, "TAKE_PROFIT_HIT")
                continue
        else:  # SHORT
            pos_pnl = (entry_price - current_price) * shares
            if current_price >= stop_loss:
                await close_position(ticker, current_price, "STOP_LOSS_HIT")
                continue
            elif current_price <= take_profit:
                await close_position(ticker, current_price, "TAKE_PROFIT_HIT")
                continue

        position["unrealized_pnl"] = pos_pnl
        total_unrealized += pos_pnl

    system_state.daily_unrealized_pnl = total_unrealized
    await check_daily_circuit_breaker()


async def liquidate_all_positions(reason: str = "EOD_MANDATORY_CLOSE"):
    """
    Flattens all positions at the end of the day or upon manual command.
    """
    for ticker in list(system_state.open_positions.keys()):
        quote = system_state.latest_quotes.get(ticker)
        exit_price = quote["price"] if quote else system_state.open_positions[ticker]["current_price"]
        await close_position(ticker, exit_price, reason)