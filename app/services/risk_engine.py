import sqlite3
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from app.core.state import system_state
from app.config import settings
from app.services.alerts import send_ntfy_alert

logger = logging.getLogger("trading_desk_v2.risk")

MAX_DAILY_LOSS = 250.0          # Circuit breaker ($)
MAX_RISK_PER_TRADE = 50.0       # Max dollar risk per trade ($)
MAX_POSITION_VALUE = 2500.0     # Max position dollar exposure ($)
COOLDOWN_MINUTES = 30           # Wait 30 mins after exiting a ticker before re-entering
MAX_TRADES_PER_TICKER_DAILY = 3 # Hard limit to prevent overtrading/churn

ticker_cooldowns: Dict[str, datetime] = {}


def init_db():
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
    """Calculates shares using risk budget AND enforces a hard maximum total dollar cap."""
    per_share_risk = abs(entry_price - stop_loss)
    min_risk = entry_price * 0.0055  # 0.55% minimum realistic risk buffer
    if per_share_risk < min_risk:
        per_share_risk = min_risk

    risk_shares = int(risk_budget / per_share_risk)
    max_capital_shares = int(MAX_POSITION_VALUE / entry_price)
    
    final_shares = min(risk_shares, max_capital_shares)
    return max(1, final_shares)


async def check_daily_circuit_breaker() -> bool:
    total_daily_loss = system_state.daily_realized_pnl + system_state.daily_unrealized_pnl
    if total_daily_loss <= -MAX_DAILY_LOSS:
        if system_state.is_active:
            system_state.is_active = False
            msg = f"🚨 CIRCUIT BREAKER TRIPPED! Daily loss ${abs(total_daily_loss):.2f} exceeded limit ${MAX_DAILY_LOSS:.2f}. System paused."
            logger.critical(msg)
            await send_ntfy_alert("Trading Desk V2 CIRCUIT BREAKER", msg, priority=5)
        return True
    return False


async def open_position(ticker: str, action: str, current_price: float, stop_loss: float, take_profit: float, reasoning: str):
    """Opens position using LIVE MARKET PRICE, strictly validating stop loss, trade counts, and cooldowns."""
    if await check_daily_circuit_breaker():
        return False

    if ticker in system_state.open_positions:
        return False

    # 1. Enforce Daily Max Trades Per Ticker Cap
    today_trades = system_state.daily_trade_counts.get(ticker, 0)
    if today_trades >= MAX_TRADES_PER_TICKER_DAILY:
        logger.info(f"Skipping entry for {ticker}: Daily cap reached ({today_trades}/{MAX_TRADES_PER_TICKER_DAILY} trades today).")
        return False

    # 2. Enforce Post-Exit Cooldown
    last_exit = ticker_cooldowns.get(ticker)
    if last_exit and datetime.utcnow() - last_exit < timedelta(minutes=COOLDOWN_MINUTES):
        remaining = int((timedelta(minutes=COOLDOWN_MINUTES) - (datetime.utcnow() - last_exit)).total_seconds() / 60)
        logger.info(f"Skipping entry for {ticker}: In cooldown for {remaining} more mins.")
        return False

    # 3. Enforce Minimum Stop Loss Distance (Minimum 0.55% to breathe past 1m noise)
    min_dist = round(current_price * 0.0055, 2)
    if action == "BUY":
        if stop_loss >= current_price or (current_price - stop_loss) < min_dist:
            stop_loss = round(current_price - min_dist, 2)
        if take_profit <= current_price:
            take_profit = round(current_price + (min_dist * 2.0), 2)
    elif action == "SELL":
        if stop_loss <= current_price or (stop_loss - current_price) < min_dist:
            stop_loss = round(current_price + min_dist, 2)
        if take_profit >= current_price:
            take_profit = round(current_price - (min_dist * 2.0), 2)

    shares = calculate_position_size(current_price, stop_loss)
    position = {
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "entry_price": current_price,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "entry_time": datetime.utcnow().isoformat(),
        "unrealized_pnl": 0.0,
        "reasoning": reasoning
    }

    system_state.open_positions[ticker] = position
    system_state.daily_trade_counts[ticker] = today_trades + 1

    alert_msg = f"{action} {shares} shares @ ${current_price:.2f} (Cap: ${shares*current_price:.2f}) | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}\nTrade #{today_trades + 1} today. Thesis: {reasoning}"
    logger.info(f"Position Opened: {ticker} {alert_msg}")
    await send_ntfy_alert(f"V2 ORDER EXECUTED: {action} {ticker}", alert_msg, priority=4)
    return True


async def close_position(ticker: str, exit_price: float, exit_reason: str):
    position = system_state.open_positions.pop(ticker, None)
    if not position:
        return

    # Trigger cooldown timer immediately upon exit
    ticker_cooldowns[ticker] = datetime.utcnow()

    shares = position["shares"]
    entry_price = position["entry_price"]
    action = position["action"]
    entry_time = position["entry_time"]
    exit_time = datetime.utcnow().isoformat()

    if action == "BUY":
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:
        pnl = (entry_price - exit_price) * shares
        pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0

    system_state.daily_realized_pnl += pnl

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
        logger.error(f"Failed to record trade: {e}")

    result_tag = "PROFIT" if pnl >= 0 else "LOSS"
    alert_msg = f"Closed {ticker} ({action}) @ ${exit_price:.2f} [{exit_reason}]\nP&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)\nDaily Realized: ${system_state.daily_realized_pnl:+.2f}"
    logger.info(f"Position Closed: {alert_msg}")
    await send_ntfy_alert(f"V2 POSITION CLOSED [{result_tag}]: {ticker}", alert_msg, priority=3)


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
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]

        if action == "BUY":
            pos_pnl = (current_price - entry_price) * shares
            if current_price <= stop_loss:
                await close_position(ticker, current_price, "STOP_LOSS_HIT")
                continue
            elif current_price >= take_profit:
                await close_position(ticker, current_price, "TAKE_PROFIT_HIT")
                continue
        else:
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
    for ticker in list(system_state.open_positions.keys()):
        quote = system_state.latest_quotes.get(ticker)
        exit_price = quote["price"] if quote else system_state.open_positions[ticker]["current_price"]
        await close_position(ticker, exit_price, reason)