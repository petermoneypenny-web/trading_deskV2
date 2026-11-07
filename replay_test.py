import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd
from unittest.mock import AsyncMock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replay_test")

logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from app.core.state import system_state
from app.services.indicators import evaluate_intraday_technical_snapshot
import app.services.risk_engine as risk_engine
from app.services.risk_engine import (
    init_db,
    open_position,
    update_open_positions_and_risk,
    liquidate_all_positions,
    ticker_cooldowns
)

risk_engine.send_ntfy_alert = AsyncMock()

TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "INTC", "TSM", "MU"]


def get_mock_session_phase(bar_time) -> str:
    """Calculates phase based on historical bar time in US Eastern."""
    if bar_time.tzinfo is None:
        et_time = bar_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
    else:
        et_time = bar_time.astimezone(ZoneInfo("America/New_York"))

    time_mins = et_time.hour * 60 + et_time.minute

    if 9 * 60 + 30 <= time_mins < 10 * 60 + 15:
        return "RTH_OPENING_FLUSH"  # 09:30 - 10:15 ET: Skip opening noise
    elif 10 * 60 + 15 <= time_mins < 11 * 60 + 30:
        return "RTH_PRIME_MORNING"   # 10:15 - 11:30 ET: Prime morning trend
    elif 11 * 60 + 30 <= time_mins < 14 * 60 + 15:
        return "RTH_MIDDAY"          # 11:30 - 14:15 ET: Midday dead zone
    elif 14 * 60 + 15 <= time_mins < 15 * 60 + 30:
        return "RTH_PRIME_AFTERNOON" # 14:15 - 15:30 ET: Afternoon squeeze
    elif 15 * 60 + 30 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"
    else:
        return "CLOSED"


def update_macro_regime_bias():
    spy_analysis = system_state.technical_analysis.get("SPY", {})
    qqq_analysis = system_state.technical_analysis.get("QQQ", {})

    spy_m = spy_analysis.get("metrics", {})
    qqq_m = qqq_analysis.get("metrics", {})

    spy_p, spy_v = spy_m.get("last_price"), spy_m.get("vwap")
    qqq_p, qqq_v = qqq_m.get("last_price"), qqq_m.get("vwap")

    bull_score = (1 if spy_p and spy_v and spy_p > spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p > qqq_v else 0)
    bear_score = (1 if spy_p and spy_v and spy_p < spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p < qqq_v else 0)

    if bull_score == 2:
        system_state.macro_bias = "BULLISH"
    elif bear_score == 2:
        system_state.macro_bias = "BEARISH"
    else:
        system_state.macro_bias = "NEUTRAL"


def get_portfolio_active_direction() -> str:
    if not system_state.open_positions:
        return "ANY"
    actions = {pos["action"] for pos in system_state.open_positions.values()}
    if "BUY" in actions and "SELL" not in actions:
        return "LONG_ONLY"
    elif "SELL" in actions and "BUY" not in actions:
        return "SHORT_ONLY"
    return "ANY"


async def run_replay(target_date_str: str = None):
    logger.info("Initializing hardened market data replay test harness...")
    init_db()

    system_state.is_active = True
    system_state.active_tickers = TICKERS
    system_state.daily_realized_pnl = 0.0
    system_state.daily_unrealized_pnl = 0.0
    system_state.daily_trade_counts = {}
    system_state.open_positions = {}
    ticker_cooldowns.clear()

    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: {target_date_str}. Use YYYY-MM-DD.")
            return
    else:
        target_date = date(2026, 11, 6)

    start_str = target_date.strftime("%Y-%m-%d")
    end_str = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    ticker_str = " ".join(TICKERS)
    logger.info(f"Downloading 1m data for [{ticker_str}] from {start_str} to {end_str}...")

    raw_df = yf.download(
        tickers=ticker_str,
        start=start_str,
        end=end_str,
        interval="1m",
        prepost=False,
        progress=False,
        threads=False,
        group_by="column"
    )

    if raw_df is None or raw_df.empty:
        logger.error(f"No 1m data returned for date {start_str}.")
        return

    all_timestamps = sorted(list(raw_df.index))
    session_timestamps = [ts for ts in all_timestamps if ts.date() == target_date]
    logger.info(f"Loaded {len(session_timestamps)} 1-minute bars for trading date: {target_date}")

    if len(session_timestamps) < 40:
        logger.error(f"Insufficient bars ({len(session_timestamps)}) for date {target_date}.")
        return

    ticker_dfs = {}
    for ticker in TICKERS:
        df_t = pd.DataFrame(index=session_timestamps)
        df_t["open"] = raw_df[("Open", ticker)].loc[session_timestamps].astype(float)
        df_t["high"] = raw_df[("High", ticker)].loc[session_timestamps].astype(float)
        df_t["low"] = raw_df[("Low", ticker)].loc[session_timestamps].astype(float)
        df_t["close"] = raw_df[("Close", ticker)].loc[session_timestamps].astype(float)
        df_t["volume"] = raw_df[("Volume", ticker)].loc[session_timestamps].astype(float)
        df_t.ffill(inplace=True)
        df_t.bfill(inplace=True)
        ticker_dfs[ticker] = df_t

    trades_opened = 0
    blocked_by_macro = 0
    blocked_by_midday = 0
    blocked_by_opening_flush = 0
    blocked_by_direction_lock = 0
    blocked_by_confirmation = 0
    signal_counts = {}
    consecutive_signals = {}

    for idx in range(30, len(session_timestamps)):
        ts = session_timestamps[idx]
        if ts.tzinfo is None:
            et_time = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
        else:
            et_time = ts.astimezone(ZoneInfo("America/New_York"))

        session_phase = get_mock_session_phase(ts)
        system_state.session_phase = session_phase

        start_idx = max(0, idx - 120)

        for ticker in TICKERS:
            sub_df = ticker_dfs[ticker].iloc[start_idx : idx + 1]
            last_close = float(sub_df["close"].iloc[-1])
            last_high = float(sub_df["high"].iloc[-1])
            last_low = float(sub_df["low"].iloc[-1])
            last_vol = float(sub_df["volume"].iloc[-1])

            system_state.latest_quotes[ticker] = {
                "price": last_close,
                "high": last_high,
                "low": last_low,
                "volume": last_vol,
                "timestamp": ts.isoformat()
            }

            analysis = evaluate_intraday_technical_snapshot(sub_df)
            system_state.technical_analysis[ticker] = analysis

        update_macro_regime_bias()
        await update_open_positions_and_risk(system_state.latest_quotes)

        if session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
            logger.info(f"[{et_time.strftime('%H:%M')} ET] EOD Auto-Close Triggered.")
            await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")
            continue

        portfolio_direction = get_portfolio_active_direction()

        for ticker in TICKERS:
            analysis = system_state.technical_analysis.get(ticker, {})
            sig = analysis.get("signal", "NEUTRAL")

            # Consecutive signal tracking
            tracker = consecutive_signals.get(ticker, {"signal": "NEUTRAL", "bars": 0})
            if sig not in ("NEUTRAL", None) and sig == tracker["signal"]:
                tracker["bars"] += 1
            else:
                tracker = {"signal": sig, "bars": 1 if sig not in ("NEUTRAL", None) else 0}
            consecutive_signals[ticker] = tracker

            if sig not in ("NEUTRAL", None):
                signal_counts[sig] = signal_counts.get(sig, 0) + 1

                # 1. Opening Flush Block (09:30 - 10:15 ET)
                if session_phase == "RTH_OPENING_FLUSH":
                    blocked_by_opening_flush += 1
                    continue

                # 2. Midday Dead Zone Block (11:30 - 14:15 ET)
                if session_phase == "RTH_MIDDAY":
                    blocked_by_midday += 1
                    continue

                if session_phase in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON") and ticker not in system_state.open_positions:
                    # 3. Consecutive Bar Signal Confirmation
                    if tracker["bars"] < 2:
                        blocked_by_confirmation += 1
                        continue

                    action = "BUY" if "BULLISH" in sig or "LONG" in sig else "SELL"

                    # 4. Macro Filter Check
                    if action == "BUY" and system_state.macro_bias == "BEARISH":
                        blocked_by_macro += 1
                        continue
                    if action == "SELL" and system_state.macro_bias == "BULLISH":
                        blocked_by_macro += 1
                        continue

                    # 5. Portfolio Direction Lock
                    if action == "BUY" and portfolio_direction == "SHORT_ONLY":
                        blocked_by_direction_lock += 1
                        continue
                    if action == "SELL" and portfolio_direction == "LONG_ONLY":
                        blocked_by_direction_lock += 1
                        continue

                    real_price = system_state.latest_quotes[ticker]["price"]
                    metrics = analysis.get("metrics", {})
                    atr = float(metrics.get("atr_14", 1.0) or 1.0)

                    min_stop_distance = max(atr * 2.2, real_price * 0.0055)
                    if action == "BUY":
                        stop_loss = round(real_price - min_stop_distance, 2)
                        take_profit = round(real_price + (min_stop_distance * 2.0), 2)
                    else:
                        stop_loss = round(real_price + min_stop_distance, 2)
                        take_profit = round(real_price - (min_stop_distance * 2.0), 2)

                    opened = await open_position(
                        ticker=ticker,
                        action=action,
                        current_price=real_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reasoning=f"Confirmed 2-bar setup: {sig} (Macro: {system_state.macro_bias})"
                    )
                    if opened:
                        trades_opened += 1
                        logger.info(f"[{et_time.strftime('%H:%M')} ET] OPENED {action} {ticker} @ ${real_price:.2f} | SL: ${stop_loss} | TP: ${take_profit}")

    logger.info("=" * 60)
    logger.info(f"REPLAY BACKTEST COMPLETED FOR: {target_date}")
    logger.info("=" * 60)
    logger.info(f"Total Technical Signals Detected: {signal_counts}")
    logger.info(f"Entries Blocked by 09:30-10:15 Opening Flush: {blocked_by_opening_flush}")
    logger.info(f"Entries Blocked by 11:30-14:15 Midday Chop: {blocked_by_midday}")
    logger.info(f"Entries Blocked by Macro Bias Filter: {blocked_by_macro}")
    logger.info(f"Entries Blocked by Portfolio Direction Lock: {blocked_by_direction_lock}")
    logger.info(f"Entries Blocked by 2-Bar Confirmation Filter: {blocked_by_confirmation}")
    logger.info(f"Total Trades Opened: {trades_opened}")
    logger.info(f"Final Realized P&L: ${system_state.daily_realized_pnl:+.2f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay intraday trading simulation for a specific date.")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format.")
    args = parser.parse_args()

    asyncio.run(run_replay(args.date))
