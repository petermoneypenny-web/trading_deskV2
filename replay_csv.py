import os
import argparse
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from unittest.mock import AsyncMock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replay_csv")

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

# Silence live alerts during backtest
risk_engine.send_ntfy_alert = AsyncMock()

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA"]
CSV_PATH = "/app/data/historical_1m_30d.csv"

def get_mock_session_phase(bar_time) -> str:
    if bar_time.tzinfo is None:
        et_time = bar_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
    else:
        et_time = bar_time.astimezone(ZoneInfo("America/New_York"))

    time_mins = et_time.hour * 60 + et_time.minute
    if 9 * 60 + 30 <= time_mins < 9 * 60 + 45:
        return "RTH_OPENING_FLUSH"
    elif 9 * 60 + 45 <= time_mins < 11 * 60 + 30:
        return "RTH_PRIME_MORNING"
    elif 11 * 60 + 30 <= time_mins < 14 * 60 + 15:
        return "RTH_MIDDAY"
    elif 14 * 60 + 15 <= time_mins < 15 * 60 + 30:
        return "RTH_PRIME_AFTERNOON"
    elif 15 * 60 + 30 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"
    else:
        return "CLOSED"

def update_macro_regime_bias():
    spy_metrics = system_state.technical_analysis.get("SPY", {}).get("metrics", {})
    qqq_metrics = system_state.technical_analysis.get("QQQ", {}).get("metrics", {})
    spy_p, spy_v = spy_metrics.get("last_price"), spy_metrics.get("vwap")
    qqq_p, qqq_v = qqq_metrics.get("last_price"), qqq_metrics.get("vwap")

    bull = (1 if spy_p and spy_v and spy_p > spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p > qqq_v else 0)
    bear = (1 if spy_p and spy_v and spy_p < spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p < qqq_v else 0)
    system_state.macro_bias = "BULLISH" if bull == 2 else ("BEARISH" if bear == 2 else "NEUTRAL")

async def run_simulation(csv_path=CSV_PATH, tickers=DEFAULT_TICKERS):
    if not os.path.exists(csv_path):
        logger.error(f"Dataset {csv_path} not found. Run download_dataset.py first.")
        return

    logger.info(f"Loading local dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    
    unique_dates = sorted(list(df["timestamp"].dt.date.unique()))
    logger.info(f"Loaded {len(df)} bars across {len(unique_dates)} trading sessions.")
    
    init_db()
    system_state.is_active = True
    system_state.active_tickers = tickers

    total_realized_pnl = 0.0
    total_trades_count = 0
    daily_results = []

    for sim_date in unique_dates:
        day_df = df[df["timestamp"].dt.date == sim_date].copy()
        if len(day_df) < 40:
            continue

        # Daily state resets
        system_state.daily_realized_pnl = 0.0
        system_state.daily_unrealized_pnl = 0.0
        system_state.daily_trade_counts = {}
        system_state.open_positions = {}
        ticker_cooldowns.clear()
        
        consecutive_signals = {}
        day_trades = 0

        # Build clean ticker DataFrames
        ticker_dfs = {}
        for t in tickers:
            t_df = pd.DataFrame(index=day_df.index)
            for m in ["open", "high", "low", "close", "volume"]:
                t_df[m] = day_df[f"{t}_{m}"].astype(float)
            t_df.ffill(inplace=True)
            t_df.bfill(inplace=True)
            ticker_dfs[t] = t_df

        for idx in range(30, len(day_df)):
            ts = day_df["timestamp"].iloc[idx]
            et_time = ts.astimezone(ZoneInfo("America/New_York"))
            session_phase = get_mock_session_phase(ts)
            system_state.session_phase = session_phase

            start_idx = max(0, idx - 120)

            for t in tickers:
                sub_df = ticker_dfs[t].iloc[start_idx : idx + 1]
                system_state.latest_quotes[t] = {
                    "price": float(sub_df["close"].iloc[-1]),
                    "high": float(sub_df["high"].iloc[-1]),
                    "low": float(sub_df["low"].iloc[-1]),
                    "volume": float(sub_df["volume"].iloc[-1]),
                    "timestamp": ts.isoformat()
                }
                system_state.technical_analysis[t] = evaluate_intraday_technical_snapshot(sub_df)

            update_macro_regime_bias()
            await update_open_positions_and_risk(system_state.latest_quotes)

            if session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
                await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")
                continue

            for t in tickers:
                analysis = system_state.technical_analysis.get(t, {})
                sig = analysis.get("signal", "NEUTRAL")

                tracker = consecutive_signals.get(t, {"signal": "NEUTRAL", "bars": 0})
                if sig != "NEUTRAL" and sig == tracker["signal"]:
                    tracker["bars"] += 1
                else:
                    tracker = {"signal": sig, "bars": 1 if sig != "NEUTRAL" else 0}
                consecutive_signals[t] = tracker

                if sig in ("NEUTRAL", None) or session_phase not in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON"):
                    continue

                if tracker["bars"] < 2 or t in system_state.open_positions:
                    continue

                action = "BUY" if "BULLISH" in sig or "LONG" in sig else "SELL"
                if action == "BUY" and system_state.macro_bias == "BEARISH":
                    continue
                if action == "SELL" and system_state.macro_bias == "BULLISH":
                    continue

                real_p = system_state.latest_quotes[t]["price"]
                atr = float(analysis.get("metrics", {}).get("atr_14", 1.0) or 1.0)
                min_stop = max(atr * 2.2, real_p * 0.0055)

                if action == "BUY":
                    stop_loss = round(real_p - min_stop, 2)
                    take_profit = round(real_p + (min_stop * 2.0), 2)
                else:
                    stop_loss = round(real_p + min_stop, 2)
                    take_profit = round(real_p - (min_stop * 2.0), 2)

                opened = await open_position(
                    ticker=t,
                    action=action,
                    current_price=real_p,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=f"CSV Replay: {sig}"
                )
                if opened:
                    day_trades += 1

        total_realized_pnl += system_state.daily_realized_pnl
        total_trades_count += day_trades
        daily_results.append({
            "date": sim_date.strftime("%Y-%m-%d"),
            "trades": day_trades,
            "pnl": system_state.daily_realized_pnl
        })
        logger.info(f"[{sim_date}] Session P&L: ${system_state.daily_realized_pnl:+.2f} ({day_trades} trades)")

    logger.info("=" * 65)
    logger.info("OFFLINE MULTI-WEEK DATASET BACKTEST COMPLETED")
    logger.info("=" * 65)
    logger.info(f"Total Trading Sessions Replayed: {len(daily_results)}")
    logger.info(f"Total Executions: {total_trades_count}")
    logger.info(f"Net Cumulative Realized P&L: ${total_realized_pnl:+.2f}")
    logger.info("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_simulation())
