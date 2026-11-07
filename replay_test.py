import os
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replay_test")

from app.core.state import system_state
from app.services.buffer import buffer_manager
from app.services.indicators import evaluate_intraday_technical_snapshot
from app.services.risk_engine import (
    init_db,
    open_position,
    update_open_positions_and_risk,
    liquidate_all_positions,
    ticker_cooldowns
)

TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "INTC", "TSM", "MU"]

def get_mock_session_phase(bar_time: datetime) -> str:
    et_time = bar_time.astimezone(ZoneInfo("America/New_York"))
    time_mins = et_time.hour * 60 + et_time.minute
    if 9 * 60 + 30 <= time_mins < 10 * 60:
        return "RTH_OPEN"
    elif 10 * 60 <= time_mins < 11 * 60 + 45:
        return "RTH_NORMAL"
    elif 11 * 60 + 45 <= time_mins < 14 * 60 + 15:
        return "RTH_MIDDAY"
    elif 14 * 60 + 15 <= time_mins < 15 * 60 + 45:
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

async def run_replay():
    logger.info("Initializing yesterday's market data replay test...")
    init_db()

    system_state.is_active = True
    system_state.active_tickers = TICKERS
    system_state.daily_realized_pnl = 0.0
    system_state.daily_unrealized_pnl = 0.0
    system_state.daily_trade_counts = {}
    system_state.open_positions = {}
    ticker_cooldowns.clear()

    ticker_str = " ".join(TICKERS)
    logger.info(f"Downloading historical 1m data for: {ticker_str}")
    raw_df = yf.download(
        tickers=ticker_str,
        period="5d",
        interval="1m",
        prepost=False,
        progress=False,
        threads=False,
        group_by="column"
    )

    if raw_df is None or raw_df.empty:
        logger.error("Failed to download historical bars.")
        return

    all_timestamps = sorted(list(raw_df.index))
    last_trading_day = all_timestamps[-1].date()
    session_timestamps = [ts for ts in all_timestamps if ts.date() == last_trading_day]
    logger.info(f"Loaded {len(session_timestamps)} 1-minute bars for trading date: {last_trading_day}")

    warmup_bars = session_timestamps[:30]
    replay_bars = session_timestamps[30:]

    for ts in warmup_bars:
        for ticker in TICKERS:
            try:
                row = pd.DataFrame([{
                    "open": float(raw_df[("Open", ticker)].loc[ts]),
                    "high": float(raw_df[("High", ticker)].loc[ts]),
                    "low": float(raw_df[("Low", ticker)].loc[ts]),
                    "close": float(raw_df[("Close", ticker)].loc[ts]),
                    "volume": float(raw_df[("Volume", ticker)].loc[ts]),
                    "timestamp": ts.isoformat()
                }], index=[ts])
                buffer_manager.append_bar(ticker, row.iloc[0].to_dict(), interval="1m")
            except Exception:
                pass

    logger.info("Buffer warmup complete. Commencing chronologic tick simulation...")

    trades_opened = 0
    blocked_by_macro = 0
    blocked_by_midday = 0

    for step_idx, ts in enumerate(replay_bars):
        current_time_et = ts.astimezone(ZoneInfo("America/New_York"))
        session_phase = get_mock_session_phase(ts)
        system_state.session_phase = session_phase

        for ticker in TICKERS:
            try:
                close_p = float(raw_df[("Close", ticker)].loc[ts])
                high_p = float(raw_df[("High", ticker)].loc[ts])
                low_p = float(raw_df[("Low", ticker)].loc[ts])
                vol_p = float(raw_df[("Volume", ticker)].loc[ts])
                if pd.isna(close_p):
                    continue

                bar_dict = {
                    "open": float(raw_df[("Open", ticker)].loc[ts]),
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": vol_p,
                    "timestamp": ts.isoformat()
                }
                buffer_manager.append_bar(ticker, bar_dict, interval="1m")

                system_state.latest_quotes[ticker] = {
                    "price": close_p,
                    "high": high_p,
                    "low": low_p,
                    "volume": vol_p,
                    "timestamp": ts.isoformat()
                }

                buf_df = buffer_manager.get_dataframe(ticker, interval="1m")
                analysis = evaluate_intraday_technical_snapshot(buf_df)
                system_state.technical_analysis[ticker] = analysis
            except Exception:
                continue

        update_macro_regime_bias()
        await update_open_positions_and_risk(system_state.latest_quotes)

        if session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
            logger.info(f"[{current_time_et.strftime('%H:%M')}] RTH_EOD_CLOSING reached. Liquidating open positions.")
            await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")
            continue

        if session_phase == "RTH_MIDDAY":
            for ticker in TICKERS:
                analysis = system_state.technical_analysis.get(ticker, {})
                if analysis.get("signal") not in ("NEUTRAL", None) and ticker not in system_state.open_positions:
                    blocked_by_midday += 1
            continue

        if session_phase in ("RTH_OPEN", "RTH_NORMAL"):
            for ticker in TICKERS:
                analysis = system_state.technical_analysis.get(ticker, {})
                signal = analysis.get("signal", "NEUTRAL")
                if signal in ("NEUTRAL", None) or ticker in system_state.open_positions:
                    continue

                real_price = system_state.latest_quotes[ticker]["price"]
                metrics = analysis.get("metrics", {})
                atr = float(metrics.get("atr_14", 1.0) or 1.0)
                action = "BUY" if "BULLISH" in signal or "LONG" in signal else "SELL"

                if action == "BUY" and system_state.macro_bias == "BEARISH":
                    blocked_by_macro += 1
                    continue
                if action == "SELL" and system_state.macro_bias == "BULLISH":
                    blocked_by_macro += 1
                    continue

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
                    reasoning=f"Replay simulated {signal} (Macro: {system_state.macro_bias})"
                )
                if opened:
                    trades_opened += 1
                    logger.info(f"[{current_time_et.strftime('%H:%M')} ET] OPENED {action} {ticker} @ ${real_price:.2f} | SL: ${stop_loss} | TP: ${take_profit}")

    logger.info("=" * 60)
    logger.info("REPLAY BACKTEST COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total Trades Opened: {trades_opened}")
    logger.info(f"Entries Blocked by Midday Chop Window: {blocked_by_midday}")
    logger.info(f"Entries Blocked by Macro Bias Filter: {blocked_by_macro}")
    logger.info(f"Final Realized P&L: ${system_state.daily_realized_pnl:+.2f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_replay())
