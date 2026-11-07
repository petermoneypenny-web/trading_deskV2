import os
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List
import yfinance as yf
import pandas as pd

# Disable yfinance internal SQLite disk-cache locking inside container
import yfinance.cache as yf_cache
try:
    os.makedirs("/tmp/py-yfinance", exist_ok=True)
    yf.set_tz_cache_location("/tmp/py-yfinance")
    if hasattr(yf_cache, "enable_cache"):
        yf_cache.enable_cache(False)
except Exception:
    pass

from app.core.state import system_state
from app.services.buffer import buffer_manager
from app.services.indicators import evaluate_intraday_technical_snapshot
from app.services.gemini_agent import evaluate_intraday_opportunity
from app.services.alerts import send_ntfy_alert
from app.services.risk_engine import (
    init_db,
    open_position,
    update_open_positions_and_risk,
    liquidate_all_positions
)
from app.config import settings

logger = logging.getLogger("trading_desk_v2.stream")

# In-memory tracking for consecutive bar signal persistence: {ticker: {"signal": str, "bars": int}}
consecutive_signals: Dict[str, Dict[str, Any]] = {}


def get_current_session_phase() -> str:
    """
    Calculates US market session phase based on Eastern Time.
    Gating:
    - 09:30 - 10:15: RTH_OPENING_FLUSH (Opening volatility noise - DO NOT ENTER)
    - 10:15 - 11:30: RTH_PRIME_MORNING (High conviction trend window)
    - 11:30 - 14:15: RTH_MIDDAY (Midday chop dead zone)
    - 14:15 - 15:30: RTH_PRIME_AFTERNOON (Afternoon trend continuation & squeeze)
    - 15:30 - 15:45: RTH_NORMAL (No new entries)
    - 15:45 - 16:00: RTH_EOD_CLOSING (Flatten all positions)
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "CLOSED_WEEKEND"

    time_mins = now_et.hour * 60 + now_et.minute

    if 4 * 60 <= time_mins < 9 * 60 + 30:
        return "PRE_MARKET"
    elif 9 * 60 + 30 <= time_mins < 10 * 60 + 15:
        return "RTH_OPENING_FLUSH"  # 09:30 - 10:15 ET: Let opening range wicks settle
    elif 10 * 60 + 15 <= time_mins < 11 * 60 + 30:
        return "RTH_PRIME_MORNING"   # 10:15 - 11:30 ET: Primary morning trend window
    elif 11 * 60 + 30 <= time_mins < 14 * 60 + 15:
        return "RTH_MIDDAY"          # 11:30 - 14:15 ET: Midday chop filter
    elif 14 * 60 + 15 <= time_mins < 15 * 60 + 30:
        return "RTH_PRIME_AFTERNOON" # 14:15 - 15:30 ET: Afternoon power hour
    elif 15 * 60 + 30 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"     # 15:45 - 16:00 ET: Auto-liquidate open trades
    elif 16 * 60 <= time_mins < 20 * 60:
        return "POST_MARKET"
    else:
        return "CLOSED"


def update_macro_regime_bias():
    """Determines broad index trend alignment using SPY & QQQ relative to VWAP."""
    spy_analysis = system_state.technical_analysis.get("SPY", {})
    qqq_analysis = system_state.technical_analysis.get("QQQ", {})

    spy_metrics = spy_analysis.get("metrics", {})
    qqq_metrics = qqq_analysis.get("metrics", {})

    spy_p = spy_metrics.get("last_price")
    spy_v = spy_metrics.get("vwap")
    qqq_p = qqq_metrics.get("last_price")
    qqq_v = qqq_metrics.get("vwap")

    bull_score = (1 if spy_p and spy_v and spy_p > spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p > qqq_v else 0)
    bear_score = (1 if spy_p and spy_v and spy_p < spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p < qqq_v else 0)

    if bull_score == 2:
        system_state.macro_bias = "BULLISH"
    elif bear_score == 2:
        system_state.macro_bias = "BEARISH"
    else:
        system_state.macro_bias = "NEUTRAL"


def get_portfolio_active_direction() -> str:
    """
    Checks open positions to enforce portfolio-wide directional consistency.
    Returns: 'LONG_ONLY', 'SHORT_ONLY', or 'ANY'
    """
    if not system_state.open_positions:
        return "ANY"

    actions = {pos["action"] for pos in system_state.open_positions.values()}
    if "BUY" in actions and "SELL" not in actions:
        return "LONG_ONLY"
    elif "SELL" in actions and "BUY" not in actions:
        return "SHORT_ONLY"
    return "ANY"


def _fetch_batched_intraday(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Downloads 1-minute bars for all watchlist tickers in a single HTTP request."""
    if not tickers:
        return {}

    ticker_str = " ".join(tickers)
    
    df = yf.download(
        tickers=ticker_str,
        period="1d",
        interval="1m",
        prepost=True,
        progress=False,
        threads=False,
        group_by="column"
    )

    if df is None or df.empty:
        df = yf.download(
            tickers=ticker_str,
            period="5d",
            interval="1m",
            prepost=True,
            progress=False,
            threads=False,
            group_by="column"
        )
        if df is not None and not df.empty:
            df = df.tail(120)

    if df is None or df.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}

    if isinstance(df.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                sub_df = pd.DataFrame(index=df.index)
                for metric in ["Open", "High", "Low", "Close", "Volume"]:
                    if (metric, ticker) in df.columns:
                        sub_df[metric] = df[(metric, ticker)]
                
                sub_df.dropna(subset=["Close"], inplace=True)
                if not sub_df.empty:
                    result[ticker] = sub_df
            except Exception as e:
                logger.warning(f"Error extracting sub-dataframe for {ticker}: {e}")
    else:
        ticker = tickers[0]
        sub_df = df.copy()
        sub_df.dropna(subset=["Close"], inplace=True)
        if not sub_df.empty:
            result[ticker] = sub_df

    return result


async def fetch_single_ticker_data(ticker: str):
    """Fetches and populates buffer/indicators for a single ticker on demand."""
    try:
        data_map = await asyncio.to_thread(_fetch_batched_intraday, [ticker])
        df = data_map.get(ticker)
        if df is not None and not df.empty:
            buffer_manager.load_initial_history(ticker, df, interval="1m")
            last_bar = buffer_manager.get_latest_bar(ticker, interval="1m")
            if last_bar:
                system_state.latest_quotes[ticker] = {
                    "price": float(last_bar["close"]),
                    "high": float(last_bar["high"]),
                    "low": float(last_bar["low"]),
                    "volume": float(last_bar["volume"]),
                    "timestamp": last_bar["timestamp"]
                }
            buffer_df = buffer_manager.get_dataframe(ticker, interval="1m")
            analysis = evaluate_intraday_technical_snapshot(buffer_df)
            system_state.technical_analysis[ticker] = analysis
            logger.info(f"Initialized buffer and indicators for {ticker}")
            return True
    except Exception as e:
        logger.error(f"Failed to fetch initial data for {ticker}: {e}")
    return False


async def sync_market_data():
    """Batched data fetch, macro regime update, risk monitoring, and gated execution."""
    active_tickers = list(system_state.active_tickers)
    if not active_tickers:
        return

    try:
        data_map = await asyncio.to_thread(_fetch_batched_intraday, active_tickers)

        for ticker in active_tickers:
            df = data_map.get(ticker)
            if df is not None and not df.empty:
                buffer_manager.load_initial_history(ticker, df, interval="1m")
                last_bar = buffer_manager.get_latest_bar(ticker, interval="1m")
                if last_bar:
                    real_price = float(last_bar["close"])
                    system_state.latest_quotes[ticker] = {
                        "price": real_price,
                        "high": float(last_bar["high"]),
                        "low": float(last_bar["low"]),
                        "volume": float(last_bar["volume"]),
                        "timestamp": last_bar["timestamp"]
                    }

                    buffer_df = buffer_manager.get_dataframe(ticker, interval="1m")
                    analysis = evaluate_intraday_technical_snapshot(buffer_df)
                    system_state.technical_analysis[ticker] = analysis

        # Update macro index bias across SPY & QQQ
        update_macro_regime_bias()

        # Trade Evaluation: Strictly gated to prime trading windows outside opening flush and midday chop
        if system_state.is_active and system_state.session_phase in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON"):
            portfolio_direction = get_portfolio_active_direction()

            for ticker in active_tickers:
                analysis = system_state.technical_analysis.get(ticker, {})
                signal = analysis.get("signal", "NEUTRAL")

                # Track consecutive bar signal persistence (require 2 consecutive bars)
                tracker = consecutive_signals.get(ticker, {"signal": "NEUTRAL", "bars": 0})
                if signal != "NEUTRAL" and signal == tracker["signal"]:
                    tracker["bars"] += 1
                else:
                    tracker = {"signal": signal, "bars": 1 if signal != "NEUTRAL" else 0}
                consecutive_signals[ticker] = tracker

                if signal in ("NEUTRAL", None) or ticker in system_state.open_positions:
                    continue

                # Enforce 2 consecutive bar confirmation
                if tracker["bars"] < 2:
                    logger.info(f"Awaiting 2-bar confirmation for {ticker} ({signal} bar {tracker['bars']}/2)")
                    continue

                decision = await evaluate_intraday_opportunity(
                    ticker=ticker,
                    session_phase=system_state.session_phase,
                    technical_analysis=analysis
                )
                system_state.agent_evaluations[ticker] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "decision": decision
                }

                action = decision.get("action")
                confidence = decision.get("confidence", 0)

                # 1. Macro Trend Gating
                if action == "BUY" and system_state.macro_bias == "BEARISH":
                    logger.info(f"Skipping BUY for {ticker}: Blocked by BEARISH macro regime.")
                    continue
                if action == "SELL" and system_state.macro_bias == "BULLISH":
                    logger.info(f"Skipping SELL for {ticker}: Blocked by BULLISH macro regime.")
                    continue

                # 2. Portfolio Directional Lock
                if action == "BUY" and portfolio_direction == "SHORT_ONLY":
                    logger.info(f"Skipping BUY for {ticker}: Portfolio locked to SHORT_ONLY while short positions are open.")
                    continue
                if action == "SELL" and portfolio_direction == "LONG_ONLY":
                    logger.info(f"Skipping SELL for {ticker}: Portfolio locked to LONG_ONLY while long positions are open.")
                    continue

                if action in ("BUY", "SELL") and confidence >= 78 and ticker not in system_state.open_positions:
                    real_price = system_state.latest_quotes[ticker]["price"]
                    metrics = analysis.get("metrics", {})
                    raw_atr = metrics.get("atr_14")
                    atr = float(raw_atr) if raw_atr and raw_atr > 0 else max(0.5, real_price * 0.003)

                    # Dynamic volatility stops: >= 2.2x ATR or >= 0.55% of stock price
                    min_stop_distance = max(atr * 2.2, real_price * 0.0055)
                    raw_sl = decision.get("stop_loss")
                    raw_tp = decision.get("take_profit")

                    if action == "BUY":
                        if not raw_sl or raw_sl >= real_price or (real_price - raw_sl) < min_stop_distance:
                            stop_loss = round(real_price - min_stop_distance, 2)
                        else:
                            stop_loss = round(float(raw_sl), 2)

                        risk_dist = real_price - stop_loss
                        if not raw_tp or raw_tp <= real_price:
                            take_profit = round(real_price + (risk_dist * 2.0), 2)
                        else:
                            take_profit = round(float(raw_tp), 2)
                    else:  # SELL / SHORT
                        if not raw_sl or raw_sl <= real_price or (raw_sl - real_price) < min_stop_distance:
                            stop_loss = round(real_price + min_stop_distance, 2)
                        else:
                            stop_loss = round(float(raw_sl), 2)

                        risk_dist = stop_loss - real_price
                        if not raw_tp or raw_tp >= real_price:
                            take_profit = round(real_price - (risk_dist * 2.0), 2)
                        else:
                            take_profit = round(float(raw_tp), 2)

                    await open_position(
                        ticker=ticker,
                        action=action,
                        current_price=real_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        reasoning=decision.get("reasoning", "Confirmed 2-bar momentum setup")
                    )

        system_state.last_fetch_error = None

    except Exception as e:
        error_msg = f"Failed in batched market data sync: {e}"
        logger.warning(error_msg)
        system_state.last_fetch_error = error_msg

    await update_open_positions_and_risk(system_state.latest_quotes)

    if system_state.session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
        logger.info("RTH_EOD_CLOSING reached. Flattening all intraday positions.")
        await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")


async def run_intraday_tick_loop():
    """Background asynchronous loop for market monitoring and buffer updates."""
    logger.info("Day-Trading Intraday Loop initialized.")
    init_db()
    
    await sync_market_data()

    while True:
        try:
            system_state.session_phase = get_current_session_phase()
            system_state.last_loop_timestamp = datetime.utcnow()

            await sync_market_data()

            # Tick loop runs every 10s during prime trading, 30s during opening flush, midday chop, or paused
            is_prime = system_state.session_phase in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON")
            sleep_duration = 10 if (system_state.is_active and is_prime) else 30
            await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop cancellation received.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(10)