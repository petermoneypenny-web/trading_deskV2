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


def get_current_session_phase() -> str:
    """Calculates US market session phase based on Eastern Time."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "CLOSED_WEEKEND"

    time_mins = now_et.hour * 60 + now_et.minute

    if 4 * 60 <= time_mins < 9 * 60 + 30:
        return "PRE_MARKET"
    elif 9 * 60 + 30 <= time_mins < 10 * 60:
        return "RTH_OPEN"
    elif 10 * 60 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"
    elif 16 * 60 <= time_mins < 20 * 60:
        return "POST_MARKET"
    else:
        return "CLOSED"


def _fetch_batched_intraday(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Downloads 1-minute bars for all watchlist tickers in a single HTTP request.
    """
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
        logger.info(f"Batched 1d download empty for [{ticker_str}]. Falling back to 5d history.")
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
                    "price": last_bar["close"],
                    "high": last_bar["high"],
                    "low": last_bar["low"],
                    "volume": last_bar["volume"],
                    "timestamp": last_bar["timestamp"]
                }
            buffer_df = buffer_manager.get_dataframe(ticker, interval="1m")
            analysis = evaluate_intraday_technical_snapshot(buffer_df)
            system_state.technical_analysis[ticker] = analysis
            logger.info(f"Successfully initialized buffer and indicators for new ticker {ticker}")
            return True
    except Exception as e:
        logger.error(f"Failed to fetch initial data for {ticker}: {e}")
    return False


async def sync_market_data():
    """Fetches intraday bars in a single batched call, updates buffers, evaluates setups, and manages execution."""
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
                        "high": last_bar["high"],
                        "low": last_bar["low"],
                        "volume": last_bar["volume"],
                        "timestamp": last_bar["timestamp"]
                    }

                    buffer_df = buffer_manager.get_dataframe(ticker, interval="1m")
                    analysis = evaluate_intraday_technical_snapshot(buffer_df)
                    system_state.technical_analysis[ticker] = analysis

                    # Evaluate Gemini and execute during active trading phases
                    if system_state.is_active and system_state.session_phase in ("RTH_OPEN", "RTH_NORMAL"):
                        if analysis.get("signal") not in ("NEUTRAL", None):
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
                            
                            if action in ("BUY", "SELL") and confidence >= 75 and ticker not in system_state.open_positions:
                                metrics = analysis.get("metrics", {})
                                raw_atr = metrics.get("atr_14")
                                atr = float(raw_atr) if raw_atr and raw_atr > 0 else max(0.5, real_price * 0.003)

                                raw_sl = decision.get("stop_loss")
                                raw_tp = decision.get("take_profit")

                                # Sanitize and enforce realistic volatility stops anchored to actual market price
                                if action == "BUY":
                                    if not raw_sl or raw_sl >= real_price or (real_price - raw_sl) < (0.5 * atr):
                                        stop_loss = round(real_price - (1.5 * atr), 2)
                                    else:
                                        stop_loss = round(float(raw_sl), 2)

                                    if not raw_tp or raw_tp <= real_price:
                                        take_profit = round(real_price + (2.5 * atr), 2)
                                    else:
                                        take_profit = round(float(raw_tp), 2)
                                else:  # SELL / SHORT
                                    if not raw_sl or raw_sl <= real_price or (raw_sl - real_price) < (0.5 * atr):
                                        stop_loss = round(real_price + (1.5 * atr), 2)
                                    else:
                                        stop_loss = round(float(raw_sl), 2)

                                    if not raw_tp or raw_tp >= real_price:
                                        take_profit = round(real_price - (2.5 * atr), 2)
                                    else:
                                        take_profit = round(float(raw_tp), 2)

                                await open_position(
                                    ticker=ticker,
                                    action=action,
                                    current_price=real_price,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    reasoning=decision.get("reasoning", "Algorithmic momentum entry")
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
    
    # Initial batch fetch on startup
    await sync_market_data()

    while True:
        try:
            system_state.session_phase = get_current_session_phase()
            system_state.last_loop_timestamp = datetime.utcnow()

            # Always sync market data during loop; frequency adjusts if active vs paused
            await sync_market_data()

            sleep_duration = 10 if system_state.is_active else 30
            await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop cancellation received.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(10)