import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd

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


def _fetch_ticker_intraday(ticker: str) -> pd.DataFrame:
    """Pulls intraday bars with 5d fallback for off-hours testing."""
    t = yf.Ticker(ticker)
    df = t.history(period="1d", interval="1m", prepost=True)
    if df is None or df.empty:
        df = t.history(period="5d", interval="1m", prepost=True)
        if df is not None and not df.empty:
            df = df.tail(120)
    return df


async def sync_market_data():
    """Fetches intraday bars, updates buffers, evaluates setups, and manages execution."""
    for ticker in system_state.active_tickers:
        try:
            df = await asyncio.to_thread(_fetch_ticker_intraday, ticker)
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
                            await open_position(
                                ticker=ticker,
                                action=action,
                                entry_price=decision["entry_price"],
                                stop_loss=decision["stop_loss"],
                                take_profit=decision["take_profit"],
                                reasoning=decision["reasoning"]
                            )

            system_state.last_fetch_error = None
        except Exception as e:
            error_msg = f"Failed to ingest/evaluate {ticker}: {e}"
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

            if system_state.is_active:
                await sync_market_data()
                logger.debug(f"Tick sync completed at {system_state.last_loop_timestamp.isoformat()}")

            sleep_duration = 10 if system_state.is_active else 60
            await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop cancellation received.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(10)