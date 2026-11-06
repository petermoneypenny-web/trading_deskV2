import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx
import yfinance as yf
import pandas as pd

from app.core.state import system_state
from app.services.buffer import buffer_manager
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
    """
    Pulls intraday bars. Falls back to period='5d' if the market is closed
    so buffers are never empty during testing or weekend preparation.
    """
    t = yf.Ticker(ticker)
    
    # Try fetching current day including extended hours
    df = t.history(period="1d", interval="1m", prepost=True)
    
    # If market is closed or no bars today, pull recent days and slice the latest session
    if df is None or df.empty:
        logger.info(f"No 1d intraday bars for {ticker}. Fetching recent multi-day history.")
        df = t.history(period="5d", interval="1m", prepost=True)
        if df is not None and not df.empty:
            # Retain the most recent session's bars up to the buffer limit
            df = df.tail(120)

    return df


async def send_ntfy_alert(title: str, message: str, priority: int = 3):
    """Sends an alert notification to ntfy.sh under the configured topic."""
    if not settings.NTFY_TOPIC:
        return
    url = f"{settings.NTFY_SERVER_URL.rstrip('/')}/{settings.NTFY_TOPIC}"
    headers = {
        "Title": title,
        "Priority": str(priority),
        "Tags": "chart_with_upwards_trend,rocket"
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, content=message.encode("utf-8"), headers=headers)
    except Exception as e:
        logger.error(f"Failed to dispatch NTFY alert: {e}")


async def sync_market_data():
    """Fetches intraday bars for all active tickers without blocking the event loop."""
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
            system_state.last_fetch_error = None
        except Exception as e:
            error_msg = f"Failed to ingest data for {ticker}: {e}"
            logger.warning(error_msg)
            system_state.last_fetch_error = error_msg


async def run_intraday_tick_loop():
    """Background asynchronous loop for market monitoring and buffer updates."""
    logger.info("Day-Trading Intraday Loop initialized.")
    
    # Warm up buffer once on startup regardless of session status
    await sync_market_data()

    while True:
        try:
            system_state.session_phase = get_current_session_phase()
            system_state.last_loop_timestamp = datetime.utcnow()

            if system_state.is_active:
                await sync_market_data()
                logger.debug(f"Market tick synced at {system_state.last_loop_timestamp.isoformat()}")

            sleep_duration = 10 if system_state.is_active else 60
            await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop cancellation received.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(10)