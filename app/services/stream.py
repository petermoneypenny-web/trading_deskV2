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
    # Check weekday (Monday=0 ... Sunday=6)
    if now_et.weekday() >= 5:
        return "CLOSED_WEEKEND"

    time_mins = now_et.hour * 60 + now_et.minute

    # Pre-market: 04:00 - 09:30 ET
    if 4 * 60 <= time_mins < 9 * 60 + 30:
        return "PRE_MARKET"
    # RTH Open surge: 09:30 - 10:00 ET (high volatility)
    elif 9 * 60 + 30 <= time_mins < 10 * 60:
        return "RTH_OPEN"
    # Regular trading hours: 10:00 - 15:45 ET
    elif 10 * 60 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    # End-of-Day liquidation window: 15:45 - 16:00 ET
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"
    # After hours: 16:00 - 20:00 ET
    elif 16 * 60 <= time_mins < 20 * 60:
        return "POST_MARKET"
    else:
        return "CLOSED"


def _fetch_ticker_intraday(ticker: str) -> pd.DataFrame:
    """
    Synchronous yfinance fetcher designed to be executed inside asyncio.to_thread.
    Pulls 1-day 1-minute historical bars.
    """
    t = yf.Ticker(ticker)
    df = t.history(period="1d", interval="1m")
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
    """Fetches intraday bars for all active tickers without blocking the main event loop."""
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
    while True:
        try:
            system_state.session_phase = get_current_session_phase()
            system_state.last_loop_timestamp = datetime.utcnow()

            if system_state.is_active:
                await sync_market_data()
                logger.debug(f"Market tick synced at {system_state.last_loop_timestamp.isoformat()}")

            # Sleep interval: 10s during active trading, 60s when market closed or paused
            sleep_duration = 10 if system_state.is_active else 60
            await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop cancellation received.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(10)