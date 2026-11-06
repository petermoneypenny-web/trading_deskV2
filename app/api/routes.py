import asyncio
import logging
import httpx
from datetime import datetime
from app.core.state import system_state
from app.config import settings

logger = logging.getLogger("trading_desk_v2.stream")


async def send_ntfy_alert(title: str, message: str, priority: int = 3):
    """
    Sends an alert notification to ntfy.sh under the configured topic.
    """
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
            response = await client.post(url, content=message.encode("utf-8"), headers=headers)
            if response.status_code != 200:
                logger.warning(f"NTFY returned status {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to dispatch NTFY alert: {e}")


async def run_intraday_tick_loop():
    """
    Background asynchronous loop for intraday candle monitoring and trade execution.
    """
    logger.info("Starting Day-Trading Intraday Loop service...")
    while True:
        try:
            if system_state.is_active:
                system_state.last_loop_timestamp = datetime.utcnow()
                # Placeholder: Ingestion, VWAP/ORB calculations, and Gemini decisions
                logger.debug(f"Tick evaluated at {system_state.last_loop_timestamp.isoformat()}")

            # 5-second polling interval
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop received cancellation signal.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(5)