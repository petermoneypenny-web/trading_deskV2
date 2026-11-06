import logging
import httpx
from app.config import settings

logger = logging.getLogger("trading_desk_v2.alerts")


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