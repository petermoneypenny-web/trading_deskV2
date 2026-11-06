import logging
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.state import system_state
from app.services.alerts import send_ntfy_alert
from app.services.risk_engine import liquidate_all_positions

logger = logging.getLogger("trading_desk_v2.scheduler")

scheduler = AsyncIOScheduler(timezone=ZoneInfo("America/New_York"))


async def scheduled_market_open():
    """Fires at 09:25 ET Monday through Friday."""
    logger.info("Scheduler Trigger: Starting Day Trading Engine for RTH Open...")
    system_state.is_active = True
    system_state.daily_realized_pnl = 0.0
    system_state.daily_unrealized_pnl = 0.0
    
    await send_ntfy_alert(
        title="🔔 Market Open: Trading Desk V2 Active",
        message="Engine activated at 09:25 ET. Monitoring 1m candle buffers for opening setups.",
        priority=4
    )


async def scheduled_market_close():
    """Fires at 16:05 ET Monday through Friday."""
    logger.info("Scheduler Trigger: Pausing Day Trading Engine for Post-Market...")
    
    # Ensure no positions remained unclosed
    if system_state.open_positions:
        await liquidate_all_positions(reason="POST_MARKET_FLATTEN")

    system_state.is_active = False
    net_pnl = system_state.daily_realized_pnl + system_state.daily_unrealized_pnl
    
    await send_ntfy_alert(
        title="🌙 Market Closed: Trading Desk V2 Paused",
        message=f"Engine paused at 16:05 ET.\nFinal Realized P&L: ${system_state.daily_realized_pnl:+.2f}\nNet P&L: ${net_pnl:+.2f}",
        priority=3
    )


def start_market_scheduler():
    """Initializes and registers Cron jobs for US market trading days."""
    # 09:25 AM ET Monday-Friday (5 mins ahead of 09:30 open)
    scheduler.add_job(
        scheduled_market_open,
        trigger=CronTrigger(hour=9, minute=25, day_of_week="mon-fri", timezone=ZoneInfo("America/New_York")),
        id="market_open_job",
        replace_existing=True
    )

    # 04:05 PM ET Monday-Friday (5 mins after 16:00 close)
    scheduler.add_job(
        scheduled_market_close,
        trigger=CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone=ZoneInfo("America/New_York")),
        id="market_close_job",
        replace_existing=True
    )

    scheduler.start()
    logger.info("APScheduler started with US/Eastern market hour triggers (09:25 open, 16:05 close).")


def stop_market_scheduler():
    """Stops the scheduler on container shutdown."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")