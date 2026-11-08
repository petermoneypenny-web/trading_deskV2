import logging
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.state import system_state
from app.services.alerts import send_ntfy_alert
from app.services.risk_engine import liquidate_all_positions, ticker_cooldowns
from app.services.stream import reset_daily_state, get_current_session_phase
from app.services.risk_config import get_risk_config

logger = logging.getLogger("trading_desk_v2.scheduler")

scheduler = AsyncIOScheduler(timezone=ZoneInfo("America/New_York"))


async def scheduled_market_open():
    """Fires at 09:25 ET Monday through Friday."""
    logger.info("Scheduler Trigger: Starting Day Trading Engine for RTH Open...")
    system_state.is_active = True
    system_state.daily_realized_pnl = 0.0
    system_state.daily_unrealized_pnl = 0.0

    # Reset every piece of per-session state: consecutive-signal trackers,
    # cooldowns, per-ticker trade counts, blocked-entry counters and the candle
    # buffers (so VWAP/ORB anchor to today's 09:30 open, not yesterday's).
    reset_daily_state()

    cfg = get_risk_config()
    await send_ntfy_alert(
        title="🔔 Market Open: Trading Desk V2 Active",
        message=(
            "Engine activated at 09:25 ET. Trade counters, P&L, cooldowns and 1m "
            f"buffers reset for the session.\n"
            f"Risk: ${cfg.max_risk_per_trade:.0f}/trade, "
            f"{cfg.max_trades_per_day} trades/day, "
            f"daily stop ${cfg.max_daily_loss:.0f}, "
            f"SL {cfg.atr_stop_multiplier:g}x ATR (min {cfg.min_stop_pct * 100:.2f}%), "
            f"target {cfg.target_rr:g}R, "
            f"time stop {cfg.max_hold_minutes}m."
        ),
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
        message=(
            f"Engine paused at 16:05 ET.\n"
            f"Final Realized P&L: ${system_state.daily_realized_pnl:+.2f}\n"
            f"Net P&L: ${net_pnl:+.2f}\n"
            f"Trades: {system_state.daily_trade_count}"
        ),
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
