import asyncio
import sqlite3
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.state import system_state
from app.config import settings
from app.services.buffer import buffer_manager
from app.services.alerts import send_ntfy_alert
from app.services.stream import fetch_single_ticker_data, reset_daily_state
from app.services.gemini_agent import evaluate_intraday_opportunity
from app.services.risk_engine import (
    open_position,
    close_position,
    liquidate_all_positions,
    portfolio_direction,
    cluster_for,
    cluster_positions,
    total_open_risk,
    session_stats,
    reset_session_stats,
    compute_risk_score,
)
from app.services.risk_config import get_risk_config, TICKER_CLUSTERS
from app.services.scheduler import scheduler

router = APIRouter()


class OrderRequest(BaseModel):
    ticker: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: Optional[str] = "Manual order entry via API"


class TickerRequest(BaseModel):
    ticker: str


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "trading-desk-v2",
        "port": 5095,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/status")
async def get_system_status():
    return {
        "is_active": system_state.is_active,
        "session_phase": system_state.session_phase,
        "active_tickers": system_state.active_tickers,
        "latest_quotes": system_state.latest_quotes,
        "open_positions": system_state.open_positions,
        "open_positions_count": len(system_state.open_positions),
        "daily_realized_pnl": round(system_state.daily_realized_pnl, 2),
        "daily_unrealized_pnl": round(system_state.daily_unrealized_pnl, 2),
        "net_daily_pnl": round(system_state.daily_realized_pnl + system_state.daily_unrealized_pnl, 2),
        "last_loop_timestamp": system_state.last_loop_timestamp.isoformat() if system_state.last_loop_timestamp else None,
        "last_fetch_error": system_state.last_fetch_error
    }


@router.get("/scheduler/jobs")
async def get_scheduled_jobs():
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger)
        })
    return {"scheduler_running": scheduler.running, "jobs": jobs}


@router.get("/risk/config")
async def get_risk_configuration():
    """The live risk parameters actually in force (sourced from rulebook.json)."""
    return {
        "source": "rulebook.json",
        "config": get_risk_config().to_dict(),
        "clusters": TICKER_CLUSTERS,
    }


@router.get("/risk/exposure")
async def get_risk_exposure():
    """Portfolio-level risk: direction lock, correlation clusters, open risk."""
    cfg = get_risk_config()
    clusters = {}
    for cluster in {cluster_for(t) for t in system_state.open_positions}:
        clusters[cluster] = cluster_positions(cluster)

    positions = []
    for ticker, pos in system_state.open_positions.items():
        positions.append({
            "ticker": ticker,
            "cluster": cluster_for(ticker),
            "action": pos["action"],
            "shares": pos["shares"],
            "entry_price": pos["entry_price"],
            "stop_loss": pos["stop_loss"],
            "take_profit": pos["take_profit"],
            "risk_usd": round(abs(pos["entry_price"] - pos["stop_loss"]) * pos["shares"], 2),
            "signal": pos.get("signal"),
            "breakeven_armed": pos.get("breakeven_armed", False),
            "max_favourable": round(pos.get("max_favourable", 0.0), 4),
        })

    return {
        "direction": portfolio_direction(),
        "open_positions": len(system_state.open_positions),
        "max_open_positions": cfg.max_total_open_positions,
        "max_positions_per_cluster": cfg.max_positions_per_cluster,
        "total_open_risk_usd": round(total_open_risk(), 2),
        "max_risk_per_trade": cfg.max_risk_per_trade,
        "max_cluster_exposure_usd": round(cfg.max_cluster_exposure_usd(), 2),
        "daily_trade_count": system_state.daily_trade_count,
        "max_trades_per_day": cfg.max_trades_per_day,
        "daily_realized_pnl": round(system_state.daily_realized_pnl, 2),
        "daily_loss_limit": cfg.max_daily_loss,
        "circuit_breaker_armed": (
            system_state.daily_realized_pnl + system_state.daily_unrealized_pnl
        ) <= -cfg.max_daily_loss,
        "clusters": clusters,
        "positions": positions,
        "entries_blocked": system_state.entries_blocked,
    }


@router.get("/stats/session")
async def get_session_stats():
    """Exit breakdown and per-signal conditional statistics for this session."""
    stats = dict(session_stats)
    by_signal = stats.pop("by_signal", {})
    trades = stats.get("trades", 0)
    resolved = stats.get("stop_loss", 0) + stats.get("take_profit", 0)

    return {
        "totals": stats,
        "by_signal": by_signal,
        "exit_rates": {
            "stop_loss_pct": round(stats.get("stop_loss", 0) / trades * 100, 1) if trades else 0.0,
            "take_profit_pct": round(stats.get("take_profit", 0) / trades * 100, 1) if trades else 0.0,
            "time_stop_pct": round(
                (stats.get("time_stop", 0) + stats.get("breakeven_stop", 0)) / trades * 100, 1
            ) if trades else 0.0,
        },
        "win_rate_resolved_pct": (
            round(stats.get("take_profit", 0) / resolved * 100, 1) if resolved else None
        ),
        "breakeven_win_rate_pct": round(100 / (1 + get_risk_config().min_risk_reward_ratio), 1),
        "expectancy_per_trade": (
            round(stats.get("net_pnl", 0.0) / trades, 2) if trades else None
        ),
    }


@router.post("/session/reset-daily")
async def reset_daily_state_endpoint():
    """Manually clears per-session state (counters, cooldowns, buffers)."""
    reset_daily_state()
    reset_session_stats()
    return {"status": "reset", "message": "Daily counters, cooldowns and candle buffers cleared."}


@router.get("/watchlist")
async def get_watchlist():
    return {
        "count": len(system_state.active_tickers),
        "tickers": system_state.active_tickers
    }


@router.post("/watchlist")
async def add_watchlist_ticker(payload: TickerRequest):
    ticker = payload.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    
    if ticker not in system_state.active_tickers:
        system_state.active_tickers.append(ticker)

    # Immediately fetch initial candle buffer asynchronously
    asyncio.create_task(fetch_single_ticker_data(ticker))

    return {"status": "added", "ticker": ticker, "tickers": system_state.active_tickers}


@router.delete("/watchlist/{ticker}")
async def remove_watchlist_ticker(ticker: str):
    ticker_upper = ticker.strip().upper()
    if ticker_upper not in system_state.active_tickers:
        raise HTTPException(status_code=404, detail=f"{ticker_upper} not found in active watchlist.")
    
    system_state.active_tickers.remove(ticker_upper)
    system_state.latest_quotes.pop(ticker_upper, None)
    system_state.technical_analysis.pop(ticker_upper, None)
    system_state.agent_evaluations.pop(ticker_upper, None)
    
    return {"status": "removed", "ticker": ticker_upper, "tickers": system_state.active_tickers}


@router.get("/positions")
async def get_positions():
    return {
        "count": len(system_state.open_positions),
        "positions": system_state.open_positions,
        "daily_realized_pnl": round(system_state.daily_realized_pnl, 2),
        "daily_unrealized_pnl": round(system_state.daily_unrealized_pnl, 2)
    }


@router.post("/orders/manual")
async def submit_manual_order(order: OrderRequest):
    ticker_upper = order.ticker.upper()
    success = await open_position(
        ticker=ticker_upper,
        action=order.action.upper(),
        entry_price=order.entry_price,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        reasoning=order.reasoning
    )
    if not success:
        raise HTTPException(status_code=400, detail="Order rejected by risk engine (circuit breaker or active position).")
    return {"status": "Order executed", "position": system_state.open_positions.get(ticker_upper)}


@router.post("/positions/{ticker}/close")
async def close_single_position(ticker: str):
    ticker_upper = ticker.upper()
    if ticker_upper not in system_state.open_positions:
        raise HTTPException(status_code=404, detail=f"No active position found for {ticker_upper}")

    quote = system_state.latest_quotes.get(ticker_upper)
    exit_price = quote["price"] if quote else system_state.open_positions[ticker_upper]["current_price"]
    await close_position(ticker_upper, exit_price, "MANUAL_CLOSE_API")
    return {"status": f"Position {ticker_upper} closed at {exit_price}"}


@router.post("/positions/flatten")
async def flatten_all():
    count = len(system_state.open_positions)
    await liquidate_all_positions(reason="MANUAL_PANIC_FLATTEN")
    return {"status": f"Flattened {count} open positions."}


@router.get("/trades/history")
async def get_trade_history(limit: int = 50):
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
    return {"count": len(rows), "trades": rows}


@router.delete("/trades/history")
async def clear_trade_history():
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        with conn:
            conn.execute("DELETE FROM trades")
        
        # VACUUM requires autocommit mode (no active transaction)
        conn.isolation_level = None
        conn.execute("VACUUM")
        conn.close()

        # Reset session P&L counters in memory
        system_state.daily_realized_pnl = 0.0
        system_state.daily_unrealized_pnl = 0.0

        return {"status": "cleared", "message": "Closed trade history wiped and daily PnL reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear trade history: {str(e)}")


@router.get("/buffer/{ticker}")
async def get_ticker_buffer(ticker: str, interval: str = Query("1m", pattern="^(1m|5m)$")):
    ticker_upper = ticker.upper()
    df = buffer_manager.get_dataframe(ticker_upper, interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No buffer data found for {ticker_upper}")

    records = df.tail(30).reset_index().to_dict(orient="records")
    return {
        "ticker": ticker_upper,
        "interval": interval,
        "bars_count": len(df),
        "recent_bars": records
    }


@router.get("/signals/{ticker}")
async def get_ticker_signals(ticker: str):
    ticker_upper = ticker.upper()
    analysis = system_state.technical_analysis.get(ticker_upper)

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail=f"No technical analysis available for {ticker_upper}"
        )

    # The analysis must come from the session being traded right now. A buffer
    # left over from a previous session survives an empty fetch (see
    # _fetch_batched_intraday), so without this check the desk can serve -- and
    # act on -- yesterday's bars while labelling them with today's clock.
    data_timestamp = analysis.get("data_timestamp")
    session_date = analysis.get("session_date")
    today_et = datetime.now(ZoneInfo("America/New_York")).date().isoformat()

    if not data_timestamp:
        raise HTTPException(
            status_code=404,
            detail=f"No priced bars available for {ticker_upper}"
        )

    if session_date and session_date != today_et:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Stale data for {ticker_upper}: last bar is {data_timestamp} "
                f"({session_date}), not the current session ({today_et})"
            )
        )

    return {
        "ticker": ticker_upper,
        "data_timestamp": data_timestamp,
        "session_date": session_date,
        "bars_analyzed": analysis.get("bars_analyzed"),
        "analysis": analysis
    }


@router.post("/evaluate/{ticker}")
async def evaluate_ticker_now(ticker: str):
    ticker_upper = ticker.upper()
    analysis = system_state.technical_analysis.get(ticker_upper)
    
    if not analysis:
        await fetch_single_ticker_data(ticker_upper)
        analysis = system_state.technical_analysis.get(ticker_upper)

    if not analysis:
        raise HTTPException(status_code=404, detail=f"No price or indicator data available for {ticker_upper}. Please wait a few seconds and retry.")

    decision = await evaluate_intraday_opportunity(
        ticker=ticker_upper,
        session_phase=system_state.session_phase,
        technical_analysis=analysis
    )
    system_state.agent_evaluations[ticker_upper] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision": decision
    }
    return {
        "ticker": ticker_upper,
        "session_phase": system_state.session_phase,
        "decision": decision
    }


@router.post("/session/toggle")
async def toggle_session():
    system_state.is_active = not system_state.is_active
    status_str = "ACTIVE" if system_state.is_active else "PAUSED"
    await send_ntfy_alert(
        title="Trading Desk V2 Status",
        message=f"Day trading engine shifted to {status_str}."
    )
    return {
        "is_active": system_state.is_active,
        "message": f"Trading engine is now {status_str}."
    }