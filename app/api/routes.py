import asyncio
import sqlite3
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.core.state import system_state
from app.config import settings
from app.services.buffer import buffer_manager
from app.services.alerts import send_ntfy_alert
from app.services.stream import fetch_single_ticker_data
from app.services.gemini_agent import evaluate_intraday_opportunity
from app.services.risk_engine import open_position, close_position, liquidate_all_positions

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

    # Immediately fetch initial candle buffer asynchronously so signals are ready right away
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
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades")
        cursor.execute("VACUUM")
        conn.commit()
    return {"status": "cleared", "message": "Closed trade history has been wiped."}


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
        raise HTTPException(status_code=404, detail=f"No technical analysis available for {ticker_upper}")
    return {
        "ticker": ticker_upper,
        "timestamp": datetime.utcnow().isoformat(),
        "analysis": analysis
    }


@router.post("/evaluate/{ticker}")
async def evaluate_ticker_now(ticker: str):
    ticker_upper = ticker.upper()
    analysis = system_state.technical_analysis.get(ticker_upper)
    
    # If buffer not yet ready, attempt on-demand fetch first
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