from fastapi import APIRouter, Query, HTTPException
from app.core.state import system_state
from app.config import settings
from app.services.buffer import buffer_manager
from app.services.stream import send_ntfy_alert
from datetime import datetime

router = APIRouter()


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
        "open_positions_count": len(system_state.open_positions),
        "daily_realized_pnl": system_state.daily_realized_pnl,
        "daily_unrealized_pnl": system_state.daily_unrealized_pnl,
        "last_loop_timestamp": system_state.last_loop_timestamp.isoformat() if system_state.last_loop_timestamp else None,
        "last_fetch_error": system_state.last_fetch_error
    }


@router.get("/buffer/{ticker}")
async def get_ticker_buffer(ticker: str, interval: str = Query("1m", regex="^(1m|5m)$")):
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


@router.post("/test-alert")
async def test_alert():
    await send_ntfy_alert(
        title="Trading Desk V2",
        message=f"Test alert received from port 5095 at {datetime.utcnow().strftime('%H:%M:%S UTC')}."
    )
    return {"status": "Alert dispatched to topic " + settings.NTFY_TOPIC}