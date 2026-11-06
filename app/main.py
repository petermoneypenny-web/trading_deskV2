import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.config import settings
from app.api.routes import router
from app.services.stream import run_intraday_tick_loop
from app.services.alerts import send_ntfy_alert
from app.services.scheduler import start_market_scheduler, stop_market_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("trading_desk_v2")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Trading Desk V2 on port 5095...")
    await send_ntfy_alert("Trading Desk V2 Online", "Container initialized on port 5095 with automated market scheduler.")
    
    # Start APScheduler for automatic US Eastern market open/close
    start_market_scheduler()
    
    # Start background data stream and risk cycle
    loop_task = asyncio.create_task(run_intraday_tick_loop())
    
    yield
    
    logger.info("Shutting down Trading Desk V2...")
    stop_market_scheduler()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Trading Desk V2 - Day Trading Engine",
    version="2.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Trading Desk V2 API Online</h1><p>Visit /api/status for JSON diagnostics.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)