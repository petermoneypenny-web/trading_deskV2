import os
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd

# Disable yfinance internal SQLite disk-cache locking inside container
import yfinance.cache as yf_cache
try:
    os.makedirs("/tmp/py-yfinance", exist_ok=True)
    yf.set_tz_cache_location("/tmp/py-yfinance")
    if hasattr(yf_cache, "enable_cache"):
        yf_cache.enable_cache(False)
except Exception:
    pass

from app.core.state import system_state
from app.services.buffer import buffer_manager
from app.services.indicators import evaluate_intraday_technical_snapshot, score_setup
from app.services.gemini_agent import evaluate_intraday_opportunity
from app.services.alerts import send_ntfy_alert
from app.services.risk_engine import (
    init_db,
    open_position,
    update_open_positions_and_risk,
    liquidate_all_positions,
    compute_stop_and_target,
    portfolio_direction,
    cluster_for,
    cluster_positions,
    ticker_cooldowns,
)
from app.services.risk_config import get_risk_config
from app.config import settings

logger = logging.getLogger("trading_desk_v2.stream")

# In-memory tracking for consecutive bar signal persistence: {ticker: {"signal": str, "bars": int}}
consecutive_signals: Dict[str, Dict[str, Any]] = {}

# LLM evaluations in flight. Kept as a set so the tick loop never awaits them:
# previously the Gemini call was awaited inline, blocking position monitoring for
# 6-11 seconds per ticker (up to ~80s per cycle with a 7-name watchlist).
_inflight_evaluations: set = set()

# Hard ceiling on how long we will wait for the LLM before falling back to the
# technical signal alone. This must comfortably exceed GEMINI_HTTP_TIMEOUT, or
# the inner client timeout fires first and every call fails with ReadTimeout.
# Because the evaluation runs in a background task, a generous deadline costs
# nothing -- the risk loop keeps ticking every 10s regardless.
LLM_TIMEOUT_SECONDS = float(getattr(settings, "LLM_DEADLINE_SECONDS", 25.0))


def reset_daily_state():
    """
    Clears per-session state. Called at market open and by the replay harness.

    Previously ``consecutive_signals`` was never reset, so the last signal of the
    previous day could carry over and satisfy the 2-bar confirmation on the very
    first bar of a new session.
    """
    consecutive_signals.clear()
    ticker_cooldowns.clear()
    system_state.daily_trade_counts = {}
    system_state.daily_trade_count = 0
    system_state.entries_blocked = {}
    buffer_manager.reset_session()
    from app.services.risk_engine import reset_session_stats
    reset_session_stats()


def get_current_session_phase() -> str:
    """
    Calculates US market session phase based on Eastern Time.
    Gating:
    - 09:30 - 10:15: RTH_OPENING_FLUSH (Opening volatility noise - DO NOT ENTER)
    - 10:15 - 11:30: RTH_PRIME_MORNING (High conviction trend window)
    - 11:30 - 14:15: RTH_MIDDAY (Midday chop dead zone)
    - 14:15 - 15:30: RTH_PRIME_AFTERNOON (Afternoon trend continuation & squeeze)
    - 15:30 - 15:45: RTH_NORMAL (No new entries)
    - 15:45 - 16:00: RTH_EOD_CLOSING (Flatten all positions)
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return "CLOSED_WEEKEND"

    time_mins = now_et.hour * 60 + now_et.minute

    if 4 * 60 <= time_mins < 9 * 60 + 30:
        return "PRE_MARKET"
    elif 9 * 60 + 30 <= time_mins < 10 * 60 + 15:
        return "RTH_OPENING_FLUSH"  # 09:30 - 10:15 ET: Let opening range wicks settle
    elif 10 * 60 + 15 <= time_mins < 11 * 60 + 30:
        return "RTH_PRIME_MORNING"   # 10:15 - 11:30 ET: Primary morning trend window
    elif 11 * 60 + 30 <= time_mins < 14 * 60 + 15:
        return "RTH_MIDDAY"          # 11:30 - 14:15 ET: Midday chop filter
    elif 14 * 60 + 15 <= time_mins < 15 * 60 + 30:
        return "RTH_PRIME_AFTERNOON" # 14:15 - 15:30 ET: Afternoon power hour
    elif 15 * 60 + 30 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"     # 15:45 - 16:00 ET: Auto-liquidate open trades
    elif 16 * 60 <= time_mins < 20 * 60:
        return "POST_MARKET"
    else:
        return "CLOSED"


def update_macro_regime_bias():
    """Determines broad index trend alignment using SPY & QQQ relative to VWAP."""
    spy_analysis = system_state.technical_analysis.get("SPY", {})
    qqq_analysis = system_state.technical_analysis.get("QQQ", {})

    spy_metrics = spy_analysis.get("metrics", {})
    qqq_metrics = qqq_analysis.get("metrics", {})

    spy_p = spy_metrics.get("last_price")
    spy_v = spy_metrics.get("vwap")
    qqq_p = qqq_metrics.get("last_price")
    qqq_v = qqq_metrics.get("vwap")

    bull_score = (1 if spy_p and spy_v and spy_p > spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p > qqq_v else 0)
    bear_score = (1 if spy_p and spy_v and spy_p < spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p < qqq_v else 0)

    if bull_score == 2:
        system_state.macro_bias = "BULLISH"
    elif bear_score == 2:
        system_state.macro_bias = "BEARISH"
    else:
        system_state.macro_bias = "NEUTRAL"


def get_portfolio_active_direction() -> str:
    """
    Checks open positions to enforce portfolio-wide directional consistency.
    Returns: 'LONG_ONLY', 'SHORT_ONLY', or 'ANY'

    NOTE: this is a point-in-time snapshot. It used to be computed once per bar
    before the ticker loop, which made it useless -- it could not block entries
    made later in the same bar. Callers must re-check it after each fill.
    """
    return portfolio_direction()


def _blocked(ticker: str, reason: str):
    system_state.entries_blocked[reason] = system_state.entries_blocked.get(reason, 0) + 1
    logger.info(f"Entry blocked for {ticker}: {reason}")


def _fetch_batched_intraday(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Downloads 1-minute bars for all watchlist tickers in a single HTTP request."""
    if not tickers:
        return {}

    ticker_str = " ".join(tickers)

    df = yf.download(
        tickers=ticker_str,
        period="1d",
        interval="1m",
        prepost=True,
        progress=False,
        threads=False,
        group_by="column"
    )

    # FIX: the old fallback silently re-downloaded 5 days and kept only the last
    # 120 bars, which mixed *previous sessions* into the session VWAP, the
    # opening range and the 2-bar confirmation count. Fail loudly instead.
    if df is None or df.empty:
        logger.warning(
            "No 1m data returned for today. Not falling back to multi-day data: "
            "stale bars from a prior session corrupt session-scoped indicators."
        )
        return {}

    result: Dict[str, pd.DataFrame] = {}

    if isinstance(df.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                sub_df = pd.DataFrame(index=df.index)
                for metric in ["Open", "High", "Low", "Close", "Volume"]:
                    if (metric, ticker) in df.columns:
                        sub_df[metric] = df[(metric, ticker)]

                sub_df.dropna(subset=["Close"], inplace=True)
                if not sub_df.empty:
                    result[ticker] = sub_df
            except Exception as e:
                logger.warning(f"Error extracting sub-dataframe for {ticker}: {e}")
    else:
        ticker = tickers[0]
        sub_df = df.copy()
        sub_df.dropna(subset=["Close"], inplace=True)
        if not sub_df.empty:
            result[ticker] = sub_df

    return result


async def fetch_single_ticker_data(ticker: str):
    """Fetches and populates buffer/indicators for a single ticker on demand."""
    try:
        data_map = await asyncio.to_thread(_fetch_batched_intraday, [ticker])
        df = data_map.get(ticker)
        if df is not None and not df.empty:
            buffer_manager.load_initial_history(ticker, df, interval="1m")
            last_bar = buffer_manager.get_latest_bar(ticker, interval="1m")
            if last_bar:
                system_state.latest_quotes[ticker] = {
                    "price": float(last_bar["close"]),
                    "high": float(last_bar["high"]),
                    "low": float(last_bar["low"]),
                    "volume": float(last_bar["volume"]),
                    "timestamp": last_bar["timestamp"]
                }
            buffer_df = buffer_manager.get_dataframe(ticker, interval="1m")
            session_start = buffer_manager.get_session_start(ticker)
            analysis = evaluate_intraday_technical_snapshot(buffer_df, session_start=session_start)
            system_state.technical_analysis[ticker] = analysis
            logger.info(f"Initialized buffer and indicators for {ticker}")
            return True
    except Exception as e:
        logger.error(f"Failed to fetch initial data for {ticker}: {e}")
    return False


def _refresh_indicators(tickers: List[str]):
    """Recomputes indicators for each ticker from its session-scoped buffer."""
    for ticker in tickers:
        buffer_df = buffer_manager.get_dataframe(ticker, interval="1m")
        if buffer_df.empty:
            continue
        session_start = buffer_manager.get_session_start(ticker)
        system_state.technical_analysis[ticker] = evaluate_intraday_technical_snapshot(
            buffer_df, session_start=session_start
        )


async def sync_market_data():
    """Batched data fetch, macro regime update, risk monitoring, and gated execution."""
    cfg = get_risk_config()
    active_tickers = list(system_state.active_tickers)
    if not active_tickers:
        return

    try:
        data_map = await asyncio.to_thread(_fetch_batched_intraday, active_tickers)

        for ticker in active_tickers:
            df = data_map.get(ticker)
            if df is not None and not df.empty:
                buffer_manager.load_initial_history(ticker, df, interval="1m")
                last_bar = buffer_manager.get_latest_bar(ticker, interval="1m")
                if last_bar:
                    real_price = float(last_bar["close"])
                    system_state.latest_quotes[ticker] = {
                        "price": real_price,
                        "high": float(last_bar["high"]),
                        "low": float(last_bar["low"]),
                        "volume": float(last_bar["volume"]),
                        "timestamp": last_bar["timestamp"]
                    }

        _refresh_indicators(active_tickers)

        # Update macro index bias across SPY & QQQ
        update_macro_regime_bias()

        system_state.last_fetch_error = None

    except Exception as e:
        error_msg = f"Failed in batched market data sync: {e}"
        logger.warning(error_msg)
        system_state.last_fetch_error = error_msg

    # Risk first, always. This runs before and independently of any LLM work.
    await update_open_positions_and_risk(system_state.latest_quotes)

    if system_state.session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
        logger.info("RTH_EOD_CLOSING reached. Flattening all intraday positions.")
        await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")
        return

    if not (system_state.is_active and system_state.session_phase in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON")):
        return

    # ---------------------------------------------------------------------
    # Candidate collection
    # ---------------------------------------------------------------------
    candidates = []
    for ticker in active_tickers:
        analysis = system_state.technical_analysis.get(ticker, {})
        signal = analysis.get("signal", "NEUTRAL")

        # Track consecutive bar signal persistence (require 2 consecutive bars)
        tracker = consecutive_signals.get(ticker, {"signal": "NEUTRAL", "bars": 0})
        if signal != "NEUTRAL" and signal == tracker["signal"]:
            tracker["bars"] += 1
        else:
            tracker = {"signal": signal, "bars": 1 if signal != "NEUTRAL" else 0}
        consecutive_signals[ticker] = tracker

        if signal in ("NEUTRAL", None):
            continue
        if ticker in system_state.open_positions:
            continue
        if ticker in _inflight_evaluations:
            continue
        if tracker["bars"] < 2:
            continue

        action = "BUY" if "BULLISH" in signal or "LONG" in signal else "SELL"

        # Macro trend gating
        if action == "BUY" and system_state.macro_bias == "BEARISH":
            _blocked(ticker, "macro_bias")
            continue
        if action == "SELL" and system_state.macro_bias == "BULLISH":
            _blocked(ticker, "macro_bias")
            continue

        # Portfolio directional lock -- checked per candidate, and re-checked
        # after every fill inside the entry loop below.
        if action == "BUY" and portfolio_direction() == "SHORT_ONLY":
            _blocked(ticker, "direction_lock")
            continue
        if action == "SELL" and portfolio_direction() == "LONG_ONLY":
            _blocked(ticker, "direction_lock")
            continue

        # Correlation cluster cap
        peers = cluster_positions(cluster_for(ticker))
        if len(peers) >= cfg.max_positions_per_cluster:
            _blocked(ticker, f"cluster_cap:{cluster_for(ticker)}")
            continue

        candidates.append({
            "ticker": ticker,
            "action": action,
            "signal": signal,
            "analysis": analysis,
            "score": analysis.get("setup_score", 0.0),
        })

    if not candidates:
        return

    # ---------------------------------------------------------------------
    # Ranked selection: take the best N rather than everything that voted.
    # ---------------------------------------------------------------------
    slots = cfg.max_trades_per_day - system_state.daily_trade_count
    slots = min(slots, cfg.max_total_open_positions - len(system_state.open_positions))
    if slots <= 0:
        for c in candidates:
            _blocked(c["ticker"], "daily_budget_or_position_cap")
        return

    candidates.sort(key=lambda c: c["score"], reverse=True)
    selected = candidates[:slots]
    for c in candidates[slots:]:
        logger.info(
            f"Deferring {c['ticker']} (score {c['score']:.3f}) -- "
            f"only {slots} slot(s) available today."
        )

    # Fire LLM evaluations concurrently; never block the risk loop on them.
    for cand in selected:
        task = asyncio.create_task(_evaluate_and_maybe_trade(cand))
        _inflight_evaluations.add(task)
        task.add_done_callback(_inflight_evaluations.discard)


async def _evaluate_and_maybe_trade(cand: Dict[str, Any]):
    """
    Runs the LLM gate for one candidate and opens the position if it passes.

    Runs as a background task so that a slow Gemini response cannot freeze
    ``update_open_positions_and_risk``.
    """
    ticker = cand["ticker"]
    try:
        decision = await asyncio.wait_for(
            evaluate_intraday_opportunity(
                ticker=ticker,
                session_phase=system_state.session_phase,
                technical_analysis=cand["analysis"],
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Gemini timed out after {LLM_TIMEOUT_SECONDS}s for {ticker}; using technical signal.")
        decision = {"action": cand["action"], "confidence": 0, "reasoning": "LLM timeout -- technical fallback"}
    except Exception as exc:
        logger.error(f"Gemini evaluation failed for {ticker}: {exc}")
        decision = {"action": cand["action"], "confidence": 0, "reasoning": f"LLM error -- technical fallback"}

    system_state.agent_evaluations[ticker] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decision": decision,
    }

    cfg = get_risk_config()
    action = (decision.get("action") or cand["action"]).upper()
    confidence = decision.get("confidence", 0) or 0

    if action not in ("BUY", "SELL"):
        logger.info(f"{ticker}: agent returned {action}; no trade.")
        return
    if confidence and confidence < 78:
        logger.info(f"{ticker}: agent confidence {confidence} below 78; no trade.")
        return

    # Macro + direction + cluster gates, re-checked at execution time.
    if action == "BUY" and system_state.macro_bias == "BEARISH":
        _blocked(ticker, "macro_bias")
        return
    if action == "SELL" and system_state.macro_bias == "BULLISH":
        _blocked(ticker, "macro_bias")
        return
    if action == "BUY" and portfolio_direction() == "SHORT_ONLY":
        _blocked(ticker, "direction_lock")
        return
    if action == "SELL" and portfolio_direction() == "LONG_ONLY":
        _blocked(ticker, "direction_lock")
        return

    real_price = system_state.latest_quotes.get(ticker, {}).get("price")
    if not real_price:
        return

    metrics = cand["analysis"].get("metrics", {})
    atr = metrics.get("atr_14")
    stop_loss, take_profit = compute_stop_and_target(action, real_price, atr, cfg)

    await open_position(
        ticker=ticker,
        action=action,
        current_price=real_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=decision.get("reasoning", "Confirmed 2-bar momentum setup"),
        atr=atr,
        atr_pct=metrics.get("atr_pct"),
        signal=cand["signal"],
        setup_score=cand["score"],
    )


async def run_intraday_tick_loop():
    """Background asynchronous loop for market monitoring and buffer updates."""
    logger.info("Day-Trading Intraday Loop initialized.")
    init_db()

    await sync_market_data()

    while True:
        try:
            system_state.session_phase = get_current_session_phase()
            system_state.last_loop_timestamp = datetime.utcnow()

            await sync_market_data()

            # Tick loop runs every 10s during prime trading, 30s during opening flush, midday chop, or paused
            is_prime = system_state.session_phase in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON")
            sleep_duration = 10 if (system_state.is_active and is_prime) else 30
            await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            logger.info("Intraday tick loop cancellation received.")
            break
        except Exception as e:
            logger.error(f"Error in intraday tick loop: {e}", exc_info=True)
            await asyncio.sleep(10)
