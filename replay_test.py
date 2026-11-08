"""
Live-data replay harness (yfinance).

Applies the same honesty fixes as ``replay_csv.py``: an injected simulated clock
so cooldowns work on bar time rather than wall-clock time, intrabar stop/target
detection against bar highs and lows, no manufactured prices, transaction costs,
session-scoped VWAP/ORB, ranked entry selection, and a summary that reports
conditional hit rates instead of a single net P&L number.
"""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd
from unittest.mock import AsyncMock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replay_test")

logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from app.core.state import system_state
from app.services.indicators import evaluate_intraday_technical_snapshot
import app.services.risk_engine as risk_engine
from app.services.risk_engine import (
    init_db,
    open_position,
    update_open_positions_and_risk,
    liquidate_all_positions,
    ticker_cooldowns,
    set_clock,
    compute_stop_and_target,
    portfolio_direction,
    cluster_for,
    cluster_positions,
    reset_session_stats,
    session_stats,
)
from app.services.risk_config import get_risk_config
from app.services.stream import reset_daily_state, consecutive_signals

risk_engine.send_ntfy_alert = AsyncMock()

TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "INTC", "TSM", "MU"]
ET = ZoneInfo("America/New_York")

_current_bar = {"ts": datetime.utcnow()}


def _simulated_now() -> datetime:
    return _current_bar["ts"]


def get_mock_session_phase(bar_time) -> str:
    if bar_time.tzinfo is None:
        et_time = bar_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
    else:
        et_time = bar_time.astimezone(ET)

    time_mins = et_time.hour * 60 + et_time.minute

    if 9 * 60 + 30 <= time_mins < 10 * 60 + 15:
        return "RTH_OPENING_FLUSH"
    elif 10 * 60 + 15 <= time_mins < 11 * 60 + 30:
        return "RTH_PRIME_MORNING"
    elif 11 * 60 + 30 <= time_mins < 14 * 60 + 15:
        return "RTH_MIDDAY"
    elif 14 * 60 + 15 <= time_mins < 15 * 60 + 30:
        return "RTH_PRIME_AFTERNOON"
    elif 15 * 60 + 30 <= time_mins < 15 * 60 + 45:
        return "RTH_NORMAL"
    elif 15 * 60 + 45 <= time_mins < 16 * 60:
        return "RTH_EOD_CLOSING"
    else:
        return "CLOSED"


def update_macro_regime_bias():
    spy_analysis = system_state.technical_analysis.get("SPY", {})
    qqq_analysis = system_state.technical_analysis.get("QQQ", {})

    spy_m = spy_analysis.get("metrics", {})
    qqq_m = qqq_analysis.get("metrics", {})

    spy_p, spy_v = spy_m.get("last_price"), spy_m.get("vwap")
    qqq_p, qqq_v = qqq_m.get("last_price"), qqq_m.get("vwap")

    bull_score = (1 if spy_p and spy_v and spy_p > spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p > qqq_v else 0)
    bear_score = (1 if spy_p and spy_v and spy_p < spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p < qqq_v else 0)

    if bull_score == 2:
        system_state.macro_bias = "BULLISH"
    elif bear_score == 2:
        system_state.macro_bias = "BEARISH"
    else:
        system_state.macro_bias = "NEUTRAL"


async def run_replay(target_date_str: str = None, use_llm: bool = False, max_llm_calls: int = 6):
    cfg = get_risk_config()
    logger.info(f"Initializing replay harness (LLM: {'enabled' if use_llm else 'disabled'})...")
    init_db()

    # Simulated clock must be installed before any position/cooldown logic runs.
    set_clock(_simulated_now)

    system_state.is_active = True
    system_state.active_tickers = TICKERS
    system_state.daily_realized_pnl = 0.0
    system_state.daily_unrealized_pnl = 0.0
    system_state.daily_trade_counts = {}
    system_state.daily_trade_count = 0
    system_state.open_positions = {}
    ticker_cooldowns.clear()
    reset_daily_state()
    reset_session_stats()

    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: {target_date_str}. Use YYYY-MM-DD.")
            return
    else:
        target_date = datetime.now(ET).date()

    start_str = target_date.strftime("%Y-%m-%d")
    end_str = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    ticker_str = " ".join(TICKERS)
    logger.info(f"Downloading 1m data for [{ticker_str}] from {start_str} to {end_str}...")

    raw_df = yf.download(
        tickers=ticker_str,
        start=start_str,
        end=end_str,
        interval="1m",
        prepost=False,
        progress=False,
        threads=False,
        group_by="column"
    )

    if raw_df is None or raw_df.empty:
        logger.error(f"No 1m data returned for date {start_str}.")
        return

    all_timestamps = sorted(list(raw_df.index))
    session_timestamps = [ts for ts in all_timestamps if ts.date() == target_date]
    logger.info(f"Loaded {len(session_timestamps)} 1-minute bars for trading date: {target_date}")

    if len(session_timestamps) < 40:
        logger.error(f"Insufficient bars ({len(session_timestamps)}) for date {start_str}.")
        return

    ticker_dfs = {}
    for ticker in TICKERS:
        df_t = pd.DataFrame(index=session_timestamps)
        for metric in ["Open", "High", "Low", "Close", "Volume"]:
            df_t[metric.lower()] = raw_df[(metric, ticker)].loc[session_timestamps].astype(float)
        # No ffill/bfill: manufactured prices create phantom stop-outs.
        df_t = df_t.dropna(subset=["open", "high", "low", "close", "volume"])
        if not df_t.empty:
            ticker_dfs[ticker] = df_t

    if not ticker_dfs:
        logger.error("No usable ticker frames after cleaning.")
        return

    trades_opened = 0
    blocked = {"opening_flush": 0, "midday": 0, "macro": 0, "direction": 0,
               "confirmation": 0, "cluster": 0, "risk_engine": 0, "budget": 0}
    signal_counts = {}
    llm_calls = 0

    for idx in range(30, len(session_timestamps)):
        ts = session_timestamps[idx]
        _current_bar["ts"] = ts.replace(tzinfo=None) if ts.tzinfo else ts
        et_time = ts.astimezone(ET)
        session_phase = get_mock_session_phase(ts)
        system_state.session_phase = session_phase

        for ticker, df_t in ticker_dfs.items():
            if idx >= len(df_t):
                continue
            row = df_t.iloc[idx]
            price = float(row["close"])
            system_state.latest_quotes[ticker] = {
                "price": price,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
                "timestamp": ts.isoformat(),
            }
            system_state.technical_analysis[ticker] = evaluate_intraday_technical_snapshot(
                df_t.iloc[: idx + 1], session_start=session_timestamps[0]
            )

        update_macro_regime_bias()
        await update_open_positions_and_risk(system_state.latest_quotes)

        if session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
            logger.info(f"[{et_time.strftime('%H:%M')} ET] EOD Auto-Close Triggered.")
            await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")
            continue

        if session_phase not in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON"):
            if session_phase == "RTH_OPENING_FLUSH":
                blocked["opening_flush"] += 1
            elif session_phase == "RTH_MIDDAY":
                blocked["midday"] += 1
            continue

        # ---- candidate collection ----------------------------------------
        candidates = []
        for ticker, df_t in ticker_dfs.items():
            analysis = system_state.technical_analysis.get(ticker, {})
            sig = analysis.get("signal", "NEUTRAL")

            tracker = consecutive_signals.get(ticker, {"signal": "NEUTRAL", "bars": 0})
            if sig != "NEUTRAL" and sig == tracker["signal"]:
                tracker["bars"] += 1
            else:
                tracker = {"signal": sig, "bars": 1 if sig != "NEUTRAL" else 0}
            consecutive_signals[ticker] = tracker

            if sig not in ("NEUTRAL", None):
                signal_counts[sig] = signal_counts.get(sig, 0) + 1

            if sig in ("NEUTRAL", None) or ticker in system_state.open_positions:
                continue
            if tracker["bars"] < 2:
                blocked["confirmation"] += 1
                continue

            action = "BUY" if "BULLISH" in sig or "LONG" in sig else "SELL"

            if action == "BUY" and system_state.macro_bias == "BEARISH":
                blocked["macro"] += 1
                continue
            if action == "SELL" and system_state.macro_bias == "BULLISH":
                blocked["macro"] += 1
                continue

            if action == "BUY" and portfolio_direction() == "SHORT_ONLY":
                blocked["direction"] += 1
                continue
            if action == "SELL" and portfolio_direction() == "LONG_ONLY":
                blocked["direction"] += 1
                continue

            if len(cluster_positions(cluster_for(ticker))) >= cfg.max_positions_per_cluster:
                blocked["cluster"] += 1
                continue

            candidates.append({
                "ticker": ticker, "action": action, "signal": sig,
                "analysis": analysis, "score": analysis.get("setup_score", 0.0),
            })

        if not candidates:
            continue

        slots = cfg.max_trades_per_day - system_state.daily_trade_count
        slots = min(slots, cfg.max_total_open_positions - len(system_state.open_positions))
        if slots <= 0:
            blocked["budget"] += len(candidates)
            continue

        candidates.sort(key=lambda c: c["score"], reverse=True)

        for cand in candidates[:slots]:
            ticker = cand["ticker"]
            action = cand["action"]
            analysis = cand["analysis"]
            metrics = analysis.get("metrics", {})
            real_price = system_state.latest_quotes[ticker]["price"]
            atr = metrics.get("atr_14")

            if use_llm and llm_calls < max_llm_calls:
                from app.services.gemini_agent import evaluate_intraday_opportunity
                llm_calls += 1
                logger.info(f"[{et_time.strftime('%H:%M')} ET] Gemini call {llm_calls}/{max_llm_calls} for {ticker}")
                decision = await evaluate_intraday_opportunity(
                    ticker=ticker,
                    session_phase=session_phase,
                    technical_analysis=analysis,
                )
                llm_action = (decision.get("action") or "HOLD").upper()
                if llm_action not in ("BUY", "SELL"):
                    logger.info(f"Gemini vetoed {ticker}: {decision.get('rejected_because') or decision.get('reasoning')}")
                    continue
                action = llm_action

            stop_loss, take_profit = compute_stop_and_target(action, real_price, atr, cfg)

            opened = await open_position(
                ticker=ticker,
                action=action,
                current_price=real_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=f"Replay: {cand['signal']} (score {cand['score']:.2f})",
                atr=atr,
                atr_pct=metrics.get("atr_pct"),
                signal=cand["signal"],
                setup_score=cand["score"],
                enforce_geometry=False,
            )
            if opened:
                trades_opened += 1
                logger.info(
                    f"[{et_time.strftime('%H:%M')} ET] OPENED {action} {ticker} @ ${real_price:.2f} "
                    f"| SL: ${stop_loss} | TP: ${take_profit}"
                )
            else:
                blocked["risk_engine"] += 1

    if system_state.open_positions:
        await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")

    # ------------------------------------------------------------------ report
    logger.info("=" * 72)
    logger.info(f"REPLAY BACKTEST COMPLETED FOR: {target_date}")
    logger.info("=" * 72)
    logger.info(f"Total Technical Signals Detected: {signal_counts}")
    logger.info(f"Bars Blocked by Opening Flush     : {blocked['opening_flush']}")
    logger.info(f"Bars Blocked by Midday Chop       : {blocked['midday']}")
    logger.info(f"Entries Blocked by Macro Filter   : {blocked['macro']}")
    logger.info(f"Entries Blocked by Direction Lock : {blocked['direction']}")
    logger.info(f"Entries Blocked by Cluster Cap    : {blocked['cluster']}")
    logger.info(f"Entries Blocked by 2-Bar Confirm  : {blocked['confirmation']}")
    logger.info(f"Entries Blocked by Budget/Caps    : {blocked['budget']}")
    logger.info(f"Entries Rejected by Risk Engine   : {blocked['risk_engine']}")
    if use_llm:
        logger.info(f"Gemini LLM Calls Dispatched       : {llm_calls}/{max_llm_calls}")
    logger.info(f"Total Trades Opened               : {trades_opened}")
    logger.info(f"Final Realized P&L                : ${system_state.daily_realized_pnl:+.2f}")

    if session_stats["trades"]:
        resolved = session_stats["stop_loss"] + session_stats["take_profit"]
        timed_out = session_stats["time_stop"] + session_stats["breakeven_stop"]
        logger.info("-" * 72)
        logger.info(f"  Stop-loss hits   : {session_stats['stop_loss']:3d}")
        logger.info(f"  Take-profit hits : {session_stats['take_profit']:3d}")
        logger.info(f"  Time stops       : {timed_out:3d}")
        logger.info(f"  Gross P&L        : ${session_stats['gross_pnl']:+.2f}")
        logger.info(f"  Costs            : ${session_stats['costs']:.2f}")
        if resolved:
            logger.info(
                f"  Win rate (resolved): {session_stats['take_profit'] / resolved * 100:.1f}% "
                f"-- break-even is {100 / (1 + cfg.min_risk_reward_ratio):.1f}%"
            )
    logger.info("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay intraday trading simulation for a specific date.")
    parser.add_argument("--date", type=str, default=None, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--with-llm", action="store_true", help="Enable live Gemini evaluation.")
    parser.add_argument("--max-llm-calls", type=int, default=6, help="Cap on Gemini calls.")
    args = parser.parse_args()

    asyncio.run(run_replay(args.date, use_llm=args.with_llm, max_llm_calls=args.max_llm_calls))
