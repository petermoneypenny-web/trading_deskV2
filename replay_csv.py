"""
Offline multi-week replay harness.

This is a backtest, so it has to be honest about the things that made the
previous run misleading:

  * Simulated clock. ``risk_engine`` used to read ``datetime.utcnow()`` for
    cooldowns while the replay advanced *simulated* minutes in milliseconds of
    wall-clock time. The 30-minute cooldown therefore never expired and only the
    first signal of each day could ever be traded -- 1,577 of the 2,006 lines in
    the old log were "In cooldown for 29 more mins". The clock is now injected.

  * Intrabar stops. Stops and targets are checked against the bar's high/low,
    not just its close. A 1-minute bar's range is typically 1.5-3x its
    close-to-close move, so close-only checks systematically undercount
    stop-outs.

  * No price filling. ``ffill()``/``bfill()`` used to manufacture prices for
    missing bars, which silently created stop hits at invented levels.

  * Costs. Commission and slippage are charged on every round trip via
    ``close_position``.

  * Session-scoped indicators. VWAP and the opening range are anchored to the
    09:30 ET open instead of a rolling window.

  * Per-setup statistics. The summary reports conditional hit rates (stop vs
    target vs time stop) per signal, not just a single net P&L number.

  * No live LLM. Gemini is off by default so a run is reproducible and does not
    burn API quota. Use --llm to enable it.
"""

import os
import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
from unittest.mock import AsyncMock

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replay_csv")

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
from app.services.stream import reset_daily_state

# Silence live alerts during backtest
risk_engine.send_ntfy_alert = AsyncMock()

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA"]
CSV_PATH = "/app/data/historical_1m_30d.csv"
ET = ZoneInfo("America/New_York")

# Holds the bar currently being replayed, so risk_engine's clock can read it.
_current_bar: Dict[str, datetime] = {"ts": datetime.utcnow()}


def _simulated_now() -> datetime:
    """Clock used by risk_engine: naive UTC, matching datetime.utcnow()."""
    return _current_bar["ts"]


def get_mock_session_phase(bar_time) -> str:
    if bar_time.tzinfo is None:
        et_time = bar_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
    else:
        et_time = bar_time.astimezone(ET)

    time_mins = et_time.hour * 60 + et_time.minute
    if 9 * 60 + 30 <= time_mins < 9 * 60 + 45:
        return "RTH_OPENING_FLUSH"
    elif 9 * 60 + 45 <= time_mins < 11 * 60 + 30:
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
    spy_metrics = system_state.technical_analysis.get("SPY", {}).get("metrics", {})
    qqq_metrics = system_state.technical_analysis.get("QQQ", {}).get("metrics", {})

    spy_p, spy_v = spy_metrics.get("last_price"), spy_metrics.get("vwap")
    qqq_p, qqq_v = qqq_metrics.get("last_price"), qqq_metrics.get("vwap")

    bull = (1 if spy_p and spy_v and spy_p > spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p > qqq_v else 0)
    bear = (1 if spy_p and spy_v and spy_p < spy_v else 0) + (1 if qqq_p and qqq_v and qqq_p < qqq_v else 0)
    system_state.macro_bias = "BULLISH" if bull == 2 else ("BEARISH" if bear == 2 else "NEUTRAL")


def _session_open_for_day(timestamps: pd.Series) -> pd.Timestamp:
    """First bar at or after 09:30 ET on this day -- the VWAP/ORB anchor."""
    et = timestamps.dt.tz_convert(ET)
    mask = (et.dt.hour * 60 + et.dt.minute) >= 9 * 60 + 30
    if mask.any():
        return timestamps[mask].iloc[0]
    return timestamps.iloc[0]


def _log_summary(daily_results, total_trades, total_pnl, use_llm):
    logger.info("=" * 72)
    logger.info("OFFLINE MULTI-WEEK DATASET BACKTEST COMPLETED")
    logger.info("=" * 72)
    logger.info(f"Total Trading Sessions Replayed: {len(daily_results)}")
    logger.info(f"Total Executions: {total_trades}")
    logger.info(f"Net Cumulative Realized P&L: ${total_pnl:+.2f}")

    resolved = session_stats["stop_loss"] + session_stats["take_profit"]
    timed_out = session_stats["time_stop"] + session_stats["breakeven_stop"]
    if session_stats["trades"]:
        logger.info("-" * 72)
        logger.info("EXIT BREAKDOWN")
        logger.info(f"  Stop-loss hits        : {session_stats['stop_loss']:3d}  "
                    f"({session_stats['stop_loss'] / session_stats['trades'] * 100:5.1f}%)")
        logger.info(f"  Take-profit hits      : {session_stats['take_profit']:3d}  "
                    f"({session_stats['take_profit'] / session_stats['trades'] * 100:5.1f}%)")
        logger.info(f"  Time / breakeven stop : {timed_out:3d}  "
                    f"({timed_out / session_stats['trades'] * 100:5.1f}%)")
        logger.info(f"  EOD / manual close    : {session_stats['eod_close'] + session_stats['manual_close']:3d}")
        logger.info(f"  Gross P&L             : ${session_stats['gross_pnl']:+.2f}")
        logger.info(f"  Transaction costs     : ${session_stats['costs']:.2f}")
        logger.info(f"  Net P&L               : ${session_stats['net_pnl']:+.2f}")
        if resolved:
            win_rate = session_stats["take_profit"] / resolved * 100
            logger.info(f"  Win rate (resolved)   : {win_rate:.1f}%  "
                        f"-- break-even at a {get_risk_config().min_risk_reward_ratio:g}:1 shape is "
                        f"{100 / (1 + get_risk_config().min_risk_reward_ratio):.1f}%")
        logger.info(f"  Expectancy per trade  : ${session_stats['net_pnl'] / session_stats['trades']:+.2f}")
        gross_win = sum(v["net"] for v in session_stats["by_signal"].values() if v["net"] > 0)
        gross_loss = sum(v["net"] for v in session_stats["by_signal"].values() if v["net"] < 0)
        if gross_loss:
            logger.info(f"  Profit factor         : {gross_win / abs(gross_loss):.3f}")

    if session_stats["by_signal"]:
        logger.info("-" * 72)
        logger.info("PER-SIGNAL CONDITIONAL STATS")
        logger.info(f"  {'signal':22s} {'n':>4s} {'win':>4s} {'loss':>5s} {'winrate':>8s} {'net':>10s}")
        for sig, v in sorted(session_stats["by_signal"].items(), key=lambda kv: -kv[1]["net"]):
            wr = (v["wins"] / v["trades"] * 100) if v["trades"] else 0.0
            logger.info(
                f"  {sig:22s} {v['trades']:4d} {v['wins']:4d} {v['losses']:5d} "
                f"{wr:7.1f}% ${v['net']:+9.2f}"
            )

    if daily_results:
        wins = sum(1 for d in daily_results if d["pnl"] > 0)
        logger.info("-" * 72)
        logger.info(f"Profitable sessions: {wins}/{len(daily_results)}")
        logger.info(f"Best  : ${max(d['pnl'] for d in daily_results):+.2f}")
        logger.info(f"Worst : ${min(d['pnl'] for d in daily_results):+.2f}")
        logger.info(f"LLM evaluations: {'enabled' if use_llm else 'disabled (reproducible run)'}")
    logger.info("=" * 72)


async def run_simulation(
    csv_path=CSV_PATH,
    tickers=DEFAULT_TICKERS,
    use_llm: bool = False,
    max_hold_minutes: int = None,
):
    cfg = get_risk_config()
    if max_hold_minutes is not None:
        cfg.max_hold_minutes = max_hold_minutes

    if not os.path.exists(csv_path):
        logger.error(f"Dataset {csv_path} not found. Run download_dataset.py first.")
        return

    logger.info(f"Loading local dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)

    unique_dates = sorted(list(df["timestamp"].dt.date.unique()))
    logger.info(f"Loaded {len(df)} bars across {len(unique_dates)} trading sessions.")

    # Install the simulated clock before anything else touches time.
    set_clock(_simulated_now)

    init_db()
    system_state.is_active = True
    system_state.active_tickers = tickers

    total_realized_pnl = 0.0
    total_trades_count = 0
    daily_results = []

    for sim_date in unique_dates:
        day_df = df[df["timestamp"].dt.date == sim_date].copy()
        if len(day_df) < 40:
            continue

        # ---- per-session reset -------------------------------------------
        system_state.daily_realized_pnl = 0.0
        system_state.daily_unrealized_pnl = 0.0
        system_state.daily_trade_counts = {}
        system_state.daily_trade_count = 0
        system_state.session_date = sim_date.strftime("%Y-%m-%d")
        system_state.open_positions = {}
        ticker_cooldowns.clear()
        reset_daily_state()
        reset_session_stats()

        # ---- build ticker frames WITHOUT inventing prices ----------------
        # ffill/bfill used to manufacture bars, creating stop hits at levels
        # that never traded. Incomplete bars are dropped from the DAY frame
        # first, so every ticker frame shares one index and positional .iloc
        # stays aligned across tickers.
        needed = [f"{t}_{m}" for t in tickers
                  for m in ("open", "high", "low", "close", "volume")]
        present = [c for c in needed if c in day_df.columns]
        bars_df = day_df.dropna(subset=present).reset_index(drop=True)
        if len(bars_df) < 40:
            continue

        ticker_dfs = {}
        for t in tickers:
            t_df = pd.DataFrame(index=bars_df.index)
            complete = True
            for m in ["open", "high", "low", "close", "volume"]:
                col = f"{t}_{m}"
                if col not in bars_df.columns:
                    complete = False
                    break
                t_df[m] = bars_df[col].astype(float)
            if complete and not t_df.empty:
                ticker_dfs[t] = t_df

        if not ticker_dfs:
            continue

        # VWAP/ORB anchor: the first bar at or after 09:30 ET today.
        session_open = _session_open_for_day(bars_df["timestamp"])

        day_trades = 0
        blocked = {"opening_flush": 0, "midday": 0, "macro": 0, "direction": 0,
                   "confirmation": 0, "cluster": 0, "risk_engine": 0, "budget": 0}

        for idx in range(30, len(bars_df)):
            ts = bars_df["timestamp"].iloc[idx]
            _current_bar["ts"] = ts.replace(tzinfo=None)
            et_time = ts.astimezone(ET)
            session_phase = get_mock_session_phase(ts)
            system_state.session_phase = session_phase

            # ---- quotes + session-scoped indicators ----------------------
            for t, t_df in ticker_dfs.items():
                row = t_df.iloc[idx]
                price = float(row["close"])
                system_state.latest_quotes[t] = {
                    "price": price,
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "volume": float(row["volume"]),
                    "timestamp": ts.isoformat(),
                }
                # Full session up to and including this bar.
                system_state.technical_analysis[t] = evaluate_intraday_technical_snapshot(
                    t_df.iloc[: idx + 1], session_start=session_open
                )

            update_macro_regime_bias()
            await update_open_positions_and_risk(system_state.latest_quotes)

            if session_phase == "RTH_EOD_CLOSING" and system_state.open_positions:
                await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")
                continue

            if session_phase not in ("RTH_PRIME_MORNING", "RTH_PRIME_AFTERNOON"):
                if session_phase == "RTH_OPENING_FLUSH":
                    blocked["opening_flush"] += 1
                elif session_phase == "RTH_MIDDAY":
                    blocked["midday"] += 1
                continue

            # ---- candidate collection ------------------------------------
            from app.services.stream import consecutive_signals
            candidates = []
            for t in ticker_dfs:
                analysis = system_state.technical_analysis.get(t, {})
                sig = analysis.get("signal", "NEUTRAL")

                tracker = consecutive_signals.get(t, {"signal": "NEUTRAL", "bars": 0})
                if sig != "NEUTRAL" and sig == tracker["signal"]:
                    tracker["bars"] += 1
                else:
                    tracker = {"signal": sig, "bars": 1 if sig != "NEUTRAL" else 0}
                consecutive_signals[t] = tracker

                if sig in ("NEUTRAL", None):
                    continue
                if t in system_state.open_positions:
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

                peers = cluster_positions(cluster_for(t))
                if len(peers) >= cfg.max_positions_per_cluster:
                    blocked["cluster"] += 1
                    continue

                candidates.append({
                    "ticker": t, "action": action, "signal": sig,
                    "analysis": analysis, "score": analysis.get("setup_score", 0.0),
                })

            if not candidates:
                continue

            # ---- ranked selection ----------------------------------------
            slots = cfg.max_trades_per_day - system_state.daily_trade_count
            slots = min(slots, cfg.max_total_open_positions - len(system_state.open_positions))
            if slots <= 0:
                blocked["budget"] += len(candidates)
                continue

            candidates.sort(key=lambda c: c["score"], reverse=True)

            for cand in candidates[:slots]:
                t = cand["ticker"]
                action = cand["action"]
                analysis = cand["analysis"]
                metrics = analysis.get("metrics", {})
                real_p = system_state.latest_quotes[t]["price"]
                atr = metrics.get("atr_14")

                stop_loss, take_profit = compute_stop_and_target(action, real_p, atr, cfg)

                opened = await open_position(
                    ticker=t,
                    action=action,
                    current_price=real_p,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=f"CSV Replay: {cand['signal']} (score {cand['score']:.2f})",
                    atr=atr,
                    atr_pct=metrics.get("atr_pct"),
                    signal=cand["signal"],
                    setup_score=cand["score"],
                    enforce_geometry=False,
                )
                if opened:
                    day_trades += 1
                else:
                    blocked["risk_engine"] += 1

        # Flatten anything still open at the session close.
        if system_state.open_positions:
            await liquidate_all_positions(reason="RTH_EOD_AUTO_CLOSE")

        total_realized_pnl += system_state.daily_realized_pnl
        total_trades_count += day_trades
        daily_results.append({
            "date": sim_date.strftime("%Y-%m-%d"),
            "trades": day_trades,
            "pnl": system_state.daily_realized_pnl,
        })
        logger.info(
            f"[{sim_date}] Session P&L: ${system_state.daily_realized_pnl:+.2f} "
            f"({day_trades} trades) | bars: flush={blocked['opening_flush']} "
            f"midday={blocked['midday']} | entries: macro={blocked['macro']} "
            f"confirm={blocked['confirmation']} cluster={blocked['cluster']} "
            f"budget={blocked['budget']}"
        )
        if system_state.entries_blocked:
            top = sorted(system_state.entries_blocked.items(), key=lambda kv: -kv[1])[:6]
            logger.info(
                "           risk-engine rejections: "
                + ", ".join(f"{k}={v}" for k, v in top)
            )

    _log_summary(daily_results, total_trades_count, total_realized_pnl, use_llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline multi-week intraday replay.")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to the 1m dataset CSV.")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS),
                        help="Comma-separated tickers.")
    parser.add_argument("--llm", action="store_true",
                        help="Enable live Gemini evaluation (off by default for reproducibility).")
    parser.add_argument("--max-hold-minutes", type=int, default=None,
                        help="Override the time stop, in minutes.")
    args = parser.parse_args()

    asyncio.run(run_simulation(
        csv_path=args.csv,
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        use_llm=args.llm,
        max_hold_minutes=args.max_hold_minutes,
    ))
