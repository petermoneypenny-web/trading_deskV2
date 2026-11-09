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
import random
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
from app.config import settings
from app.services.stream import reset_daily_state

# Silence live alerts during backtest
risk_engine.send_ntfy_alert = AsyncMock()

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA"]
CSV_PATH = "/app/data/historical_1m_30d.csv"
ET = ZoneInfo("America/New_York")

# Holds the bar currently being replayed, so risk_engine's clock can read it.
_current_bar: Dict[str, datetime] = {"ts": datetime.utcnow()}

# Placebo-arm RNG. Seeded per run so a sweep of seeds is reproducible.
_rng = random.Random(0)


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


# --------------------------------------------------------------------------
# Cross-session aggregation
# --------------------------------------------------------------------------
# ``session_stats`` is reset every session (correctly -- it is per-day state),
# but the summary used to read it *after* the loop, so every aggregate in the
# report described the final session only. A 21-session run reported 67 trades
# and +$803.96 while its own exit breakdown described 2 trades and +$91.64.
# These helpers fold each session into a running total instead.
def _new_cum() -> Dict[str, Any]:
    return {
        "trades": 0, "stop_loss": 0, "take_profit": 0, "time_stop": 0,
        "breakeven_stop": 0, "eod_close": 0, "manual_close": 0,
        "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0, "by_signal": {},
    }


def _accumulate(cum: Dict[str, Any], s: Dict[str, Any]) -> None:
    for k in ("trades", "stop_loss", "take_profit", "time_stop",
              "breakeven_stop", "eod_close", "manual_close"):
        cum[k] += int(s.get(k, 0) or 0)
    for k in ("gross_pnl", "costs", "net_pnl"):
        cum[k] += float(s.get(k, 0.0) or 0.0)
    for sig, v in (s.get("by_signal") or {}).items():
        d = cum["by_signal"].setdefault(
            sig, {"trades": 0, "wins": 0, "losses": 0, "net": 0.0})
        for k in ("trades", "wins", "losses"):
            d[k] += int(v.get(k, 0) or 0)
        d["net"] += float(v.get("net", 0.0) or 0.0)


def _trades_from_db(db_path: str) -> List[Dict[str, Any]]:
    """Every closed trade recorded by this run, as plain dicts.

    ``close_position`` already persists ticker, action, exit reason, signal,
    cluster, hold time and P&L, so the per-ticker/per-action/per-exit tables
    come from the journal rather than from a second bookkeeping path that could
    drift from it.
    """
    if not db_path or not os.path.exists(db_path):
        return []
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trades").fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not read trade journal {db_path}: {exc}")
        return []


def _max_drawdown(daily_pnls: List[float]) -> Tuple[float, float]:
    """Peak-to-trough drawdown of the compounded equity curve."""
    if not daily_pnls:
        return 0.0, 0.0
    equity, peak, worst, worst_pct = 1.0, 1.0, 0.0, 0.0
    for pnl in daily_pnls:
        equity *= (1.0 + pnl / 25_000.0)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        if dd > worst:
            worst, worst_pct = dd, dd * 100.0
    return worst * 25_000.0, worst_pct


def _table(rows: List[Tuple[str, Dict[str, Any]]],
           cols: Tuple[Tuple[str, str], ...]) -> List[str]:
    """Fixed-width text table. ``cols`` is ((header, kind), ...) where kind is
    one of 'n' (count), 'pct' or 'usd'. Keeps the report diffable between runs."""
    def _cell(v: Any, kind: str) -> str:
        if v is None:
            return "-"
        if kind == "n":
            return str(v)
        if kind == "pct":
            return f"{v:.1f}%"
        if kind == "usd":
            return f"${v:+,.2f}"
        return str(v)

    widths = [len(h) for h, _ in cols]
    for name, vals in rows:
        widths[0] = max(widths[0], len(str(name)))
        for i, (k, kind) in enumerate(cols[1:], start=1):
            widths[i] = max(widths[i], len(_cell(vals.get(k), kind)))
    out = ["  " + "  ".join(h.ljust(w) for h, w in zip([c[0] for c in cols], widths))]
    out.append("  " + "  ".join("-" * w for w in widths))
    for name, vals in rows:
        cells = [str(name)] + [_cell(vals.get(k), kind) for k, kind in cols[1:]]
        out.append("  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return out


def _group(trades: List[Dict[str, Any]], key: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Aggregate the journal by ``key``. Values stay numeric so results.json is
    machine-readable; only the log table formats them."""
    agg: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        name = str(t.get(key) or "UNKNOWN")
        d = agg.setdefault(name, {"n": 0, "wins": 0, "net": 0.0})
        pnl = float(t.get("pnl") or 0.0)
        d["n"] += 1
        d["net"] += pnl
        d["wins"] += 1 if pnl >= 0 else 0
    out = []
    for name, d in sorted(agg.items(), key=lambda kv: -kv[1]["net"]):
        out.append((name, {
            "n": d["n"],
            "winrate": round(d["wins"] / d["n"] * 100, 1) if d["n"] else 0.0,
            "net": round(d["net"], 2),
            "exp": round(d["net"] / d["n"], 2) if d["n"] else 0.0,
        }))
    return out


def _log_summary(cum, by_ticker, by_action, by_exit, daily_results,
                 total_trades, total_pnl, use_llm, cfg, drawdown_usd,
                 drawdown_pct, journal_rows):
    logger.info("=" * 78)
    logger.info("OFFLINE MULTI-WEEK DATASET BACKTEST COMPLETED")
    logger.info("=" * 78)
    logger.info(f"Trading sessions replayed : {len(daily_results)}")
    logger.info(f"Executions                : {total_trades}")
    logger.info(f"Net cumulative realized   : ${total_pnl:+.2f}")
    if journal_rows:
        logger.info(f"Closed trades in journal  : {len(journal_rows)}")

    resolved = cum["stop_loss"] + cum["take_profit"]
    timed_out = cum["time_stop"] + cum["breakeven_stop"]
    if cum["trades"]:
        logger.info("-" * 78)
        logger.info("EXIT BREAKDOWN (whole run, not the last session)")
        logger.info(f"  Stop-loss hits        : {cum['stop_loss']:4d}  "
                    f"({cum['stop_loss'] / cum['trades'] * 100:5.1f}%)")
        logger.info(f"  Take-profit hits      : {cum['take_profit']:4d}  "
                    f"({cum['take_profit'] / cum['trades'] * 100:5.1f}%)")
        logger.info(f"  Time stop             : {cum['time_stop']:4d}  "
                    f"({cum['time_stop'] / cum['trades'] * 100:5.1f}%)")
        logger.info(f"  Breakeven stop        : {cum['breakeven_stop']:4d}  "
                    f"({cum['breakeven_stop'] / cum['trades'] * 100:5.1f}%)")
        logger.info(f"  EOD / manual close    : {cum['eod_close'] + cum['manual_close']:4d}")
        logger.info(f"  Gross P&L             : ${cum['gross_pnl']:+.2f}")
        logger.info(f"  Transaction costs     : ${cum['costs']:.2f}")
        logger.info(f"  Net P&L               : ${cum['net_pnl']:+.2f}")
        if resolved:
            win_rate = cum["take_profit"] / resolved * 100
            logger.info(f"  Win rate (resolved)   : {win_rate:.1f}%  "
                        f"-- break-even at a {cfg.min_risk_reward_ratio:g}:1 shape is "
                        f"{100 / (1 + cfg.min_risk_reward_ratio):.1f}%")
        logger.info(f"  Expectancy per trade  : ${cum['net_pnl'] / cum['trades']:+.2f}")
        gross_win = sum(v["net"] for v in cum["by_signal"].values() if v["net"] > 0)
        gross_loss = sum(v["net"] for v in cum["by_signal"].values() if v["net"] < 0)
        if gross_loss:
            logger.info(f"  Profit factor         : {gross_win / abs(gross_loss):.3f}")
        logger.info(f"  Max drawdown          : ${drawdown_usd:.2f} ({drawdown_pct:.2f}%)")

    if by_ticker:
        logger.info("-" * 78)
        logger.info("PER-TICKER")
        for line in _table(by_ticker, (("ticker", "s"), ("n", "n"), ("winrate", "pct"),
                             ("net", "usd"), ("exp", "usd"))):
            logger.info(line)
    if by_action:
        logger.info("-" * 78)
        logger.info("PER-ACTION")
        for line in _table(by_action, (("action", "s"), ("n", "n"), ("winrate", "pct"),
                             ("net", "usd"), ("exp", "usd"))):
            logger.info(line)
    if by_exit:
        logger.info("-" * 78)
        logger.info("PER-EXIT-REASON")
        for line in _table(by_exit, (("exit_reason", "s"), ("n", "n"), ("winrate", "pct"),
                           ("net", "usd"), ("exp", "usd"))):
            logger.info(line)

    if cum["by_signal"]:
        logger.info("-" * 78)
        logger.info("PER-SIGNAL CONDITIONAL STATS")
        logger.info(f"  {'signal':22s} {'n':>4s} {'win':>4s} {'loss':>5s} "
                    f"{'winrate':>8s} {'net':>11s} {'exp':>9s}")
        for sig, v in sorted(cum["by_signal"].items(), key=lambda kv: -kv[1]["net"]):
            wr = (v["wins"] / v["trades"] * 100) if v["trades"] else 0.0
            exp = (v["net"] / v["trades"]) if v["trades"] else 0.0
            logger.info(f"  {sig:22s} {v['trades']:4d} {v['wins']:4d} {v['losses']:5d} "
                        f"{wr:7.1f}% ${v['net']:+10.2f} ${exp:+8.2f}")

    if daily_results:
        wins = sum(1 for d in daily_results if d["pnl"] > 0)
        logger.info("-" * 78)
        logger.info(f"Profitable sessions      : {wins}/{len(daily_results)}")
        logger.info(f"Best session             : ${max(d['pnl'] for d in daily_results):+.2f}")
        logger.info(f"Worst session            : ${min(d['pnl'] for d in daily_results):+.2f}")
        logger.info(f"LLM evaluations          : "
                    f"{'enabled' if use_llm else 'disabled (reproducible, zero API quota)'}")
    logger.info("=" * 78)


async def run_simulation(
    csv_path=CSV_PATH,
    tickers=DEFAULT_TICKERS,
    use_llm: bool = False,
    max_hold_minutes: int = None,
    db_path: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    verbose: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
    random_entries: bool = False,
    random_rate: float = 0.01,
    seed: int = 0,
) -> Optional[Dict[str, Any]]:
    """Replay a multi-week 1m dataset with the LLM switched off.

    Returns a plain-dict summary (also written to the log) so a caller can run
    many configurations in one process and compare them. ``None`` on failure.
    """
    if not verbose:
        logger.setLevel(logging.WARNING)

    if random_entries:
        _rng.seed(seed)

    # reload=True is load-bearing. ``get_risk_config()`` caches a process-wide
    # singleton, and a sweep reuses worker processes, so without the reload the
    # second configuration a worker sees would inherit the first one's values
    # and silently produce a chimera (hold_20m was coming back with
    # atr_stop_multiplier=2.0 from the previous task).
    cfg = get_risk_config(reload=True)
    if config_overrides:
        from app.services.risk_config import RiskConfig, set_risk_config
        merged = cfg.to_dict()
        merged.update(config_overrides)
        cfg = RiskConfig.from_dict(merged)
        set_risk_config(cfg)
    if max_hold_minutes is not None:
        cfg.max_hold_minutes = max_hold_minutes

    if not os.path.exists(csv_path):
        logger.error(f"Dataset {csv_path} not found. Run download_dataset.py first.")
        return None

    # A backtest must never write into the live journal. Without this the
    # replayed trades land in the same SQLite file the running container uses,
    # which corrupts real trading history and makes every subsequent report
    # wrong. Default to a dedicated file.
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                               "backtest_trades.db")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    settings.DATABASE_URL = f"sqlite:///{os.path.abspath(db_path)}"

    logger.info(f"Loading local dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)

    if date_from or date_to:
        dates = df["timestamp"].dt.date
        if date_from:
            df = df[dates >= datetime.strptime(date_from, "%Y-%m-%d").date()]
        if date_to:
            df = df[dates <= datetime.strptime(date_to, "%Y-%m-%d").date()]
        df = df.reset_index(drop=True)
        if df.empty:
            logger.error(f"No bars between {date_from} and {date_to}.")
            return None

    unique_dates = sorted(list(df["timestamp"].dt.date.unique()))
    logger.info(f"Loaded {len(df)} bars across {len(unique_dates)} trading sessions.")

    # Install the simulated clock before anything else touches time.
    set_clock(_simulated_now)

    init_db()
    # Start from an empty journal so the report describes *this* run only.
    import sqlite3
    with sqlite3.connect(settings.DATABASE_URL.replace("sqlite:///", "")) as conn:
        conn.execute("DELETE FROM trades")
        conn.commit()

    system_state.is_active = True
    system_state.active_tickers = tickers

    total_realized_pnl = 0.0
    total_trades_count = 0
    daily_results = []
    cum = _new_cum()

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
        # The daily circuit breaker flips is_active to False when a session
        # breaches max_daily_loss, and nothing in here ever flips it back. Left
        # alone, one bad day would silently pause every later session in the
        # run. Restore it per session so each day is judged on its own P&L.
        system_state.is_active = True
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

                if random_entries:
                    # ---- PLACEBO ARM ------------------------------------
                    # Identical engine, gates, stops, exits and costs. The only
                    # difference is that the entry decision is a coin flip at a
                    # fixed rate instead of a confirmed signal. If the signal
                    # carries information this arm must do materially worse;
                    # if it performs the same, the signal is noise and no
                    # amount of stop or target tuning will rescue it.
                    # Note the score is randomised too, otherwise ranking
                    # random entries by the real setup_score would leak signal
                    # back into the placebo arm.
                    if t in system_state.open_positions:
                        continue
                    if _rng.random() >= random_rate:
                        continue
                    action = _rng.choice(("BUY", "SELL"))
                    sig = f"RANDOM_{action}"
                else:
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
                    "analysis": analysis,
                    "score": _rng.random() if random_entries
                             else analysis.get("setup_score", 0.0),
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
        _accumulate(cum, session_stats)

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

    # ------------------------------------------------------------------ report
    journal = _trades_from_db(db_path)
    dd_usd, dd_pct = _max_drawdown([d["pnl"] for d in daily_results])
    _log_summary(cum, _group(journal, "ticker"), _group(journal, "action"),
                 _group(journal, "exit_reason"), daily_results,
                 total_trades_count, total_realized_pnl, use_llm, cfg,
                 dd_usd, dd_pct, journal)

    return {
        "sessions": len(daily_results),
        "trades": total_trades_count,
        "net_pnl": round(total_realized_pnl, 2),
        "gross_pnl": round(cum["gross_pnl"], 2),
        "costs": round(cum["costs"], 2),
        "stop_loss": cum["stop_loss"],
        "take_profit": cum["take_profit"],
        "time_stop": cum["time_stop"],
        "breakeven_stop": cum["breakeven_stop"],
        "eod_close": cum["eod_close"],
        "win_rate_resolved": round(cum["take_profit"] / resolved * 100, 1)
        if (resolved := cum["stop_loss"] + cum["take_profit"]) else None,
        "expectancy": round(cum["net_pnl"] / cum["trades"], 2) if cum["trades"] else 0.0,
        "profit_factor": round(
            sum(v["net"] for v in cum["by_signal"].values() if v["net"] > 0)
            / abs(sum(v["net"] for v in cum["by_signal"].values() if v["net"] < 0)), 3
        ) if any(v["net"] < 0 for v in cum["by_signal"].values()) else None,
        "max_drawdown_usd": round(dd_usd, 2),
        "max_drawdown_pct": round(dd_pct, 2),
        "profitable_sessions": sum(1 for d in daily_results if d["pnl"] > 0),
        "daily": daily_results,
        "by_signal": cum["by_signal"],
        "by_ticker": {name: v for name, v in _group(journal, "ticker")},
        "by_action": {name: v for name, v in _group(journal, "action")},
        "by_exit": {name: v for name, v in _group(journal, "exit_reason")},
        "config": cfg.to_dict(),
        "use_llm": use_llm,
        "tickers": list(tickers),
        "bars": int(len(df)),
        "random_entries": bool(random_entries),
        "random_rate": float(random_rate) if random_entries else None,
        "seed": int(seed) if random_entries else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline multi-week intraday replay.")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Path to the 1m dataset CSV.")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS),
                        help="Comma-separated tickers.")
    parser.add_argument("--llm", action="store_true",
                        help="Enable live Gemini evaluation (off by default for reproducibility).")
    parser.add_argument("--max-hold-minutes", type=int, default=None,
                        help="Override the time stop, in minutes.")
    parser.add_argument("--db", type=str, default=None,
                        help="Trade journal for this run. Defaults to "
                             "backtest_trades.db next to the CSV, so a backtest "
                             "never writes into the live database.")
    parser.add_argument("--from", dest="date_from", type=str, default=None,
                        help="First session to replay, YYYY-MM-DD (inclusive).")
    parser.add_argument("--to", dest="date_to", type=str, default=None,
                        help="Last session to replay, YYYY-MM-DD (inclusive).")
    parser.add_argument("--quiet", action="store_true",
                        help="Only warnings/errors. Used when sweeping many configs.")
    parser.add_argument("--random-entries", action="store_true",
                        help="PLACEBO ARM: enter at random bars with a random "
                             "direction instead of on signal. Everything else "
                             "(gates, stops, exits, costs) is identical.")
    parser.add_argument("--random-rate", type=float, default=0.01,
                        help="Per-ticker-per-bar entry probability for the "
                             "placebo arm. Default 0.01.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the placebo arm.")
    args = parser.parse_args()

    asyncio.run(run_simulation(
        csv_path=args.csv,
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        use_llm=args.llm,
        max_hold_minutes=args.max_hold_minutes,
        db_path=args.db,
        date_from=args.date_from,
        date_to=args.date_to,
        verbose=not args.quiet,
        random_entries=args.random_entries,
        random_rate=args.random_rate,
        seed=args.seed,
    ))
