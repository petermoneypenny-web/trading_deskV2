"""
Month-scale batch backtest -- no LLM, zero API quota.

``replay_csv.py`` replays one configuration over a whole multi-week dataset and
prints a report. That is enough to answer "what happened", but the interesting
question after tuning the rulebook is "which of these settings actually earns
its keep, and is that just curve-fitting?". This script runs the *same*
engine over many configurations, splits the month into in-sample and
out-of-sample halves, and ranks the configs on what they did out of sample.

Why it is built this way:

  * One subprocess per configuration. The engine keeps session state in module
    globals (``system_state``, ``ticker_cooldowns``, ``consecutive_signals``,
    the ``RiskConfig`` singleton). Running two configurations in one process
    would leak state between them and quietly corrupt the second result. A
    fresh process per run is the only cheap way to guarantee isolation.

  * In-sample / out-of-sample split, always. A sweep of 15 configs over one
    month will always produce a winner by chance. Splitting the month means the
    ranking is based on dates the configuration was not chosen on.

  * A private trade journal per run. The replay writes closed trades to SQLite;
    without a per-run file every configuration would append to the same table
    and the live database.

  * No LLM anywhere. Every number here is reproducible for free.

Usage
-----
    # single run over the whole month, same as replay_csv.py
    python3 backtest_month.py --csv data/historical_1m_30d.csv

    # sweep everything, 4 workers, split the month 60/40
    python3 backtest_month.py --csv data/historical_1m_30d.csv --sweep all --jobs 4 --split 0.6

    # just the stop geometry and target settings
    python3 backtest_month.py --csv data/historical_1m_30d.csv --sweep atr_2.0,rr_2.0

    # ad-hoc override without editing rulebook.json
    python3 backtest_month.py --csv data/historical_1m_30d.csv --set target_rr=3.0,max_hold_minutes=30

Results are written to ``data/backtests/<label>/`` as results.json,
summary.csv and daily_pnl.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest_month")
logging.getLogger("replay_csv").setLevel(logging.ERROR)

# --------------------------------------------------------------------------
# Named sweeps
# --------------------------------------------------------------------------
# Each entry is a set of RiskConfig overrides applied on top of rulebook.json.
# They are chosen to probe exactly the levers the profitability work moved, so
# the report says whether each change was load-bearing or incidental.
SWEEPS: Dict[str, Dict[str, Any]] = {
    "baseline": {},

    # stop geometry
    "atr_2.0":       {"atr_stop_multiplier": 2.0},
    "atr_3.0":       {"atr_stop_multiplier": 3.0},
    "minstop_0.004": {"min_stop_pct": 0.004},
    "minstop_0.015": {"min_stop_pct": 0.015},

    # target
    "rr_2.0":        {"target_rr": 2.0},
    "rr_3.5":        {"target_rr": 3.5},

    # trade management
    "hold_20m":      {"max_hold_minutes": 20},
    "hold_90m":      {"max_hold_minutes": 90},
    "no_breakeven":  {"breakeven_trigger_r": 99.0},

    # sizing
    "risk_25":       {"max_risk_per_trade": 25.0},
    "risk_100":      {"max_risk_per_trade": 100.0},

    # breadth
    "pos_2":         {"max_total_open_positions": 2},
    "pos_5":         {"max_total_open_positions": 5},
    "cluster_3":     {"max_positions_per_cluster": 3},
    "trades_12":     {"max_trades_per_day": 12},

    # liquidity gate
    "relvol_off":    {"min_relative_volume": 0.0},
    "relvol_1.2":    {"min_relative_volume": 1.2},

    # costs
    "costs_2x":      {"commission_per_share": 0.01, "slippage_pct": 0.0004},
}


def _parse_set(pairs: List[str]) -> Dict[str, Any]:
    """``target_rr=3.0,max_hold_minutes=30`` -> dict, with light type coercion."""
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw.strip()
    return out


# --------------------------------------------------------------------------
# Worker (runs in its own process)
# --------------------------------------------------------------------------
def _run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one configuration. Never raises: failures come back as errors."""
    import asyncio

    from replay_csv import run_simulation

    started = time.time()
    # Each configuration runs in its own process; without this the children
    # spam the console with the same per-bar chatter the parent suppressed.
    logging.getLogger().setLevel(logging.ERROR)
    logging.disable(logging.INFO)
    # ...but silence is worse than noise when a run takes minutes: print a
    # start marker so the operator can see work being dispatched. print()
    # bypasses logging.disable().
    print(f"    >> start {task['label']} [{task['phase']}]", flush=True)
    try:
        result = asyncio.run(run_simulation(
            csv_path=task["csv"],
            tickers=task["tickers"],
            use_llm=False,
            db_path=task["db"],
            date_from=task["date_from"],
            date_to=task["date_to"],
            verbose=False,
            config_overrides=task["overrides"],
            max_hold_minutes=task.get("max_hold_minutes"),
            random_entries=task.get("random_entries", False),
            random_rate=task.get("random_rate", 0.01),
            seed=task.get("seed", 0),
        ))
    except Exception as exc:  # pragma: no cover - defensive
        return {"label": task["label"], "phase": task["phase"],
                "overrides": task["overrides"], "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.time() - started, 1)}

    if result is None:
        return {"label": task["label"], "phase": task["phase"],
                "overrides": task["overrides"], "error": "no result (see log)",
                "elapsed_s": round(time.time() - started, 1)}

    result["label"] = task["label"]
    result["phase"] = task["phase"]
    result["overrides"] = task["overrides"]
    result["elapsed_s"] = round(time.time() - started, 1)
    return result


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def _session_dates(csv_path: str) -> List[str]:
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["timestamp"])
    dates = pd.to_datetime(df["timestamp"], utc=True).dt.date.unique()
    return [d.strftime("%Y-%m-%d") for d in sorted(dates)]


def _split_dates(dates: List[str], split: float) -> Tuple[Optional[str], Optional[str]]:
    """Cut the session list into an in-sample and an out-of-sample range."""
    if split <= 0 or split >= 1 or len(dates) < 4:
        return None, None
    cut = max(1, int(len(dates) * split))
    if cut >= len(dates):
        return None, None
    return dates[cut - 1], dates[cut]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt(v: Any, kind: str = "f") -> str:
    if v is None:
        return "-"
    if kind == "s":
        return str(v)
    if kind == "usd":
        return f"${v:+,.2f}"
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "n":
        return str(v)
    return f"{v:.3f}"


def _print_table(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]]) -> None:
    """cols: [(header, kind)]"""
    widths = [len(h) for h, _ in cols]
    cells: List[List[str]] = []
    for r in rows:
        line = [_fmt(r.get(k), kind) for k, kind in cols]
        widths = [max(w, len(c)) for w, c in zip(widths, line)]
        cells.append(line)
    header = "  " + "  ".join(h.ljust(w) for h, w in zip([c[0] for c in cols], widths))
    logger.info(header)
    logger.info("  " + "  ".join("-" * w for w in widths))
    for line in cells:
        logger.info("  " + "  ".join(c.ljust(w) for c, w in zip(line, widths)))


def _rank(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return sorted([r for r in rows if r.get(key) is not None], key=lambda r: -r[key])


def _persist(out_dir: str, payload: Dict[str, Any]) -> None:
    """Write every report file. Called after *each* run finishes, so a run the
    user aborts with Ctrl-C still leaves a readable partial result behind."""
    runs = payload["runs"]
    is_rows = [r for r in runs if r.get("phase") == "is" and not r.get("error")]
    oos_rows = [r for r in runs if r.get("phase") == "oos" and not r.get("error")]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    cols = [("label", "s"), ("trades", "n"), ("net_pnl", "usd"),
            ("expectancy", "usd"), ("win_rate_resolved", "pct"),
            ("profit_factor", "f"), ("max_drawdown_usd", "usd"),
            ("profitable_sessions", "n"), ("stop_loss", "n"),
            ("take_profit", "n"), ("time_stop", "n")]
    keys = [c[0] for c in cols]
    for fname, rows in (("summary_in_sample.csv", is_rows), ("summary_out_of_sample.csv", oos_rows)):
        if not rows:
            continue
        with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

    daily: Dict[str, Dict[str, float]] = {}
    for run in payload["runs"]:
        if run.get("error") or not run.get("daily"):
            continue
        tag = f"{run['label']}|{run['phase']}"
        for d in run["daily"]:
            daily.setdefault(tag, {})[d["date"]] = round(float(d["pnl"]), 2)
    if daily:
        dates = sorted({d for v in daily.values() for d in v})
        with open(os.path.join(out_dir, "daily_pnl.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["run"] + dates)
            for tag, by_date in daily.items():
                w.writerow([tag] + [by_date.get(d, "") for d in dates])

    if not payload.get("_partial"):
        logger.info(f"Reports written to {out_dir}/")


def _print_verdict(payload: Dict[str, Any]) -> None:
    """The honest part: did the in-sample winner survive out of sample?"""
    is_rows = [r for r in payload["runs"] if r.get("phase") == "is" and not r.get("error")]
    oos_rows = [r for r in payload["runs"] if r.get("phase") == "oos" and not r.get("error")]
    if not oos_rows:
        return

    logger.info("=" * 78)
    logger.info("OUT-OF-SAMPLE RANKING (the number that counts)")
    logger.info("=" * 78)
    _print_table(_rank(oos_rows, "expectancy"), [
        ("label", "s"), ("trades", "n"), ("net_pnl", "usd"),
        ("expectancy", "usd"), ("win_rate_resolved", "pct"),
        ("profit_factor", "f"), ("max_drawdown_usd", "usd"),
    ])

    if is_rows:
        logger.info("-" * 78)
        logger.info("IN-SAMPLE (for reference only -- these were chosen on this data)")
        _print_table(_rank(is_rows, "expectancy"), [
            ("label", "s"), ("trades", "n"), ("net_pnl", "usd"),
            ("expectancy", "usd"), ("win_rate_resolved", "pct"),
            ("profit_factor", "f"),
        ])

        is_by = {r["label"]: r for r in is_rows}
        oos_by = {r["label"]: r for r in oos_rows}
        decays = []
        for label, i in is_by.items():
            o = oos_by.get(label)
            if not o or i.get("expectancy") is None or o.get("expectancy") is None:
                continue
            if i["expectancy"] > 0 > o["expectancy"]:
                decays.append((label, i["expectancy"], o["expectancy"]))
        if decays:
            logger.info("-" * 78)
            logger.info("OVERFIT WARNING -- profitable in sample, losing out of sample:")
            for label, i, o in sorted(decays, key=lambda x: x[2]):
                logger.info(f"    {label:16s} in-sample ${i:+.2f}/trade -> "
                            f"out-of-sample ${o:+.2f}/trade")
            logger.info("  A sweep over one month will always find a winner by chance.")
            logger.info("  Treat only the configs that survive both halves as real.")


def _print_null_distribution(real: Optional[Dict[str, Any]],
                             null: List[Dict[str, Any]]) -> None:
    """Compare the real run against a distribution of placebo runs.

    The question is not "is the strategy profitable" -- with 15 sessions that is
    unanswerable -- but "does entering on the signal beat entering at random".
    That comparison shares the same market, the same gates and the same costs,
    so it is far more powerful than comparing either arm to zero.
    """
    if real is None or not null:
        return
    exps = [r["expectancy"] for r in null if r.get("expectancy") is not None]
    if len(exps) < 3:
        logger.warning("Too few placebo runs to form a distribution.")
        return

    import statistics as st
    mean, sd = st.mean(exps), (st.stdev(exps) if len(exps) > 1 else 0.0)
    r = real.get("expectancy") or 0.0

    logger.info("=" * 78)
    logger.info("PLACEBO TEST -- does the entry signal beat random?")
    logger.info("=" * 78)
    logger.info(f"  Real (signal)      : {real['trades']:3d} trades, "
                f"expectancy ${r:+.2f}, net ${real['net_pnl']:+.2f}")
    logger.info(f"  Placebo (random)   : {len(exps)} seeds, "
                f"{st.mean([x['trades'] for x in null]):.1f} trades avg")
    logger.info(f"  Placebo expectancy : mean ${mean:+.2f}, sd ${sd:.2f}, "
                f"range ${min(exps):+.2f} .. ${max(exps):+.2f}")
    lo = sorted(exps)[max(0, int(len(exps) * 0.025))]
    hi = sorted(exps)[min(len(exps) - 1, int(len(exps) * 0.975))]
    logger.info(f"  Placebo 95% band   : ${lo:+.2f} .. ${hi:+.2f}")
    logger.info(f"  Real vs placebo    : ${r - mean:+.2f}/trade "
                f"({(r - mean) / abs(mean) * 100:+.0f}%)" if mean else
                f"  Real vs placebo    : ${r - mean:+.2f}/trade")

    beat = sum(1 for e in exps if e >= r)
    p = beat / (len(exps) + 1)
    logger.info(f"  Placebo seeds >= real : {beat}/{len(exps)}  "
                f"-> one-sided p ~ {p:.3f}")
    if sd > 0:
        logger.info(f"  z-score             : {(r - mean) / sd:+.2f}")

    if p <= 0.05 and r > mean:
        logger.info("  VERDICT: the signal beats random at the 5% level.")
    elif p <= 0.10 and r > mean:
        logger.info("  VERDICT: weak evidence the signal beats random. "
                    "Needs more sessions.")
    elif r <= mean:
        logger.info("  VERDICT: RANDOM ENTRIES DO AT LEAST AS WELL AS THE SIGNAL.")
        logger.info("            The entry trigger carries no detectable edge at")
        logger.info("            this horizon. Tuning stops/targets will not fix it.")
    else:
        logger.info("  VERDICT: inconclusive -- the placebo spread is too wide "
                    "relative to the gap.")
    logger.info(f"  Caveat: {len(exps)} seeds gives a p-value resolution of "
                f"~{1/(len(exps)+1):.3f}. Treat as directional.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch backtest over a month of 1m data with no LLM calls.")
    parser.add_argument("--csv", type=str,
                        default=os.environ.get("REPLAY_CSV", "data/historical_1m_30d.csv"),
                        help="Wide-format 1m dataset: timestamp + <TICKER>_{open,high,low,close,volume}.")
    parser.add_argument("--tickers", type=str, default="SPY,QQQ,AAPL,NVDA",
                        help="Comma-separated tickers present in the CSV.")
    parser.add_argument("--sweep", type=str, default="baseline",
                        help="Comma-separated sweep names, or 'all'.")
    parser.add_argument("--set", dest="sets", action="append", default=[],
                        help="Ad-hoc override key=value (repeatable, comma-separated ok).")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Parallel configurations. Each gets its own process.")
    parser.add_argument("--split", type=float, default=0.6,
                        help="Fraction of sessions used as in-sample (0 or 1 disables).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory. Defaults to data/backtests/<label>.")
    parser.add_argument("--label", type=str, default=None, help="Name for this batch.")
    parser.add_argument("--max-hold-minutes", type=int, default=None,
                        help="Override the time stop for every configuration.")
    parser.add_argument("--random", type=int, default=0, metavar="N",
                        help="PLACEBO TEST: run N random-entry seeds alongside "
                             "the real baseline and report whether the signal "
                             "beats random. 0 disables (default).")
    parser.add_argument("--random-rate", type=float, default=0.01,
                        help="Per-ticker-per-bar entry probability for the "
                             "placebo arm. Tune until its trade count roughly "
                             "matches the real run.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error(f"Dataset not found: {args.csv}")
        logger.error("Fetch one first (Yahoo serves ~30 days of 1m bars, so a month is the ceiling):")
        logger.error("    docker compose exec trading-desk-v2 python3 download_dataset.py --days 30")
        logger.error("or locally:")
        logger.error("    python3 download_dataset.py --days 30 --out data/historical_1m_30d.csv")
        return 2

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    # ---- resolve the configurations to run --------------------------------
    names = [n.strip() for n in args.sweep.split(",") if n.strip()]
    if "all" in names:
        names = list(SWEEPS)
    configs: List[Tuple[str, Dict[str, Any]]] = []
    for name in names:
        if name not in SWEEPS:
            logger.error(f"Unknown sweep {name!r}. Available: {', '.join(SWEEPS)}")
            return 2
        configs.append((name, dict(SWEEPS[name])))
    for pair in args.sets:
        configs.append(("custom:" + pair, _parse_set(pair.split(","))))
    if not configs:
        logger.error("Nothing to run.")
        return 2

    dates = _session_dates(args.csv)
    if len(dates) < 4:
        logger.error(f"Only {len(dates)} session(s) in {args.csv}; need at least 4.")
        return 2
    is_to, oos_from = _split_dates(dates, args.split)
    if is_to is None:
        logger.warning("Split disabled or too few sessions -- running the whole month per config.")

    label = args.label or (args.sweep if args.sweep != "all" else "sweep_all")
    label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    out_dir = args.out or os.path.join("data", "backtests", f"{label}_{len(dates)}s")

    logger.info("=" * 78)
    logger.info("MONTH-SCALE BATCH BACKTEST (no LLM)")
    logger.info("=" * 78)
    logger.info(f"Dataset        : {args.csv}")
    logger.info(f"Sessions       : {len(dates)} ({dates[0]} .. {dates[-1]})")
    logger.info(f"Tickers        : {', '.join(tickers)}")
    logger.info(f"Configurations : {len(configs)}")
    logger.info(f"Workers        : {args.jobs}")
    if args.random > 0:
        logger.info(f"Period         : {dates[0]} .. {dates[-1]} "
                    f"(both arms use the whole period)")
    elif is_to:
        logger.info(f"In-sample      : {dates[0]} .. {is_to} "
                    f"({sum(1 for d in dates if d <= is_to)} sessions)")
        logger.info(f"Out-of-sample  : {oos_from} .. {dates[-1]} "
                    f"({sum(1 for d in dates if d >= oos_from)} sessions)")
    else:
        logger.info("Split          : disabled (whole month per configuration)")

    # ---- build the task list ---------------------------------------------
    tasks: List[Dict[str, Any]] = []

    def _task(label, phase, overrides, dfrom, dto, **extra):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        t = {
            "label": label, "phase": phase, "overrides": overrides,
            "csv": os.path.abspath(args.csv), "tickers": tickers,
            "db": os.path.abspath(os.path.join(out_dir, f"journal_{safe}_{phase}.db")),
            "date_from": dfrom, "date_to": dto,
            "max_hold_minutes": args.max_hold_minutes,
        }
        t.update(extra)
        return t

    if args.random > 0 and args.sweep != "baseline":
        logger.warning(f"--random ignores --sweep {args.sweep!r}; the placebo test "
                       f"always uses the rulebook as-is.")

    if args.random > 0:
        # Placebo arm. The real baseline is included as the reference, and the
        # random seeds form the null distribution it is judged against. Both
        # arms run over the same dates with the same rulebook, so the only
        # difference is the entry decision.
        logger.info(f"Placebo test     : {args.random} random seeds at rate "
                    f"{args.random_rate} per ticker-bar")
        tasks.append(_task("REAL_signal", "real", {}, None, None))
        for seed in range(1, args.random + 1):
            tasks.append(_task(f"random_s{seed}", "random", {}, None, None,
                               random_entries=True, random_rate=args.random_rate,
                               seed=seed))
    else:
        for name, overrides in configs:
            for phase, dfrom, dto in (("is", None, is_to), ("oos", oos_from, None),
                                      ("all", None, None)):
                if phase == "is" and is_to is None:
                    continue
                if phase == "oos" and oos_from is None:
                    continue
                if phase == "all" and is_to is not None:
                    continue
                tasks.append(_task(name, phase, overrides, dfrom, dto))

    os.makedirs(out_dir, exist_ok=True)

    def _stub() -> Dict[str, Any]:
        """Payload skeleton, shared by the pre-run stub and every incremental
        write so a partially-finished sweep is always readable on disk."""
        return {
            "generated": datetime.utcnow().isoformat() + "Z",
            "dataset": os.path.abspath(args.csv),
            "tickers": tickers,
            "sessions": len(dates),
            "session_dates": dates,
            "split": {"in_sample_through": is_to, "out_of_sample_from": oos_from},
            "wall_seconds": 0.0,
            "runs": [],
            "_partial": True,
        }

    # Land a stub immediately. The output file then exists and can be watched
    # for progress before the first run completes, which on a NAS is minutes.
    _persist(out_dir, _stub())

    logger.info(f"Runs           : {len(tasks)}")
    logger.info(f"Output         : {out_dir}")
    logger.info("-" * 78)

    # ---- execute ----------------------------------------------------------
    runs: List[Dict[str, Any]] = []
    t0 = time.time()
    stop_hb = threading.Event()

    def _heartbeat() -> None:
        """A full sweep can run for the better part of an hour. Without this the
        operator has no way to tell 'slow' from 'dead'."""
        while not stop_hb.wait(30):
            elapsed = time.time() - t0
            logger.info(f"    ... {len(runs)}/{len(tasks)} runs finished, "
                        f"{elapsed / 60:.1f} min elapsed")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    pool = ProcessPoolExecutor(max_workers=max(1, args.jobs))
    try:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            task = futures[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                res = {"label": task["label"], "phase": task["phase"],
                       "overrides": task["overrides"],
                       "error": f"{type(exc).__name__}: {exc}"}
            runs.append(res)
            if res.get("error"):
                logger.info(f"[{done}/{len(tasks)}] {res['label']:18s} {res['phase']:>3s}  "
                            f"ERROR: {res['error']}")
            else:
                logger.info(
                    f"[{done}/{len(tasks)}] {res['label']:18s} {res['phase']:>3s}  "
                    f"trades={res['trades']:3d}  net={_fmt(res['net_pnl'], 'usd'):>12s}  "
                    f"exp={_fmt(res['expectancy'], 'usd'):>9s}  "
                    f"wr={_fmt(res.get('win_rate_resolved'), 'pct'):>6s}  "
                    f"({res.get('elapsed_s', 0):.0f}s)")
            # Persist after every run so an aborted sweep is still readable.
            _persist(out_dir, _stub() | {
                "wall_seconds": round(time.time() - t0, 1),
                "runs": runs,
            })
    except KeyboardInterrupt:
        # A full sweep can run for the better part of an hour, so aborting must
        # be quick and must leave the completed runs behind rather than a
        # traceback. shutdown(wait=False) does not stop workers that are mid
        # run, and the executor's management thread is non-daemon, so the
        # interpreter would otherwise block until they finish.
        stop_hb.set()
        logger.warning("")
        logger.warning("Interrupted. Saving the runs that completed...")
        _persist(out_dir, _stub() | {
            "wall_seconds": round(time.time() - t0, 1),
            "runs": runs,
        })
        logger.warning(f"Partial results: {out_dir}/results.json "
                       f"({len(runs)}/{len(tasks)} runs).")
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # pragma: no cover - Python < 3.9
            pool.shutdown(wait=False)
        sys.stdout.flush()
        os._exit(130)
    finally:
        stop_hb.set()

    runs.sort(key=lambda r: (r.get("phase", ""), r.get("label", "")))
    payload = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "dataset": os.path.abspath(args.csv),
        "tickers": tickers,
        "sessions": len(dates),
        "session_dates": dates,
        "split": {"in_sample_through": is_to, "out_of_sample_from": oos_from},
        "wall_seconds": round(time.time() - t0, 1),
        "runs": runs,
    }

    is_rows = [r for r in runs if r.get("phase") == "is" and not r.get("error")]
    oos_rows = [r for r in runs if r.get("phase") == "oos" and not r.get("error")]
    all_rows = [r for r in runs if r.get("phase") == "all" and not r.get("error")]

    if all_rows:
        logger.info("=" * 78)
        logger.info("WHOLE-MONTH RESULTS")
        logger.info("=" * 78)
        _print_table(_rank(all_rows, "expectancy"), [
            ("label", "s"), ("trades", "n"), ("net_pnl", "usd"),
            ("expectancy", "usd"), ("win_rate_resolved", "pct"),
            ("profit_factor", "f"), ("max_drawdown_usd", "usd"),
        ])

    _print_verdict(payload)
    if args.random > 0:
        real_runs = [r for r in runs if r.get("phase") == "real" and not r.get("error")]
        null_runs = [r for r in runs if r.get("phase") == "random" and not r.get("error")]
        _print_null_distribution(real_runs[0] if real_runs else None, null_runs)
        rt = real_runs[0]["trades"] if real_runs else 0
        if null_runs:
            import statistics as _st
            nt = _st.mean([r["trades"] for r in null_runs])
            if rt and abs(nt - rt) / rt > 0.25:
                logger.warning("")
                logger.warning(f"Placebo trade count ({nt:.1f}) differs from the real "
                               f"run ({rt}) by more than 25%. Re-run with "
                               f"--random-rate adjusted so the two arms take a "
                               f"comparable number of trades.")
    payload.pop("_partial", None)
    _persist(out_dir, payload)

    logger.info("=" * 78)
    logger.info(f"{len(tasks)} run(s) in {time.time() - t0:.0f}s "
                f"({(time.time() - t0) / max(1, len(tasks)):.0f}s each). "
                f"LLM calls: 0.")
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
