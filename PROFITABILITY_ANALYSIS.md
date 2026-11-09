# Trading Desk V2 — Why It Isn't Profitable

**Analysis of:** `petermoneypenny-web/trading_deskV2` @ `ec8df4c` + `replay_test_results.txt`
**Scope:** `replay_csv.py` run (17 sessions, 68 trades) and the `--with-llm` run for 2026-11-04.

---

## 1. Headline: the strategy has a structurally negative edge

| Metric | Value |
|---|---|
| Sessions replayed | 17 (2026-10-15 → 2026-11-06) |
| Trades executed | 68 (exactly 4 per session) |
| Net realized P&L | **−$75.82** |
| Expectancy per trade | **−$1.11** |
| Profit factor (gross win / gross loss) | **0.843** |
| Stop-loss hits | 34 (50.0%) → −$460.76 |
| Take-profit hits | 11 (16.2%) → +$308.28 |
| Never resolved, force-closed at EOD | 23 (33.8%) → +$76.66 (7 of the 23 were losses) |
| Mean risk per trade | $12.93 — **only 26% of the `MAX_RISK_PER_TRADE = $50` budget** |
| Mean stop distance | **0.560%** of price |
| Mean target distance | **1.121%** of price (⇒ R:R is hard-wired to exactly 2.00) |

### The single sentence that explains the loss

You risk **0.56%** to make **1.12%** on a **1-minute** chart, and your entries are *less* predictive than a coin flip.

For a driftless random walk with barriers at −0.56% and +1.12%, the probability the
target is hit first is `0.56 / (0.56 + 1.12) = 33.3%` — which is exactly the break-even
win rate for a 2:1 reward:risk. You are achieving **16.2%**, and only 24.4% even among
trades that actually resolved. **Your entry signal is worse than random**, so a 2:1
payoff cannot save it.

The 2026-11-04 LLM run is the cleanest confirmation: 4 entries at 10:03–10:19 ET,
all long SPY/NVDA/TSM/INTC, **all four stopped out**, **−$77.87 in a single session**.

---

## 2. Root causes, ranked by damage

### 🔴 #1 — The stop loss is inside 1-minute noise (0.55% floor)

`app/services/risk_engine.py:48` and `:93`:

```python
min_risk = entry_price * 0.0055      # 0.55% minimum "realistic" risk buffer
...
min_dist = round(current_price * 0.0055, 2)
```

The intent was "don't set a stop tighter than 0.55%". In practice the 0.55% floor is
**the binding constraint on nearly every trade** (mean observed stop = 0.560%), and
`atr * 2.2` is silently discarded whenever it is smaller. On a 1-minute bar for NVDA,
QQQ or AAPL, a 0.55% adverse excursion is routine noise — it is not a thesis
invalidation. That is why 50% of trades are stopped out.

**This is the #1 driver of the −$460.76 in stop losses.**

---

### 🔴 #2 — The reward is derived from the broken stop, so R:R is a lie

`app/services/stream.py:291–314` and `replay_test.py:255–261`:

```python
min_stop_distance = max(atr * 2.2, real_price * 0.0055)
take_profit = round(real_price + (risk_dist * 2.0), 2)
```

The target is mechanically set to **exactly 2× the stop**, regardless of whether a
1.12% move is plausible in the remaining session. A "2:1 reward:risk" is only an edge
if your win rate exceeds 33%. Yours is 16%. The ratio is arithmetic, not an edge.

---

### 🔴 #3 — Position sizing is broken: the risk budget is dead code

`app/services/risk_engine.py:45–56`:

```python
risk_shares      = int(risk_budget / per_share_risk)   # 50 / (0.0055 * price) = 9090/price
max_capital_shares = int(MAX_POSITION_VALUE / entry_price)  # 2500 / price
final_shares = min(risk_shares, max_capital_shares)
```

Because the stop is always the 0.55% floor, `risk_shares` (≈9090/price) is **always
larger** than `max_capital_shares` (2500/price). The capital cap therefore binds on
**every single trade**, and actual risk per trade is pinned at
`$2500 × 0.0055 = $13.75`.

Consequences:
- `MAX_RISK_PER_TRADE = $50` is never used (26% utilisation).
- `MAX_DAILY_LOSS = $250` circuit breaker is **unreachable**: 4 tickers × $13.75 = $55
  max daily risk. Worst observed session was −$53.59. The breaker has never once been
  relevant — it is decorative risk management.
- The account mathematically cannot grow: with ~$13.75 of risk per trade you need a
  ~90% win rate to make $50/day.

---

### 🔴 #4 — 34% of trades never resolve: no time stop, no trade management

23 of 68 trades hit neither barrier and were force-closed at 15:45 ET at an average of
**+0.142%**. They consumed a full day of capital to generate $3.33 each. There is:

- no time-based exit ("if it isn't working in N minutes, get out"),
- no move-to-breakeven after a partial gain,
- no trailing stop,
- no scaling out.

A trade that drifts around flat for 5 hours is not a trade, it is a parked bet.

---

### 🟠 #5 — "ORB" and "VWAP" are computed on a rolling window, not the session

`app/services/indicators.py:85–94`:

```python
def calculate_orb(df, bars_count=15):
    opening_slice = df.iloc[:bars_count]      # <-- first 15 bars of WHATEVER df is passed
```

and `app/services/buffer.py:15` keeps only the **last 120 one-minute bars**, while
`replay_test.py:172` feeds a rolling `iloc[idx-120 : idx+1]` window.

So the "Opening Range Breakout high/low" is the high/low of **the last 15 minutes**,
and "VWAP" is a **rolling 120-minute VWAP**, not the session-anchored VWAP. Both
recompute every bar and both hug price, so:

- `last_close > orb_high` fires constantly → one of the three "momentum" votes is
  almost always true,
- rolling VWAP produces far more crossovers than real session VWAP.

Evidence: on 2026-11-04 the detector produced **1,101 directional signals** across
7 tickers × 360 bars = 2,520 evaluations — i.e. **44% of all ticker-bars generated a
"BULLISH_MOMENTUM" or "BEARISH_MOMENTUM" signal**. A genuine ORB + VWAP + RSI filter
should fire on perhaps 5–15% of bars. This is signal spam, and it is the direct cause
of the 4-trades-per-day churn.

---

### 🟠 #6 — The book is one correlated bet wearing four costumes

Every session opens SPY, QQQ, NVDA and AAPL **simultaneously** (~$9,600 of gross
exposure). Intraday, SPY/QQQ/NVDA/AAPL are 0.7–0.95 correlated, so this is one
directional bet at 4× size, not a diversified portfolio.

Worse, **14 of the 17 sessions held longs and shorts at the same time** (e.g.
2026-10-16: QQQ short, SPY short, AAPL short, NVDA long) — self-hedging, double
churn, no edge. The "portfolio direction lock"
(`app/services/stream.py:233`, `replay_test.py:200`) is computed **once per bar,
before the ticker loop**, so it cannot block entries made later in the same bar. It is
structurally incapable of doing its job.

Per-ticker results confirm the concentration story:

| Ticker | P&L | n | TP | SL | Exp/trade |
|---|---|---|---|---|---|
| AAPL | **+$169.31** | 17 | 6 | 2 | +$9.96 |
| QQQ | −$50.46 | 17 | 2 | 10 | −$2.97 |
| NVDA | −$85.97 | 17 | 1 | 10 | −$5.06 |
| SPY | −$108.70 | 17 | 2 | 12 | −$6.39 |

One ticker carries the entire result; the three index/semi names lose −$245 combined.

---

### 🟠 #7 — The signal itself is a "buy the pop" vote with no context

`app/services/indicators.py:128–160` scores three binary conditions and fires on
`bias_score >= 2`:

1. `close > VWAP and ema9 > ema21`
2. `close > orb_high` (rolling — see #5)
3. `rsi > 55`

There is no volatility regime filter, no relative-strength ranking, no volume
confirmation, no requirement that the move be *early* rather than extended. Because
condition 3 is literally "RSI above 55 on a 1-minute chart", the entry systematically
buys after a 3-bar pop — which is precisely the population that mean-reverts.

Also `calculate_rsi` (`indicators.py:64`) does `rsi.fillna(50.0)`, so a stock with
**zero down-bars** (avg_loss = 0, RSI should be 100) is scored as **50 = neutral**.
Momentum is being thrown away in exactly the strongest cases.

---

### 🟡 #8 — The Gemini layer adds latency and no edge

From the 2026-11-04 run:

- 1,101 signals → 564 blocked by the midday filter, 63 by 2-bar confirmation,
  25 by macro bias, **280 vetoed by Gemini** → **4 trades opened**.
- All 4 approved trades lost money (−$77.87).
- Every Gemini thesis is the same template ("price above VWAP, EMA 9 > 21, macro
  bullish") — it is echoing your prompt back at you, not adding information.
- Each call took **6–11 seconds**, and it is `await`ed *inside* the tick loop
  (`app/services/stream.py:255`). With 7 tickers that is up to ~80 seconds per cycle
  during which **no open position is monitored at all** — the risk engine is frozen
  while you wait on an LLM.
- `rulebook.json`'s `min_risk_reward_ratio: 1.4` is never enforced against the LLM's
  output, and `confidence >= 78` is the only quality gate.

---

### 🟡 #9 — The replay harness does not measure your strategy

Even the −$75.82 number is not trustworthy:

| Issue | Where | Effect |
|---|---|---|
| **Cooldown uses wall-clock `datetime.utcnow()` while bars advance simulated time** | `risk_engine.py:87, 134` | 17 sessions × 390 bars replay in ~8 real seconds, so the 30-minute cooldown **never expires**. Only 4 trades/session are possible. **1,577 of 2,006 log lines (78%) are "In cooldown for 29 more mins" spam.** Your entire 68-trade sample is just "the first signal of each day". |
| Stop/target checked against the **bar close only** | `risk_engine.py:167–196` | A 1-minute bar's range is typically 1.5–3× its close-to-close move, so **stop-outs are systematically undercounted**. The true 50% stop rate is worse. |
| Stop fills assumed at the exact stop price | same | No gap/slippage through the stop. |
| `ffill()` / `bfill()` on prices | `replay_test.py:144–145` | Stale prices used for SL/TP checks. |
| Zero commission, zero spread, zero market impact | everywhere | With ~7 shares/trade the dollar cost is small here, but it is unmodelled and it flips marginal trades. |
| Live Gemini API called mid-backtest | `replay_test.py --with-llm` | Non-reproducible, 6–11 s per call, and it perturbs the loop timing. |
| `consecutive_signals` never reset per session | `stream.py:36`, `scheduler.py:22–33` | Stale carry-over can pass the 2-bar confirmation on the **first bar of a new day**. |
| `period="1d"` falls back to `period="5d"` `tail(120)` | `stream.py:127–140` | Silently mixes **previous sessions' bars** into "session VWAP", ORB and the confirmation count. |
| `prepost=True` in live, `prepost=False` in replay | `stream.py:127` vs `replay_test.py:117` | Live, pre-market bars pollute the ORB and session VWAP. |

**Statistical honesty:** 17 sessions with 4 correlated trades each is ~17 *independent*
observations. Mean −$4.46/day, sd $30.86, **t = −0.60** — the −$75.80 is not
statistically significant on its own. But the *mechanism* (profit factor 0.84, 16.2%
win rate against a 33.3% break-even, 78% of the log being cooldown spam) is a
structural defect that will not improve with more of the same data.

---

### 🟡 #10 — `rulebook.json` is dead configuration

`grep` confirms **nothing in the codebase reads `rulebook.json`**. These limits exist
only on paper:

- `max_total_open_positions: 10` — no aggregate exposure cap exists.
- `min_risk_reward_ratio: 1.4` — never validated.
- `min_dollar_volume: 25_000_000` — **there is no volume/liquidity filter at all**,
  which is why entries can happen in dead, wide-spread moments.
- `max_stop_distance_multiplier: 2.0` — contradicted by the code's own 2.2×.

---

## 3. Also worth knowing (from `journal.csv`, the V1 book)

19 trades, 14 closed, 5 still open, realized **+$7.36** on ~$3,700 of gross exposure.
18 of 19 were shorts. 8 stop losses vs 6 take profits. Median hold **44 hours**
(mean 55, max 238) — this is swing/position trading, not day trading, so V1 and V2
are not comparable strategies. It does confirm one pattern though: **short-biased
momentum entries with ~1.45–1.55 R:R also failed to clear break-even.**

---

## 4. What to fix, in order of expected impact

1. **Anchor ORB and VWAP to the session** (09:30 ET → now). Pass a `session_start`
   into `evaluate_intraday_technical_snapshot` and slice; stop using the 120-bar
   buffer's first 15 rows as "the opening range". Expect signal count to drop by
   ~70–90%.
2. **Re-derive the stop from volatility, not from a 0.55% floor.**
   `stop = max(3.0 × ATR(1m), 1.0 × ATR(5m), 0.8% × price)` and let the target be
   `≥ 2.5 × stop` **or** replace the fixed target with a trailing stop. Then
   *measure* the historical hit rate of the setup and refuse to trade any setup whose
   out-of-sample win rate is below its break-even.
3. **Fix position sizing so risk binds.** After widening the stop, `risk_shares` will
   fall below the capital cap and `MAX_RISK_PER_TRADE` will start doing something.
   Re-scale `MAX_DAILY_LOSS` to the real per-trade risk so the circuit breaker can
   actually fire.
4. **Add a time stop and breakeven management.** Exit if the trade hasn't reached
   +0.5R within N minutes; move the stop to entry at +1R. This converts the 34% of
   trades that currently drift to +0.14% at EOD into something meaningful.
5. **De-correlate the book.** Cluster SPY/QQQ (index) and NVDA/TSM/INTC/MU (semis)
   and cap aggregate risk per cluster; forbid mixed-direction books by re-evaluating
   the direction lock after each entry.
6. **Cut trade count.** The data says 1 good trade beats 4 correlated ones. Rank
   setups by a scored metric and take only the best 1–2 per day.
7. **Make the replay harness honest.** Inject a simulated clock into `risk_engine`
   (fixes the cooldown artifact and removes 78% of your log noise), check SL/TP
   against bar high/low, drop `ffill`/`bfill`, add spread + commission, reset
   `consecutive_signals` per session, and snapshot LLM responses instead of calling
   the live API. Then report **per-setup conditional hit rates**, not just net P&L.
8. **Wire up `rulebook.json`** — especially `min_dollar_volume` and
   `max_total_open_positions` — or delete it so it stops pretending to be risk
   management.
9. **Demote or delete the LLM gate.** Run it concurrently with a hard timeout so it
   never blocks `update_open_positions_and_risk`, and require it to output a
   *quantified* edge you can backtest. Right now it is a random 25% filter costing
   6–11 s per ticker per cycle.
10. **Fix the small correctness bugs:** `rsi.fillna(50.0)` should be 100 when
    `avg_loss == 0`; `calculate_atr` should seed with the 14-bar mean rather than the
    first true range; `calculate_position_size`'s `max(1, final_shares)` can breach
    `MAX_POSITION_VALUE` for stocks above $2,500.

---

## 5. Bottom line

The bot is not losing because of bad luck or bad data. It is losing because it
**enters on a 44%-of-bars signal that has no predictive power, risks 0.56% of price
on a 1-minute chart, aims for a mechanically-derived 1.12%, sizes every position at
26% of its intended risk, holds four correlated copies of the same bet, and lets a
third of its trades drift to a flat EOD close.** The −$75.82 over 17 sessions is the
symptom; the 16.2% win rate against a 33.3% break-even is the disease.

Fix the stop width, the ORB/VWAP anchoring, and the position sizing first — those
three changes account for the majority of the loss.

---

# PART 2 — What was actually changed (implemented)

All ten fixes from Part 1 have now been implemented on top of `master` V10
(which already contained your pullback filter in `indicators.py`). Verified with
35 logic checks plus an end-to-end replay run on a synthetic 3-session dataset.

> **Note on the numbers:** the synthetic dataset is a random walk, so its P&L is
> meaningless. What it proves is that the harness runs, that the simulated
> cooldown now expires on bar time (you can watch it count down and then re-enter
> the same ticker), and that the new summary reports exit breakdown, costs and
> per-signal conditional statistics. **Re-run `replay_csv.py` on your real
> `historical_1m_30d.csv` to get the real answer** — this sandbox has no access
> to Yahoo Finance.

## Files changed

| File | Change |
|---|---|
| `app/services/risk_config.py` | **NEW** — loads `rulebook.json` into a validated, typed config |
| `app/services/indicators.py` | Session-anchored VWAP/ORB, RSI + ATR correctness, liquidity metrics, setup scoring |
| `app/services/risk_engine.py` | Volatility stops, honest R:R gate, fixed sizing, time/breakeven stops, cluster caps, costs, injected clock |
| `app/services/buffer.py` | 120 → 400 bars (full session), pre-market filtering, session anchor |
| `app/services/stream.py` | Non-blocking LLM, ranked entry selection, per-entry gates, daily reset |
| `app/services/scheduler.py` | Full per-session state reset at market open |
| `app/services/gemini_agent.py` | Output validation, quantified-edge requirement, real timeout |
| `app/services/state.py` | `daily_trade_count`, `session_date`, `entries_blocked` |
| `app/api/routes.py` | `/risk/config`, `/risk/exposure`, `/stats/session`, `/session/reset-daily` |
| `replay_csv.py`, `replay_test.py` | Rewritten harnesses — see Fix 7 |
| `rulebook.json` | Corrected and extended; now actually read by the code |

## Fix-by-fix

### Fix 1 — ORB and VWAP anchored to the session ✅
`indicators.py` gained a `session_start` parameter threaded through
`calculate_vwap`, `calculate_orb` and `evaluate_intraday_technical_snapshot`.
`buffer.py` now holds **400** one-minute bars (a full RTH session) instead of
120, and drops pre-market bars on load, so the "opening range" really is
09:30–09:45 and VWAP really is session-anchored.

### Fix 2 — Volatility-derived stop and target ✅
`compute_stop_and_target()` is now the single source of truth:

```
stop = clamp(max(atr * 2.5, price * 0.8%), ceiling = price * 3.0%)
target = entry ± stop * 2.5
```

The 0.55% floor is gone (raised to 0.80%), the target is a configurable 2.5R
rather than a mechanical 2× the stop, and `open_position` now **rejects** any
trade whose realised reward:risk falls below `min_risk_reward_ratio`.

### Fix 3 — Position sizing ✅
`calculate_position_size` now logs which cap binds, and `RiskConfig.reconcile()`
raises `max_position_value` to at least `max_risk_per_trade / min_stop_pct` when
it is too low to express the risk budget. Measured: risk per trade went from
**$12.93 (26% of budget) to the full $50**. `MAX_DAILY_LOSS` was also rescaled
from $250 (unreachable) to $120, which the per-trade caps can actually reach.

### Fix 4 — Time stop and breakeven management ✅
`_manage_position()` tracks the best favourable excursion, moves the stop to
breakeven after +1R, and closes any trade still unresolved after
`max_hold_minutes` (45) with `TIME_STOP` / `BREAKEVEN_STOP`. This converts the
34% of trades that used to drift to a flat EOD close into something with an exit.

### Fix 5 — Correlation clusters ✅
New `TICKER_CLUSTERS` map (INDEX / SEMIS / MEGA_TECH). `open_position` enforces
`max_positions_per_cluster`, and the **direction lock is now enforced inside
`open_position`** rather than in the entry loops, so it holds on every path
(API, manual order, replay) — previously it was computed once per bar and could
not block same-bar entries.

### Fix 6 — Fewer, better trades ✅
`score_setup()` ranks candidates on VWAP proximity, RSI health, relative volume
and ATR regime. Both harnesses and `stream.py` now sort candidates by score and
take only the top N, subject to `max_trades_per_day`. Added the missing
`min_dollar_volume` and `min_relative_volume` liquidity gates.

### Fix 7 — Honest replay harness ✅
* **Injected clock** (`set_clock`) — cooldowns run on simulated bar time, so the
  30-minute cooldown actually expires. The old run produced 1,577 lines of
  cooldown spam out of 2,006; the new run shows `cooldown=39` as a clean
  histogram entry and re-enters the same ticker later in the session.
* **Intrabar stops** — `_resolve_exit` uses the bar's high/low, not just the close.
* **No manufactured prices** — incomplete bars are dropped from the day frame
  instead of `ffill`/`bfill`.
* **Costs** — commission + slippage charged on every round trip.
* **Per-signal conditional stats** — the summary now reports stop/target/time-stop
  counts, win rate on resolved trades, break-even win rate, expectancy, profit
  factor, and a per-signal table.
* **No live LLM by default** — `--llm` opts in.
* The misleading 5-day fallback download was removed.

### Fix 8 — `rulebook.json` is live ✅
Nothing read it before. `risk_config.py` loads it, validates it, applies
environment overrides (`RISK_MAX_DAILY_LOSS=...`), and `GET /api/risk/config`
shows exactly what is in force. Two keys were corrected: `atr_stop_multiplier`
1.8 → 2.5 (1.8 was *tighter* than the 2.2 the code actually used) and
`max_stop_distance_multiplier` → `max_stop_distance_pct` (an ATR-multiple ceiling
can never bind when the floor is also an ATR multiple).

### Fix 9 — LLM demoted ✅
Gemini now runs as a **background task** with an 8s deadline, so it can never
freeze `update_open_positions_and_risk` (previously up to ~80s per cycle with a
7-name watchlist). Its output is validated: stop on the correct side, at least
2.0× ATR away, reward:risk above the rulebook minimum, and a **quantified
`edge`** field — no number, no trade. HTTP timeout reduced 12s → 6s.

### Fix 10 — Correctness bugs ✅
* `calculate_rsi` returns **100** with zero down-bars (was collapsing to 50,
  discarding the strongest momentum reading).
* `calculate_atr` seeds with the **mean of the first 14 true ranges** per Wilder,
  instead of the first bar's TR, and pads the warm-up window.
* `calculate_position_size` returns **0** rather than 1 share when the position
  cap cannot be met, so a >$6,250 position can no longer slip through.
* `init_db()` creates its data directory on demand.
* `from typing import Dict, List, Any, Any` duplicate-import typo in `stream.py`.

## Verification performed here

* All 12 modules import cleanly.
* **35/35 logic checks pass** — stop geometry, sizing, session-anchored VWAP/ORB,
  RSI/ATR edge cases, injected clock, cooldown expiry, cluster cap, direction
  lock, R:R rejection, intrabar stop, time stop, breakeven arming, cost
  accounting, risk-score gate.
* `replay_csv.py` runs end-to-end on a synthetic 3-session dataset and emits the
  new summary.

## What still needs your data

1. **Re-run the replay on real data** and compare against the old −$75.82 /
   16.2% win rate. The break-even win rate at the new 2.5R shape is **28.6%**
   (was 33.3%) — but only if the wider stops actually reduce the 50% stop-out
   rate, which is an empirical question this sandbox cannot answer.
2. **Set `equity`** in `rulebook.json` to your real account size — every
   percentage-based cap derives from it. $25,000 is a placeholder.
3. **Tune `max_hold_minutes`** — 45 is a guess; the right value depends on how
   long your setups actually take to work.
4. **Watch the first live sessions.** The new engine will take *fewer* trades
   (ranked, capped at 6/day, 1 per cluster) and hold them longer.

---

# PART 3 — Month-scale backtesting without the LLM

Added after the two regressions (short Gemini timeout, opening-spike-biased
`relative_volume`) were fixed. The question was: *can we test a full month of
1-minute data with zero Gemini API calls?*

`replay_csv.py` already replayed multi-week datasets with the LLM off, but it
was not usable for this at scale, and it had a reporting defect that would have
misled any month-long run.

## Four bugs found while building it

**1. The summary described the last session, not the run.**
`session_stats` is reset every session (correctly — it is per-day state), but
`_log_summary` read it *after* the loop. A 21-session run reported
**67 trades and +$803.96**, while its own exit breakdown described
**2 trades and +$91.64**. The win rate, profit factor and per-signal tables
were all single-day numbers presented as month-wide ones. Fixed by folding each
session into a running total, and by reading per-ticker/per-action/per-exit
tables from the trade journal (`close_position` already persists them) rather
than from a second bookkeeping path that could drift.

**2. A backtest wrote into the live trade journal.**
`init_db()` and `close_position` both use `settings.DATABASE_URL`, which in the
container is the same SQLite file the running engine uses. Every replay
therefore appended fabricated trades to real trading history, and every
subsequent report was wrong. Replays now default to a private
`backtest_trades.db` next to the CSV (`--db` to override).

**3. Parameter overrides leaked between sweep runs.**
`get_risk_config()` caches a process-wide singleton. A sweep reuses worker
processes, so the second configuration a worker saw inherited the first one's
values — `hold_20m` was coming back with `atr_stop_multiplier=2.0` left over
from the previous task. Silently produced chimeric results. Fixed with
`get_risk_config(reload=True)` at the top of every run.

**4. One bad day paused the rest of the month.**
`check_daily_circuit_breaker()` sets `system_state.is_active = False`, and
nothing in the replay ever set it back. Once a session breached
`max_daily_loss`, every later session ran with the system flagged paused. The
replay's entry path does not read `is_active`, so this did not corrupt results
— but it made the "System paused" log a lie and left a live trap for anyone who
later wires the replay through the real signal loop. Now restored per session.

## What was added

| File | Purpose |
|---|---|
| `backtest_month.py` | Batch runner: many configurations, one dataset, no LLM |
| `replay_csv.py` | Now returns a result dict; `--db`, `--from`, `--to`, `--quiet` |
| `download_dataset.py` | `--out` so data can be fetched outside the container |

`backtest_month.py`:

* **One subprocess per configuration.** The engine keeps session state in
  module globals (`system_state`, `ticker_cooldowns`, `consecutive_signals`,
  the `RiskConfig` singleton). Running two configurations in one process leaks
  state between them; a fresh process per run is the only cheap guarantee.
* **In-sample / out-of-sample split, always on.** A sweep of 19 configs over one
  month will always find a winner by chance. `--split 0.6` replays the first 60%
  of sessions and the last 40% separately, and ranks on what each configuration
  did on dates it was not chosen on. Configs that are profitable in-sample and
  losing out-of-sample are called out explicitly.
* **19 named sweeps** covering the levers the profitability work actually moved:
  stop geometry (`atr_2.0`, `minstop_0.004`), target (`rr_2.0`, `rr_3.5`), trade
  management (`hold_20m`, `no_breakeven`), sizing (`risk_25`, `risk_100`),
  breadth (`pos_2`, `pos_5`, `cluster_3`, `trades_12`), the liquidity gate
  (`relvol_off`, `relvol_1.2`) and costs (`costs_2x`).
* **Ad-hoc overrides** with `--set target_rr=3.0,max_hold_minutes=30`, so a
  parameter can be tested without editing `rulebook.json`.
* **Persisted reports**: `results.json`, `summary_in_sample.csv`,
  `summary_out_of_sample.csv`, `daily_pnl.csv` under `data/backtests/<label>/`.
  The JSON holds numbers, not formatted strings, so it can be loaded and
  compared programmatically.

## Progress, and what happens if you abort

A full sweep is 38 runs on 15 sessions and takes **20–60 minutes** on a NAS,
during which nothing is printed until the first run lands — which read as a
hang. Three things now make that legible:

* `>> start <config> [phase]` is printed the moment a worker picks up a task,
  so dispatch is visible immediately.
* A heartbeat line every 30 s: `... 7/38 runs finished, 12.4 min elapsed`.
* `results.json` is written **before** the first run and rewritten after every
  one, so the file can be watched for progress and a partial sweep is never
  lost. Ctrl-C exits immediately (no waiting for in-flight workers) with
  `Partial results: <dir>/results.json (n/38 runs).` rather than a traceback.

## Measured on a synthetic 1-month dataset

4 tickers × 21 sessions × 390 bars, U-shaped volume, mean-reverting price:

* **133 s per configuration**, 12 runs across 4 workers in 433 s wall clock.
* Sweeping 6 configurations produced an overfit warning on 2 of them — the
  machinery works and is willing to say "this one is noise".
* `pos_5` correctly came back identical to `baseline`: with 4 tickers spread
  across 3 correlation clusters and `max_positions_per_cluster = 1`, the cluster
  cap binds before `max_total_open_positions`, so raising the latter changes
  nothing. That is the de-correlation fix doing its job.

## How to run it on the QNAP

```bash
# 1. fetch a month of 1m data (Yahoo caps 1m at ~30 days)
docker compose exec trading-desk-v2 python3 download_dataset.py --days 30

# 2. single run, whole month, no LLM
python3 replay_csv.py --csv data/historical_1m_30d.csv

# 3. sweep everything, ranked out-of-sample
python3 backtest_month.py --csv data/historical_1m_30d.csv --sweep all --jobs 4 --split 0.6

# 4. one specific question
python3 backtest_month.py --csv data/historical_1m_30d.csv --set target_rr=3.0 --split 0
```

Every number these produce is deterministic and costs nothing in API quota.

## What this still cannot tell you

The synthetic dataset above is a random walk with a U-shaped volume profile. It
proves the *machinery* is correct — the aggregation, the split, the isolation,
the override handling — and it produced a tidy +$803.96 that means absolutely
nothing about the market. Only real Yahoo data answers the economic question,
and that has to be fetched from the QNAP.

---

# PART 4 — Real-data results, and what they actually say

First run on real Yahoo data (15 sessions, 2026-10-19 → 2026-11-06, 77 trades).
This replaces the synthetic numbers in Part 3, which proved the machinery and
nothing else.

## Headline

| | |
|---|---|
| Net P&L | **−$367.14** |
| Gross P&L (before costs) | −$169.62 |
| Transaction costs | **$197.52** |
| Expectancy | −$4.77/trade |
| Profitable sessions | 5/15 |
| Max drawdown | $443.97 (1.78%) |

## The finding that matters: there is no edge to tune

**Costs exceed the gross loss.** $2.57/trade of cost against −$2.20/trade of
gross P&L. Strip the costs out entirely and the strategy still loses — just
more slowly.

Narrower still: the 65 time-stopped trades produced **−$27.67 gross over 65
trades, i.e. −$0.43/trade**. That is a coin flip. The entire gross loss comes
from the **5 stop-outs at −$51.58 each (−$257.89)**. Everything else washes out
and then pays $2.57 to find that out.

## The 2.5R target is unreachable

**Take-profit was hit 0 times in 77 trades**, and 0 times across all 19 sweep
configurations. Every stop sits at exactly 0.800% of price (`min_stop_pct`), so
the target is exactly 2.000% — and a 2% move in 45 minutes on SPY/QQQ/NVDA does
not happen. 84.4% of trades exit on the time stop.

This also means `atr_stop_multiplier` is **dead configuration**: ATR × 2.5 on a
1-minute chart is well under the 0.8% floor, so the floor binds on all 77
trades. Raising or lowering the multiplier changes nothing — confirmed by
`atr_2.0` and `atr_3.0` returning results identical to baseline.

## Which parameters do anything

Six of the nineteen sweeps returned numbers **identical to baseline** —
`atr_2.0`, `atr_3.0`, `rr_2.0`, `rr_3.5`, `no_breakeven`, `pos_5`. Those levers
are non-binding, so tuning them is motion without progress.

Of the ones that do bind, out-of-sample:

| config | net | note |
|---|---|---|
| `relvol_1.2` | **+$3.18** | 7 trades instead of 26 — the only non-losing run, and 7 trades is noise |
| `risk_25` | −$127.96 | half the risk, half the loss. No edge created. |
| `minstop_0.015` | −$152.16 | wider stops cut stop-outs from 4 to 1 |
| baseline | −$256.28 | |
| `hold_90m` | −$334.87 | holding longer converts time stops into stop losses |
| `risk_100` | −$395.94 | |
| `costs_2x` | −$323.11 | costs are already the binding constraint |

**18 of 19 configurations lose money.** When nothing in the parameter space
works, the problem is the entry signal, not the parameters.

## Bug found and fixed: BREAKEVEN_STOP was a mislabelled time stop

`_manage_position()` labelled the time-stop exit `BREAKEVEN_STOP` whenever the
breakeven stop had been armed, while still exiting at the market price. Because
`_resolve_exit()` can only return `STOP_LOSS_HIT` or `TAKE_PROFIT_HIT`, that
label could only ever have come from the time-stop branch.

The report showed three "breakeven stops" at **+$36.34 average** — a breakeven
exit should be roughly flat. They were time stops that happened to still be in
profit. Corrected: TIME_STOP 68 trades −$85.38, BREAKEVEN_STOP 0 trades.

Side effect: the breakeven mechanism never actually produced an exit in this
dataset, so its value is **untested**, not proven.

## What this means

The strategy is a coin flip that pays $2.57 to play. To break even it needs a
gross edge of at least $2.57/trade and currently has −$2.20.

The lever that actually moved the number was **trading less** (`relvol_1.2`,
7 trades) — which is what you would expect if the entry signal has no
predictive power at a 45-minute horizon on 1-minute momentum. Widening stops
(`minstop_0.015`) and halving risk (`risk_25`) both reduce the damage without
creating edge.

Before any further parameter work, the question to answer is whether
`BULLISH_MOMENTUM` / `BEARISH_MOMENTUM` with 2-bar confirmation predicts
anything at all. Suggested test: run the replay with entries taken at random
bars instead of on signal, and compare. If random performs the same, the signal
is the problem and no amount of stop tuning will fix it.