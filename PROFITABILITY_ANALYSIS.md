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
