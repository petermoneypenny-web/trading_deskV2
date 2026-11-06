# -*- coding: utf-8 -*-
import csv
import json
import logging
import math
import os
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types
from google.genai.errors import APIError
import pandas as pd
from pydantic import BaseModel, Field
import yfinance as yf

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.8-flash"


def is_us_market_open() -> bool:
    """Checks if US stock exchanges are open (Mon-Fri 09:30-16:00 ET)."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    return dtime(9, 30) <= now_et.time() < dtime(16, 0)


class AnalystReport(BaseModel):
    ticker: str
    catalyst_summary: str = Field(description="Summary of primary catalyst from live news")
    sentiment_score: int = Field(description="Score between -100 (bearish) and +100 (bullish)")
    sentiment_drivers: List[str] = Field(description="Key drivers influencing sentiment")
    key_support: float = Field(description="Key technical support level based on structure")
    key_resistance: float = Field(description="Key technical resistance level based on structure")
    overall_bias: str = Field(description="BULLISH, BEARISH, or NEUTRAL")


class SkepticEvaluation(BaseModel):
    decision: str = Field(description="PASS, SHRINK, or KILL")
    risk_score_1_to_10: int = Field(description="Risk assessment from 1 (safest) to 10 (highest risk)")
    kill_reason: Optional[str] = Field(default="", description="Fatal flaw if KILL, else empty")
    adversarial_take: str = Field(description="Direct, blunt reason why this setup might fail")


class TradeDecisionCard(BaseModel):
    ticker: str
    action: str = Field(description="BUY, SELL, or PASS")
    entry_price: float = Field(description="Recommended entry limit/market price")
    stop_loss: float = Field(description="Mandatory hard stop-loss positioned beyond structural support/resistance")
    take_profit: float = Field(description="Mandatory target price")
    risk_reward_ratio: float = Field(description="Calculated Risk:Reward ratio")
    position_size_usd: float = Field(description="Capital allocated in USD")
    top_3_reasons: List[str] = Field(description="3 clear bullet points for the decision")
    invalidation_criteria: str = Field(description="Specific technical or fundamental condition that invalidates thesis")


class UnifiedEvaluation(BaseModel):
    catalyst_summary: str = Field(description="Summary of primary catalyst from live news")
    sentiment_score: int = Field(description="Score between -100 (bearish) and +100 (bullish)")
    sentiment_drivers: List[str] = Field(description="Key drivers influencing sentiment")
    key_support: float = Field(description="Key technical support level based on structure")
    key_resistance: float = Field(description="Key technical resistance level based on structure")
    overall_bias: str = Field(description="BULLISH, BEARISH, or NEUTRAL")

    skeptic_decision: str = Field(description="PASS, SHRINK, or KILL")
    risk_score_1_to_10: int = Field(description="Risk assessment from 1 (safest) to 10 (highest risk)")
    kill_reason: Optional[str] = Field(default="", description="Fatal flaw if KILL, else empty")
    adversarial_take: str = Field(description="Direct, blunt reason why this setup might fail")

    action: str = Field(description="BUY, SELL, or PASS")
    entry_price: float = Field(description="Recommended entry limit/market price")
    stop_loss: float = Field(description="Mandatory hard stop-loss positioned beyond structural support/resistance")
    take_profit: float = Field(description="Mandatory target price")
    risk_reward_ratio: float = Field(description="Calculated Risk:Reward ratio")
    position_size_usd: float = Field(description="Capital allocated in USD")
    top_3_reasons: List[str] = Field(description="3 clear bullet points for the decision")
    invalidation_criteria: str = Field(description="Specific technical or fundamental condition that invalidates thesis")


class TradingAgentHub:
    def __init__(self, account_size: float = 1000.0, risk_per_trade_pct: float = 1.0):
        self.account_size = account_size
        self.risk_per_trade_pct = risk_per_trade_pct
        self.base_dir = Path(__file__).resolve().parent
        self.project_root = self.base_dir.parent
        self.journal_file = str(self._resolve_file_path("journal.csv"))
        self.rulebook_path = str(self._resolve_file_path("rulebook.json"))
        self.rulebook = self._load_rulebook()
        self._ensure_journal_exists()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ [WARNING] GEMINI_API_KEY environment variable is not set! LLM analysis steps will require key.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def _resolve_file_path(self, filename: str) -> Path:
        candidates = [
            Path(filename),
            self.project_root / filename,
            self.base_dir / filename,
            Path("/app") / filename,
            Path("/app/tools") / filename,
            Path("/share/Container/trading_desk") / filename,
        ]
        for p in candidates:
            if p.is_file():
                return p
        return self.project_root / filename

    def _load_rulebook(self) -> dict:
        default_rules = {
            "min_risk_reward_ratio": 1.4,
            "max_risk_score": 6,
            "min_stock_price": 10.0,
            "min_dollar_volume": 25_000_000.0,
            "max_portfolio_risk_pct": 3.0,
            "max_open_trades_per_ticker": 1,
            "max_total_open_positions": 10,
            "atr_stop_multiplier": 1.8,
            "max_stop_distance_multiplier": 2.0
        }
        if os.path.exists(self.rulebook_path):
            try:
                with open(self.rulebook_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    default_rules.update(data)
                    return default_rules
            except Exception as e:
                print(f"⚠️ Error reading rulebook at {self.rulebook_path}: {e}")
        return default_rules

    def _ensure_journal_exists(self):
        if not os.path.exists(self.journal_file):
            try:
                with open(self.journal_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "trade_id", "timestamp", "ticker", "action", "entry_price",
                        "stop_loss", "take_profit", "risk_reward_ratio", "position_size_usd",
                        "status", "exit_price", "exit_timestamp", "realized_pnl", "exit_reason",
                        "top_reasons", "invalidation_criteria"
                    ])
            except Exception as e:
                print(f"⚠️ Error creating journal file at {self.journal_file}: {e}")

    def get_journal_records(self) -> List[dict]:
        self._ensure_journal_exists()
        records = []
        try:
            with open(self.journal_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        except Exception as e:
            print(f"Error reading journal: {e}")
        return records

    def get_active_tickers(self) -> List[str]:
        records = self.get_journal_records()
        return [
            r["ticker"] for r in records
            if (r.get("status") or "").upper() in ("OPEN", "PENDING")
        ]

    def get_live_prices(self, tickers: List[str]) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        unique_tickers = list(set([t.strip().upper() for t in tickers if t]))
        if not unique_tickers:
            return prices

        for ticker in unique_tickers:
            try:
                stock = yf.Ticker(ticker)
                curr_price = getattr(stock.fast_info, "last_price", None)
                if curr_price is None or math.isnan(curr_price):
                    hist = stock.history(period="2d", interval="1d")
                    if not hist.empty:
                        curr_price = float(hist["Close"].iloc[-1])
                if curr_price is not None and not math.isnan(curr_price):
                    prices[ticker] = round(float(curr_price), 2)
            except Exception as e:
                print(f"⚠️ Error fetching current price for {ticker}: {e}")
        return prices

    def log_trade(self, card_dict: dict, force_market_execution: bool = True) -> Optional[str]:
        self._ensure_journal_exists()
        active_tickers = self.get_active_tickers()
        ticker = card_dict.get("ticker", "").strip().upper()

        max_per_ticker = self.rulebook.get("max_open_trades_per_ticker", 1)
        max_total = self.rulebook.get("max_total_open_positions", 10)

        if active_tickers.count(ticker) >= max_per_ticker:
            print(f"🚫 [REJECTED] Active order already open for {ticker}.")
            return None

        if len(active_tickers) >= max_total:
            print(f"🚫 [REJECTED] Max portfolio capacity reached ({max_total} positions).")
            return None

        trade_id = f"TRD-{int(time.time())}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reasons_joined = " | ".join(card_dict.get("top_3_reasons", []))

        live_prices = self.get_live_prices([ticker])
        current_price = live_prices.get(ticker)
        if current_price is None or current_price <= 0:
            current_price = float(card_dict.get("current_price", card_dict.get("entry_price", 0.0)))

        action = card_dict.get("action", "BUY").upper()
        orig_entry = float(card_dict.get("entry_price", current_price))
        orig_stop = float(card_dict.get("stop_loss", 0.0))
        orig_target = float(card_dict.get("take_profit", 0.0))

        if force_market_execution:
            final_entry = current_price
            initial_status = "OPEN"

            if orig_entry > 0 and orig_stop > 0 and orig_target > 0:
                sl_pct = abs(orig_entry - orig_stop) / orig_entry
                tp_pct = abs(orig_target - orig_entry) / orig_entry

                if action == "BUY":
                    stop_loss = round(final_entry * (1.0 - sl_pct), 2)
                    take_profit = round(final_entry * (1.0 + tp_pct), 2)
                else:
                    stop_loss = round(final_entry * (1.0 + sl_pct), 2)
                    take_profit = round(final_entry * (1.0 - tp_pct), 2)
            else:
                stop_loss = round(final_entry * 0.95, 2) if action == "BUY" else round(final_entry * 1.05, 2)
                take_profit = round(final_entry * 1.10, 2) if action == "BUY" else round(final_entry * 0.90, 2)

            risk = abs(final_entry - stop_loss)
            reward = abs(take_profit - final_entry)
            rr_ratio = round(reward / risk, 2) if risk > 0 else 1.4

            max_risk_usd = self.account_size * (self.risk_per_trade_pct / 100.0)
            if risk > 0:
                position_size_usd = round((max_risk_usd / risk) * final_entry, 2)
            else:
                position_size_usd = float(card_dict.get("position_size_usd", 100.0))
        else:
            final_entry = orig_entry
            stop_loss = orig_stop
            take_profit = orig_target
            rr_ratio = float(card_dict.get("risk_reward_ratio", 1.4))
            position_size_usd = float(card_dict.get("position_size_usd", 100.0))
            is_pending = (action == "BUY" and final_entry < current_price * 0.998) or \
                         (action == "SELL" and final_entry > current_price * 1.002)
            initial_status = "PENDING" if is_pending else "OPEN"

        with open(self.journal_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_id,
                now_str,
                ticker,
                action,
                final_entry,
                stop_loss,
                take_profit,
                rr_ratio,
                position_size_usd,
                initial_status,
                "", "", "", "",
                reasons_joined,
                card_dict.get("invalidation_criteria", "")
            ])
        print(f"✅ [ORDER LOGGED] {ticker} {action} at ${final_entry:.2f} (Status: {initial_status})")
        return trade_id

    def close_position_now(self, identifier: str) -> Optional[dict]:
        self._ensure_journal_exists()
        trades = self.get_journal_records()
        closed_trade = None
        for t in trades:
            status = (t.get("status") or "").upper()
            is_match = (t.get("trade_id") == identifier or t.get("ticker") == identifier)
            if is_match and status in ("OPEN", "PENDING"):
                ticker = t.get("ticker")
                live_prices = self.get_live_prices([ticker])
                entry = float(t.get("entry_price", 0.0))
                curr_price = live_prices.get(ticker, entry)
                size_usd = float(t.get("position_size_usd", 0.0))
                action = t.get("action", "BUY").upper()
                shares = size_usd / entry if entry > 0 else 0.0
                pnl = (curr_price - entry) * shares if action == "BUY" else (entry - curr_price) * shares
                t["status"] = "CLOSED"
                t["exit_price"] = f"{curr_price:.2f}"
                t["exit_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t["realized_pnl"] = f"{pnl:.2f}"
                t["exit_reason"] = "MANUAL_SELL"
                closed_trade = t
                break
        if closed_trade:
            fieldnames = [
                "trade_id", "timestamp", "ticker", "action", "entry_price",
                "stop_loss", "take_profit", "risk_reward_ratio", "position_size_usd",
                "status", "exit_price", "exit_timestamp", "realized_pnl", "exit_reason",
                "top_reasons", "invalidation_criteria"
            ]
            with open(self.journal_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)
        return closed_trade

    def adjust_take_profit(self, identifier: str, reduction_pct: float = 0.01) -> Optional[dict]:
        self._ensure_journal_exists()
        trades = self.get_journal_records()
        updated_trade = None

        for t in trades:
            status = (t.get("status") or "").upper()
            is_match = (t.get("trade_id") == identifier or t.get("ticker") == identifier)
            if is_match and status in ("OPEN", "PENDING"):
                old_tp = float(t.get("take_profit", 0.0))
                entry = float(t.get("entry_price", 0.0))
                stop = float(t.get("stop_loss", 0.0))
                
                new_tp = round(old_tp * (1.0 - reduction_pct), 2)
                t["take_profit"] = str(new_tp)

                risk = abs(entry - stop)
                reward = abs(new_tp - entry)
                if risk > 0:
                    t["risk_reward_ratio"] = f"{round(reward / risk, 2)}"

                updated_trade = t
                break

        if updated_trade:
            fieldnames = [
                "trade_id", "timestamp", "ticker", "action", "entry_price",
                "stop_loss", "take_profit", "risk_reward_ratio", "position_size_usd",
                "status", "exit_price", "exit_timestamp", "realized_pnl", "exit_reason",
                "top_reasons", "invalidation_criteria"
            ]
            with open(self.journal_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)
            print(f"🎯 [TP Adjusted] {updated_trade.get('ticker')} Take-Profit lowered to ${updated_trade.get('take_profit')}")

        return updated_trade

    def check_open_positions(self, force: bool = False) -> List[dict]:
        self._ensure_journal_exists()
        trades = self.get_journal_records()
        active_trades = [
            t for t in trades 
            if (t.get("status") or "").upper() in ("OPEN", "PENDING")
        ]
        if not active_trades:
            return []

        active_tickers = [t.get("ticker") for t in active_trades if t.get("ticker")]
        live_prices = self.get_live_prices(active_tickers)

        closed_events = []
        updated = False

        for t in active_trades:
            ticker = t.get("ticker")
            curr_price = live_prices.get(ticker)
            if curr_price is None or math.isnan(curr_price):
                continue

            entry = float(t.get("entry_price", 0))
            stop = float(t.get("stop_loss", 0))
            target = float(t.get("take_profit", 0))
            size_usd = float(t.get("position_size_usd", 0))
            action = t.get("action", "BUY")
            status = t.get("status", "OPEN").upper()

            if status == "PENDING":
                filled = (action == "BUY" and curr_price <= entry) or (action == "SELL" and curr_price >= entry)
                if filled:
                    t["status"] = "OPEN"
                    updated = True
                    print(f"🎯 [ORDER FILLED] {ticker} filled at ${curr_price:.2f}")
                continue

            hit_stop = (action == "BUY" and curr_price <= stop) or (action == "SELL" and curr_price >= stop)
            hit_target = (action == "BUY" and curr_price >= target) or (action == "SELL" and curr_price <= target)

            if hit_stop or hit_target:
                reason = "TAKE_PROFIT" if hit_target else "STOP_LOSS"
                shares = size_usd / entry if entry > 0 else 0
                pnl = (curr_price - entry) * shares if action == "BUY" else (entry - curr_price) * shares

                t["status"] = "CLOSED"
                t["exit_price"] = f"{curr_price:.2f}"
                t["exit_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                t["realized_pnl"] = f"{pnl:.2f}"
                t["exit_reason"] = reason
                updated = True

                closed_events.append({
                    "ticker": ticker,
                    "reason": reason,
                    "price": curr_price,
                    "pnl": pnl
                })

        if updated:
            fieldnames = [
                "trade_id", "timestamp", "ticker", "action", "entry_price",
                "stop_loss", "take_profit", "risk_reward_ratio", "position_size_usd",
                "status", "exit_price", "exit_timestamp", "realized_pnl", "exit_reason",
                "top_reasons", "invalidation_criteria"
            ]
            with open(self.journal_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trades)

        return closed_events

    def _load_custom_watchlist(self) -> List[str]:
        candidate_paths = []
        env_path = os.environ.get("WATCHLIST_PATH")
        if env_path:
            candidate_paths.append(Path(env_path))

        candidate_paths.extend([
            Path("/share/Container/trading_desk/watchlist.csv"),
            Path("/app/watchlist.csv"),
            Path("/app/tools/watchlist.csv"),
            Path("watchlist.csv"),
            self.project_root / "watchlist.csv",
            self.base_dir / "watchlist.csv",
        ])

        target_path: Optional[Path] = None
        for p in candidate_paths:
            try:
                if p.is_file():
                    target_path = p
                    break
            except Exception:
                continue

        if not target_path:
            return []

        tickers: List[str] = []
        seen = set()
        try:
            with open(target_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    for col in row:
                        raw_val = col.strip().upper().replace('"', '').replace("'", "")
                        tokens = [t.strip() for t in raw_val.split(";") if t.strip()]
                        for token in tokens:
                            token = token.replace('.', '-')
                            if token and token not in ("TICKER", "SYMBOL") and not token.startswith("#"):
                                if token not in seen:
                                    seen.add(token)
                                    tickers.append(token)
        except Exception as e:
            print(f"⚠️ Error reading watchlist: {e}")
            return []

        return tickers

    def _get_universe_tickers(self, universe: str) -> List[str]:
        u = (universe or "TOP50_VOLUME").upper()
        if "CUSTOM" in u or "WATCHLIST" in u:
            custom_tickers = self._load_custom_watchlist()
            if custom_tickers:
                return custom_tickers

        if "NASDAQ" in u:
            return [
                "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST",
                "ASML", "AMD", "NFLX", "ADBE", "QCOM", "TXN", "INTU", "AMAT", "ISRG",
                "BKNG", "LRCX", "ADI", "MDLZ", "PANW", "REGN", "VRTX", "KLAC", "SNPS"
            ]
        elif "SP500" in u:
            return [
                "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO",
                "JPM", "TSLA", "UNH", "XOM", "V", "PG", "MA", "JNJ", "HD", "COST"
            ]
        else:
            return [
                "NVDA", "TSLA", "AMD", "AAPL", "AMZN", "PLTR", "META", "MSFT", "MARA",
                "COIN", "SOFI", "INTC", "GOOGL", "NFLX", "UBER", "AVGO", "HOOD", "SMCI"
            ]

    def scout_scan(self, universe: str = "TOP50_VOLUME", max_candidates: int = 8, force: bool = False) -> List[dict]:
        if not force and not is_us_market_open():
            now_et = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S %Z")
            print(f"⏸️ [Scout] US Market is closed (ET: {now_et}). Aborting scan.")
            return []

        active_tickers = set(self.get_active_tickers())
        tickers = [t for t in self._get_universe_tickers(universe) if t not in active_tickers]
        if not tickers:
            return []

        min_price = float(self.rulebook.get("min_stock_price", 10.0))
        min_dollar_vol = float(self.rulebook.get("min_dollar_volume", 25_000_000.0))
        evaluated = []

        batch_size = 100
        total_tickers = len(tickers)
        for i in range(0, total_tickers, batch_size):
            chunk = tickers[i:i + batch_size]
            try:
                data = yf.download(
                    tickers=chunk,
                    period="3mo",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False
                )
            except Exception as e:
                print(f"⚠️ Batch download error: {e}")
                continue

            if data is None or data.empty:
                continue

            is_multi = isinstance(data.columns, pd.MultiIndex)

            for ticker in chunk:
                try:
                    if is_multi:
                        if ticker in data.columns.levels[0]:
                            df = data[ticker].dropna()
                        elif data.columns.nlevels > 1 and ticker in data.columns.get_level_values(0):
                            df = data.xs(ticker, axis=1, level=0).dropna()
                        elif data.columns.nlevels > 1 and ticker in data.columns.get_level_values(-1):
                            df = data.xs(ticker, axis=1, level=-1).dropna()
                        else:
                            continue
                    else:
                        df = data.dropna()

                    if df.empty or len(df) < 50:
                        continue

                    high = df['High']
                    low = df['Low']
                    close = df['Close']
                    prev_close_series = close.shift(1)

                    tr1 = high - low
                    tr2 = (high - prev_close_series).abs()
                    tr3 = (low - prev_close_series).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr_14 = float(tr.rolling(14).mean().iloc[-1])

                    sma_50 = float(close.rolling(50).mean().iloc[-1])

                    curr_price = float(close.iloc[-1])
                    if curr_price < min_price or math.isnan(curr_price):
                        continue

                    prev_close = float(prev_close_series.iloc[-1])
                    if prev_close <= 0 or math.isnan(prev_close):
                        continue

                    volume = float(df['Volume'].iloc[-1])
                    dollar_vol = curr_price * volume
                    if dollar_vol < min_dollar_vol:
                        continue

                    pct_change = abs((curr_price - prev_close) / prev_close)
                    current_range = float(tr.iloc[-1])
                    expansion = (current_range / atr_14) if atr_14 > 0 else 1.0
                    momentum_score = pct_change * expansion

                    is_bullish_trend = curr_price >= sma_50
                    trend_label = "UPTREND" if is_bullish_trend else "DOWNTREND"

                    evaluated.append({
                        "ticker": ticker,
                        "current_price": round(curr_price, 2),
                        "volume": int(volume),
                        "previous_close": round(prev_close, 2),
                        "dollar_volume": round(dollar_vol, 2),
                        "pct_change": round(pct_change, 4),
                        "atr_14": round(atr_14, 2),
                        "sma_50": round(sma_50, 2),
                        "trend": trend_label,
                        "momentum_score": round(momentum_score, 4)
                    })
                except Exception:
                    continue

        if not evaluated:
            return []

        evaluated.sort(key=lambda x: x.get("momentum_score", x["dollar_volume"]), reverse=True)
        return evaluated[:max_candidates]

    def evaluate_candidate_unified(self, candidate: dict, max_retries: int = 4) -> Tuple[AnalystReport, SkepticEvaluation, TradeDecisionCard]:
        """Unified multi-agent evaluation in a single Gemini API call with rate-limit and backoff protection."""
        if not self.client:
            raise ValueError("Cannot run unified evaluation: GEMINI_API_KEY is not configured.")

        ticker = candidate.get("ticker")
        price = candidate.get("current_price", 0.0)
        atr = candidate.get("atr_14", 1.0)
        trend = candidate.get("trend", "UPTREND")
        sma50 = candidate.get("sma_50", price)

        atr_mult = float(self.rulebook.get("atr_stop_multiplier", 1.8))
        min_rr = float(self.rulebook.get("min_risk_reward_ratio", 1.4))
        max_stop_distance = round(atr * atr_mult, 2)
        max_loss = self.account_size * (self.risk_per_trade_pct / 100.0)

        prompt = f"""
You are the Executive Trading Committee consisting of:
1. Lead Market Analyst
2. Chief Risk Officer (Skeptic)
3. Quantitative Execution Strategist

Ticker: {ticker}
Current Price: ${price:.2f}
50-Day SMA: ${sma50:.2f}
Regime: {trend}
14-day ATR: ${atr:.2f}

DELIBERATION GUIDELINES:
- ANALYST: Analyze technical swing support and resistance levels aligned with {trend}. Actionable levels must be within ~1.5x to 1.8x ATR (${atr:.2f}) from current price.
- SKEPTIC: Evaluate the setup with extreme skepticism.
  * If Analyst bias contradicts macro {trend} (e.g. BUY in DOWNTREND), assign skeptic_decision: KILL and risk_score_1_to_10 >= 8.
  * Assign 'PASS' (Score 1-4) ONLY to textbook trend-continuation setups with clean consolidation.
  * Assign 'SHRINK' (Score 5-6) if price is slightly extended, near major resistance, or has conflicting signals.
  * Assign 'KILL' (Score 7-10) to overextended runs, chop, earnings risk, or ambiguous levels.
- EXECUTION STRATEGIST:
  * Hard stop-loss anchored beyond key support/resistance. The stop distance (|entry_price - stop_loss|) MUST NOT exceed ${max_stop_distance:.2f}.
  * Account Size: ${self.account_size}, Max Risk: {self.risk_per_trade_pct}%. Max loss: ${max_loss:.2f}.
    Position Size = (Max Risk $ / abs(entry_price - stop_loss)) * entry_price.
  * Take-Profit target must achieve a Risk:Reward ratio >= {min_rr}.
  * If Skeptic decision is KILL, you MUST output action 'PASS'.
    If Skeptic decision is PASS or SHRINK, output 'BUY' or 'SELL' ONLY IF R:R >= {min_rr} and stop distance <= ${max_stop_distance:.2f}. Otherwise output 'PASS'.
"""
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=UnifiedEvaluation,
                        temperature=0.2
                    )
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "resource" in err_str or "quota" in err_str
                if attempt < max_retries:
                    wait_time = 4 * attempt if is_rate_limit else 2 * attempt
                    time.sleep(wait_time)
                else:
                    raise e

        u = UnifiedEvaluation.model_validate_json(response.text)

        report = AnalystReport(
            ticker=ticker,
            catalyst_summary=u.catalyst_summary,
            sentiment_score=u.sentiment_score,
            sentiment_drivers=u.sentiment_drivers,
            key_support=u.key_support,
            key_resistance=u.key_resistance,
            overall_bias=u.overall_bias
        )

        skeptic = SkepticEvaluation(
            decision=u.skeptic_decision,
            risk_score_1_to_10=u.risk_score_1_to_10,
            kill_reason=u.kill_reason,
            adversarial_take=u.adversarial_take
        )

        card = TradeDecisionCard(
            ticker=ticker,
            action=u.action,
            entry_price=u.entry_price,
            stop_loss=u.stop_loss,
            take_profit=u.take_profit,
            risk_reward_ratio=u.risk_reward_ratio,
            position_size_usd=u.position_size_usd,
            top_3_reasons=u.top_3_reasons,
            invalidation_criteria=u.invalidation_criteria
        )

        return report, skeptic, card

    def run_analysts(self, candidate: dict) -> AnalystReport:
        if not self.client:
            raise ValueError("Cannot run analyst: GEMINI_API_KEY is not configured.")

        ticker = candidate.get("ticker")
        price = candidate.get("current_price", 0.0)
        atr = candidate.get("atr_14", 0.0)
        trend = candidate.get("trend", "UPTREND")
        sma50 = candidate.get("sma_50", price)

        prompt = f"""
You are the Lead Market Analyst.
Analyze ticker {ticker} trading at ${price:.2f} (50-Day Trend SMA: ${sma50:.2f}, Regime: {trend}, 14-day ATR: ${atr:.2f}).
Identify technical swing support and resistance levels suitable for a multi-day position aligned with {trend}.
Focus on actionable levels within roughly 1.5x to 1.8x of the 14-day ATR (${atr:.2f}) from current price.
Avoid selecting levels that are excessively far away (e.g. > 1.8x ATR).
"""
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalystReport,
                temperature=0.2
            )
        )
        return AnalystReport.model_validate_json(response.text)

    def run_skeptic(self, candidate: dict, report: AnalystReport) -> SkepticEvaluation:
        if not self.client:
            raise ValueError("Cannot run skeptic: GEMINI_API_KEY is not configured.")

        ticker = candidate.get("ticker")
        price = candidate.get("current_price", 0.0)
        atr = candidate.get("atr_14", 0.0)
        trend = candidate.get("trend", "UPTREND")
        sma50 = candidate.get("sma_50", price)

        prompt = f"""
You are an exceptionally conservative Chief Risk Officer. Your mandate is capital preservation.
Candidate: {ticker} (Current Price: ${price:.2f}, 50 SMA: ${sma50:.2f}, Regime: {trend}, Daily ATR: ${atr:.2f})
Analyst Bias: {report.overall_bias} (Sentiment: {report.sentiment_score})
Support: {report.key_support} | Resistance: {report.key_resistance}

Evaluate the setup with extreme skepticism:
- If Analyst Bias contradicts the macro {trend} (e.g. trying to BUY a stock in a DOWNTREND), assign decision: KILL and risk score >= 8.
- Assign 'PASS' (Score 1-4) ONLY to textbook trend-continuation setups with clean consolidation, clear catalysts, and tight risk boundaries.
- Assign 'SHRINK' (Score 5-6) if the price is even slightly extended from the 50 SMA, approaching major overhead resistance, or has conflicting sentiment.
- Assign 'KILL' (Score 7-10) to any setup chasing an overextended green/red streak, chop, earnings risk, or ambiguous support/resistance. When in doubt, KILL.
"""
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SkepticEvaluation,
                temperature=0.2
            )
        )
        return SkepticEvaluation.model_validate_json(response.text)

    def generate_decision_card(self, candidate: dict, report: AnalystReport, skeptic: SkepticEvaluation) -> Optional[TradeDecisionCard]:
        if not self.client:
            raise ValueError("Cannot generate decision card: GEMINI_API_KEY is not configured.")

        ticker = candidate.get("ticker")
        price = candidate.get("current_price", 0.0)
        atr = candidate.get("atr_14", 1.0)
        
        atr_mult = float(self.rulebook.get("atr_stop_multiplier", 1.8))
        min_rr = float(self.rulebook.get("min_risk_reward_ratio", 1.4))
        max_stop_distance = round(atr * atr_mult, 2)

        prompt = f"""
Synthesize the technical trade card for {ticker} at ${price:.2f} (14-day ATR: ${atr:.2f}).
Analyst Report: {report.model_dump_json()}
Skeptic Critique: {skeptic.model_dump_json()}

STRICT RISK & EXECUTION RULES:
1. Define a hard stop-loss anchored beyond key support/resistance, but keep the stop distance (|entry_price - stop_loss|) within ${max_stop_distance:.2f} (approx 1.5x to 1.8x ATR).
2. The stop distance MUST NOT exceed ${max_stop_distance:.2f}.
3. Account Size: ${self.account_size}, Max Risk: {self.risk_per_trade_pct}%. Max loss is ${self.account_size * (self.risk_per_trade_pct / 100):.2f}.
   Position Size = Max Risk $ / abs(entry_price - stop_loss) * entry_price.
4. Calculate realistic Take-Profit based on technical resistance/support to target a Risk:Reward ratio >= {min_rr}.
5. If Skeptic decision is KILL, you MUST output action 'PASS'.
   If Skeptic decision is PASS or SHRINK, output 'BUY' or 'SELL' if structural R:R >= {min_rr} and stop distance <= ${max_stop_distance:.2f}. Otherwise output 'PASS'.
"""
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TradeDecisionCard,
                temperature=0.2
            )
        )
        return TradeDecisionCard.model_validate_json(response.text)
