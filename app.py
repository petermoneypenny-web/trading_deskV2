import os
import io
import asyncio
import json
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, Response, FileResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from tools.trading_agent import TradingAgentHub

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

app = FastAPI(title="Engineering Efficiency Hub")

@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="48" fill="#10b981" stroke="#059669" stroke-width="4"/>
  <text x="50%" y="54%" font-size="62" font-weight="900" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" fill="#ffffff" dominant-baseline="middle" text-anchor="middle">$</text>
</svg>"""
    return Response(content=svg_icon, media_type="image/svg+xml")

trading_hub = TradingAgentHub(account_size=1000.0)

scheduler = BackgroundScheduler()

SETTINGS_PATH = "hub_settings.json"
SKEPTIC_STATS_PATH = "skeptic_stats.json"

def load_settings() -> dict:
    default_settings = {
        "ntfy_topic": "cm-hub-desk-8492",
        "auto_execute": True
    }
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r") as f:
                saved = json.load(f)
                default_settings.update(saved)
                return default_settings
        except Exception:
            pass
    return default_settings

def save_settings(settings: dict):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

current_settings = load_settings()

scheduler_state = {
    "is_running": False,
    "interval_hours": 4,
    "universe": "TOP50_VOLUME",
    "last_run": None
}

latest_scan_cache = {
    "timestamp": None,
    "universe": None,
    "results": []
}

def load_skeptic_stats() -> dict:
    default_stats = {
        "total_evaluations": 0,
        "decisions": {"PASS": 0, "SHRINK": 0, "KILL": 0},
        "score_distribution": {str(i): 0 for i in range(1, 11)},
        "recent_evaluations": []
    }
    if os.path.exists(SKEPTIC_STATS_PATH):
        try:
            with open(SKEPTIC_STATS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k in ["decisions", "score_distribution"]:
                    if k in saved:
                        default_stats[k].update(saved[k])
                default_stats["total_evaluations"] = saved.get("total_evaluations", 0)
                default_stats["recent_evaluations"] = saved.get("recent_evaluations", [])
                return default_stats
        except Exception:
            pass
    return default_stats

def record_skeptic_evaluation(ticker: str, score: int, decision: str, adversarial_take: str):
    stats = load_skeptic_stats()
    stats["total_evaluations"] += 1
    dec_key = (decision or "KILL").upper()
    if dec_key not in stats["decisions"]:
        stats["decisions"][dec_key] = 0
    stats["decisions"][dec_key] += 1

    clamped_score = max(1, min(10, int(score)))
    score_str = str(clamped_score)
    stats["score_distribution"][score_str] = stats["score_distribution"].get(score_str, 0) + 1

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "score": clamped_score,
        "decision": dec_key,
        "take": adversarial_take[:100]
    }
    stats["recent_evaluations"].insert(0, entry)
    stats["recent_evaluations"] = stats["recent_evaluations"][:50]

    with open(SKEPTIC_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

def print_skeptic_distribution_terminal():
    stats = load_skeptic_stats()
    tot = stats["total_evaluations"]
    if tot == 0:
        return

    print("\n" + "=" * 65)
    print("📊 SKEPTIC RISK SCORE DISTRIBUTION & VETO PROFILE")
    print("=" * 65)
    d = stats["decisions"]
    print(f"Decisions Breakdown: PASS: {d.get('PASS',0)} | SHRINK: {d.get('SHRINK',0)} | KILL: {d.get('KILL',0)} | Total: {tot}")
    print("-" * 65)

    for i in range(1, 11):
        count = stats["score_distribution"].get(str(i), 0)
        pct = (count / tot * 100) if tot > 0 else 0
        bar = "█" * int(pct // 4)
        if i <= 4:
            color = "\033[92m"
        elif i <= 6:
            color = "\033[93m"
        else:
            color = "\033[91m"
        print(f"Score {i:>2}/10: {color}{bar:<25}\033[0m {count:>3} ({pct:>5.1f}%)")
    print("=" * 65 + "\n")


class ScanRequest(BaseModel):
    universe: Optional[str] = "TOP50_VOLUME"

class TradeExecutionRequest(BaseModel):
    ticker: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size_usd: float
    top_3_reasons: Optional[List[str]] = []
    invalidation_criteria: Optional[str] = ""

class SchedulerConfig(BaseModel):
    interval_hours: int = 4
    universe: Optional[str] = "TOP50_VOLUME"

class NtfyConfig(BaseModel):
    ntfy_topic: str

class AutoExecuteConfig(BaseModel):
    enabled: bool

class WatchlistUpdateRequest(BaseModel):
    tickers: List[str]

class AdjustTakeProfitRequest(BaseModel):
    identifier: str
    reduction_pct: Optional[float] = 0.01

class ManualSellRequest(BaseModel):
    identifier: str


def send_email_alert(subject: str, body: str):
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 465))
    smtp_user = os.environ.get("SMTP_USER", "peter.moneypenny@gmail.com")
    smtp_pass = os.environ.get("SMTP_PASS", "").replace(" ", "")
    recipient = os.environ.get("ALERT_EMAIL_TO", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, recipient]):
        print("⚠️ [Email Alert] Missing SMTP configuration. Email skipped.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = f"Trading Desk <{smtp_user}>"
        msg["To"] = recipient
        msg["Subject"] = f"[Trading Desk] {subject}"
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)

        print(f"📧 [Email Alert Sent] To: {recipient} | Subject: {subject}")
    except smtplib.SMTPAuthenticationError:
        print("❌ [Email Error] Gmail authentication failed.")
    except Exception as e:
        print(f"❌ [Email Error] Failed to send email alert: {e}")


def send_ntfy_push(title: str, message: str, tags: str = "chart_with_upwards_trend", priority: str = "default"):
    topic = current_settings.get("ntfy_topic", "").strip()
    if not topic:
        return

    url = f"https://ntfy.sh/{topic}"
    try:
        title.encode("ascii")
        safe_title = title
    except UnicodeEncodeError:
        encoded_title = base64.b64encode(title.encode("utf-8")).decode("ascii")
        safe_title = f"=?utf-8?b?{encoded_title}?="

    headers = {
        "Title": safe_title,
        "Priority": priority,
        "Tags": tags
    }
    try:
        resp = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=5)
        resp.raise_for_status()
        print(f"📡 [ntfy] Notification sent: {title}")
    except Exception as e:
        print(f"Failed to send ntfy alert: {e}")


def print_detailed_scout_candidates(candidates: List[dict]):
    if not candidates:
        print("🎯 [Scout] No candidates passed screening filters.")
        return

    print("\n" + "=" * 95)
    print(f"🎯 [Scout] Screened {len(candidates)} Priority Candidates for AI Evaluation:")
    print("=" * 95)
    header = f"{'#':<3} {'Ticker':<8} {'Price':<10} {'50 SMA':<10} {'Regime':<12} {'ATR(14)':<10} {'$ Vol':<12} {'Momentum':<10}"
    print(header)
    print("-" * 95)
    for idx, c in enumerate(candidates, 1):
        ticker = c.get("ticker", "N/A")
        price = f"${c.get('current_price', 0.0):.2f}"
        sma50 = f"${c.get('sma_50', 0.0):.2f}"
        trend = c.get("trend", "UNKNOWN")
        trend_color = "\033[92m" if trend == "UPTREND" else "\033[91m"
        trend_str = f"{trend_color}{trend:<12}\033[0m"
        atr = f"${c.get('atr_14', 0.0):.2f}"
        d_vol = f"${(c.get('dollar_volume', 0.0) / 1_000_000.0):.1f}M"
        mom = f"{c.get('momentum_score', 0.0):.4f}"

        print(f"{idx:<3} {ticker:<8} {price:<10} {sma50:<10} {trend_str} {atr:<10} {d_vol:<12} {mom:<10}")
    print("=" * 95 + "\n")


def print_detailed_evaluation_summary(cand: dict, skeptic, card):
    min_rr = trading_hub.rulebook.get("min_risk_reward_ratio", 1.4)
    max_risk = trading_hub.rulebook.get("max_risk_score", 6)
    min_price = trading_hub.rulebook.get("min_stock_price", 10.0)
    atr_mult = trading_hub.rulebook.get("atr_stop_multiplier", 1.8)

    p_val = cand.get("current_price", 0.0)
    p_pass = p_val >= min_price
    p_col = "\033[92m[PASS]\033[0m" if p_pass else "\033[91m[FAIL]\033[0m"

    r_val = getattr(skeptic, "risk_score_1_to_10", 10)
    decision = getattr(skeptic, "decision", "UNKNOWN")
    r_pass = (r_val <= max_risk) and (decision in ("PASS", "SHRINK"))
    r_col = "\033[92m[PASS]\033[0m" if r_pass else "\033[91m[FAIL]\033[0m"

    sl_val = getattr(card, "stop_loss", 0.0) if card else 0.0
    action = getattr(card, "action", "PASS") if card else "PASS"
    atr_val = cand.get("atr_14", 1.0)
    max_allowed_dist = round(atr_val * atr_mult, 2)
    actual_dist = round(abs(p_val - sl_val), 2) if sl_val > 0 else 0.0

    sl_pass = card is not None and sl_val > 0 and action in ("BUY", "SELL") and (0 < actual_dist <= max_allowed_dist)
    sl_col = "\033[92m[PASS]\033[0m" if sl_pass else "\033[91m[FAIL]\033[0m"
    sl_str = f"${sl_val:.2f} (Dist: ${actual_dist:.2f} <= ${max_allowed_dist:.2f})" if sl_val > 0 else "None defined"

    rr_val = getattr(card, "risk_reward_ratio", 0.0) if card else 0.0
    rr_pass = rr_val >= min_rr and action in ("BUY", "SELL")
    rr_col = "\033[92m[PASS]\033[0m" if rr_pass else "\033[91m[FAIL]\033[0m"

    print(f"\n🛡️ Candidate {cand.get('ticker')} Evaluation Summary:")
    print(f"   Reasoning: {getattr(skeptic, 'adversarial_take', 'Risk evaluation completed')}")
    print(f"   {p_col} Price >= ${min_price:.2f}: ${p_val:.2f}")
    print(f"   {r_col} Skeptic PASS/SHRINK & Score <= {max_risk}/10: {r_val}/10 (Decision: {decision})")
    print(f"   {sl_col} Defined Stop-Loss: {sl_str}")
    print(f"   {rr_col} Risk/Reward Ratio >= {min_rr}: {rr_val:.2f} (Action: {action})")

    if not r_pass:
        print("   \033[91m🚫 [VETOED]\033[0m Trade rejected and killed by Skeptic Risk Officer.\n")
    elif not (sl_pass and rr_pass):
        print("   \033[91m🚫 [DISCARDED]\033[0m Trade discarded due to structural Stop-Loss or R:R invalidation.\n")
    else:
        print("   \033[92m✅ [APPROVED]\033[0m All rulebook gates passed successfully!\n")


def get_total_realized_pnl() -> float:
    records = trading_hub.get_journal_records()
    total = 0.0
    for r in records:
        status = (r.get("status") or "").upper()
        if "CLOSED" in status:
            try:
                total += float(r.get("realized_pnl") or r.get("pnl_usd") or 0.0)
            except (ValueError, TypeError):
                continue
    return round(total, 2)


def trigger_position_check():
    closed_events = trading_hub.check_open_positions(force=True)
    if not closed_events:
        return {"status": "success", "closed_positions": []}

    total_realized = get_total_realized_pnl()
    total_sign = "+" if total_realized >= 0 else "-"

    for ev in closed_events:
        is_win = ev["pnl"] >= 0
        tag = "tada,moneybag" if is_win else "warning,small_red_triangle_down"
        title = f"Trade Closed: {ev['ticker']} ({ev['reason']})"
        body = (
            f"Exit: ${ev['price']:.2f} | PnL: ${ev['pnl']:+.2f}\n"
            f"Total Realized PnL: {total_sign}${abs(total_realized):.2f}"
        )
        
        send_ntfy_push(title, body, tags=tag, priority="high" if is_win else "default")
        send_email_alert(title, f"{title}\n{body}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    return {"status": "success", "closed_positions": closed_events, "total_realized_pnl": total_realized}


scheduler.add_job(
    trigger_position_check,
    "interval",
    seconds=30,
    id="position_check_job",
    replace_existing=True
)
scheduler.start()


def _evaluate_single_candidate(cand: dict) -> Optional[dict]:
    """Parallel worker evaluating a single candidate setup via Gemini."""
    max_risk = trading_hub.rulebook.get("max_risk_score", 6)
    min_rr = trading_hub.rulebook.get("min_risk_reward_ratio", 1.4)
    min_price = trading_hub.rulebook.get("min_stock_price", 10.0)
    atr_mult = trading_hub.rulebook.get("atr_stop_multiplier", 1.8)

    try:
        report, skeptic, card = trading_hub.evaluate_candidate_unified(cand)

        record_skeptic_evaluation(
            ticker=cand.get("ticker"),
            score=getattr(skeptic, "risk_score_1_to_10", 10),
            decision=getattr(skeptic, "decision", "KILL"),
            adversarial_take=getattr(skeptic, "adversarial_take", "")
        )

        p_val = cand.get("current_price", 0.0)
        sl_val = getattr(card, "stop_loss", 0.0) if card else 0.0
        action = getattr(card, "action", "PASS") if card else "PASS"
        atr_val = cand.get("atr_14", 1.0)
        max_allowed_dist = round(atr_val * atr_mult, 2)
        actual_dist = round(abs(p_val - sl_val), 2) if sl_val > 0 else 0.0

        p_passed = p_val >= min_price
        skeptic_passed = (skeptic.decision in ("PASS", "SHRINK")) and (skeptic.risk_score_1_to_10 <= max_risk)
        sl_passed = (card is not None) and (sl_val > 0) and (action in ("BUY", "SELL")) and (0 < actual_dist <= max_allowed_dist)
        card_passed = (card is not None) and (action in ("BUY", "SELL")) and (card.risk_reward_ratio >= min_rr)
        
        is_approved = p_passed and skeptic_passed and sl_passed and card_passed

        return {
            "cand": cand,
            "report": report,
            "skeptic": skeptic,
            "card": card,
            "is_approved": is_approved
        }
    except Exception as e:
        print(f"Error evaluating {cand.get('ticker')}: {e}")
        return None


def run_automated_scan_job():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n⏰ [{now_str}] Scheduled cycle running for universe: {scheduler_state['universe']}...")
    scheduler_state["last_run"] = now_str

    trigger_position_check()

    candidates = trading_hub.scout_scan(universe=scheduler_state["universe"], max_candidates=8)
    print_detailed_scout_candidates(candidates)

    auto_exec = current_settings.get("auto_execute", False)
    cards = []

    if candidates:
        max_threads = min(len(candidates), 8)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            eval_results = list(executor.map(_evaluate_single_candidate, candidates))

        for res in eval_results:
            if not res:
                continue

            cand = res["cand"]
            report = res["report"]
            skeptic = res["skeptic"]
            card = res["card"]
            is_approved = res["is_approved"]

            print_detailed_evaluation_summary(cand, skeptic, card)

            auto_purchased = False
            if is_approved and auto_exec:
                logged_id = trading_hub.log_trade(card.model_dump())
                if logged_id:
                    auto_purchased = True
                    print(f"⚡ [AUTO-EXECUTE] Order booked for {card.ticker} ({card.action})")
                    title = f"Auto-Order Executed: {card.ticker} ({card.action})"
                    body = f"Entry: ${card.entry_price:.2f} | Target: ${card.take_profit:.2f} | Stop: ${card.stop_loss:.2f} | Size: ${card.position_size_usd:.2f}"
                    send_ntfy_push(title, body, tags="zap,moneybag,white_check_mark", priority="high")
                    send_email_alert(title, f"{title}\n{body}\nTrade ID: {logged_id}\nTimestamp: {now_str}")
            elif is_approved:
                title = f"Setup Approved: {card.ticker} ({card.action})"
                body = f"Entry: ${card.entry_price:.2f} | Target: ${card.take_profit:.2f} | Stop: ${card.stop_loss:.2f} | R/R: {card.risk_reward_ratio}"
                send_ntfy_push(title, body, tags="chart_with_upwards_trend,dart", priority="high")
                send_email_alert(title, f"{title}\n{body}\nTimestamp: {now_str}")

            cards.append({
                "candidate": cand,
                "analyst": report.model_dump(),
                "skeptic": skeptic.model_dump(),
                "card": card.model_dump() if card else None,
                "approved_for_trade": is_approved,
                "auto_purchased": auto_purchased
            })

    print_skeptic_distribution_terminal()

    latest_scan_cache["timestamp"] = now_str
    latest_scan_cache["universe"] = scheduler_state["universe"]
    latest_scan_cache["results"] = cards


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="index.html not found in templates/")
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/trading/skeptic-distribution")
async def get_skeptic_distribution():
    return load_skeptic_stats()


@app.post("/api/trading/scan")
async def run_scanner(payload: ScanRequest):
    candidates = trading_hub.scout_scan(universe=payload.universe, max_candidates=8, force=True)
    print_detailed_scout_candidates(candidates)

    cards = []
    auto_exec = current_settings.get("auto_execute", False)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if candidates:
        max_threads = min(len(candidates), 8)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            eval_results = list(executor.map(_evaluate_single_candidate, candidates))

        for res in eval_results:
            if not res:
                continue

            cand = res["cand"]
            report = res["report"]
            skeptic = res["skeptic"]
            card = res["card"]
            is_approved = res["is_approved"]

            print_detailed_evaluation_summary(cand, skeptic, card)

            auto_purchased = False
            if is_approved and auto_exec:
                logged_id = trading_hub.log_trade(card.model_dump())
                if logged_id:
                    auto_purchased = True
                    title = f"Auto-Order Executed: {card.ticker} ({card.action})"
                    body = f"Entry: ${card.entry_price:.2f} | Target: ${card.take_profit:.2f} | Stop: ${card.stop_loss:.2f} | Size: ${card.position_size_usd:.2f}"
                    send_ntfy_push(title, body, tags="zap,moneybag,white_check_mark", priority="high")
                    send_email_alert(title, f"{title}\n{body}\nTrade ID: {logged_id}\nTimestamp: {now_str}")

            cards.append({
                "candidate": cand,
                "analyst": report.model_dump(),
                "skeptic": skeptic.model_dump(),
                "card": card.model_dump() if card else None,
                "approved_for_trade": is_approved,
                "auto_purchased": auto_purchased
            })

    print_skeptic_distribution_terminal()

    latest_scan_cache["timestamp"] = now_str
    latest_scan_cache["universe"] = payload.universe
    latest_scan_cache["results"] = cards

    return {"results": cards}

@app.get("/api/trading/latest-scan")
async def get_latest_scan():
    return latest_scan_cache

@app.get("/api/trading/watchlist")
async def get_watchlist():
    tickers = trading_hub._load_custom_watchlist()
    return {"count": len(tickers), "tickers": tickers}

@app.post("/api/trading/watchlist")
async def update_watchlist(payload: WatchlistUpdateRequest):
    target_path = Path("/app/watchlist.csv")
    if not target_path.exists():
        target_path = Path("watchlist.csv")
    
    clean_tickers = [t.strip().upper() for t in payload.tickers if t.strip()]
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("ticker\n" + "\n".join(clean_tickers) + "\n")
    return {"status": "success", "count": len(clean_tickers)}

@app.post("/api/trading/execute")
async def execute_trade(payload: TradeExecutionRequest):
    logged_id = trading_hub.log_trade(payload.model_dump())
    title = f"Paper Trade Executed: {payload.ticker}"
    body = f"Order: {payload.action} | Entry: ${payload.entry_price} | Stop: ${payload.stop_loss} | Target: ${payload.take_profit}"
    send_ntfy_push(title, body, tags="white_check_mark,briefcase")
    send_email_alert(title, f"{title}\n{body}\nTrade ID: {logged_id}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return {"status": "success", "message": f"Order for {payload.ticker} logged."}

@app.post("/api/trading/position/sell")
async def sell_position_endpoint(payload: ManualSellRequest):
    updated = trading_hub.close_position_now(payload.identifier)
    if not updated:
        raise HTTPException(status_code=404, detail="Open position not found for given identifier")

    ticker = updated.get("ticker", "UNKNOWN")
    try:
        exit_p = float(updated.get("exit_price") or 0.0)
    except (ValueError, TypeError):
        exit_p = 0.0

    try:
        pnl = float(updated.get("realized_pnl") or updated.get("pnl_usd") or 0.0)
    except (ValueError, TypeError):
        pnl = 0.0

    total_realized = get_total_realized_pnl()
    total_sign = "+" if total_realized >= 0 else "-"
    is_win = pnl >= 0
    tag = "tada,moneybag" if is_win else "warning,small_red_triangle_down"

    title = f"Trade Closed: {ticker} (MANUAL_SELL)"
    body = (
        f"Exit: ${exit_p:.2f} | PnL: ${pnl:+.2f}\n"
        f"Total Realized PnL: {total_sign}${abs(total_realized):.2f}"
    )

    send_ntfy_push(title, body, tags=tag, priority="high" if is_win else "default")
    send_email_alert(title, f"{title}\n{body}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {"status": "success", "trade": updated}

@app.post("/api/trading/position/adjust-tp")
async def adjust_position_tp(payload: AdjustTakeProfitRequest):
    updated = trading_hub.adjust_take_profit(payload.identifier, reduction_pct=payload.reduction_pct or 0.01)
    if not updated:
        raise HTTPException(status_code=404, detail="Open position not found for given identifier")
    
    trigger_position_check()
    return {"status": "success", "trade": updated}

@app.get("/api/trading/journal")
async def get_journal():
    records = trading_hub.get_journal_records()
    return {"trades": records}

@app.get("/api/trading/journal/download")
async def download_journal():
    if os.path.exists(trading_hub.journal_file):
        return FileResponse(
            trading_hub.journal_file,
            media_type="text/csv",
            filename=f"trading_journal_{datetime.now().strftime('%Y%m%d')}.csv"
        )
    raise HTTPException(status_code=404, detail="Journal file not found")

@app.get("/api/trading/trades/categorized")
async def get_categorized_trades():
    records = trading_hub.get_journal_records()
    now = datetime.now()
    cutoff_24h = now - timedelta(hours=24)

    open_trades = []
    recent_closed = []
    archived_trades = []
    all_closed = []

    for raw in records:
        status_str = (raw.get("status") or "OPEN").strip().upper()
        is_closed = "CLOSED" in status_str
        is_open = "OPEN" in status_str and not is_closed

        r = dict(raw)
        pnl = raw.get("realized_pnl") or raw.get("pnl_usd") or "0.0"
        r["realized_pnl"] = pnl
        r["pnl_usd"] = pnl

        rr = raw.get("risk_reward_ratio") or raw.get("risk_reward") or "0.0"
        r["risk_reward_ratio"] = rr
        r["risk_reward"] = rr

        reasons = raw.get("top_reasons") or raw.get("notes") or raw.get("invalidation_criteria") or "--"
        r["top_reasons"] = reasons
        r["notes"] = reasons

        exit_reason = raw.get("exit_reason") or ""
        if not exit_reason and is_closed:
            if "(" in status_str and ")" in status_str:
                exit_reason = status_str.split("(")[1].split(")")[0].strip()
            else:
                exit_reason = "CLOSED"
        r["exit_reason"] = exit_reason

        exit_ts_str = raw.get("exit_timestamp") or raw.get("timestamp")
        r["exit_timestamp"] = exit_ts_str

        if is_open:
            r["status"] = "OPEN"
            open_trades.append(r)
        elif is_closed:
            r["status"] = "CLOSED"
            all_closed.append(r)
            is_recent = False
            try:
                exit_dt = datetime.strptime(exit_ts_str, "%Y-%m-%d %H:%M:%S")
                if exit_dt >= cutoff_24h:
                    is_recent = True
            except Exception:
                is_recent = True

            if is_recent:
                recent_closed.append(r)
            else:
                archived_trades.append(r)

    if open_trades:
        active_tickers = [t["ticker"] for t in open_trades if t.get("ticker")]
        live_quotes = trading_hub.get_live_prices(active_tickers)
        for t in open_trades:
            ticker = t.get("ticker")
            if ticker in live_quotes:
                t["current_price"] = live_quotes[ticker]
            else:
                try:
                    t["current_price"] = float(t.get("entry_price", 0.0))
                except (ValueError, TypeError):
                    t["current_price"] = 0.0

    open_trades.reverse()
    recent_closed.reverse()
    archived_trades.reverse()

    return {
        "open_trades": open_trades,
        "recent_closed_24h": recent_closed,
        "archived_trades_over_24h": archived_trades,
        "all_closed_trades": all_closed
    }

@app.get("/api/trading/check-positions")
def check_positions_endpoint():
    return trigger_position_check()


@app.post("/api/trading/scheduler/start")
async def start_scheduler(config: SchedulerConfig):
    hours = max(1, config.interval_hours)
    scheduler_state["is_running"] = True
    scheduler_state["interval_hours"] = hours
    scheduler_state["universe"] = config.universe

    if scheduler.get_job("trading_scan_job"):
        scheduler.remove_job("trading_scan_job")

    scheduler.add_job(
        run_automated_scan_job,
        "interval",
        hours=hours,
        id="trading_scan_job",
        replace_existing=True
    )

    if not scheduler.running:
        scheduler.start()

    return {"status": "started", "interval_hours": hours, "universe": config.universe}

@app.post("/api/trading/scheduler/stop")
async def stop_scheduler():
    scheduler_state["is_running"] = False
    if scheduler.get_job("trading_scan_job"):
        scheduler.remove_job("trading_scan_job")
    return {"status": "stopped"}

@app.get("/api/trading/scheduler/status")
async def get_scheduler_status():
    job = scheduler.get_job("trading_scan_job")
    next_run = str(job.next_run_time) if job and job.next_run_time else None
    return {
        "is_running": scheduler_state["is_running"],
        "interval_hours": scheduler_state["interval_hours"],
        "universe": scheduler_state["universe"],
        "last_run": scheduler_state["last_run"],
        "next_run": next_run
    }

@app.get("/api/trading/ntfy/config")
async def get_ntfy_config():
    return {"ntfy_topic": current_settings.get("ntfy_topic", "")}

@app.post("/api/trading/ntfy/config")
async def save_ntfy_config(payload: NtfyConfig):
    current_settings["ntfy_topic"] = payload.ntfy_topic.strip()
    save_settings(current_settings)
    send_ntfy_push("Hub Connected", "ntfy notifications successfully linked to Efficiency Hub Desk!", tags="white_check_mark,bell")
    return {"status": "success", "ntfy_topic": current_settings["ntfy_topic"]}

@app.get("/api/trading/auto-execute/config")
async def get_auto_execute_config():
    return {"auto_execute": current_settings.get("auto_execute", False)}

@app.post("/api/trading/auto-execute/config")
async def set_auto_execute_config(payload: AutoExecuteConfig):
    current_settings["auto_execute"] = payload.enabled
    save_settings(current_settings)
    return {"status": "success", "auto_execute": current_settings["auto_execute"]}

@app.get("/api/trading/test-ntfy")
async def test_ntfy_endpoint():
    send_ntfy_push("Test Alert: Auto-Order", "Live test message from Efficiency Hub Desk", tags="zap,test_tube,rocket", priority="high")
    send_email_alert("Test Alert: Auto-Order", "This is a test notification confirming email delivery from Efficiency Hub Desk.")
    return {"status": "dispatched", "topic": current_settings.get("ntfy_topic")}

@app.get("/api/trading/test-email")
async def test_email_endpoint():
    send_email_alert("Manual Test Email", "Testing Gmail SMTP dispatch from FastAPI endpoint.")
    return {"status": "dispatched", "recipient": os.environ.get("ALERT_EMAIL_TO", "peter.moneypenny@gmail.com")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5085, reload=True)
