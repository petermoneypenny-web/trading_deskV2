import json
import logging
from typing import Dict, Any, Optional
import httpx

from app.config import settings
from app.core.state import system_state
from app.services.risk_config import get_risk_config

logger = logging.getLogger("trading_desk_v2.gemini")

BASE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_NAME = getattr(settings, "GEMINI_MODEL", "gemini-flash-latest")

# Measured latency for this model/prompt on a typical connection is 6-11s, so the
# original 12s client timeout was already marginal. 20s is safe headroom.
# Override with GEMINI_HTTP_TIMEOUT in the environment.
# NOTE: this is NOT what protects the trading loop any more -- the caller applies
# LLM_DEADLINE_SECONDS via asyncio.wait_for, and the call runs in a background
# task, so position monitoring continues regardless.
HTTP_TIMEOUT_SECONDS = float(getattr(settings, "GEMINI_HTTP_TIMEOUT", 20.0))

SYSTEM_INSTRUCTION = """You are an institutional intraday momentum trader and execution agent.
Your objective is to evaluate real-time technical indicators, VWAP positioning, Opening Range Breakout (ORB) levels, and momentum metrics to issue precise day-trading decisions.

Strict Rules:
1. Trade Types: Intraday Scalp / Momentum Day Trade only. Never suggest holding overnight.
2. Market Trend Alignment: Do NOT recommend BUY if Broad Market Macro Bias is BEARISH. Do NOT recommend SELL/SHORT if Broad Market Macro Bias is BULLISH.
3. Long Entries: Price must be above VWAP AND fast EMA (9) must be >= slow EMA (21).
4. Short Entries: Price must be below VWAP AND fast EMA (9) must be <= slow EMA (21).
5. Risk-Reward Ratio: Minimum target reward must be at least the configured minimum ratio.
6. Stop Loss: Set stops beyond market noise. Stop distance must be AT LEAST 2.0x ATR from entry price. Never set stops closer than 0.5% of stock price.
7. Quantified edge: you must state, in "edge", the single measurable reason this setup should continue -- for example the distance in ATR units to the nearest structural level, or the ORB width as a percentage of price. If you cannot state a number, answer HOLD.
8. Output: Output strictly a valid raw JSON object matching the requested schema. No conversational preamble, explanation, or markdown fences."""


def _hold(price: float, reason: str) -> Dict[str, Any]:
    return {
        "action": "HOLD",
        "confidence": 0,
        "entry_price": price,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "risk_reward_ratio": 0.0,
        "reasoning": reason,
    }


def _validate_decision(
    decision: Dict[str, Any],
    last_price: float,
    atr_val: float,
) -> Optional[str]:
    """
    Rejects LLM output that does not actually satisfy the rules it was given.

    Previously the model's stop_loss/take_profit were used verbatim whenever they
    were merely on the right side of the entry, and ``risk_reward_ratio`` was
    never checked against the rulebook's ``min_risk_reward_ratio``.
    """
    cfg = get_risk_config()
    action = (decision.get("action") or "").upper()
    if action not in ("BUY", "SELL"):
        return None  # HOLD needs no validation

    stop = decision.get("stop_loss")
    target = decision.get("take_profit")
    if not stop or not target:
        return "missing stop_loss/take_profit"

    stop = float(stop)
    target = float(target)

    if action == "BUY":
        if stop >= last_price:
            return "stop must be below entry for a BUY"
        if target <= last_price:
            return "target must be above entry for a BUY"
        risk = last_price - stop
        reward = target - last_price
    else:
        if stop <= last_price:
            return "stop must be above entry for a SELL"
        if target >= last_price:
            return "target must be below entry for a SELL"
        risk = stop - last_price
        reward = last_price - target

    if risk <= 0:
        return "non-positive risk"

    min_stop = max(atr_val * 2.0, last_price * 0.005)
    if risk < min_stop:
        return f"stop ${risk:.4f} closer than the ${min_stop:.4f} minimum"

    rr = reward / risk
    if rr < cfg.min_risk_reward_ratio:
        return f"reward:risk {rr:.2f} below the {cfg.min_risk_reward_ratio:.2f} minimum"

    if not decision.get("edge"):
        return "no quantified edge stated"

    return None


async def evaluate_intraday_opportunity(
    ticker: str,
    session_phase: str,
    technical_analysis: Dict[str, Any],
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Asynchronously queries Gemini Flash to evaluate an intraday setup.

    Pass ``enabled=False`` to skip the network call entirely (used by the replay
    harness so a backtest is reproducible and does not burn API quota).
    """
    metrics = technical_analysis.get("metrics", {})
    last_price = metrics.get("last_price")
    signal = technical_analysis.get("signal", "NEUTRAL")

    if not enabled:
        return _hold(last_price or 0.0, "LLM evaluation disabled for this run.")

    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not configured. Defaulting to HOLD.")
        return _hold(last_price or 0.0, "Gemini API key is not configured in settings.")

    if not last_price:
        return _hold(0.0, "Missing price data for technical evaluation.")

    cfg = get_risk_config()
    macro_bias = getattr(system_state, "macro_bias", "NEUTRAL")
    atr_val = float(metrics.get("atr_14") or 1.0)

    user_prompt = f"""Evaluate day trading opportunity for ticker: {ticker}
Market Session Phase: {session_phase}
Broad Market Macro Bias (SPY/QQQ): {macro_bias}

Current Ticker Indicators:
- Current Live Price: {last_price}
- VWAP: {metrics.get('vwap')}
- Distance from VWAP: {metrics.get('dist_from_vwap_pct')}%
- Fast EMA (9): {metrics.get('ema_9')}
- Slow EMA (21): {metrics.get('ema_21')}
- RSI (14): {metrics.get('rsi_14')}
- ATR (14): {atr_val}
- ORB High: {metrics.get('orb_high')}
- ORB Low: {metrics.get('orb_low')}
- Relative Volume: {metrics.get('relative_volume')}
- Algorithmic Signal: {signal}
- Bias Score: {technical_analysis.get('bias_score', 0)}
- Setup Score: {technical_analysis.get('setup_score')}

Required JSON schema:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 80,
  "entry_price": {last_price},
  "stop_loss": <numeric price at least 2.0x ATR from entry>,
  "take_profit": <numeric price target with >= {cfg.min_risk_reward_ratio:g}x risk-reward>,
  "risk_reward_ratio": {cfg.min_risk_reward_ratio:g},
  "edge": "<one measurable number and what it represents>",
  "reasoning": "1 to 2 sentences explaining key conviction and confluence"
}}
Return ONLY raw JSON."""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    url = f"{BASE_GEMINI_URL}/{MODEL_NAME}:generateContent?key={settings.GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                err_text = response.text
                logger.error(f"Gemini API returned status {response.status_code}: {err_text}")
                return _hold(last_price, f"Gemini API error ({response.status_code}): {err_text[:140]}")

            result_data = response.json()

            # A 200 does NOT guarantee content. Gemini returns an empty (or
            # absent) candidates array when the prompt is blocked, and an empty
            # parts array when the model emits nothing -- and the old code
            # indexed straight through, so the resulting IndexError propagated to
            # the generic handler below. IndexError's str() is the empty string,
            # so the desk reported "Evaluation error:" and then nothing at all.
            # Diagnose the response instead of assuming its shape.
            candidates = result_data.get("candidates") or []
            if not candidates:
                block = (result_data.get("promptFeedback") or {}).get("blockReason")
                reason = block or "empty candidates array"
                logger.error(
                    f"Gemini returned no candidates for {ticker} "
                    f"({reason}): {json.dumps(result_data)[:400]}"
                )
                return _hold(last_price, f"Gemini returned no content ({reason}).")

            parts = (candidates[0].get("content") or {}).get("parts") or []
            text_part = next((pt for pt in parts if pt.get("text")), None)
            if text_part is None:
                finish = candidates[0].get("finishReason") or "no text part"
                logger.error(
                    f"Gemini returned no text for {ticker} "
                    f"({finish}): {json.dumps(result_data)[:400]}"
                )
                return _hold(last_price, f"Gemini returned no text ({finish}).")

            raw_text = text_part["text"].strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed_decision = json.loads(raw_text.strip())
            # Force entry_price to the real live price regardless of LLM output
            parsed_decision["entry_price"] = last_price

            rejection = _validate_decision(parsed_decision, last_price, atr_val)
            if rejection:
                logger.info(f"Gemini decision for {ticker} rejected: {rejection}")
                parsed_decision["action"] = "HOLD"
                parsed_decision["rejected_because"] = rejection
            return parsed_decision

    except Exception as e:
        logger.error(f"Failed to query Gemini agent for {ticker}: {e}", exc_info=True)
        # Never report a blank reason: several exceptions (IndexError,
        # asyncio.TimeoutError) stringify to "", which is how a bare
        # "Evaluation error:" reached the UI. Always name the type.
        detail = str(e) or f"{type(e).__name__} (no message)"
        return _hold(last_price, f"Evaluation error: {detail}")
