import json
import logging
from typing import Dict, Any, Optional
import httpx

from app.config import settings
from app.core.state import system_state

logger = logging.getLogger("trading_desk_v2.gemini")

BASE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_NAME = getattr(settings, "GEMINI_MODEL", "gemini-flash-latest")

SYSTEM_INSTRUCTION = """You are an institutional intraday momentum trader and execution agent.
Your objective is to evaluate real-time technical indicators, VWAP positioning, Opening Range Breakout (ORB) levels, and momentum metrics to issue precise day-trading decisions.

Strict Rules:
1. Trade Types: Intraday Scalp / Momentum Day Trade only. Never suggest holding overnight.
2. Market Trend Alignment: Do NOT recommend BUY if Broad Market Macro Bias is BEARISH. Do NOT recommend SELL/SHORT if Broad Market Macro Bias is BULLISH.
3. Long Entries: Price must be above VWAP AND fast EMA (9) must be >= slow EMA (21).
4. Short Entries: Price must be below VWAP AND fast EMA (9) must be <= slow EMA (21).
5. Risk-Reward Ratio: Minimum target reward must be at least 1.8x the risk distance.
6. Stop Loss: Set stops beyond market noise. Stop distance must be AT LEAST 2.0x ATR from entry price. Never set stops closer than 0.5% of stock price.
7. Output: Output strictly a valid raw JSON object matching the requested schema. No conversational preamble, explanation, or markdown fences.
"""


async def evaluate_intraday_opportunity(
    ticker: str,
    session_phase: str,
    technical_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Asynchronously queries Gemini Flash to evaluate an intraday setup.
    """
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not configured. Defaulting to HOLD.")
        return {
            "action": "HOLD",
            "confidence": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 0.0,
            "reasoning": "Gemini API key is not configured in settings."
        }

    metrics = technical_analysis.get("metrics", {})
    last_price = metrics.get("last_price")
    signal = technical_analysis.get("signal", "NEUTRAL")

    if not last_price:
        return {
            "action": "HOLD",
            "confidence": 0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 0.0,
            "reasoning": "Missing price data for technical evaluation."
        }

    macro_bias = getattr(system_state, "macro_bias", "NEUTRAL")
    atr_val = metrics.get("atr_14", 1.0) or 1.0

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
- Algorithmic Signal: {signal}
- Bias Score: {technical_analysis.get('bias_score', 0)}

Required JSON schema:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 80,
  "entry_price": {last_price},
  "stop_loss": <numeric price at least 2.0x ATR from entry>,
  "take_profit": <numeric price target with >= 1.8x risk-reward>,
  "risk_reward_ratio": 1.85,
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
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                err_text = response.text
                logger.error(f"Gemini API returned status {response.status_code}: {err_text}")
                return {
                    "action": "HOLD",
                    "confidence": 0,
                    "entry_price": last_price,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "risk_reward_ratio": 0.0,
                    "reasoning": f"Gemini API error ({response.status_code}): {err_text[:140]}"
                }

            result_data = response.json()
            raw_text = result_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed_decision = json.loads(raw_text.strip())
            # Force entry_price to the real live price regardless of LLM output
            parsed_decision["entry_price"] = last_price
            return parsed_decision

    except Exception as e:
        logger.error(f"Failed to query Gemini agent for {ticker}: {e}", exc_info=True)
        return {
            "action": "HOLD",
            "confidence": 0,
            "entry_price": last_price,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 0.0,
            "reasoning": f"Evaluation error: {str(e)}"
        }