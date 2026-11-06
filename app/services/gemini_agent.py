import json
import logging
from typing import Dict, Any, Optional
import httpx

from app.config import settings

logger = logging.getLogger("trading_desk_v2.gemini")

BASE_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

SYSTEM_INSTRUCTION = """You are an institutional intraday momentum trader and execution agent.
Your objective is to evaluate real-time technical indicators, VWAP positioning, Opening Range Breakout (ORB) levels, and momentum metrics to issue precise day-trading decisions.

Rules:
1. Trade Types: Intraday Scalp / Day Trade only. Never suggest holding overnight.
2. Long Entries: Price must be above VWAP or demonstrating confirmed mean-reversion at support.
3. Short Entries: Price must be below VWAP or breaking down through key support.
4. Risk-Reward Ratio: Minimum target reward must be at least 1.5x the risk distance (Entry to Stop Loss).
5. Stop Loss: Anchor stops strictly around technical invalidation levels (e.g., VWAP cross, recent swing high/low, or 1.5x ATR).
6. Response Format: You must output ONLY a valid JSON object matching the requested schema. No conversational filler or markdown fences.
"""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["BUY", "SELL", "HOLD"]
        },
        "confidence": {
            "type": "integer",
            "description": "Confidence level between 0 and 100"
        },
        "entry_price": {
            "type": "number",
            "description": "Suggested entry price"
        },
        "stop_loss": {
            "type": "number",
            "description": "Hard stop-loss level"
        },
        "take_profit": {
            "type": "number",
            "description": "Target profit exit level"
        },
        "risk_reward_ratio": {
            "type": "number",
            "description": "Calculated reward-to-risk ratio (e.g. 1.8)"
        },
        "reasoning": {
            "type": "string",
            "description": "1 to 2 sentences explaining key technical conviction"
        }
    },
    "required": ["action", "confidence", "entry_price", "stop_loss", "take_profit", "risk_reward_ratio", "reasoning"]
}


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

    user_prompt = f"""Evaluate day trading opportunity for ticker: {ticker}
Market Session Phase: {session_phase}
Current Indicator Snapshot:
- Last Price: {metrics.get('last_price')}
- VWAP: {metrics.get('vwap')}
- Distance from VWAP: {metrics.get('dist_from_vwap_pct')}%
- Fast EMA (9): {metrics.get('ema_9')}
- Slow EMA (21): {metrics.get('ema_21')}
- RSI (14): {metrics.get('rsi_14')}
- ATR (14): {metrics.get('atr_14')}
- ORB High: {metrics.get('orb_high')}
- ORB Low: {metrics.get('orb_low')}
- Algorithmic Signal: {signal}
- Bias Score: {technical_analysis.get('bias_score', 0)}

Determine action (BUY, SELL, or HOLD), dynamic stop loss, and target profit. Output strictly according to the specified schema."""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
            "response_schema": DECISION_SCHEMA
        }
    }

    url = f"{BASE_GEMINI_URL}/{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
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
                    "reasoning": f"Gemini API error ({response.status_code}): {err_text[:120]}"
                }

            result_data = response.json()
            raw_text = result_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed_decision = json.loads(raw_text)
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