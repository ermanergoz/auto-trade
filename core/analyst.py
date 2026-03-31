"""AI Analyst — LLM-powered deep analysis of screener candidates."""

import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, AI_MODEL,
    AI_CONFIDENCE_THRESHOLD, DEFAULT_STOP_LOSS_PCT, DEFAULT_TAKE_PROFIT_PCT,
)
from core.models import Signal, Action, TradeType

logger = logging.getLogger(__name__)

# Token usage tracking for cost control
_daily_token_usage = {"input": 0, "output": 0, "date": None}
_DAILY_COST_ALERT = 2.0  # USD


def _reset_daily_usage_if_needed() -> None:
    today = datetime.now().date().isoformat()
    if _daily_token_usage["date"] != today:
        _daily_token_usage["input"] = 0
        _daily_token_usage["output"] = 0
        _daily_token_usage["date"] = today


def _estimate_cost() -> float:
    """Rough cost estimate based on token usage."""
    # Approximate pricing (varies by model)
    input_cost = _daily_token_usage["input"] / 1_000_000 * 3.0
    output_cost = _daily_token_usage["output"] / 1_000_000 * 15.0
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are a professional stock analyst. Analyze this stock and provide a trading recommendation.

## Stock: {ticker} ({exchange})

## Recent Price Action (last 5 days)
{price_action}

## Technical Indicators
{indicators}

## News Headlines
{news}

## Instructions
Based on the data above, provide your analysis as a JSON object with these exact fields:
- "action": "buy", "sell", or "hold"
- "confidence": integer 0-100 (how confident you are)
- "entry_price": recommended entry price (float)
- "stop_loss": stop-loss price (float)
- "take_profit": take-profit target (float)
- "trade_type": "day" or "swing"
- "reasoning": 2-3 sentence explanation

Consider:
- Risk/reward ratio (minimum 1.5:1)
- Current market conditions and momentum
- Volume confirmation
- Multiple indicator alignment
- News sentiment

Only recommend "buy" or "sell" if you have high conviction. Default to "hold" if unclear."""


def _build_prompt(
    ticker: str,
    exchange: str,
    df: pd.DataFrame,
    indicator_values: dict,
    news: list[str],
) -> str:
    """Build the analysis prompt with all context."""
    # Recent price action
    recent = df.tail(5)
    price_lines = []
    for idx, row in recent.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        price_lines.append(
            f"  {date_str}: O={row['open']:.2f} H={row['high']:.2f} "
            f"L={row['low']:.2f} C={row['close']:.2f} V={row['volume']:,.0f}"
        )
    price_action = "\n".join(price_lines) if price_lines else "No recent data"

    # Indicators
    if indicator_values:
        indicators = "\n".join(f"  - {k}: {v}" for k, v in indicator_values.items())
    else:
        indicators = "  No indicator data available"

    # News
    if news:
        news_text = "\n".join(f"  - {h}" for h in news[:5])
    else:
        news_text = "  No recent news available"

    return ANALYSIS_PROMPT.format(
        ticker=ticker,
        exchange=exchange,
        price_action=price_action,
        indicators=indicators,
        news=news_text,
    )


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _call_claude(prompt: str) -> Optional[dict]:
    """Call Claude API with structured output via tool_use."""
    if not ANTHROPIC_API_KEY:
        logger.error("No ANTHROPIC_API_KEY configured")
        return None

    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    tool_schema = {
        "name": "trading_recommendation",
        "description": "Provide a structured trading recommendation",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "entry_price": {"type": "number"},
                "stop_loss": {"type": "number"},
                "take_profit": {"type": "number"},
                "trade_type": {"type": "string", "enum": ["day", "swing"]},
                "reasoning": {"type": "string"},
            },
            "required": [
                "action", "confidence", "entry_price",
                "stop_loss", "take_profit", "trade_type", "reasoning",
            ],
        },
    }

    response = client.messages.create(
        model=AI_MODEL,
        max_tokens=1024,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "trading_recommendation"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Track usage
    _reset_daily_usage_if_needed()
    _daily_token_usage["input"] += response.usage.input_tokens
    _daily_token_usage["output"] += response.usage.output_tokens

    # Extract tool use result
    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return None


def _call_openai(prompt: str) -> Optional[dict]:
    """Call OpenAI API with structured output via function calling."""
    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY configured")
        return None

    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    function_schema = {
        "name": "trading_recommendation",
        "description": "Provide a structured trading recommendation",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "entry_price": {"type": "number"},
                "stop_loss": {"type": "number"},
                "take_profit": {"type": "number"},
                "trade_type": {"type": "string", "enum": ["day", "swing"]},
                "reasoning": {"type": "string"},
            },
            "required": [
                "action", "confidence", "entry_price",
                "stop_loss", "take_profit", "trade_type", "reasoning",
            ],
        },
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,
        functions=[function_schema],
        function_call={"name": "trading_recommendation"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Track usage
    _reset_daily_usage_if_needed()
    if response.usage:
        _daily_token_usage["input"] += response.usage.prompt_tokens
        _daily_token_usage["output"] += response.usage.completion_tokens

    msg = response.choices[0].message
    if msg.function_call:
        return json.loads(msg.function_call.arguments)

    return None


def _call_llm(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """Call the configured LLM with retry logic."""
    use_claude = ANTHROPIC_API_KEY and ("claude" in AI_MODEL.lower() or not OPENAI_API_KEY)

    for attempt in range(1, max_retries + 1):
        try:
            if use_claude:
                result = _call_claude(prompt)
            else:
                result = _call_openai(prompt)

            if result and _validate_response(result):
                return result
            else:
                logger.warning("Invalid LLM response on attempt %d", attempt)

        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt, max_retries, e)

    return None


def _validate_response(data: dict) -> bool:
    """Validate LLM response has all required fields with valid values."""
    required = ["action", "confidence", "entry_price", "stop_loss", "take_profit", "reasoning"]
    for field in required:
        if field not in data:
            return False

    if data["action"] not in ("buy", "sell", "hold"):
        return False
    if not isinstance(data["confidence"], (int, float)) or not (0 <= data["confidence"] <= 100):
        return False
    if data["entry_price"] <= 0 or data["stop_loss"] <= 0 or data["take_profit"] <= 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_candidate(
    ticker: str,
    exchange: str,
    df: pd.DataFrame,
    indicator_values: dict,
    news: list[str],
) -> Optional[Signal]:
    """Analyze a single screener candidate with the LLM.

    Pure function — receives all data, doesn't fetch anything.

    Returns a Signal if confidence >= threshold, else None.
    """
    # Check cost
    estimated_cost = _estimate_cost()
    if estimated_cost > _DAILY_COST_ALERT:
        logger.warning("Daily AI cost estimate: $%.2f (alert at $%.2f)", estimated_cost, _DAILY_COST_ALERT)

    prompt = _build_prompt(ticker, exchange, df, indicator_values, news)
    result = _call_llm(prompt)

    if not result:
        logger.warning("No valid LLM response for %s", ticker)
        return None

    if result["confidence"] < AI_CONFIDENCE_THRESHOLD:
        logger.info(
            "Skipping %s: confidence %d < threshold %d",
            ticker, result["confidence"], AI_CONFIDENCE_THRESHOLD,
        )
        return None

    action_map = {"buy": Action.BUY, "sell": Action.SELL, "hold": Action.HOLD}
    action = action_map.get(result["action"], Action.HOLD)

    if action == Action.HOLD:
        return None

    trade_type_map = {"day": TradeType.DAY, "swing": TradeType.SWING}
    trade_type = trade_type_map.get(result.get("trade_type", "day"), TradeType.DAY)

    return Signal(
        ticker=ticker,
        action=action,
        confidence=float(result["confidence"]),
        entry_price=float(result["entry_price"]),
        stop_loss=float(result["stop_loss"]),
        take_profit=float(result["take_profit"]),
        reasoning=result.get("reasoning", ""),
        source="ai",
        exchange=exchange,
        trade_type=trade_type,
        indicator_values=indicator_values,
    )


def analyze_batch(
    candidates: list[dict],
) -> list[Signal]:
    """Analyze a batch of screener candidates sequentially.

    Args:
        candidates: List of dicts with keys:
            ticker, exchange, df, indicator_values, news

    Returns list of approved Signal objects.
    """
    signals = []

    for i, c in enumerate(candidates, 1):
        logger.info("Analyzing candidate %d/%d: %s", i, len(candidates), c["ticker"])
        signal = analyze_candidate(
            ticker=c["ticker"],
            exchange=c["exchange"],
            df=c["df"],
            indicator_values=c.get("indicator_values", {}),
            news=c.get("news", []),
        )
        if signal:
            signals.append(signal)
            logger.info(
                "AI approved %s: %s confidence=%d",
                signal.ticker, signal.action.value, signal.confidence,
            )

    logger.info(
        "AI analysis complete: %d/%d candidates approved (est. cost: $%.2f)",
        len(signals), len(candidates), _estimate_cost(),
    )
    return signals


def get_daily_cost_estimate() -> float:
    """Return the estimated AI cost for today."""
    _reset_daily_usage_if_needed()
    return _estimate_cost()
