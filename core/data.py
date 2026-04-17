"""Market data service — IBKR primary, YFinance fallback, Tavily news."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd
import yfinance as yf
from ib_insync import IB, Contract

from config.settings import TAVILY_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple TTL cache (thread-safe)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()
_DEFAULT_TTL = 300  # 5 minutes


def _cache_get(key: str) -> Optional[object]:
    with _cache_lock:
        if key in _cache:
            expiry, value = _cache[key]
            if time.time() < expiry:
                # Return copies to prevent caller mutation from corrupting
                # cached data shared across multiple callers
                if isinstance(value, pd.DataFrame):
                    return value.copy()
                if isinstance(value, dict):
                    return value.copy()
                if isinstance(value, list):
                    return value.copy()
                return value
            del _cache[key]
        return None


def _cache_set(key: str, value: object, ttl: int = _DEFAULT_TTL) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + ttl, value)


def clear_cache() -> None:
    """Clear all cached data."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# IBKR Historical Data (primary source)
# ---------------------------------------------------------------------------

def get_historical_data(
    ib: IB,
    contract: Contract,
    duration: str = "30 D",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
    use_rth: bool = True,
) -> pd.DataFrame:
    """Fetch historical OHLCV data from IBKR.

    Args:
        ib: Connected IB instance.
        contract: Stock contract (use connection.create_contract()).
        duration: How far back, e.g. "30 D", "6 M", "1 Y".
        bar_size: Bar granularity, e.g. "1 day", "1 hour", "5 mins".
        what_to_show: Data type — "TRADES", "MIDPOINT", "BID", "ASK".
        use_rth: Regular trading hours only.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
    """
    cache_key = f"hist:{contract.symbol}:{contract.exchange}:{duration}:{bar_size}:{what_to_show}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
    )

    if not bars:
        logger.warning("No historical data returned for %s", contract.symbol)
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    _cache_set(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# YFinance Fallback (backtest mode only)
# ---------------------------------------------------------------------------

def get_historical_data_yfinance(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    market: str = "US",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch historical data via YFinance. Fallback for backtest mode.

    auto_adjust=True (default) returns split- and dividend-adjusted OHLC so
    the price series is continuous through corporate actions. Without this,
    a 4-for-1 split shows as a 75% crash and poisons backtests.
    """
    cache_key = f"yf:{ticker}:{market}:{period}:{interval}:adj={auto_adjust}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    yf_ticker = ticker

    try:
        data = yf.download(
            yf_ticker, period=period, interval=interval,
            progress=False, auto_adjust=auto_adjust,
        )
    except Exception as e:
        logger.error("YFinance download failed for %s: %s", yf_ticker, e)
        return pd.DataFrame()

    if data.empty:
        logger.warning("No YFinance data returned for %s", yf_ticker)
        return pd.DataFrame()

    # Normalize column names. yfinance can return a MultiIndex whose level
    # order varies across versions — (field, ticker) in some releases,
    # (ticker, field) in others. Detect which level carries the OHLC field
    # names by locating "Close" and flatten to that level.
    if isinstance(data.columns, pd.MultiIndex):
        field_level = None
        for level in range(data.columns.nlevels):
            if "Close" in data.columns.get_level_values(level):
                field_level = level
                break
        if field_level is None:
            logger.error(
                "YFinance returned MultiIndex columns with no 'Close' level for %s: %s",
                yf_ticker, data.columns.tolist(),
            )
            return pd.DataFrame()
        data.columns = data.columns.get_level_values(field_level)

    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"

    _cache_set(cache_key, df, ttl=3600)  # cache longer for historical
    return df


# ---------------------------------------------------------------------------
# Split/dividend adjustment helpers
# ---------------------------------------------------------------------------

def detect_unadjusted_splits(
    df: pd.DataFrame,
    threshold: float = 0.3,
) -> list[dict]:
    """Find days where close-to-close change looks like an unadjusted split.

    Returns a list of {'date', 'ratio', 'type'} dicts. A forward split
    (e.g. 4-for-1) makes price drop sharply; ratio is the inferred divisor
    (prev_close / curr_close). A reverse split makes price jump; ratio is
    the same formula (gives a value < 1).

    threshold: minimum absolute fractional change to flag (0.3 = 30%).
    """
    if df is None or df.empty or len(df) < 2:
        return []

    closes = df["close"].values
    index = df.index
    events: list[dict] = []

    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        curr_close = closes[i]
        if prev_close <= 0 or curr_close <= 0:
            continue

        pct_change = (curr_close - prev_close) / prev_close
        if abs(pct_change) < threshold:
            continue

        ratio = prev_close / curr_close
        events.append({
            "date": index[i],
            "ratio": ratio,
            "type": "forward" if ratio > 1 else "reverse",
        })

    return events


def adjust_for_splits(
    df: pd.DataFrame,
    splits: dict,
) -> pd.DataFrame:
    """Retroactively adjust OHLC and volume for split events.

    Args:
        df: DataFrame with OHLC + volume columns, indexed by date.
        splits: {split_date: ratio} mapping.
                ratio > 1  → forward split (e.g. 4.0 for 4-for-1):
                             pre-split price /= ratio, volume *= ratio.
                ratio < 1  → reverse split (e.g. 0.1 for 1-for-10):
                             pre-split price /= ratio (i.e. multiplied),
                             volume *= ratio (i.e. divided).
                ratio == 1 → no-op.

    Rows on and after split_date are unchanged. Multiple splits apply
    cumulatively to all earlier rows. Returns a new DataFrame.
    """
    if df is None or df.empty or not splits:
        return df.copy() if df is not None else df

    out = df.copy()
    price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]

    for split_date, ratio in sorted(splits.items()):
        if ratio == 1.0 or ratio <= 0:
            continue
        mask = out.index < split_date
        if not mask.any():
            continue
        for col in price_cols:
            out.loc[mask, col] = out.loc[mask, col] / ratio
        if "volume" in out.columns:
            out.loc[mask, "volume"] = out.loc[mask, "volume"] * ratio

    return out


# ---------------------------------------------------------------------------
# Real-time Quotes (IBKR)
# ---------------------------------------------------------------------------

def get_realtime_quote(ib: IB, contract: Contract) -> Optional[dict]:
    """Request a snapshot quote from IBKR.

    Returns dict with last, bid, ask, volume or None on failure.
    """
    cache_key = f"quote:{contract.symbol}:{contract.exchange}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ib.reqMktData(contract, snapshot=True)
    ib.sleep(2)  # allow time for snapshot data

    ticker = ib.ticker(contract)
    # Cancel snapshot to free IBKR slot — required before requesting again
    ib.cancelMktData(contract)

    if ticker is None:
        logger.warning("No quote data for %s", contract.symbol)
        return None

    result = {
        "last": ticker.last if ticker.last == ticker.last else None,  # NaN check
        "bid": ticker.bid if ticker.bid == ticker.bid else None,
        "ask": ticker.ask if ticker.ask == ticker.ask else None,
        "volume": ticker.volume if ticker.volume == ticker.volume else None,
        "timestamp": datetime.now(timezone.utc),
    }

    # Don't cache or return quotes where all price fields are None
    # (halted/illiquid stocks) — callers checking `if quote:` would get True
    if result["last"] is None and result["bid"] is None and result["ask"] is None:
        logger.warning("All price fields are None for %s — returning None", contract.symbol)
        return None

    _cache_set(cache_key, result, ttl=30)  # short TTL for quotes
    return result


_realtime_subscriptions: dict[int, set[str]] = {}
_realtime_callbacks: dict[int, Callable] = {}
_realtime_lock = threading.Lock()


def subscribe_realtime(
    ib: IB,
    contract: Contract,
    callback: Callable,
) -> None:
    """Subscribe to streaming real-time data via IBKR.

    The callback receives ticker updates. Guards against duplicate
    subscriptions per IB instance; resubscribes after reconnect.
    """
    with _realtime_lock:
        ib_id = id(ib)
        if ib_id not in _realtime_subscriptions:
            _realtime_subscriptions[ib_id] = set()
        key = f"{contract.symbol}:{contract.exchange}"
        if key in _realtime_subscriptions[ib_id]:
            logger.debug("Already subscribed to %s, skipping duplicate", key)
            return
        _realtime_subscriptions[ib_id].add(key)
        ib.reqMktData(contract)
        # Register callback once per IB instance. On reconnect,
        # clear_realtime_subscriptions removes the old callback before
        # new subscriptions re-add it, preventing duplicate registrations.
        if len(_realtime_subscriptions[ib_id]) == 1:
            ib.pendingTickersEvent += callback
            _realtime_callbacks[ib_id] = callback
    logger.info("Subscribed to real-time data for %s", contract.symbol)


def unsubscribe_realtime(ib: IB, contract: Contract) -> None:
    """Cancel streaming data for a contract."""
    with _realtime_lock:
        key = f"{contract.symbol}:{contract.exchange}"
        ib_id = id(ib)
        if ib_id in _realtime_subscriptions:
            _realtime_subscriptions[ib_id].discard(key)
            # Remove event callback when last subscription is removed
            if not _realtime_subscriptions[ib_id] and ib_id in _realtime_callbacks:
                ib.pendingTickersEvent -= _realtime_callbacks.pop(ib_id)
    ib.cancelMktData(contract)


def clear_realtime_subscriptions(ib: IB) -> None:
    """Clear all subscription tracking for an IB instance.

    Must be called after reconnection — IBKR drops all subscriptions
    on disconnect, so the tracking set must be reset to allow
    re-subscribing. Also removes the event callback to prevent
    duplicate registrations on re-subscribe.
    """
    with _realtime_lock:
        ib_id = id(ib)
        if ib_id in _realtime_subscriptions:
            _realtime_subscriptions[ib_id].clear()
        if ib_id in _realtime_callbacks:
            cb = _realtime_callbacks.pop(ib_id)
            ib.pendingTickersEvent -= cb


# ---------------------------------------------------------------------------
# News (Tavily primary, yfinance fallback)
# ---------------------------------------------------------------------------

_NEWS_TTL = 3600  # 1 hour — reduces Tavily API usage
_NEWS_FAILURE_TTL = 60  # Retry sooner when news fetch fails


def _get_news_yfinance(ticker: str, max_results: int = 5) -> list[str]:
    """Fetch recent news headlines via yfinance (free, no API key)."""
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return []
        headlines = []
        for item in news[:max_results]:
            # yfinance >=1.2: title nested under item["content"]["title"]
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        logger.debug("yfinance news fetch failed for %s: %s", ticker, e)
        return []


def _get_news_tavily(ticker: str, max_results: int = 5) -> Optional[list[str]]:
    """Fetch news via Tavily. Returns None on failure (triggers fallback),
    [] if Tavily is not configured, or a list of headlines on success."""
    if not TAVILY_API_KEY:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        query = f"{ticker} stock"
        response = client.search(query, max_results=max_results, search_depth="basic")
        return [r.get("title", "") for r in response.get("results", []) if r.get("title")]
    except Exception as e:
        logger.warning("Tavily failed for %s: %s — falling back to yfinance", ticker, e)
        return None


def get_news(ticker: str, market: str = "US", max_results: int = 5) -> list[str]:
    """Fetch recent news headlines for a ticker.

    Tries Tavily first (richer results), falls back to yfinance when
    Tavily is unconfigured, errors out (e.g. rate limit), or returns
    no results. Returns a list of headline strings.
    """
    cache_key = f"news:{ticker}:{market}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("News for %s: cache hit (%d headlines)", ticker, len(cached))
        return cached

    # Try Tavily first. None signals an error (rate limit, network, etc.);
    # [] means Tavily unconfigured or returned no results. In both cases
    # we fall back to yfinance so the AI still gets some signal.
    headlines = _get_news_tavily(ticker, max_results)
    if headlines:
        logger.info("News for %s: Tavily OK (%d headlines)", ticker, len(headlines))
        _cache_set(cache_key, headlines, ttl=_NEWS_TTL)
        return headlines

    if headlines is None:
        logger.info("News for %s: Tavily failed — trying yfinance fallback", ticker)
    elif TAVILY_API_KEY:
        logger.info("News for %s: Tavily empty — trying yfinance fallback", ticker)
    else:
        logger.debug("News for %s: Tavily unconfigured — using yfinance", ticker)

    yf_headlines = _get_news_yfinance(ticker, max_results)
    if yf_headlines:
        logger.info("News for %s: yfinance OK (%d headlines)", ticker, len(yf_headlines))
        _cache_set(cache_key, yf_headlines, ttl=_NEWS_TTL)
        return yf_headlines

    logger.warning("News for %s: no headlines from Tavily or yfinance", ticker)
    _cache_set(cache_key, [], ttl=_NEWS_FAILURE_TTL)
    return []


def get_macro_news(max_results: int = 5) -> list[str]:
    """Fetch broad market/political/macro headlines (not stock-specific).

    Called once per scan cycle and shared across all candidates.
    Uses Tavily with a macro-focused query that includes social media coverage.
    """
    cache_key = "macro_news"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            query = "stock market political regulatory macroeconomic trade policy news twitter"
            response = client.search(query, max_results=max_results, search_depth="basic")
            headlines = [r.get("title", "") for r in response.get("results", []) if r.get("title")]
            _cache_set(cache_key, headlines, ttl=_NEWS_TTL)
            logger.info("Tavily macro/X news: fetched %d headlines", len(headlines))
            return headlines
        except Exception as e:
            logger.warning("Tavily macro/X news fetch failed: %s", e)

    headlines: list[str] = []
    _cache_set(cache_key, headlines, ttl=_NEWS_FAILURE_TTL)
    return headlines


# ---------------------------------------------------------------------------
# Analyst Recommendations (yfinance)
# ---------------------------------------------------------------------------

_ANALYST_TTL = 86400  # 24 hours — analyst ratings change infrequently


def get_analyst_recommendation(ticker: str) -> dict | None:
    """Fetch analyst consensus recommendation via yfinance.

    Returns dict with keys:
        consensus: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell" | None
        details: {strong_buy, buy, hold, sell, strong_sell} counts

    Returns None if no data is available or on error.
    """
    cache_key = f"analyst:{ticker}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        summary = yf.Ticker(ticker).recommendations_summary
        if summary is None or summary.empty:
            return None

        row = summary.iloc[0]
        counts = {
            "strong_buy": int(row.get("strongBuy", 0)),
            "buy": int(row.get("buy", 0)),
            "hold": int(row.get("hold", 0)),
            "sell": int(row.get("sell", 0)),
            "strong_sell": int(row.get("strongSell", 0)),
        }

        # Consensus = rating with the most analysts
        consensus_key = max(counts, key=counts.get)
        result = {"consensus": consensus_key, "details": counts}
        _cache_set(cache_key, result, ttl=_ANALYST_TTL)
        logger.info(
            "Analyst recommendation for %s: %s (%s)",
            ticker, consensus_key, counts,
        )
        return result
    except Exception as e:
        logger.warning("Failed to fetch analyst recommendation for %s: %s", ticker, e)
        return None
