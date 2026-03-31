"""Market data service — IBKR primary, YFinance fallback, Tavily news."""

import logging
import time
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
import yfinance as yf
from ib_insync import IB, Contract

from config.settings import TAVILY_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, object]] = {}
_DEFAULT_TTL = 300  # 5 minutes


def _cache_get(key: str) -> Optional[object]:
    if key in _cache:
        expiry, value = _cache[key]
        if time.time() < expiry:
            return value
        del _cache[key]
    return None


def _cache_set(key: str, value: object, ttl: int = _DEFAULT_TTL) -> None:
    _cache[key] = (time.time() + ttl, value)


def clear_cache() -> None:
    """Clear all cached data."""
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
    cache_key = f"hist:{contract.symbol}:{contract.exchange}:{duration}:{bar_size}"
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
) -> pd.DataFrame:
    """Fetch historical data via YFinance. Fallback for backtest mode.

    For BIST tickers, auto-appends .IS suffix if not present.
    """
    cache_key = f"yf:{ticker}:{market}:{period}:{interval}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    yf_ticker = ticker
    if market.upper() == "BIST" and not ticker.endswith(".IS"):
        yf_ticker = f"{ticker}.IS"

    try:
        data = yf.download(yf_ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        logger.error("YFinance download failed for %s: %s", yf_ticker, e)
        return pd.DataFrame()

    if data.empty:
        logger.warning("No YFinance data returned for %s", yf_ticker)
        return pd.DataFrame()

    # Normalize column names (yfinance can return MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"

    _cache_set(cache_key, df, ttl=3600)  # cache longer for historical
    return df


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
    if ticker is None:
        logger.warning("No quote data for %s", contract.symbol)
        return None

    result = {
        "last": ticker.last if ticker.last == ticker.last else None,  # NaN check
        "bid": ticker.bid if ticker.bid == ticker.bid else None,
        "ask": ticker.ask if ticker.ask == ticker.ask else None,
        "volume": ticker.volume if ticker.volume == ticker.volume else None,
        "timestamp": datetime.now(),
    }

    _cache_set(cache_key, result, ttl=30)  # short TTL for quotes
    return result


def subscribe_realtime(
    ib: IB,
    contract: Contract,
    callback: Callable,
) -> None:
    """Subscribe to streaming real-time data via IBKR.

    The callback receives ticker updates.
    """
    ib.reqMktData(contract)
    ib.pendingTickersEvent += callback
    logger.info("Subscribed to real-time data for %s", contract.symbol)


def unsubscribe_realtime(ib: IB, contract: Contract) -> None:
    """Cancel streaming data for a contract."""
    ib.cancelMktData(contract)


# ---------------------------------------------------------------------------
# News (Tavily API)
# ---------------------------------------------------------------------------

def get_news(ticker: str, market: str = "US", max_results: int = 5) -> list[str]:
    """Fetch recent news headlines for a ticker via Tavily API.

    Returns a list of headline strings.
    """
    if not TAVILY_API_KEY:
        logger.debug("No Tavily API key configured — skipping news")
        return []

    cache_key = f"news:{ticker}:{market}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)

        query = f"{ticker} stock"
        if market.upper() == "BIST":
            query = f"{ticker} BIST hisse"

        response = client.search(query, max_results=max_results, search_depth="basic")
        headlines = [r.get("title", "") for r in response.get("results", [])]

        _cache_set(cache_key, headlines, ttl=900)  # 15 min cache
        return headlines

    except Exception as e:
        logger.error("Tavily news fetch failed for %s: %s", ticker, e)
        return []
