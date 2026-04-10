"""Stock universe builder — discovers tradeable tickers, filters financials."""

import json
import logging
import urllib.request
from datetime import date
from pathlib import Path
from typing import Optional

import yfinance as yf
from ib_insync import IB, Stock, ScannerSubscription

from config.settings import (
    AI_MODEL, DATA_DIR, EXCLUDED_COUNTRIES, EXCLUDED_SECTORS, EXCLUDED_TICKERS,
    FINANCIAL_KEYWORDS, DEFENSE_KEYWORDS, MIN_DAILY_VOLUME, MIN_MARKET_CAP,
    OLLAMA_HOST,
)
from core.models import StockInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def _cache_path(market: str) -> Path:
    return DATA_DIR / f"universe_{market.lower()}_{date.today().isoformat()}.json"


# ---------------------------------------------------------------------------
# Build universe via IBKR scanner
# ---------------------------------------------------------------------------

def build_universe(
    ib: IB,
    markets: list[str],
) -> list[StockInfo]:
    """Build the full tradeable stock universe for given markets.

    Tries IBKR scanner first; falls back to cached/static list if needed.
    Filters out financial sector stocks and applies liquidity thresholds.
    """
    all_stocks: list[StockInfo] = []

    for market in markets:
        market = market.upper()

        # Try cache first (universe doesn't change intraday)
        cached = load_cached_universe(market)
        if cached:
            # Re-apply filter to catch exclusion rule changes since cache was written
            cached = _filter_universe(cached)
            logger.info("Loaded %d cached %s stocks", len(cached), market)
            all_stocks.extend(cached)
            continue

        # Try IBKR scanner
        stocks = _scan_ibkr(ib, market)

        if not stocks:
            logger.warning(
                "IBKR scanner returned nothing for %s — using static fallback", market
            )
            stocks = _static_fallback(market)

        # Enrich with sector/name from contract details (needed for filtering)
        stocks = _enrich_with_contract_details(ib, stocks)

        # YFinance fallback for stocks still missing sector data
        stocks = _fill_missing_sectors(stocks)

        # Filter
        stocks = _filter_universe(stocks)
        logger.info("Universe for %s: %d stocks after filtering", market, len(stocks))

        # Cache for the day
        cache_universe(stocks, market)
        all_stocks.extend(stocks)

    return all_stocks


def _scan_ibkr(ib: IB, market: str) -> list[StockInfo]:
    """Use IBKR scanner to get stocks for a market."""
    stocks: list[StockInfo] = []

    if market == "US":
        # Run multiple scan types to build a larger universe.
        # Each scan returns ~50 results; dedup combines them.
        scans = [
            ("MOST_ACTIVE", "STK.US.MAJOR"),
            ("TOP_PERC_GAIN", "STK.US.MAJOR"),
            ("TOP_PERC_LOSE", "STK.US.MAJOR"),
            ("HOT_BY_VOLUME", "STK.US.MAJOR"),
            ("TOP_OPEN_PERC_GAIN", "STK.US.MAJOR"),
            ("TOP_OPEN_PERC_LOSE", "STK.US.MAJOR"),
            ("HIGH_VS_13W_HL", "STK.US.MAJOR"),
            ("LOW_VS_13W_HL", "STK.US.MAJOR"),
            ("TOP_TRADE_COUNT", "STK.US.MAJOR"),
            ("TOP_TRADE_RATE", "STK.US.MAJOR"),
        ]
    else:
        logger.warning("Unknown market: %s", market)
        return []

    seen_tickers: set[str] = set()

    for scan_code, location_code in scans:
        try:
            sub = ScannerSubscription(
                instrument="STK",
                locationCode=location_code,
                scanCode=scan_code,
                numberOfRows=50,
            )
            results = ib.reqScannerData(sub)

            added = 0
            for item in results:
                contract = item.contractDetails.contract
                if contract.symbol in seen_tickers:
                    continue
                seen_tickers.add(contract.symbol)

                details = item.contractDetails

                sector = getattr(details, "industry", "") or getattr(details, "category", "") or ""
                market_cap = 0.0  # IBKR scanner doesn't directly give market cap
                avg_volume = 0.0

                exchange = getattr(contract, "primaryExchange", "") or "SMART"

                stocks.append(StockInfo(
                    ticker=contract.symbol,
                    exchange=exchange,
                    sector=sector,
                    market_cap=market_cap,
                    avg_volume=avg_volume,
                    currency=contract.currency,
                    name=getattr(details, "longName", ""),
                ))
                added += 1

            logger.info(
                "IBKR scanner %s returned %d results (%d new)",
                scan_code, len(results), added,
            )
        except Exception as e:
            logger.error("IBKR scanner %s failed: %s", scan_code, e)

    return stocks


def _enrich_with_contract_details(
    ib: IB, stocks: list[StockInfo],
) -> list[StockInfo]:
    """Fetch contract details to fill in missing sector/name info.

    The IBKR scanner doesn't return sector data, so we need to call
    reqContractDetails for each stock to get it. This is essential for
    the financial sector filter to work.

    Respects IBKR pacing with small sleeps between requests.
    """
    need_enrichment = [s for s in stocks if not s.sector or s.sector in ("", "Unknown")]

    if not need_enrichment:
        return stocks

    logger.info("Enriching %d stocks with contract details...", len(need_enrichment))

    enriched_count = 0
    for i, stock in enumerate(need_enrichment, 1):
        try:
            contract = Stock(stock.ticker, "SMART", "USD")
            details_list = ib.reqContractDetails(contract)
            if details_list:
                d = details_list[0]
                stock.sector = getattr(d, "industry", "") or getattr(d, "category", "") or ""
                stock.name = getattr(d, "longName", "") or ""
                enriched_count += 1

            # IBKR pacing: small delay per request to avoid pacing violations
            ib.sleep(0.05)
            if i % 50 == 0:
                logger.info("Enriched %d/%d stocks...", i, len(need_enrichment))
                ib.sleep(1)

        except Exception as e:
            logger.debug("Could not enrich %s: %s", stock.ticker, e)

    logger.info(
        "Enrichment complete: %d/%d stocks got sector data",
        enriched_count, len(need_enrichment),
    )
    return stocks


# ---------------------------------------------------------------------------
# YFinance sector fallback
# ---------------------------------------------------------------------------

def _classify_sector_yfinance(ticker: str) -> tuple[Optional[str], Optional[str]]:
    """Look up sector and country via yfinance for a single ticker.

    For regular stocks, returns the sector directly.
    For ETFs, classifies by category: Equity ETF (kept), Bond/Leveraged/Non-Stock ETF (excluded).

    Returns (sector, country) or (None, None) on failure.
    """
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or None
        country = info.get("country") or None
        if sector:
            return sector, country

        # For ETFs: use the category field to classify
        quote_type = info.get("quoteType", "")
        category = info.get("category", "")
        if quote_type == "ETF" and category:
            cat_lower = category.lower()
            if any(kw in cat_lower for kw in ("bond", "income", "treasury", "debt", "money market")):
                return "Bond ETF", country
            elif any(kw in cat_lower for kw in ("leverag", "inverse")):
                return "Leveraged ETF", country
            elif any(kw in cat_lower for kw in ("commodity", "volatility", "crypto", "currency")):
                return "Non-Stock ETF", country
            else:
                return "Equity ETF", country

        return None, None
    except Exception as e:
        logger.debug("yfinance lookup failed for %s: %s", ticker, e)
        return None, None


def _classify_sector_ollama(
    ticker: str, name: str = "", max_retries: int = 2,
) -> tuple[Optional[str], Optional[str]]:
    """Use Ollama LLM to classify sector and country for a ticker.

    Returns (sector, country) or (None, None) on failure.
    """
    prompt = (
        f"What sector and country does the stock ticker '{ticker}'"
        f"{f' ({name})' if name else ''} belong to?\n\n"
        "If this is an ETF, leveraged product, warrant, unit, right, or "
        "preferred share — return sector as 'ETF' or 'Non-Stock'.\n\n"
        'Return JSON: {{"sector": "...", "country": "..."}}\n'
        "Use standard sector names: Technology, Healthcare, Energy, "
        "Consumer Cyclical, Consumer Defensive, Industrials, Materials, "
        "Utilities, Real Estate, Communication, Financials, ETF, Non-Stock."
    )
    for attempt in range(1, max_retries + 1):
        try:
            payload = json.dumps({
                "model": AI_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"num_predict": 128},
            }).encode()

            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            response = urllib.request.urlopen(req, timeout=30)
            result = json.loads(response.read())
            data = json.loads(result["response"])

            sector = data.get("sector") or None
            country = data.get("country") or None
            if sector:
                return sector, country
        except Exception as e:
            logger.debug(
                "Ollama sector lookup failed for %s (attempt %d): %s",
                ticker, attempt, e,
            )
    return None, None


def _fill_missing_sectors(stocks: list[StockInfo]) -> list[StockInfo]:
    """Fill sector/country for stocks missing it: yfinance -> Ollama -> exclude.

    Stocks that remain unclassified after all fallbacks are excluded entirely.
    """
    result = []
    need_lookup = 0
    excluded = 0
    ollama_used = 0

    for stock in stocks:
        if stock.sector and stock.sector not in ("", "Unknown"):
            result.append(stock)
            continue

        need_lookup += 1

        # Try yfinance first
        sector, country = _classify_sector_yfinance(stock.ticker)
        if sector:
            stock.sector = sector
            stock.country = country or ""
            result.append(stock)
            continue

        # Try Ollama as last resort
        sector, country = _classify_sector_ollama(stock.ticker, stock.name)
        if sector:
            stock.sector = sector
            stock.country = country or ""
            ollama_used += 1
            result.append(stock)
            continue

        excluded += 1
        logger.warning(
            "Excluding %s: no sector data from IBKR, yfinance, or Ollama",
            stock.ticker,
        )

    if need_lookup:
        logger.info(
            "Sector fallback: %d lookups, %d via Ollama, %d excluded",
            need_lookup, ollama_used, excluded,
        )
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_universe(stocks: list[StockInfo]) -> list[StockInfo]:
    """Remove financial sector stocks, excluded countries, and apply liquidity filters."""
    filtered = []
    for s in stocks:
        # Exclude specific tickers
        if s.ticker in EXCLUDED_TICKERS:
            continue

        # Exclude by country
        if s.country and s.country in EXCLUDED_COUNTRIES:
            continue

        # Exclude financial and defense sectors
        if _is_excluded_sector(s.sector):
            continue

        # Also check company name for defense/financial keywords — catches
        # companies with generic sector labels like "Industrials" that are
        # actually defense contractors (e.g., Raytheon, Lockheed Martin)
        if s.name and _is_excluded_by_name(s.name):
            continue

        # Apply liquidity filters only if data is available
        # (scanner results may not have volume/market_cap, let them through)
        if s.avg_volume > 0 and s.avg_volume < MIN_DAILY_VOLUME:
            continue
        if s.market_cap > 0 and s.market_cap < MIN_MARKET_CAP:
            continue

        filtered.append(s)

    return filtered


def _is_excluded_sector(sector: str) -> bool:
    """Check if the sector should be excluded (financials, defense, non-equity ETFs)."""
    if not sector:
        return False
    sector_lower = sector.lower()

    # Exclude non-equity ETFs (bond, leveraged, commodity, etc.)
    # "Equity ETF" intentionally NOT in this list — those are kept
    non_equity_etf_types = ["bond etf", "leveraged etf", "non-stock etf"]
    if any(t in sector_lower for t in non_equity_etf_types):
        return True

    for excluded in EXCLUDED_SECTORS:
        if excluded.lower() in sector_lower:
            return True
    if any(kw in sector_lower for kw in FINANCIAL_KEYWORDS):
        return True
    return any(kw in sector_lower for kw in DEFENSE_KEYWORDS)


def _is_excluded_by_name(name: str) -> bool:
    """Check if company name contains defense or financial keywords.

    Catches companies with generic sector labels (e.g. "Industrials")
    that are actually defense contractors or financial firms.
    """
    name_lower = name.lower()
    for kw in DEFENSE_KEYWORDS:
        if kw in name_lower:
            return True
    for kw in FINANCIAL_KEYWORDS:
        if kw in name_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Static fallback tickers
# ---------------------------------------------------------------------------

def _static_fallback(market: str) -> list[StockInfo]:
    """Return a static list of well-known tickers when IBKR scanner is unavailable."""
    if market == "US":
        tickers = [
            # Technology
            ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
            ("AMZN", "Consumer Cyclical"), ("META", "Technology"),
            ("NVDA", "Technology"), ("TSM", "Technology"), ("AVGO", "Technology"),
            ("ORCL", "Technology"), ("CRM", "Technology"), ("AMD", "Technology"),
            ("INTC", "Technology"), ("QCOM", "Technology"), ("TXN", "Technology"),
            ("AMAT", "Technology"), ("MU", "Technology"), ("NOW", "Technology"),
            ("ADBE", "Technology"), ("SNPS", "Technology"), ("CDNS", "Technology"),
            # Healthcare
            ("UNH", "Healthcare"), ("JNJ", "Healthcare"), ("LLY", "Healthcare"),
            ("PFE", "Healthcare"), ("ABBV", "Healthcare"), ("MRK", "Healthcare"),
            ("TMO", "Healthcare"), ("ABT", "Healthcare"), ("DHR", "Healthcare"),
            ("BMY", "Healthcare"), ("AMGN", "Healthcare"), ("GILD", "Healthcare"),
            # Consumer
            ("TSLA", "Consumer Cyclical"), ("HD", "Consumer Cyclical"),
            ("NKE", "Consumer Cyclical"), ("MCD", "Consumer Cyclical"),
            ("SBUX", "Consumer Cyclical"), ("TGT", "Consumer Cyclical"),
            ("LOW", "Consumer Cyclical"), ("BKNG", "Consumer Cyclical"),
            ("PG", "Consumer Defensive"), ("KO", "Consumer Defensive"),
            ("PEP", "Consumer Defensive"), ("COST", "Consumer Defensive"),
            ("WMT", "Consumer Defensive"), ("CL", "Consumer Defensive"),
            # Industrials
            ("CAT", "Industrials"), ("HON", "Industrials"),
            ("UPS", "Industrials"), ("DE", "Industrials"),
            ("GE", "Industrials"), ("MMM", "Industrials"),
            ("UNP", "Industrials"),
            # Energy
            ("XOM", "Energy"), ("CVX", "Energy"), ("COP", "Energy"),
            ("SLB", "Energy"), ("EOG", "Energy"), ("MPC", "Energy"),
            # Materials
            ("LIN", "Materials"), ("APD", "Materials"), ("SHW", "Materials"),
            ("FCX", "Materials"), ("NEM", "Materials"),
            # Communication
            ("DIS", "Communication"), ("CMCSA", "Communication"),
            ("NFLX", "Communication"), ("T", "Communication"),
            ("VZ", "Communication"), ("TMUS", "Communication"),
            # Utilities
            ("NEE", "Utilities"), ("DUK", "Utilities"), ("SO", "Utilities"),
            # Real Estate
            ("PLD", "Real Estate"), ("AMT", "Real Estate"), ("SPG", "Real Estate"),
        ]
        return [
            StockInfo(
                ticker=t, exchange="SMART", sector=s,
                market_cap=0, avg_volume=0, currency="USD",
            )
            for t, s in tickers
        ]

    return []


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def cache_universe(stocks: list[StockInfo], market: str) -> None:
    """Save the universe to a daily JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(market)

    data = [
        {
            "ticker": s.ticker,
            "exchange": s.exchange,
            "sector": s.sector,
            "market_cap": s.market_cap,
            "avg_volume": s.avg_volume,
            "currency": s.currency,
            "name": s.name,
            "country": s.country,
        }
        for s in stocks
    ]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Cached %d %s stocks to %s", len(stocks), market, path)


def load_cached_universe(market: str) -> Optional[list[StockInfo]]:
    """Load today's cached universe if it exists."""
    path = _cache_path(market)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return [
            StockInfo(
                ticker=d["ticker"],
                exchange=d["exchange"],
                sector=d.get("sector", ""),
                market_cap=d.get("market_cap", 0),
                avg_volume=d.get("avg_volume", 0),
                currency=d.get("currency", "USD"),
                name=d.get("name", ""),
                country=d.get("country", ""),
            )
            for d in data
        ]
    except Exception as e:
        logger.error("Failed to load cached universe: %s", e)
        return None


def get_tickers_for_market(
    universe: list[StockInfo], market: str,
) -> list[StockInfo]:
    """Filter universe to a specific market."""
    market = market.upper()
    if market == "US":
        us_exchanges = {"SMART", "NYSE", "NASDAQ", "ARCA", "BATS", "IEX", "AMEX", "ISLAND"}
        return [s for s in universe if s.exchange in us_exchanges]
    logger.warning("Unknown market '%s' in get_tickers_for_market — returning empty list", market)
    return []
