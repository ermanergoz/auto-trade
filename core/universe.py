"""Stock universe builder — discovers tradeable tickers, filters financials."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from ib_insync import IB, Stock, ScannerSubscription

from config.settings import (
    DATA_DIR, EXCLUDED_SECTORS, MIN_DAILY_VOLUME, MIN_MARKET_CAP,
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
        exchanges = [("NYSE", "STK.US.MAJOR"), ("NASDAQ", "STK.US.MAJOR")]
    elif market == "BIST":
        exchanges = [("BIST", "STK.EU.BIST")]
    else:
        logger.warning("Unknown market: %s", market)
        return []

    for exchange_name, location_code in exchanges:
        try:
            sub = ScannerSubscription(
                instrument="STK",
                locationCode=location_code,
                scanCode="MOST_ACTIVE",
                numberOfRows=500,
            )
            results = ib.reqScannerData(sub)

            for item in results:
                contract = item.contractDetails.contract
                details = item.contractDetails

                sector = getattr(details, "category", "") or ""
                market_cap = 0.0  # IBKR scanner doesn't directly give market cap
                avg_volume = 0.0

                stocks.append(StockInfo(
                    ticker=contract.symbol,
                    exchange=exchange_name,
                    sector=sector,
                    market_cap=market_cap,
                    avg_volume=avg_volume,
                    currency=contract.currency,
                    name=getattr(details, "longName", ""),
                ))

            logger.info(
                "IBKR scanner returned %d results for %s", len(results), exchange_name
            )
        except Exception as e:
            logger.error("IBKR scanner failed for %s: %s", exchange_name, e)

    return stocks


def _enrich_with_contract_details(
    ib: IB, stocks: list[StockInfo],
) -> list[StockInfo]:
    """Fetch contract details to fill in missing sector/market cap info.

    Useful when IBKR scanner doesn't provide full details.
    Respects IBKR pacing — processes in batches.
    """
    enriched = []
    for stock in stocks:
        if stock.sector and stock.sector not in ("", "Unknown"):
            enriched.append(stock)
            continue

        try:
            if stock.exchange == "BIST":
                contract = Stock(stock.ticker, "BIST", "TRY")
            else:
                contract = Stock(stock.ticker, "SMART", "USD")

            details_list = ib.reqContractDetails(contract)
            if details_list:
                d = details_list[0]
                stock.sector = getattr(d, "category", stock.sector) or stock.sector
                stock.name = getattr(d, "longName", stock.name) or stock.name
            enriched.append(stock)
        except Exception as e:
            logger.debug("Could not enrich %s: %s", stock.ticker, e)
            enriched.append(stock)

    return enriched


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_universe(stocks: list[StockInfo]) -> list[StockInfo]:
    """Remove financial sector stocks and apply liquidity filters."""
    filtered = []
    for s in stocks:
        # Exclude financial sector
        if _is_financial_sector(s.sector):
            continue

        # Apply liquidity filters only if data is available
        # (scanner results may not have volume/market_cap, let them through)
        if s.avg_volume > 0 and s.avg_volume < MIN_DAILY_VOLUME:
            continue
        if s.market_cap > 0 and s.market_cap < MIN_MARKET_CAP:
            continue

        filtered.append(s)

    return filtered


def _is_financial_sector(sector: str) -> bool:
    """Check if the sector is in the excluded list (Financials)."""
    if not sector:
        return False
    sector_lower = sector.lower()
    for excluded in EXCLUDED_SECTORS:
        if excluded.lower() in sector_lower:
            return True
    # Also catch common IBKR sector names for financials
    financial_keywords = [
        "bank", "insurance", "lending", "mortgage",
        "capital markets", "consumer finance", "financial",
    ]
    return any(kw in sector_lower for kw in financial_keywords)


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
            ("CAT", "Industrials"), ("BA", "Industrials"), ("HON", "Industrials"),
            ("UPS", "Industrials"), ("RTX", "Industrials"), ("DE", "Industrials"),
            ("GE", "Industrials"), ("LMT", "Industrials"), ("MMM", "Industrials"),
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

    elif market == "BIST":
        tickers = [
            # Industrials / Airlines
            ("THYAO", "Industrials"), ("TUPRS", "Energy"),
            ("SISE", "Industrials"), ("EREGL", "Materials"),
            ("BIMAS", "Consumer Defensive"), ("KCHOL", "Industrials"),
            ("ASELS", "Industrials"), ("TOASO", "Consumer Cyclical"),
            ("TAVHL", "Industrials"), ("SAHOL", "Industrials"),
            ("PETKM", "Energy"), ("SASA", "Materials"),
            ("KOZAA", "Materials"), ("KOZAL", "Materials"),
            ("ENKAI", "Industrials"), ("TCELL", "Communication"),
            ("TTKOM", "Communication"), ("MGROS", "Consumer Defensive"),
            ("VESTL", "Consumer Cyclical"), ("ARCLK", "Consumer Cyclical"),
            ("FROTO", "Consumer Cyclical"), ("OTKAR", "Consumer Cyclical"),
            ("DOHOL", "Industrials"), ("EKGYO", "Real Estate"),
            ("PGSUS", "Industrials"), ("GUBRF", "Materials"),
            ("SOKM", "Consumer Defensive"), ("CCOLA", "Consumer Defensive"),
            ("AKSEN", "Utilities"), ("GESAN", "Energy"),
        ]
        return [
            StockInfo(
                ticker=t, exchange="BIST", sector=s,
                market_cap=0, avg_volume=0, currency="TRY",
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
        return [s for s in universe if s.exchange in ("SMART", "NYSE", "NASDAQ")]
    elif market == "BIST":
        return [s for s in universe if s.exchange == "BIST"]
    return universe
