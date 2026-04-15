"""Tests for core/universe.py."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.models import StockInfo
from core.universe import (
    _classify_sector_yfinance,
    _classify_sector_ollama,
    _fill_missing_sectors,
    _filter_universe,
    _is_excluded_sector,
    _scan_ibkr,
    _static_fallback,
    cache_universe,
    load_cached_universe,
    get_tickers_for_market,
)


class TestFinancialFilter:
    def test_detects_financials(self):
        assert _is_excluded_sector("Financials") is True
        assert _is_excluded_sector("Financial Services") is True
        assert _is_excluded_sector("Banks") is True
        assert _is_excluded_sector("Insurance") is True
        assert _is_excluded_sector("Consumer Finance") is True
        assert _is_excluded_sector("Capital Markets") is True

    def test_detects_lending_and_interest_businesses(self):
        assert _is_excluded_sector("Lending Services") is True
        assert _is_excluded_sector("Mortgage Finance") is True
        assert _is_excluded_sector("Consumer Lending") is True
        assert _is_excluded_sector("Microfinance") is True
        assert _is_excluded_sector("Payday Loans") is True
        assert _is_excluded_sector("Credit Services") is True
        assert _is_excluded_sector("Debt Collection") is True

    def test_allows_non_financials(self):
        assert _is_excluded_sector("Technology") is False
        assert _is_excluded_sector("Healthcare") is False
        assert _is_excluded_sector("Energy") is False
        assert _is_excluded_sector("Industrials") is False
        assert _is_excluded_sector("") is False

    def test_case_insensitive(self):
        assert _is_excluded_sector("FINANCIALS") is True
        assert _is_excluded_sector("banking") is True

    def test_excludes_non_equity_etfs(self):
        assert _is_excluded_sector("Bond ETF") is True
        assert _is_excluded_sector("Leveraged ETF") is True
        assert _is_excluded_sector("Non-Stock ETF") is True

    def test_keeps_equity_etfs(self):
        assert _is_excluded_sector("Equity ETF") is False

    def test_detects_defense_and_military(self):
        assert _is_excluded_sector("Aerospace & Defense") is True
        assert _is_excluded_sector("Defense") is True
        assert _is_excluded_sector("Defence") is True
        assert _is_excluded_sector("Military Equipment") is True
        assert _is_excluded_sector("Weapons & Ammunition") is True
        assert _is_excluded_sector("Arms Manufacturer") is True
        assert _is_excluded_sector("Missile Systems") is True
        assert _is_excluded_sector("Combat Systems") is True
        assert _is_excluded_sector("Ordnance & Accessories") is True

    def test_defense_case_insensitive(self):
        assert _is_excluded_sector("AEROSPACE & DEFENSE") is True
        assert _is_excluded_sector("military") is True

    def test_allows_non_defense_industrials(self):
        assert _is_excluded_sector("Industrials") is False
        assert _is_excluded_sector("Industrial Machinery") is False
        assert _is_excluded_sector("Aerospace Parts") is False


class TestFilterUniverse:
    def test_removes_financials(self):
        stocks = [
            StockInfo("AAPL", "SMART", "Technology", 0, 0),
            StockInfo("JPM", "SMART", "Financials", 0, 0),
            StockInfo("BAC", "SMART", "Banks", 0, 0),
            StockInfo("MSFT", "SMART", "Technology", 0, 0),
        ]
        filtered = _filter_universe(stocks)
        tickers = {s.ticker for s in filtered}
        assert "JPM" not in tickers
        assert "BAC" not in tickers
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_removes_defense(self):
        stocks = [
            StockInfo("AAPL", "SMART", "Technology", 0, 0),
            StockInfo("LMT", "SMART", "Aerospace & Defense", 0, 0),
            StockInfo("RTX", "SMART", "Defense", 0, 0),
            StockInfo("CAT", "SMART", "Industrials", 0, 0),
        ]
        filtered = _filter_universe(stocks)
        tickers = {s.ticker for s in filtered}
        assert "LMT" not in tickers
        assert "RTX" not in tickers
        assert "AAPL" in tickers
        assert "CAT" in tickers

    def test_removes_excluded_countries(self):
        stocks = [
            StockInfo("AAPL", "SMART", "Technology", 0, 0, country="United States"),
            StockInfo("NEWIL", "SMART", "Technology", 0, 0, country="Israel"),
            StockInfo("MSFT", "SMART", "Technology", 0, 0, country="United States"),
        ]
        filtered = _filter_universe(stocks)
        tickers = {s.ticker for s in filtered}
        assert "NEWIL" not in tickers
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_empty_country_passes(self):
        """Stocks without country info should not be excluded by country filter."""
        stocks = [StockInfo("UNK", "SMART", "Technology", 0, 0, country="")]
        filtered = _filter_universe(stocks)
        assert len(filtered) == 1

    def test_low_volume_filtered(self):
        stocks = [
            StockInfo("TINY", "SMART", "Technology", 0, 50_000),  # below MIN_DAILY_VOLUME
            StockInfo("BIG", "SMART", "Technology", 0, 500_000),
        ]
        filtered = _filter_universe(stocks)
        tickers = {s.ticker for s in filtered}
        assert "TINY" not in tickers
        assert "BIG" in tickers

    def test_zero_volume_passes(self):
        """Stocks with no volume info (0) should pass through."""
        stocks = [StockInfo("UNK", "SMART", "Technology", 0, 0)]
        filtered = _filter_universe(stocks)
        assert len(filtered) == 1

    def test_equity_etf_passes_filter(self):
        stocks = [StockInfo("SPY", "SMART", "Equity ETF", 0, 0)]
        filtered = _filter_universe(stocks)
        assert len(filtered) == 1

    def test_bond_etf_filtered(self):
        stocks = [StockInfo("HYG", "SMART", "Bond ETF", 0, 0)]
        filtered = _filter_universe(stocks)
        assert len(filtered) == 0

    def test_leveraged_etf_filtered(self):
        stocks = [StockInfo("TQQQ", "SMART", "Leveraged ETF", 0, 0)]
        filtered = _filter_universe(stocks)
        assert len(filtered) == 0


class TestStaticFallback:
    def test_us_fallback_has_stocks(self):
        stocks = _static_fallback("US")
        assert len(stocks) > 50
        tickers = {s.ticker for s in stocks}
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "NVDA" in tickers

    def test_us_fallback_no_financials(self):
        stocks = _static_fallback("US")
        for s in stocks:
            assert not _is_excluded_sector(s.sector), f"{s.ticker} is financial"

    def test_unknown_market(self):
        stocks = _static_fallback("MOON")
        assert stocks == []


class TestCacheIO:
    def test_cache_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.universe.DATA_DIR", tmp_path)
        monkeypatch.setattr(
            "core.universe._cache_path",
            lambda m: tmp_path / f"test_{m}.json",
        )

        stocks = [
            StockInfo("AAPL", "SMART", "Technology", 3e12, 50e6, "USD", "Apple Inc"),
            StockInfo("MSFT", "SMART", "Technology", 2e12, 30e6, "USD", "Microsoft"),
        ]
        cache_universe(stocks, "TEST")

        loaded = load_cached_universe("TEST")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].ticker == "AAPL"
        assert loaded[1].ticker == "MSFT"


class TestYFinanceSectorFallback:
    @patch("core.universe.yf.Ticker")
    def test_classify_sector_returns_sector_and_country(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "sector": "Technology",
            "country": "United States",
        }
        sector, country = _classify_sector_yfinance("AAPL")
        assert sector == "Technology"
        assert country == "United States"
        mock_ticker_cls.assert_called_once_with("AAPL")

    @patch("core.universe.yf.Ticker")
    def test_classify_sector_returns_none_on_empty_info(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {}
        sector, country = _classify_sector_yfinance("FAKE")
        assert sector is None
        assert country is None

    @patch("core.universe.yf.Ticker")
    def test_classify_sector_returns_none_on_exception(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = property(
            lambda self: (_ for _ in ()).throw(Exception("network error"))
        )
        mock_ticker_cls.side_effect = Exception("network error")
        sector, country = _classify_sector_yfinance("FAKE")
        assert sector is None
        assert country is None

    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_enriches_stock(self, mock_classify):
        mock_classify.return_value = ("Technology", "United States")
        stocks = [StockInfo("AAPL", "SMART", "", 0, 0)]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 1
        assert result[0].sector == "Technology"
        assert result[0].country == "United States"

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_excludes_unclassifiable(self, mock_classify, mock_ollama):
        mock_classify.return_value = (None, None)
        mock_ollama.return_value = (None, None)
        stocks = [StockInfo("FAKE", "SMART", "", 0, 0)]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 0

    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_skips_already_enriched(self, mock_classify):
        stocks = [StockInfo("AAPL", "SMART", "Healthcare", 0, 0)]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 1
        assert result[0].sector == "Healthcare"
        mock_classify.assert_not_called()

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_mixed(self, mock_classify, mock_ollama):
        def side_effect(ticker):
            if ticker == "GOOG":
                return ("Technology", "United States")
            return (None, None)
        mock_classify.side_effect = side_effect
        mock_ollama.return_value = (None, None)

        stocks = [
            StockInfo("AAPL", "SMART", "Healthcare", 0, 0),  # already has sector
            StockInfo("GOOG", "SMART", "", 0, 0),             # yfinance succeeds
            StockInfo("FAKE", "SMART", "", 0, 0),             # all fallbacks fail
        ]
        result = _fill_missing_sectors(stocks)
        tickers = {s.ticker for s in result}
        assert tickers == {"AAPL", "GOOG"}
        assert len(result) == 2

    @patch("core.universe.yf.Ticker")
    def test_classify_bond_etf(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "quoteType": "ETF",
            "category": "High Yield Bond",
            "country": "United States",
        }
        sector, country = _classify_sector_yfinance("HYG")
        assert sector == "Bond ETF"

    @patch("core.universe.yf.Ticker")
    def test_classify_leveraged_etf(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "quoteType": "ETF",
            "category": "Trading--Leveraged Equity",
        }
        sector, _ = _classify_sector_yfinance("TQQQ")
        assert sector == "Leveraged ETF"

    @patch("core.universe.yf.Ticker")
    def test_classify_inverse_etf(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "quoteType": "ETF",
            "category": "Trading--Inverse Equity",
        }
        sector, _ = _classify_sector_yfinance("SQQQ")
        assert sector == "Leveraged ETF"

    @patch("core.universe.yf.Ticker")
    def test_classify_commodity_etf(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "quoteType": "ETF",
            "category": "Commodity",
        }
        sector, _ = _classify_sector_yfinance("USO")
        assert sector == "Non-Stock ETF"

    @patch("core.universe.yf.Ticker")
    def test_classify_equity_etf(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "quoteType": "ETF",
            "category": "Large Blend",
            "country": "United States",
        }
        sector, country = _classify_sector_yfinance("SPY")
        assert sector == "Equity ETF"
        assert country == "United States"

    @patch("core.universe._classify_sector_yfinance")
    def test_financial_from_yfinance_gets_filtered(self, mock_classify):
        """End-to-end: yfinance identifies a financial, then _filter_universe removes it."""
        mock_classify.return_value = ("Financial Services", "United States")
        stocks = [StockInfo("JPM", "SMART", "", 0, 0)]
        stocks = _fill_missing_sectors(stocks)
        assert len(stocks) == 1
        assert stocks[0].sector == "Financial Services"
        # Now the filter should catch it
        stocks = _filter_universe(stocks)
        assert len(stocks) == 0


class TestOllamaSectorFallback:
    @patch("core.universe.urllib.request.urlopen")
    def test_classify_sector_ollama_returns_sector_and_country(self, mock_urlopen):
        import json
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": json.dumps({"sector": "Technology", "country": "United States"}),
        }).encode()
        mock_urlopen.return_value = mock_response

        sector, country = _classify_sector_ollama("AAPL", "APPLE INC")
        assert sector == "Technology"
        assert country == "United States"

    @patch("core.universe.urllib.request.urlopen")
    def test_classify_sector_ollama_returns_none_on_failure(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection refused")
        sector, country = _classify_sector_ollama("FAKE", "FAKE CORP")
        assert sector is None
        assert country is None

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_uses_ollama_when_yfinance_fails(self, mock_yf, mock_ollama):
        mock_yf.return_value = (None, None)
        mock_ollama.return_value = ("Energy", "Canada")
        stocks = [StockInfo("FAKE", "SMART", "", 0, 0, name="FAKE ENERGY CORP")]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 1
        assert result[0].sector == "Energy"
        assert result[0].country == "Canada"
        mock_ollama.assert_called_once()

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_excludes_when_all_fail(self, mock_yf, mock_ollama):
        mock_yf.return_value = (None, None)
        mock_ollama.return_value = (None, None)
        stocks = [StockInfo("FAKE", "SMART", "", 0, 0)]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 0


class TestGetTickersForMarket:
    def test_filter_us(self):
        universe = [
            StockInfo("AAPL", "SMART", "Tech", 0, 0),
            StockInfo("MSFT", "NYSE", "Tech", 0, 0),
        ]
        us = get_tickers_for_market(universe, "US")
        assert len(us) == 2


class TestScannerTimeout:
    """Regression: a hung reqScannerData call used to freeze build_universe
    for hours. Each scanner call must be bounded by a timeout so one bad
    scan can't stall the rest."""

    def _make_ib_with_scans(self, scan_behaviors):
        """Build a fake IB whose reqScannerDataAsync follows a list of behaviors.

        Each behavior is either:
          - a list of mock items (returned normally)
          - the string "hang" (simulates indefinite block via TimeoutError)
        """
        import asyncio

        ib = MagicMock()
        call_idx = {"i": 0}

        async def fake_scan_async(sub):
            idx = call_idx["i"]
            call_idx["i"] += 1
            behavior = scan_behaviors[idx] if idx < len(scan_behaviors) else []
            if behavior == "hang":
                await asyncio.sleep(60)  # longer than any sane timeout
            return behavior

        ib.reqScannerDataAsync = fake_scan_async
        return ib, call_idx

    def test_hanging_scanner_does_not_block_subsequent_scans(self):
        """If scanner #3 hangs, scanners #4..#10 must still run."""
        # Scanner 3 hangs; the other 9 return one result each (different tickers).
        behaviors = []
        for i in range(10):
            if i == 2:
                behaviors.append("hang")
            else:
                item = MagicMock()
                item.contractDetails.contract.symbol = f"TICK{i}"
                item.contractDetails.contract.currency = "USD"
                item.contractDetails.contract.primaryExchange = "NASDAQ"
                item.contractDetails.industry = "Technology"
                item.contractDetails.category = ""
                item.contractDetails.longName = f"Test {i}"
                behaviors.append([item])

        ib, counter = self._make_ib_with_scans(behaviors)

        # With a short timeout, the hanging scanner must be abandoned and the
        # rest of the scans must complete. The whole call should finish in
        # well under 30s — proving it's bounded, not hanging for hours.
        import time
        start = time.monotonic()
        result = _scan_ibkr(ib, "US", scan_timeout=0.5)
        elapsed = time.monotonic() - start

        assert elapsed < 10, f"_scan_ibkr took {elapsed}s — not bounded"
        assert counter["i"] == 10, f"Only {counter['i']}/10 scanners ran"
        tickers = {s.ticker for s in result}
        # 9 good scanners returned unique tickers
        assert len(tickers) == 9
