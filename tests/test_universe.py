"""Tests for core/universe.py."""

import json
from pathlib import Path

import pytest

from core.models import StockInfo
from core.universe import (
    _filter_universe,
    _is_financial_sector,
    _static_fallback,
    cache_universe,
    load_cached_universe,
    get_tickers_for_market,
)


class TestFinancialFilter:
    def test_detects_financials(self):
        assert _is_financial_sector("Financials") is True
        assert _is_financial_sector("Financial Services") is True
        assert _is_financial_sector("Banks") is True
        assert _is_financial_sector("Insurance") is True
        assert _is_financial_sector("Consumer Finance") is True
        assert _is_financial_sector("Capital Markets") is True

    def test_allows_non_financials(self):
        assert _is_financial_sector("Technology") is False
        assert _is_financial_sector("Healthcare") is False
        assert _is_financial_sector("Energy") is False
        assert _is_financial_sector("Industrials") is False
        assert _is_financial_sector("") is False

    def test_case_insensitive(self):
        assert _is_financial_sector("FINANCIALS") is True
        assert _is_financial_sector("banking") is True


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
            assert not _is_financial_sector(s.sector), f"{s.ticker} is financial"

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


class TestGetTickersForMarket:
    def test_filter_us(self):
        universe = [
            StockInfo("AAPL", "SMART", "Tech", 0, 0),
            StockInfo("MSFT", "NYSE", "Tech", 0, 0),
        ]
        us = get_tickers_for_market(universe, "US")
        assert len(us) == 2
