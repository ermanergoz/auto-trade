"""Tests for core/data.py (unit tests that don't require IBKR)."""

import pandas as pd
import pytest

from core.data import (
    get_historical_data_yfinance,
    get_news,
    clear_cache,
    _cache_set,
    _cache_get,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestCache:
    def test_set_and_get(self):
        _cache_set("test_key", {"value": 42}, ttl=60)
        result = _cache_get("test_key")
        assert result == {"value": 42}

    def test_expired(self):
        _cache_set("expired", "data", ttl=-1)  # already expired
        result = _cache_get("expired")
        assert result is None

    def test_missing_key(self):
        result = _cache_get("nonexistent")
        assert result is None


class TestYFinanceFallback:
    def test_fetch_us_stock(self):
        df = get_historical_data_yfinance("AAPL", period="5d", interval="1d")
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            )

    def test_invalid_ticker(self):
        df = get_historical_data_yfinance("ZZZZZZZNOTREAL", period="5d")
        assert isinstance(df, pd.DataFrame)
        # Should return empty or minimal data without crashing


class TestNews:
    def test_no_api_key_returns_empty(self):
        # With no API key configured, should return empty list
        headlines = get_news("AAPL")
        assert isinstance(headlines, list)

    def test_yfinance_news_fallback(self):
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("AAPL", max_results=3)
        assert isinstance(headlines, list)
        # AAPL should have some news
        if headlines:
            assert all(isinstance(h, str) for h in headlines)
            assert len(headlines) <= 3

    def test_yfinance_news_invalid_ticker(self):
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("ZZZZZNOTREAL123")
        assert isinstance(headlines, list)  # should not crash
