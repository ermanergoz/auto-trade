"""Tests for core/data.py (unit tests that don't require IBKR)."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from core.data import (
    get_historical_data_yfinance,
    get_news,
    get_macro_news,
    clear_cache,
    _cache_set,
    _cache_get,
    _cache,
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
    @patch("core.data.yf")
    def test_fetch_us_stock(self, mock_yf):
        """Test yfinance fetch with mocked network call."""
        dates = pd.date_range("2024-01-10", periods=5, freq="D")
        mock_df = pd.DataFrame({
            "Open": [150.0] * 5, "High": [155.0] * 5,
            "Low": [148.0] * 5, "Close": [152.0] * 5,
            "Volume": [1_000_000] * 5,
        }, index=dates)
        mock_df.index.name = "Date"
        mock_yf.download.return_value = mock_df

        df = get_historical_data_yfinance("AAPL", period="5d", interval="1d")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
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


class TestNewsPriority:
    """Verify yfinance-first strategy: yfinance is tried before Tavily."""

    @patch("core.data._get_news_yfinance")
    @patch("core.data.TAVILY_API_KEY", "fake-key")
    def test_yfinance_called_first_when_it_returns_results(self, mock_yf):
        """When yfinance returns headlines, Tavily should NOT be called."""
        mock_yf.return_value = ["AAPL hits new high", "Apple revenue beats"]
        headlines = get_news("AAPL")
        mock_yf.assert_called_once_with("AAPL", 5)
        assert headlines == ["AAPL hits new high", "Apple revenue beats"]

    @patch("core.data.TAVILY_API_KEY", "fake-key")
    @patch("core.data._get_news_yfinance")
    def test_tavily_used_as_fallback_when_yfinance_empty(self, mock_yf):
        """When yfinance returns nothing, Tavily should be tried."""
        mock_yf.return_value = []
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": "Apple from Tavily"}]
        }
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        import sys
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            headlines = get_news("AAPL")
            mock_yf.assert_called_once()
            mock_client.search.assert_called_once()
            assert headlines == ["Apple from Tavily"]

    @patch("core.data.TAVILY_API_KEY", "")
    @patch("core.data._get_news_yfinance")
    def test_no_tavily_key_uses_yfinance_only(self, mock_yf):
        """Without Tavily API key, only yfinance is used."""
        mock_yf.return_value = ["MSFT earnings strong"]
        headlines = get_news("MSFT")
        mock_yf.assert_called_once()
        assert headlines == ["MSFT earnings strong"]


class TestNewsCacheTTL:
    """Verify cache TTLs are long enough to reduce API calls."""

    @patch("core.data._get_news_yfinance")
    def test_stock_news_cached_for_one_hour(self, mock_yf):
        """Stock news TTL should be 3600s (1 hour), not 900s."""
        mock_yf.return_value = ["Headline 1"]
        get_news("AAPL")
        # Check the cache entry has correct TTL (expiry ~3600s from now)
        import time
        cache_key = "news:AAPL:US"
        assert cache_key in _cache
        expiry, _ = _cache[cache_key]
        remaining = expiry - time.time()
        assert remaining > 3500, f"Stock news TTL too short: {remaining:.0f}s (expected ~3600)"

    @patch("core.data.TAVILY_API_KEY", "fake-key")
    def test_macro_news_cached_for_one_hour(self):
        """Macro news TTL should be 3600s (1 hour), not 900s."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": "Fed holds rates"}]
        }
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        import sys
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            get_macro_news()
            import time
            assert "macro_news" in _cache
            expiry, _ = _cache["macro_news"]
            remaining = expiry - time.time()
            assert remaining > 3500, f"Macro news TTL too short: {remaining:.0f}s (expected ~3600)"


class TestNewsFailureCacheTTL:
    """Verify that failed news fetches use a short cache TTL for faster retry."""

    @patch("core.data.TAVILY_API_KEY", "")
    @patch("core.data._get_news_yfinance")
    def test_failed_stock_news_uses_short_ttl(self, mock_yf):
        """When all news sources fail, cache TTL should be ~60s, not 3600s."""
        mock_yf.return_value = []  # yfinance returns nothing, no Tavily key
        get_news("FAIL_TICKER")
        import time
        cache_key = "news:FAIL_TICKER:US"
        assert cache_key in _cache
        expiry, _ = _cache[cache_key]
        remaining = expiry - time.time()
        assert remaining < 120, f"Failure TTL too long: {remaining:.0f}s (expected ~60)"
        assert remaining > 30, f"Failure TTL too short: {remaining:.0f}s (expected ~60)"

    @patch("core.data.TAVILY_API_KEY", "")
    def test_failed_macro_news_uses_short_ttl(self):
        """When macro news fetch fails, cache TTL should be ~60s."""
        get_macro_news()
        import time
        assert "macro_news" in _cache
        expiry, _ = _cache["macro_news"]
        remaining = expiry - time.time()
        assert remaining < 120, f"Failure TTL too long: {remaining:.0f}s (expected ~60)"
        assert remaining > 30, f"Failure TTL too short: {remaining:.0f}s (expected ~60)"


class TestYFinanceNewsFormat:
    """Verify _get_news_yfinance handles both old and new yfinance response formats."""

    @patch("yfinance.Ticker")
    def test_yfinance_v1_2_nested_content_format(self, mock_ticker):
        """yfinance >=1.2 nests title under item['content']['title']."""
        mock_ticker.return_value.news = [
            {"content": {"title": "AAPL hits record high"}, "id": "1"},
            {"content": {"title": "Apple launches new product"}, "id": "2"},
        ]
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("AAPL", max_results=5)
        assert headlines == ["AAPL hits record high", "Apple launches new product"]

    @patch("yfinance.Ticker")
    def test_yfinance_old_flat_format(self, mock_ticker):
        """Older yfinance versions return title at top level."""
        mock_ticker.return_value.news = [
            {"title": "MSFT earnings beat", "link": "http://..."},
            {"title": "Azure growth strong", "link": "http://..."},
        ]
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("MSFT", max_results=5)
        assert headlines == ["MSFT earnings beat", "Azure growth strong"]

    @patch("yfinance.Ticker")
    def test_yfinance_empty_news(self, mock_ticker):
        """Empty news list should return empty."""
        mock_ticker.return_value.news = []
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("ZZZZ", max_results=5)
        assert headlines == []

    @patch("yfinance.Ticker")
    def test_yfinance_none_news(self, mock_ticker):
        """None news should return empty."""
        mock_ticker.return_value.news = None
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("ZZZZ", max_results=5)
        assert headlines == []

    @patch("yfinance.Ticker")
    def test_yfinance_respects_max_results(self, mock_ticker):
        """Should only return up to max_results headlines."""
        mock_ticker.return_value.news = [
            {"content": {"title": f"Headline {i}"}} for i in range(10)
        ]
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("AAPL", max_results=3)
        assert len(headlines) == 3

    @patch("yfinance.Ticker")
    def test_yfinance_skips_empty_titles(self, mock_ticker):
        """Items with no title should be skipped."""
        mock_ticker.return_value.news = [
            {"content": {"title": "Real headline"}},
            {"content": {"title": ""}},
            {"content": {}},
            {"id": "no-content-key"},
        ]
        from core.data import _get_news_yfinance
        headlines = _get_news_yfinance("AAPL", max_results=10)
        assert headlines == ["Real headline"]


class TestNewsStockMapping:
    """Verify each ticker gets its own news — no cross-contamination."""

    @patch("core.data._get_news_yfinance")
    def test_different_tickers_get_different_news(self, mock_yf):
        """Each ticker must receive its own headlines, not another ticker's."""
        def yf_side_effect(ticker, max_results=5):
            return {
                "AAPL": ["Apple news 1", "Apple news 2"],
                "MSFT": ["Microsoft news 1", "Microsoft news 2"],
                "TSLA": ["Tesla news 1", "Tesla news 2"],
            }.get(ticker, [])

        mock_yf.side_effect = yf_side_effect

        aapl_news = get_news("AAPL")
        msft_news = get_news("MSFT")
        tsla_news = get_news("TSLA")

        assert "Apple news 1" in aapl_news
        assert "Microsoft news 1" not in aapl_news
        assert "Tesla news 1" not in aapl_news

        assert "Microsoft news 1" in msft_news
        assert "Apple news 1" not in msft_news

        assert "Tesla news 1" in tsla_news
        assert "Apple news 1" not in tsla_news

    @patch("core.data._get_news_yfinance")
    def test_cached_news_returns_correct_ticker(self, mock_yf):
        """After caching, retrieving news for a ticker returns THAT ticker's news."""
        mock_yf.side_effect = lambda t, max_results=5: {
            "AAPL": ["Apple cached"],
            "GOOG": ["Google cached"],
        }.get(t, [])

        # First calls populate cache
        get_news("AAPL")
        get_news("GOOG")

        # Second calls should hit cache — verify correct mapping
        aapl_again = get_news("AAPL")
        goog_again = get_news("GOOG")

        assert aapl_again == ["Apple cached"]
        assert goog_again == ["Google cached"]
        # yfinance only called once per ticker (cache hit on second)
        assert mock_yf.call_count == 2


class TestNewsAnalystIntegration:
    """Verify news flows correctly from get_news() through to the LLM prompt."""

    @patch("core.data._get_news_yfinance")
    @patch("core.analyst._call_llm")
    def test_correct_news_in_prompt_per_ticker(self, mock_llm, mock_yf):
        """When analyzing multiple stocks, each prompt must contain only
        that stock's news, not another stock's headlines."""
        from core.analyst import analyze_batch

        mock_yf.side_effect = lambda t, max_results=5: {
            "AAPL": ["Apple beats earnings"],
            "MSFT": ["Microsoft cloud growth"],
        }.get(t, [])

        mock_llm.return_value = {
            "action": "buy", "confidence": 85,
            "entry_price": 150.0, "stop_loss": 145.0, "take_profit": 160.0,
            "trade_type": "day", "reasoning": "Strong",
        }

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame({
            "open": [100 + i * 0.5 for i in range(30)],
            "high": [101 + i * 0.5 for i in range(30)],
            "low": [99 + i * 0.5 for i in range(30)],
            "close": [100.5 + i * 0.5 for i in range(30)],
            "volume": [1_000_000] * 30,
        }, index=dates)

        aapl_news = get_news("AAPL")
        msft_news = get_news("MSFT")

        candidates = [
            {"ticker": "AAPL", "exchange": "SMART", "df": df,
             "indicator_values": {"RSI": 28}, "news": aapl_news},
            {"ticker": "MSFT", "exchange": "SMART", "df": df,
             "indicator_values": {"RSI": 32}, "news": msft_news},
        ]

        analyze_batch(candidates, macro_news=["Fed holds rates"])

        # Check each prompt got the right news
        assert mock_llm.call_count == 2
        aapl_prompt = mock_llm.call_args_list[0][0][0]
        msft_prompt = mock_llm.call_args_list[1][0][0]

        assert "Apple beats earnings" in aapl_prompt
        assert "Microsoft cloud growth" not in aapl_prompt

        assert "Microsoft cloud growth" in msft_prompt
        assert "Apple beats earnings" not in msft_prompt

        # Both should have macro news
        assert "Fed holds rates" in aapl_prompt
        assert "Fed holds rates" in msft_prompt
