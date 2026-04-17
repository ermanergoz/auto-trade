"""Tests for core/data.py (unit tests that don't require IBKR)."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from core.data import (
    get_historical_data_yfinance,
    get_news,
    get_macro_news,
    get_analyst_recommendation,
    clear_cache,
    _cache_set,
    _cache_get,
    _cache,
)

# New split/dividend adjustment helpers (implemented in Feature 1)
from core.data import (
    detect_unadjusted_splits,
    adjust_for_splits,
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
    """Verify Tavily-first strategy: Tavily is tried before yfinance."""

    @patch("core.data._get_news_yfinance")
    @patch("core.data.TAVILY_API_KEY", "fake-key")
    def test_tavily_called_first_when_it_returns_results(self, mock_yf):
        """When Tavily returns headlines, yfinance should NOT be called."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": "AAPL hits new high"}, {"title": "Apple revenue beats"}]
        }
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        import sys
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            headlines = get_news("AAPL")
            mock_client.search.assert_called_once()
            mock_yf.assert_not_called()
            assert headlines == ["AAPL hits new high", "Apple revenue beats"]

    @patch("core.data.TAVILY_API_KEY", "fake-key")
    @patch("core.data._get_news_yfinance")
    def test_yfinance_used_as_fallback_when_tavily_errors(self, mock_yf):
        """When Tavily raises (e.g. rate limit), yfinance should be tried."""
        mock_yf.return_value = ["AAPL from yfinance"]
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("usage limit exceeded")
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        import sys
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            headlines = get_news("AAPL")
            mock_client.search.assert_called_once()
            mock_yf.assert_called_once_with("AAPL", 5)
            assert headlines == ["AAPL from yfinance"]

    @patch("core.data.TAVILY_API_KEY", "fake-key")
    @patch("core.data._get_news_yfinance")
    def test_yfinance_used_as_fallback_when_tavily_empty(self, mock_yf):
        """When Tavily returns no results, yfinance should be tried."""
        mock_yf.return_value = ["AAPL from yfinance"]
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        import sys
        with patch.dict(sys.modules, {"tavily": mock_tavily_module}):
            headlines = get_news("AAPL")
            mock_yf.assert_called_once_with("AAPL", 5)
            assert headlines == ["AAPL from yfinance"]

    @patch("core.data.TAVILY_API_KEY", "")
    @patch("core.data._get_news_yfinance")
    def test_no_tavily_key_uses_yfinance_only(self, mock_yf):
        """Without Tavily API key, yfinance is used directly."""
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


class TestCacheMutation:
    """Cached dicts must not be corrupted by caller mutation."""

    def test_mutating_cached_dict_does_not_corrupt_cache(self):
        """If a caller mutates a returned dict, the cache should be unaffected."""
        _cache_set("mut_test", {"key": "original"}, ttl=60)
        result1 = _cache_get("mut_test")
        result1["key"] = "mutated"  # caller mutates the result

        result2 = _cache_get("mut_test")
        assert result2["key"] == "original", (
            "Cache was corrupted by caller mutation — dicts must be copied on read"
        )

    def test_mutating_cached_list_does_not_corrupt_cache(self):
        """If a caller mutates a returned list, the cache should be unaffected."""
        _cache_set("list_test", ["headline1", "headline2"], ttl=60)
        result1 = _cache_get("list_test")
        result1.append("injected")  # caller mutates the result

        result2 = _cache_get("list_test")
        assert len(result2) == 2, (
            "Cache was corrupted by caller mutation — lists must be copied on read"
        )

    def test_dataframe_copy_still_works(self):
        """DataFrame copy behavior should still work correctly."""
        import pandas as pd
        df = pd.DataFrame({"close": [100, 101, 102]})
        _cache_set("df_test", df, ttl=60)
        result1 = _cache_get("df_test")
        result1["close"].iloc[0] = 999

        result2 = _cache_get("df_test")
        assert result2["close"].iloc[0] == 100, (
            "DataFrame cache copy is broken"
        )


class TestAnalystRecommendation:
    """Tests for get_analyst_recommendation() — yfinance analyst consensus."""

    @patch("yfinance.Ticker")
    def test_returns_consensus_buy(self, mock_ticker):
        """When most analysts say buy, consensus should be 'buy'."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 5, "buy": 10, "hold": 3, "sell": 1, "strongSell": 0,
        }])
        result = get_analyst_recommendation("AAPL")
        assert result is not None
        assert result["consensus"] == "buy"

    @patch("yfinance.Ticker")
    def test_returns_consensus_sell(self, mock_ticker):
        """When most analysts say sell, consensus should be 'sell'."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 0, "buy": 1, "hold": 2, "sell": 10, "strongSell": 3,
        }])
        result = get_analyst_recommendation("BAD_STOCK")
        assert result is not None
        assert result["consensus"] == "sell"

    @patch("yfinance.Ticker")
    def test_returns_consensus_strong_sell(self, mock_ticker):
        """When most analysts say strong sell, consensus should be 'strong_sell'."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 0, "buy": 0, "hold": 1, "sell": 2, "strongSell": 15,
        }])
        result = get_analyst_recommendation("TERRIBLE")
        assert result is not None
        assert result["consensus"] == "strong_sell"

    @patch("yfinance.Ticker")
    def test_returns_consensus_hold(self, mock_ticker):
        """When most analysts say hold, consensus should be 'hold'."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 1, "buy": 2, "hold": 15, "sell": 1, "strongSell": 0,
        }])
        result = get_analyst_recommendation("MEH")
        assert result is not None
        assert result["consensus"] == "hold"

    @patch("yfinance.Ticker")
    def test_returns_consensus_strong_buy(self, mock_ticker):
        """When most analysts say strong buy, consensus should be 'strong_buy'."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 20, "buy": 3, "hold": 1, "sell": 0, "strongSell": 0,
        }])
        result = get_analyst_recommendation("HOT")
        assert result is not None
        assert result["consensus"] == "strong_buy"

    @patch("yfinance.Ticker")
    def test_returns_none_on_empty_data(self, mock_ticker):
        """When yfinance returns empty DataFrame, return None."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame()
        result = get_analyst_recommendation("UNKNOWN")
        assert result is None

    @patch("yfinance.Ticker")
    def test_returns_none_on_none_data(self, mock_ticker):
        """When yfinance returns None, return None."""
        mock_ticker.return_value.recommendations_summary = None
        result = get_analyst_recommendation("UNKNOWN")
        assert result is None

    @patch("yfinance.Ticker")
    def test_returns_none_on_exception(self, mock_ticker):
        """When yfinance throws, return None gracefully."""
        mock_ticker.side_effect = Exception("Network error")
        result = get_analyst_recommendation("FAIL")
        assert result is None

    @patch("yfinance.Ticker")
    def test_includes_detail_counts(self, mock_ticker):
        """Result should include the raw analyst counts."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 5, "buy": 10, "hold": 3, "sell": 1, "strongSell": 0,
        }])
        result = get_analyst_recommendation("AAPL")
        assert result["details"]["strong_buy"] == 5
        assert result["details"]["buy"] == 10
        assert result["details"]["hold"] == 3
        assert result["details"]["sell"] == 1
        assert result["details"]["strong_sell"] == 0

    @patch("yfinance.Ticker")
    def test_cached_on_second_call(self, mock_ticker):
        """Second call for same ticker should use cache."""
        mock_ticker.return_value.recommendations_summary = pd.DataFrame([{
            "strongBuy": 5, "buy": 10, "hold": 3, "sell": 1, "strongSell": 0,
        }])
        get_analyst_recommendation("CACHED")
        get_analyst_recommendation("CACHED")
        # yfinance Ticker should only be called once
        assert mock_ticker.call_count == 1


# ---------------------------------------------------------------------------
# Feature 1: Split/dividend adjustment
# ---------------------------------------------------------------------------

class TestYFinanceAutoAdjust:
    """yf.download must pass auto_adjust=True so prices are already
    adjusted for splits and dividends. Without this, a 4-for-1 split
    would look like a -75% crash in backtest.
    """

    @patch("core.data.yf")
    def test_auto_adjust_true_is_passed(self, mock_yf):
        """yf.download must be called with auto_adjust=True."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_df = pd.DataFrame({
            "Open": [100.0] * 5, "High": [101.0] * 5,
            "Low": [99.0] * 5, "Close": [100.0] * 5,
            "Volume": [1_000_000] * 5,
        }, index=dates)
        mock_yf.download.return_value = mock_df

        get_historical_data_yfinance("AAPL", period="5d")

        mock_yf.download.assert_called_once()
        kwargs = mock_yf.download.call_args.kwargs
        assert kwargs.get("auto_adjust") is True, (
            "auto_adjust must be True — otherwise splits/dividends "
            "corrupt backtest price continuity"
        )

    @patch("core.data.yf")
    def test_multiindex_with_fields_on_level_1(self, mock_yf):
        """yfinance versions differ on which level carries the field name.

        Some versions return columns as MultiIndex(field, ticker); others
        return MultiIndex(ticker, field). The old code called
        get_level_values(0) unconditionally, which crashes with KeyError if
        level 0 is the ticker. The correct behavior: detect which level
        contains 'Close' and flatten using that level.
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        # Ticker on level 0, field on level 1 (the problematic ordering)
        cols = pd.MultiIndex.from_tuples(
            [("AAPL", "Open"), ("AAPL", "High"), ("AAPL", "Low"),
             ("AAPL", "Close"), ("AAPL", "Volume")]
        )
        mock_df = pd.DataFrame(
            [[100.0, 101.0, 99.0, 100.5, 1_000_000]] * 5,
            index=dates, columns=cols,
        )
        mock_yf.download.return_value = mock_df

        df = get_historical_data_yfinance("AAPL", period="5d")

        # Columns must be the standardized lowercase field names
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].iloc[0] == 100.5

    @patch("core.data.yf")
    def test_multiindex_with_fields_on_level_0(self, mock_yf):
        """The other yfinance ordering: field on level 0, ticker on level 1."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_tuples(
            [("Open", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"),
             ("Close", "AAPL"), ("Volume", "AAPL")]
        )
        mock_df = pd.DataFrame(
            [[100.0, 101.0, 99.0, 100.5, 1_000_000]] * 5,
            index=dates, columns=cols,
        )
        mock_yf.download.return_value = mock_df

        df = get_historical_data_yfinance("AAPL", period="5d")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].iloc[0] == 100.5

    @patch("core.data.yf")
    def test_split_like_drop_is_not_in_adjusted_data(self, mock_yf):
        """Simulate the scenario where yfinance (with auto_adjust=True)
        returns a continuous series even through a split date — the pre-split
        prices are scaled down so the series never shows a fake drop.
        """
        # With auto_adjust=True, Apple's 4-for-1 split on 2020-08-31
        # appears as a smooth price trajectory, not a -75% gap.
        dates = pd.date_range("2020-08-28", periods=5, freq="B")
        mock_df = pd.DataFrame({
            "Open": [124.0, 125.0, 126.0, 127.0, 128.0],
            "High": [125.0, 126.0, 127.0, 128.0, 129.0],
            "Low": [123.0, 124.0, 125.0, 126.0, 127.0],
            "Close": [124.5, 125.5, 126.5, 127.5, 128.5],
            "Volume": [50_000_000] * 5,
        }, index=dates)
        mock_yf.download.return_value = mock_df

        df = get_historical_data_yfinance("AAPL", period="5d")
        closes = df["close"].values
        daily_pct = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        # No day should show more than a 5% gap in adjusted data
        assert all(abs(p) < 0.05 for p in daily_pct), (
            f"Adjusted data should be continuous, got daily changes: {daily_pct}"
        )


class TestDetectUnadjustedSplits:
    """detect_unadjusted_splits(df, threshold=0.3) -> list of dicts.

    Each dict: {'date': Timestamp, 'ratio': float, 'type': 'forward'|'reverse'}
    where ratio is the inferred split ratio (4.0 for 4-for-1, 0.1 for 1-for-10).
    """

    def test_detects_4_for_1_split(self):
        """Apple-style 4-for-1 split: price drops from ~$500 to ~$125."""
        dates = pd.date_range("2020-08-28", periods=5, freq="B")
        df = pd.DataFrame({
            "open":  [499.0, 500.0, 125.0, 126.0, 127.0],
            "high":  [505.0, 505.0, 128.0, 129.0, 130.0],
            "low":   [495.0, 495.0, 123.0, 124.0, 125.0],
            "close": [500.0, 500.0, 125.0, 126.0, 127.0],
            "volume": [1_000_000] * 5,
        }, index=dates)

        events = detect_unadjusted_splits(df)
        assert len(events) == 1
        assert events[0]["type"] == "forward"
        assert events[0]["ratio"] == pytest.approx(4.0, rel=0.1)

    def test_detects_2_for_1_split(self):
        """2-for-1 split: $200 → $100."""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({
            "open":  [198.0, 200.0, 100.0, 101.0],
            "high":  [202.0, 203.0, 102.0, 103.0],
            "low":   [197.0, 199.0, 99.0, 100.0],
            "close": [200.0, 200.0, 100.0, 101.0],
            "volume": [500_000] * 4,
        }, index=dates)

        events = detect_unadjusted_splits(df)
        assert len(events) == 1
        assert events[0]["ratio"] == pytest.approx(2.0, rel=0.1)

    def test_detects_reverse_split_1_for_10(self):
        """Reverse split: $5 → $50."""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({
            "open":  [5.10, 5.00, 50.0, 51.0],
            "high":  [5.15, 5.05, 52.0, 53.0],
            "low":   [4.95, 4.95, 49.0, 50.0],
            "close": [5.00, 5.00, 50.0, 51.0],
            "volume": [100_000] * 4,
        }, index=dates)

        events = detect_unadjusted_splits(df)
        assert len(events) == 1
        assert events[0]["type"] == "reverse"
        assert events[0]["ratio"] == pytest.approx(0.1, rel=0.1)

    def test_does_not_flag_normal_volatility(self):
        """5% daily swings should not be flagged as splits."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        closes = [100.0]
        for i in range(19):
            closes.append(closes[-1] * (1 + (0.03 if i % 2 == 0 else -0.03)))
        df = pd.DataFrame({
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [1_000_000] * 20,
        }, index=dates)

        events = detect_unadjusted_splits(df)
        assert events == []

    def test_does_not_flag_earnings_drop(self):
        """A 10-15% earnings-related drop is below the split threshold."""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 88.0, 89.0],
            "high":  [102.0, 103.0, 90.0, 91.0],
            "low":   [99.0, 100.0, 87.0, 88.0],
            "close": [101.0, 101.0, 88.0, 89.0],  # -13% drop
            "volume": [1_000_000] * 4,
        }, index=dates)
        assert detect_unadjusted_splits(df) == []

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert detect_unadjusted_splits(df) == []

    def test_single_row(self):
        """Need at least 2 bars to detect a gap."""
        dates = pd.date_range("2024-01-01", periods=1)
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.0], "volume": [1_000_000],
        }, index=dates)
        assert detect_unadjusted_splits(df) == []

    def test_detects_multiple_splits(self):
        """Two splits in one series should both be flagged."""
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        df = pd.DataFrame({
            "open":  [400.0, 400.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0],
            "high":  [405.0, 405.0, 102.0, 102.0, 102.0, 102.0, 52.0, 52.0],
            "low":   [395.0, 395.0, 98.0, 98.0, 98.0, 98.0, 48.0, 48.0],
            "close": [400.0, 400.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0],
            "volume": [1_000_000] * 8,
        }, index=dates)

        events = detect_unadjusted_splits(df)
        assert len(events) == 2

    def test_custom_threshold_excludes_small_moves(self):
        """threshold=0.5 should only flag >=50% moves."""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({
            "open":  [100.0, 100.0, 60.0, 61.0],  # -40% drop
            "high":  [102.0, 102.0, 62.0, 63.0],
            "low":   [98.0, 98.0, 58.0, 59.0],
            "close": [100.0, 100.0, 60.0, 61.0],
            "volume": [1_000_000] * 4,
        }, index=dates)

        # With default threshold (0.3), 40% drop IS flagged
        assert len(detect_unadjusted_splits(df, threshold=0.3)) == 1
        # With stricter threshold (0.5), 40% drop is NOT flagged
        assert detect_unadjusted_splits(df, threshold=0.5) == []

    def test_each_event_includes_date(self):
        """Each flagged event should include its date for diagnostics."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [400.0, 100.0, 101.0],
            "high":  [405.0, 102.0, 103.0],
            "low":   [395.0, 99.0, 100.0],
            "close": [400.0, 100.0, 101.0],
            "volume": [1_000_000] * 3,
        }, index=dates)
        events = detect_unadjusted_splits(df)
        assert len(events) == 1
        assert events[0]["date"] == dates[1]


class TestAdjustForSplits:
    """adjust_for_splits(df, splits) where splits is {date: ratio}.

    For forward split (ratio > 1, e.g. 4-for-1 = 4.0):
      pre-split OHLC / ratio, volume * ratio.
    For reverse split (ratio < 1, e.g. 1-for-10 = 0.1):
      pre-split OHLC / ratio (i.e. multiplied), volume * ratio (divided).
    Post-split data is UNCHANGED.
    """

    def test_applies_4_for_1_split(self):
        """Pre-split prices divided by 4, volume multiplied by 4."""
        dates = pd.date_range("2020-08-28", periods=4, freq="B")
        df = pd.DataFrame({
            "open":  [400.0, 400.0, 100.0, 101.0],
            "high":  [405.0, 405.0, 102.0, 103.0],
            "low":   [395.0, 395.0, 98.0, 99.0],
            "close": [400.0, 400.0, 100.0, 101.0],
            "volume": [1_000_000] * 4,
        }, index=dates)

        split_date = dates[2]  # Split effective on this date
        adjusted = adjust_for_splits(df, {split_date: 4.0})

        # Before split: prices / 4, volume * 4
        assert adjusted.loc[dates[0], "close"] == pytest.approx(100.0)
        assert adjusted.loc[dates[1], "close"] == pytest.approx(100.0)
        assert adjusted.loc[dates[0], "volume"] == pytest.approx(4_000_000)

        # On and after split: unchanged
        assert adjusted.loc[dates[2], "close"] == pytest.approx(100.0)
        assert adjusted.loc[dates[3], "close"] == pytest.approx(101.0)
        assert adjusted.loc[dates[3], "volume"] == pytest.approx(1_000_000)

    def test_applies_reverse_split_1_for_10(self):
        """Pre-split prices multiplied by 10, volume divided by 10."""
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({
            "open":  [5.0, 5.0, 50.0, 51.0],
            "high":  [5.1, 5.1, 52.0, 53.0],
            "low":   [4.9, 4.9, 48.0, 49.0],
            "close": [5.0, 5.0, 50.0, 51.0],
            "volume": [10_000_000] * 4,
        }, index=dates)

        split_date = dates[2]
        adjusted = adjust_for_splits(df, {split_date: 0.1})

        # Before split: prices * 10, volume / 10
        assert adjusted.loc[dates[0], "close"] == pytest.approx(50.0)
        assert adjusted.loc[dates[1], "close"] == pytest.approx(50.0)
        assert adjusted.loc[dates[0], "volume"] == pytest.approx(1_000_000)

        # On/after: unchanged
        assert adjusted.loc[dates[2], "close"] == pytest.approx(50.0)
        assert adjusted.loc[dates[3], "volume"] == pytest.approx(10_000_000)

    def test_adjusts_all_ohlc_columns(self):
        """Open, high, low, close should all be scaled identically."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [200.0, 100.0, 101.0],
            "high":  [210.0, 105.0, 106.0],
            "low":   [195.0, 95.0, 96.0],
            "close": [200.0, 100.0, 101.0],
            "volume": [2_000_000] * 3,
        }, index=dates)

        adjusted = adjust_for_splits(df, {dates[1]: 2.0})
        # Row 0 (pre-split) all prices halved
        assert adjusted.loc[dates[0], "open"] == pytest.approx(100.0)
        assert adjusted.loc[dates[0], "high"] == pytest.approx(105.0)
        assert adjusted.loc[dates[0], "low"] == pytest.approx(97.5)
        assert adjusted.loc[dates[0], "close"] == pytest.approx(100.0)

    def test_empty_splits_is_noop(self):
        """An empty splits dict should leave the DataFrame unchanged."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0],
            "high":  [101.0, 102.0, 103.0],
            "low":   [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000_000] * 3,
        }, index=dates)

        adjusted = adjust_for_splits(df, {})
        pd.testing.assert_frame_equal(adjusted, df)

    def test_does_not_mutate_input(self):
        """adjust_for_splits should return a new DataFrame, not mutate input."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [200.0, 100.0, 101.0],
            "high":  [210.0, 105.0, 106.0],
            "low":   [195.0, 95.0, 96.0],
            "close": [200.0, 100.0, 101.0],
            "volume": [2_000_000] * 3,
        }, index=dates)
        original = df.copy()

        adjust_for_splits(df, {dates[1]: 2.0})
        pd.testing.assert_frame_equal(df, original)

    def test_applies_multiple_splits_cumulatively(self):
        """Two splits should apply cumulatively to earlier data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        # Split 1: 2-for-1 on date[2]. Split 2: 2-for-1 on date[4].
        # date[0..1] should be scaled by 1/4 (both splits apply).
        # date[2..3] should be scaled by 1/2 (only second split applies).
        # date[4] unchanged.
        df = pd.DataFrame({
            "open":  [400.0, 400.0, 200.0, 200.0, 100.0],
            "high":  [410.0, 410.0, 205.0, 205.0, 102.0],
            "low":   [390.0, 390.0, 195.0, 195.0, 98.0],
            "close": [400.0, 400.0, 200.0, 200.0, 100.0],
            "volume": [1_000_000] * 5,
        }, index=dates)

        adjusted = adjust_for_splits(df, {dates[2]: 2.0, dates[4]: 2.0})

        # Before both splits: scaled by 1/4
        assert adjusted.loc[dates[0], "close"] == pytest.approx(100.0)
        assert adjusted.loc[dates[0], "volume"] == pytest.approx(4_000_000)
        # Between splits: scaled by 1/2
        assert adjusted.loc[dates[2], "close"] == pytest.approx(100.0)
        assert adjusted.loc[dates[2], "volume"] == pytest.approx(2_000_000)
        # After both: unchanged
        assert adjusted.loc[dates[4], "close"] == pytest.approx(100.0)
        assert adjusted.loc[dates[4], "volume"] == pytest.approx(1_000_000)

    def test_split_date_not_in_index_is_skipped(self):
        """If a split date doesn't fall in the DataFrame, skip it silently."""
        dates = pd.date_range("2024-06-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0],
            "high":  [101.0, 102.0, 103.0],
            "low":   [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000_000] * 3,
        }, index=dates)

        # Split date is earlier than any data
        adjusted = adjust_for_splits(df, {pd.Timestamp("2020-01-01"): 2.0})
        pd.testing.assert_frame_equal(adjusted, df)

    def test_preserves_column_order_and_index(self):
        """Adjusted DataFrame must keep the same columns and index."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [200.0, 100.0, 101.0],
            "high":  [210.0, 105.0, 106.0],
            "low":   [195.0, 95.0, 96.0],
            "close": [200.0, 100.0, 101.0],
            "volume": [2_000_000] * 3,
        }, index=dates)

        adjusted = adjust_for_splits(df, {dates[1]: 2.0})
        assert list(adjusted.columns) == list(df.columns)
        pd.testing.assert_index_equal(adjusted.index, df.index)

    def test_ratio_of_one_is_noop(self):
        """Ratio 1.0 means no split — should not change data."""
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0],
            "high":  [101.0, 102.0, 103.0],
            "low":   [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000_000] * 3,
        }, index=dates)

        adjusted = adjust_for_splits(df, {dates[1]: 1.0})
        pd.testing.assert_frame_equal(adjusted, df)

    def test_empty_dataframe(self):
        """Adjusting an empty DataFrame should return empty."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        adjusted = adjust_for_splits(df, {pd.Timestamp("2024-01-01"): 2.0})
        assert adjusted.empty


class TestRoundTripDetectAndAdjust:
    """detect_unadjusted_splits + adjust_for_splits should fix bad data."""

    def test_round_trip_removes_split_gap(self):
        """After detecting and adjusting, no split-sized gaps should remain."""
        dates = pd.date_range("2020-08-28", periods=5, freq="B")
        df = pd.DataFrame({
            "open":  [500.0, 500.0, 125.0, 126.0, 127.0],
            "high":  [505.0, 505.0, 128.0, 129.0, 130.0],
            "low":   [495.0, 495.0, 123.0, 124.0, 125.0],
            "close": [500.0, 500.0, 125.0, 126.0, 127.0],
            "volume": [1_000_000] * 5,
        }, index=dates)

        events = detect_unadjusted_splits(df)
        assert len(events) == 1

        splits = {e["date"]: e["ratio"] for e in events}
        adjusted = adjust_for_splits(df, splits)

        # After adjustment, no day should show >20% single-day change
        closes = adjusted["close"].values
        daily_pct = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        assert all(abs(p) < 0.2 for p in daily_pct), (
            f"Adjusted series still has large gaps: {daily_pct}"
        )
