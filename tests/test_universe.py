"""Tests for core/universe.py."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.models import StockInfo
from core.universe import (
    _classify_sector_yfinance,
    _classify_sector_ollama,
    _classify_sector_gemini,
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

    def test_blank_sector_excluded_fail_closed(self):
        """Unknown sector must be excluded — defense-in-depth for cached/stale entries.

        A stock loaded from cache with a blank sector string has no verified
        classification. The safety contract is 'never trade a stock we cannot
        confirm is not financial/defense.' Returning False here would let an
        unclassified financial slip through the filter.
        """
        assert _is_excluded_sector("") is True
        assert _is_excluded_sector(None) is True  # type: ignore[arg-type]
        assert _is_excluded_sector("   ") is True

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

    def test_blank_sector_from_cache_excluded(self):
        """A cached stock with missing sector must be dropped (safety default).

        _fill_missing_sectors drops unclassified stocks, but when build_universe
        loads from cache (day 2+) it only reapplies _filter_universe. Any cached
        entry with sector="" would otherwise bypass both filters.
        """
        stocks = [
            StockInfo("AAPL", "SMART", "Technology", 0, 0),
            StockInfo("MYSTERY", "SMART", "", 0, 0),  # unclassified
        ]
        filtered = _filter_universe(stocks)
        tickers = {s.ticker for s in filtered}
        assert "AAPL" in tickers
        assert "MYSTERY" not in tickers

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
    @patch("core.universe._classify_sector_gemini")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_excludes_unclassifiable(
        self, mock_classify, mock_gemini, mock_ollama,
    ):
        mock_classify.return_value = (None, None)
        mock_gemini.return_value = (None, None)
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
    @patch("core.universe._classify_sector_gemini")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_mixed(self, mock_classify, mock_gemini, mock_ollama):
        def side_effect(ticker):
            if ticker == "GOOG":
                return ("Technology", "United States")
            return (None, None)
        mock_classify.side_effect = side_effect
        mock_gemini.return_value = (None, None)
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
    @patch("core.universe._classify_sector_gemini")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_uses_ollama_when_yfinance_and_gemini_fail(
        self, mock_yf, mock_gemini, mock_ollama,
    ):
        mock_yf.return_value = (None, None)
        mock_gemini.return_value = (None, None)
        mock_ollama.return_value = ("Energy", "Canada")
        stocks = [StockInfo("FAKE", "SMART", "", 0, 0, name="FAKE ENERGY CORP")]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 1
        assert result[0].sector == "Energy"
        assert result[0].country == "Canada"
        mock_ollama.assert_called_once()

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_gemini")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_excludes_when_all_fail(
        self, mock_yf, mock_gemini, mock_ollama,
    ):
        mock_yf.return_value = (None, None)
        mock_gemini.return_value = (None, None)
        mock_ollama.return_value = (None, None)
        stocks = [StockInfo("FAKE", "SMART", "", 0, 0)]
        result = _fill_missing_sectors(stocks)
        assert len(result) == 0


class TestGeminiSectorFallback:
    """_classify_sector_gemini + its integration into _fill_missing_sectors.

    Contract mirrors core.analyst._call_gemini:
      - No key OR _gemini_exhausted latched -> (None, None) without HTTP call
      - 401/403 latches _gemini_exhausted for the process
      - 429 with permanent-exhaustion markers latches the flag
      - Transient HTTP/network errors return (None, None) without latching
    """

    @pytest.fixture
    def reset_gemini_state(self):
        """Clear every piece of process-wide Gemini rotation state before and
        after each test. The per-key exhaustion flag and rotation cursor live
        in module globals; without clearing them, state from one test (e.g.
        a 401 that latches the per-key flag) leaks into the next."""
        from core import analyst as _a
        _a._gemini_exhausted.clear()
        _a._gemini_key_index = 0
        _a._gemini_key_exhausted.clear()
        yield
        _a._gemini_exhausted.clear()
        _a._gemini_key_index = 0
        _a._gemini_key_exhausted.clear()

    def _gemini_body(self, json_text: str) -> bytes:
        """Build a successful Gemini generateContent response envelope."""
        return json.dumps({
            "candidates": [{"content": {"parts": [{"text": json_text}]}}],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 20},
        }).encode()

    @patch("core.analyst._gemini_keys", ["fake-key"])
    @patch("core.analyst.urllib.request.urlopen")
    def test_classify_sector_gemini_returns_sector_and_country(
        self, mock_urlopen, reset_gemini_state,
    ):
        mock_response = MagicMock()
        mock_response.read.return_value = self._gemini_body(
            json.dumps({"sector": "Technology", "country": "United States"})
        )
        mock_urlopen.return_value = mock_response

        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector == "Technology"
        assert country == "United States"

    @patch("core.analyst._gemini_keys", [])
    def test_classify_sector_gemini_returns_none_without_api_key(
        self, reset_gemini_state,
    ):
        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector is None
        assert country is None

    @patch("core.analyst._gemini_keys", ["fake-key"])
    @patch("core.analyst.urllib.request.urlopen")
    def test_classify_sector_gemini_returns_none_when_exhausted(
        self, mock_urlopen, reset_gemini_state,
    ):
        """When _gemini_exhausted is latched, skip the HTTP call entirely."""
        from core.analyst import _gemini_exhausted
        _gemini_exhausted.set()

        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector is None
        assert country is None
        mock_urlopen.assert_not_called()

    @patch("core.analyst._gemini_keys", ["fake-key"])
    @patch("core.analyst.urllib.request.urlopen")
    def test_classify_sector_gemini_returns_none_on_transient_http_error(
        self, mock_urlopen, reset_gemini_state,
    ):
        """Transient 5xx or network errors do NOT latch the exhaustion flag."""
        import urllib.error
        from core.analyst import _gemini_exhausted

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="x", code=500, msg="srv", hdrs=None, fp=None,
        )
        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector is None
        assert country is None
        assert not _gemini_exhausted.is_set(), "500 must not latch exhaustion"

    @patch("core.analyst._gemini_keys", ["fake-key"])
    @patch("core.analyst.urllib.request.urlopen")
    def test_classify_sector_gemini_latches_exhausted_on_auth_failure(
        self, mock_urlopen, reset_gemini_state,
    ):
        """401/403 latches _gemini_exhausted so subsequent calls short-circuit."""
        import io
        import urllib.error
        from core.analyst import _gemini_exhausted

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="x", code=401, msg="unauthorized",
            hdrs=None, fp=io.BytesIO(b"API key not valid"),
        )
        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector is None
        assert country is None
        assert _gemini_exhausted.is_set(), "401 must latch exhaustion"

    @patch("core.analyst._gemini_keys", ["fake-key"])
    @patch("core.analyst.urllib.request.urlopen")
    def test_classify_sector_gemini_latches_exhausted_on_quota_depletion(
        self, mock_urlopen, reset_gemini_state,
    ):
        """429 matching permanent-exhaustion markers latches the flag."""
        import io
        import urllib.error
        from core.analyst import _gemini_exhausted

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="x", code=429, msg="quota",
            hdrs=None, fp=io.BytesIO(b"free tier limit reached"),
        )
        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector is None
        assert country is None
        assert _gemini_exhausted.is_set(), "permanent 429 must latch exhaustion"

    @patch("core.analyst._gemini_keys", ["fake-key"])
    @patch("core.analyst.urllib.request.urlopen")
    def test_classify_sector_gemini_transient_429_does_not_latch(
        self, mock_urlopen, reset_gemini_state,
    ):
        """Per-minute rate limits (transient) must not latch the flag."""
        import io
        import urllib.error
        from core.analyst import _gemini_exhausted

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="x", code=429, msg="rate",
            hdrs=None, fp=io.BytesIO(b"Quota exceeded per minute"),
        )
        sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector is None
        assert country is None
        assert not _gemini_exhausted.is_set(), "transient 429 must not latch"

    # ---- _fill_missing_sectors routing order ----

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_gemini")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_prefers_gemini_over_ollama(
        self, mock_yf, mock_gemini, mock_ollama, reset_gemini_state=None,
    ):
        """When yfinance fails and Gemini succeeds, Ollama must NOT be called."""
        mock_yf.return_value = (None, None)
        mock_gemini.return_value = ("Technology", "United States")
        mock_ollama.return_value = ("SHOULD NOT BE USED", "X")

        stocks = [StockInfo("FAKE", "SMART", "", 0, 0, name="FAKE TECH CORP")]
        result = _fill_missing_sectors(stocks)

        assert len(result) == 1
        assert result[0].sector == "Technology"
        assert result[0].country == "United States"
        mock_gemini.assert_called_once()
        mock_ollama.assert_not_called()

    @patch("core.universe._classify_sector_ollama")
    @patch("core.universe._classify_sector_gemini")
    @patch("core.universe._classify_sector_yfinance")
    def test_fill_missing_sectors_falls_back_to_ollama_when_gemini_returns_none(
        self, mock_yf, mock_gemini, mock_ollama,
    ):
        """yfinance None -> Gemini None -> Ollama used."""
        mock_yf.return_value = (None, None)
        mock_gemini.return_value = (None, None)
        mock_ollama.return_value = ("Energy", "Canada")

        stocks = [StockInfo("FAKE", "SMART", "", 0, 0, name="FAKE ENERGY CORP")]
        result = _fill_missing_sectors(stocks)

        assert len(result) == 1
        assert result[0].sector == "Energy"
        mock_gemini.assert_called_once()
        mock_ollama.assert_called_once()


class TestSectorLookupRoutesThroughRotation:
    """Regression for the 2026-04-28 production bug.

    Before the API-key consolidation, `_classify_sector_gemini` issued
    its own `urllib.request.urlopen` call against a single key
    (`?key={GEMINI_API_KEY}`), bypassing the multi-key rotation that
    `core.analyst._call_gemini_payload` implements. Combined with the
    singular-var gate, sector lookups silently fell through to Ollama
    for every cache miss in the universe scan ("0 via Gemini, 13 via
    Ollama" in the 04-28 log).

    Contract after the fix: sector lookups must go through
    `core.analyst._call_gemini_payload`, picking up rotation, RPM 429
    fallthrough, and the global `_gemini_exhausted` short-circuit
    automatically.
    """

    @pytest.fixture
    def reset_full_gemini_state(self):
        """Clear every piece of process-wide rotation state so tests can
        not leak the cursor or per-key flags into each other."""
        from core import analyst as _a
        _a._gemini_exhausted.clear()
        _a._gemini_key_index = 0
        _a._gemini_key_exhausted.clear()
        yield
        _a._gemini_exhausted.clear()
        _a._gemini_key_index = 0
        _a._gemini_key_exhausted.clear()

    def test_sector_lookup_routes_through_call_gemini_payload(
        self, reset_full_gemini_state,
    ):
        """The sector helper must call analyst._call_gemini_payload once per
        lookup rather than building a one-key URL itself."""
        with patch("core.universe.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["KEY_A"]), \
             patch("core.analyst._call_gemini_payload") as mock_call:
            mock_call.return_value = {
                "sector": "Technology", "country": "United States",
            }
            sector, country = _classify_sector_gemini("AAPL", "APPLE INC")
        assert sector == "Technology"
        assert country == "United States"
        assert mock_call.call_count == 1, (
            "Sector lookup must dispatch through _call_gemini_payload "
            "(rotation wrapper). Direct urlopen calls bypass multi-key rotation."
        )

    def test_sector_lookup_rotates_keys_on_rpm_429(self, reset_full_gemini_state):
        """When key A returns RPM 429, the rotation in _call_gemini_payload
        must advance to key B and the sector lookup must succeed."""
        import io
        import urllib.error
        from core import analyst as _a

        rpm_body = b'{"error":{"code":429,"message":"requests per minute"}}'
        rpm_err = urllib.error.HTTPError(
            url="x", code=429, msg="rate", hdrs=None, fp=io.BytesIO(rpm_body),
        )
        success_body = json.dumps({
            "candidates": [{"content": {"parts": [{"text":
                json.dumps({"sector": "Technology", "country": "US"})
            }]}}],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 20},
        }).encode()
        success_resp = MagicMock()
        success_resp.read.return_value = success_body

        # Patch urlopen on the analyst module — that's where _call_gemini
        # lives after the consolidation. Pre-fix, universe.py owns its own
        # urlopen import and this patch site won't intercept anything.
        with patch("core.universe.AI_PROVIDER", "gemini"), \
             patch("core.analyst._gemini_keys", ["KEY_A", "KEY_B"]), \
             patch("core.analyst.urllib.request.urlopen") as mu:
            mu.side_effect = [rpm_err, success_resp]
            sector, country = _classify_sector_gemini("AAPL", "APPLE INC")

        assert sector == "Technology"
        assert mu.call_count == 2, (
            "Must retry on second key after first 429 RPM"
        )
        urls = [mu.call_args_list[i][0][0].full_url for i in range(2)]
        assert "KEY_A" in urls[0], f"first attempt should use KEY_A, got {urls[0]}"
        assert "KEY_B" in urls[1], f"second attempt should use KEY_B, got {urls[1]}"
        # No flags latched on transient RPM
        assert not _a._gemini_exhausted.is_set()


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
