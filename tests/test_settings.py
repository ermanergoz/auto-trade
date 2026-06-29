"""Tests for config/settings.py validation."""

from unittest.mock import patch

import config.settings as settings
from config.settings import validate_settings, is_paper_mode


class TestValidateSettings:
    """Verify startup validation catches invalid configuration."""

    def test_valid_defaults_pass(self):
        """Default settings should pass validation."""
        errors = validate_settings()
        assert errors == [], f"Default settings should be valid, got: {errors}"

    def test_invalid_port_detected(self):
        with patch("config.settings.IBKR_PORT", 9999):
            errors = validate_settings()
            assert any("IBKR_PORT" in e for e in errors)

    def test_zero_positions_detected(self):
        with patch("config.settings.MAX_OPEN_POSITIONS", 0):
            errors = validate_settings()
            assert any("MAX_OPEN_POSITIONS" in e for e in errors)

    def test_negative_risk_reward_detected(self):
        with patch("config.settings.MIN_RISK_REWARD_RATIO", -1):
            errors = validate_settings()
            assert any("MIN_RISK_REWARD_RATIO" in e for e in errors)

    def test_zero_scan_interval_detected(self):
        with patch("config.settings.SCAN_INTERVAL_MINUTES", 0):
            errors = validate_settings()
            assert any("SCAN_INTERVAL_MINUTES" in e for e in errors)

    def test_invalid_sector_concentration_detected(self):
        with patch("config.settings.MAX_SECTOR_CONCENTRATION_PCT", 0):
            errors = validate_settings()
            assert any("MAX_SECTOR_CONCENTRATION_PCT" in e for e in errors)

    def test_invalid_take_profit_detected(self):
        with patch("config.settings.DEFAULT_TAKE_PROFIT_PCT", -1):
            errors = validate_settings()
            assert any("DEFAULT_TAKE_PROFIT_PCT" in e for e in errors)

    def test_negative_circuit_breaker_detected(self):
        with patch("config.settings.CIRCUIT_BREAKER_LOSSES", -1):
            errors = validate_settings()
            assert any("CIRCUIT_BREAKER_LOSSES" in e for e in errors)

    # --- Intraday-margin framework (post-2026-06-04) bounds -----------------

    def test_invalid_reg_t_min_equity_detected(self):
        """REG_T_MIN_EQUITY_USD <= 0 must be rejected."""
        with patch("config.settings.REG_T_MIN_EQUITY_USD", 0):
            errors = validate_settings()
            assert any("REG_T_MIN_EQUITY_USD" in e for e in errors)

    def test_negative_reg_t_min_equity_detected(self):
        with patch("config.settings.REG_T_MIN_EQUITY_USD", -100.0):
            errors = validate_settings()
            assert any("REG_T_MIN_EQUITY_USD" in e for e in errors)

    def test_intraday_maintenance_pct_above_100_detected(self):
        with patch("config.settings.INTRADAY_MAINTENANCE_MARGIN_PCT", 150.0):
            errors = validate_settings()
            assert any("INTRADAY_MAINTENANCE_MARGIN_PCT" in e for e in errors)

    def test_intraday_maintenance_pct_zero_detected(self):
        with patch("config.settings.INTRADAY_MAINTENANCE_MARGIN_PCT", 0):
            errors = validate_settings()
            assert any("INTRADAY_MAINTENANCE_MARGIN_PCT" in e for e in errors)

    def test_invalid_margin_regime_detected(self):
        with patch("config.settings.MARGIN_REGIME", "wishful"):
            errors = validate_settings()
            assert any("MARGIN_REGIME" in e for e in errors)

    def test_valid_margin_regimes_pass(self):
        for regime in ("intraday", "legacy_pdt", "both"):
            with patch("config.settings.MARGIN_REGIME", regime):
                errors = validate_settings()
                assert not any("MARGIN_REGIME" in e for e in errors), (
                    f"regime {regime!r} should be accepted, got {errors}"
                )

    def test_negative_legacy_pdt_threshold_detected(self):
        with patch("config.settings.LEGACY_PDT_THRESHOLD_USD", -1.0):
            errors = validate_settings()
            assert any("LEGACY_PDT_THRESHOLD_USD" in e for e in errors)


class TestMarginConstants:
    """The eliminated $5k PDT gate must be gone; the new framework present."""

    def test_obsolete_pdt_threshold_removed(self):
        assert not hasattr(settings, "PDT_PROTECTION_THRESHOLD_USD")

    def test_intraday_margin_defaults(self):
        assert settings.REG_T_MIN_EQUITY_USD == 2000.0
        assert settings.INTRADAY_MAINTENANCE_MARGIN_PCT == 25.0

    def test_margin_regime_default_is_valid(self):
        assert settings.MARGIN_REGIME in ("intraday", "legacy_pdt", "both")

    def test_legacy_pdt_threshold_is_correct_25k_not_5k(self):
        """The legacy counter must use the real $25k PDT threshold, never $5k."""
        assert settings.LEGACY_PDT_THRESHOLD_USD == 25000.0


class TestPaperMode:
    """is_paper_mode() is the safety gate — must have coverage."""

    def test_default_port_is_paper(self):
        with patch("config.settings.IBKR_PORT", 7497):
            assert is_paper_mode() is True

    def test_live_port_is_not_paper(self):
        with patch("config.settings.IBKR_PORT", 7496):
            assert is_paper_mode() is False
