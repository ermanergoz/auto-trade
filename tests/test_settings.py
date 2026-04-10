"""Tests for config/settings.py validation."""

from unittest.mock import patch

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


class TestPaperMode:
    """is_paper_mode() is the safety gate — must have coverage."""

    def test_default_port_is_paper(self):
        with patch("config.settings.IBKR_PORT", 7497):
            assert is_paper_mode() is True

    def test_live_port_is_not_paper(self):
        with patch("config.settings.IBKR_PORT", 7496):
            assert is_paper_mode() is False
