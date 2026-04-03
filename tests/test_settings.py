"""Tests for config/settings.py validation."""

from unittest.mock import patch

from config.settings import validate_settings


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
