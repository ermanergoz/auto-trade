"""Tests for core/connection.py (unit tests that don't require IBKR)."""

from ib_insync import Stock

from core.connection import create_contract


def test_create_us_contract():
    contract = create_contract("AAPL", "US")
    assert isinstance(contract, Stock)
    assert contract.symbol == "AAPL"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"


def test_create_bist_contract():
    contract = create_contract("THYAO", "BIST")
    assert isinstance(contract, Stock)
    assert contract.symbol == "THYAO"
    assert contract.exchange == "BIST"
    assert contract.currency == "TRY"


def test_create_contract_case_insensitive():
    contract = create_contract("MSFT", "us")
    assert contract.exchange == "SMART"

    contract = create_contract("GARAN", "bist")
    assert contract.exchange == "BIST"
