"""IBKR connection manager via ib_insync."""

import logging
import time

from ib_insync import IB, Stock, Contract

from config.settings import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_TIMEOUT

logger = logging.getLogger(__name__)


def connect(
    host: str = IBKR_HOST,
    port: int = IBKR_PORT,
    client_id: int = IBKR_CLIENT_ID,
    timeout: int = IBKR_TIMEOUT,
) -> IB:
    """Connect to TWS/IB Gateway. Returns an IB instance.

    Raises ConnectionError if unable to connect within timeout.
    """
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        logger.info(
            "Connected to IBKR at %s:%s (clientId=%s, paper=%s)",
            host, port, client_id, port == 7497,
        )
        return ib
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to IBKR at {host}:{port}. "
            "Ensure TWS or IB Gateway is running with API connections enabled."
        ) from e


def disconnect(ib: IB) -> None:
    """Clean disconnect from IBKR."""
    if ib.isConnected():
        ib.disconnect()
        logger.info("Disconnected from IBKR")


def ensure_connected(
    ib: IB,
    host: str = IBKR_HOST,
    port: int = IBKR_PORT,
    client_id: int = IBKR_CLIENT_ID,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> IB:
    """Reconnect if the connection has dropped. Returns the IB instance.

    IBKR drops connections after inactivity or TWS restart.
    """
    if ib.isConnected():
        return ib

    logger.warning("IBKR connection lost — attempting reconnect...")
    for attempt in range(1, max_retries + 1):
        try:
            ib.connect(host, port, clientId=client_id, timeout=IBKR_TIMEOUT)
            logger.info("Reconnected to IBKR (attempt %d)", attempt)
            return ib
        except Exception:
            logger.warning("Reconnect attempt %d/%d failed", attempt, max_retries)
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise ConnectionError(
        f"Could not reconnect to IBKR after {max_retries} attempts"
    )


def create_contract(ticker: str, exchange: str) -> Stock:
    """Create an ib_insync Stock contract for the given ticker and exchange.

    US stocks use SMART routing; BIST stocks use BIST exchange with TRY currency.
    """
    exchange_upper = exchange.upper()

    if exchange_upper == "BIST":
        return Stock(ticker, "BIST", "TRY")
    else:
        # US stocks — use SMART routing for best execution
        return Stock(ticker, "SMART", "USD")


def qualify_contracts(ib: IB, contracts: list[Contract]) -> list[Contract]:
    """Qualify contracts to resolve ambiguous tickers.

    Particularly important for BIST stocks. Returns only successfully
    qualified contracts.
    """
    qualified = []
    for contract in contracts:
        try:
            results = ib.qualifyContracts(contract)
            if results:
                qualified.append(contract)
            else:
                logger.warning("Could not qualify contract: %s", contract)
        except Exception as e:
            logger.warning("Error qualifying %s: %s", contract, e)
    return qualified


def get_account_summary(ib: IB) -> dict:
    """Fetch key account summary values from IBKR."""
    summary = ib.accountSummary()
    result = {}
    for item in summary:
        if item.tag in (
            "NetLiquidation", "TotalCashValue", "GrossPositionValue",
            "UnrealizedPnL", "RealizedPnL", "BuyingPower",
        ):
            result[item.tag] = float(item.value)
    return result
