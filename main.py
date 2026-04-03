"""Auto-trader entry point."""

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.table import Table

from config.settings import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, TIMEZONE, MARKETS, is_paper_mode,
    IBC_PATH, IBC_INI, TWS_PATH, TWS_VERSION, IBC_USERID, IBC_PASSWORD,
)
from core.connection import connect, disconnect, get_account_summary
from core.portfolio import init_db

console = Console()
logger = logging.getLogger("auto_trader")


def setup_logging(mode: str) -> None:
    """Configure logging to file and console."""
    from config.settings import LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"trader_{date_str}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy HTTP request logs from Telegram polling
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Starting auto-trader in %s mode", mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto Stock Trader")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest", "dry-run"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--market",
        choices=["us", "all"],
        default="all",
        help="Market to trade (default: all)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan cycle then exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass market hours check (orders queue for next open)",
    )
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Use IBC Watchdog to auto-start gateway and reconnect on restarts",
    )
    parser.add_argument(
        "--backtest-tickers",
        nargs="+",
        default=None,
        help="Tickers for backtest mode (e.g. AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--backtest-start",
        default="",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--backtest-end",
        default="",
        help="Backtest end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Initial capital for backtest (default: 100000)",
    )
    return parser.parse_args()


def display_account_summary(summary: dict) -> None:
    """Pretty-print account summary with rich."""
    table = Table(title="Account Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in summary.items():
        table.add_row(key, f"${value:,.2f}")

    console.print(table)


def _resolve_markets(market_arg: str) -> list[str]:
    if market_arg == "all":
        return MARKETS
    return [market_arg.upper()]


def run_backtest_mode(args: argparse.Namespace) -> None:
    """Run the backtesting engine."""
    from backtest.engine import run_backtest, BacktestConfig
    from backtest.report import calculate_metrics, display_metrics

    console.print("[yellow]Backtest mode — no IBKR connection needed[/yellow]")

    # Default tickers if none specified
    tickers = args.backtest_tickers
    if not tickers:
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "JPM", "UNH", "HD",
        ]
        console.print(f"Using default tickers: {', '.join(tickers)}")

    market = "US"

    config = BacktestConfig(
        tickers=tickers,
        market=market,
        start_date=args.backtest_start,
        end_date=args.backtest_end,
        initial_capital=args.capital,
    )

    console.print(f"\nRunning backtest: {len(tickers)} tickers, ${config.initial_capital:,.0f} capital...")
    portfolio = run_backtest(config)

    metrics = calculate_metrics(
        portfolio.trades, portfolio.equity_curve, config.initial_capital,
    )
    console.print()
    display_metrics(metrics)


def run_watchdog_mode(args: argparse.Namespace, markets: list[str]) -> None:
    """Run with IBC Watchdog — auto-starts gateway and reconnects on restarts."""
    import nest_asyncio
    from ib_insync import IB
    from ib_insync.ibcontroller import IBC, Watchdog

    # Allow nested event loops so sync ib_insync calls work inside Watchdog callbacks
    nest_asyncio.apply()

    trading_mode = "paper" if is_paper_mode() else "live"

    ibc = IBC(
        twsVersion=TWS_VERSION,
        gateway=True,
        tradingMode=trading_mode,
        twsPath=TWS_PATH,
        ibcPath=IBC_PATH,
        ibcIni=IBC_INI,
        userid=IBC_USERID,
        password=IBC_PASSWORD,
    )

    ib = IB()
    watchdog = Watchdog(
        controller=ibc,
        ib=ib,
        host=IBKR_HOST,
        port=IBKR_PORT,
        clientId=IBKR_CLIENT_ID,
        appStartupTime=60,
        appTimeout=40,
        retryDelay=10,
    )

    first_connect = [True]

    def on_connected(watchdog: Watchdog):
        _ib = watchdog.ib
        summary = get_account_summary(_ib)

        if first_connect[0]:
            first_connect[0] = False
            logger.info("Watchdog connected to IBKR")
            display_account_summary(summary)

            from notifications.telegram import notify_startup, start_listener, update_status
            notify_startup(args.mode, summary)
            start_listener()
            update_status("startup_complete")

            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
            console.print(f"\nLocal time ({TIMEZONE}): {now.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"Mode: [bold]{args.mode}[/bold] | Markets: [bold]{', '.join(markets)}[/bold]")

            console.print("\n[cyan]Starting scheduler...[/cyan]")
            from core.scheduler import start_scheduler
            start_scheduler(_ib, markets, mode=args.mode, force=args.force)
        else:
            logger.info("Watchdog reconnected to IBKR after gateway restart")
            from notifications.telegram import update_status
            update_status("reconnected", "Gateway restarted — reconnected")

    watchdog.startedEvent += on_connected

    console.print("[cyan]Starting IBC Watchdog — gateway will auto-start and reconnect...[/cyan]")
    watchdog.start()

    try:
        IB.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down watchdog...[/yellow]")
        watchdog.stop()


def main() -> None:
    args = parse_args()
    setup_logging(args.mode)

    # Safety check: live mode requires explicit confirmation
    if args.mode == "live":
        console.print(
            "[bold red]WARNING: Live trading mode selected. Real money will be used![/bold red]"
        )
        confirm = input("Type 'CONFIRM LIVE' to proceed: ")
        if confirm != "CONFIRM LIVE":
            console.print("Aborted.")
            sys.exit(0)

    # Initialize database
    init_db()
    logger.info("Portfolio database initialized")

    # Backtest mode doesn't need IBKR connection
    if args.mode == "backtest":
        run_backtest_mode(args)
        return

    markets = _resolve_markets(args.market)

    # Watchdog mode: IBC manages gateway lifecycle
    if args.watchdog:
        run_watchdog_mode(args, markets)
        return

    # Direct connection mode (gateway must already be running)
    paper = is_paper_mode()
    mode_label = "PAPER" if paper else "LIVE"
    console.print(f"Connecting to IBKR ({mode_label}) at {IBKR_HOST}:{IBKR_PORT}...")

    try:
        ib = connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
    except ConnectionError as e:
        console.print(f"[bold red]Connection failed:[/bold red] {e}")
        console.print(
            "\n[yellow]Make sure TWS or IB Gateway is running with API "
            "connections enabled on the correct port.[/yellow]"
        )
        sys.exit(1)

    try:
        # Display account info
        summary = get_account_summary(ib)
        display_account_summary(summary)

        # Start Telegram notifications + listener
        from notifications.telegram import notify_startup, start_listener, update_status
        notify_startup(args.mode, summary)
        start_listener()
        update_status("startup_complete")

        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)
        console.print(f"\nLocal time ({TIMEZONE}): {now.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Mode: [bold]{args.mode}[/bold] | Markets: [bold]{', '.join(markets)}[/bold]")

        if args.once:
            console.print("\n[cyan]Running single scan cycle...[/cyan]")
            from core.scheduler import run_scan_cycle
            summary = run_scan_cycle(ib, markets, mode=args.mode, force=args.force)
            console.print(f"\nScan complete: {summary}")
        else:
            console.print("\n[cyan]Starting scheduler...[/cyan]")
            from core.scheduler import start_scheduler
            start_scheduler(ib, markets, mode=args.mode, force=args.force)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        disconnect(ib)
        console.print("Disconnected from IBKR. Goodbye.")


if __name__ == "__main__":
    main()
