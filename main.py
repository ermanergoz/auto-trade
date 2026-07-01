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
from core.portfolio import init_db, verify_db

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
    # Multi-fold rolling walk-forward (HRN-02). Window sizes are in TRADING days
    # and mirror backtest.engine.DEFAULT_WF_* (2y IS, ~12-month OOS, stepped by
    # the OOS length so OOS windows are adjacent and non-overlapping).
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help=(
            "Run a multi-fold rolling walk-forward instead of a single backtest "
            "(reports per-fold metrics, IS->OOS degradation, and Walk-Forward "
            "Efficiency, flagging WFE < 0.5 as fail). Requires --mode backtest."
        ),
    )
    parser.add_argument(
        "--wf-is-days",
        type=int,
        default=504,
        help="Walk-forward in-sample length in trading days (default: 504 ~= 2y)",
    )
    parser.add_argument(
        "--wf-oos-days",
        type=int,
        default=252,
        help=(
            "Walk-forward out-of-sample length in trading days (default: 252 "
            "~= 12 months). Kept at 9-12 months on purpose: each fold consumes a "
            "fixed 60-bar indicator warmup before it can trade, so a short OOS "
            "window (e.g. 6 months / ~125 bars) would leave only ~65 tradable "
            "bars and starve the >=30-trade statistical gate."
        ),
    )
    parser.add_argument(
        "--wf-step-days",
        type=int,
        default=252,
        help=(
            "Trading days to advance the window between folds (default: 252 = one "
            "OOS window, so OOS segments are adjacent and non-overlapping)"
        ),
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


def _extract_ibkr_fills(ib) -> list[dict]:
    """Convert ib.fills() to plain dicts for reconcile_positions.

    ib_insync populates ib.fills() during connection sync, giving us the
    actual execution prices for orders that filled while the bot was
    offline. Using those beats guessing with stop_loss.
    """
    try:
        fills = ib.fills()
    except Exception as e:
        logger.debug("Could not fetch ib.fills(): %s", e)
        return []
    result = []
    for f in fills:
        try:
            result.append({
                "ticker": f.contract.symbol,
                "side": f.execution.side,
                "shares": float(f.execution.shares),
                "price": float(f.execution.price),
                "time": f.execution.time,
                "realized_pnl": float(getattr(f.commissionReport, "realizedPNL", 0.0) or 0.0),
            })
        except Exception as e:
            logger.debug("Skipping unparseable fill: %s", e)
    return result


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


def run_walk_forward_mode(args: argparse.Namespace) -> None:
    """Run the multi-fold rolling walk-forward harness (HRN-02).

    Mirrors run_backtest_mode's config construction (tickers / dates / capital,
    honoring the 5y history_period default and sub-$25k capital) but dispatches
    to the rolling walk-forward and its per-fold + WFE report instead of a single
    backtest. The OOS windows default to ~12 months so each fold can clear the
    >=30-trade gate after its 60-bar warmup is consumed.
    """
    from backtest.engine import rolling_walk_forward, BacktestConfig
    from backtest.report import display_walk_forward

    console.print("[yellow]Walk-forward mode — no IBKR connection needed[/yellow]")

    tickers = args.backtest_tickers
    if not tickers:
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "JPM", "UNH", "HD",
        ]
        console.print(f"Using default tickers: {', '.join(tickers)}")

    config = BacktestConfig(
        tickers=tickers,
        market="US",
        start_date=args.backtest_start,
        end_date=args.backtest_end,
        initial_capital=args.capital,
    )

    console.print(
        f"\nRunning rolling walk-forward: {len(tickers)} tickers, "
        f"${config.initial_capital:,.0f} capital, "
        f"IS={args.wf_is_days}d / OOS={args.wf_oos_days}d / step={args.wf_step_days}d "
        "(trading days)..."
    )
    result = rolling_walk_forward(
        config,
        is_days=args.wf_is_days,
        oos_days=args.wf_oos_days,
        step_days=args.wf_step_days,
    )
    console.print()
    display_walk_forward(result)


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

    def _reconcile_and_reattach(_ib):
        """Sync DB with IBKR and reattach exit handlers."""
        from core.portfolio import reconcile_positions, get_open_positions, get_daily_pnl
        from core.executor import reattach_exit_handlers, import_ibkr_positions
        ibkr_positions = [
            {"ticker": p.contract.symbol, "quantity": int(p.position)}
            for p in _ib.positions()
        ]
        ibkr_fills = _extract_ibkr_fills(_ib)
        recon = reconcile_positions(ibkr_positions, auto_fix=True, ibkr_fills=ibkr_fills)
        if recon["auto_closed"]:
            logger.warning("Auto-closed orphaned positions: %s", recon["auto_closed"])
        if recon["orphaned_ibkr"]:
            import_ibkr_positions(_ib, recon["orphaned_ibkr"])
        if recon["in_sync"] and not recon["auto_closed"]:
            logger.info("Position reconciliation OK: %d in sync", recon["db_count"])
        reattach_exit_handlers(_ib)
        return recon

    # The on_connected callback MUST return quickly so Watchdog.runAsync can
    # proceed to register its disconnect/timeout handlers.  The scheduler
    # runs in the main thread after the first connection.
    first_connect = [True]

    def on_connected(watchdog: Watchdog):
        _ib = watchdog.ib
        summary = get_account_summary(_ib)

        if first_connect[0]:
            first_connect[0] = False
            logger.info("Watchdog connected to IBKR")
            display_account_summary(summary)

            recon = _reconcile_and_reattach(_ib)
            if recon["auto_closed"]:
                console.print(
                    f"[yellow]Auto-closed orphaned DB positions: {recon['auto_closed']} "
                    f"(filled at IBKR while bot was offline)[/yellow]"
                )

            from core.portfolio import get_open_positions, get_daily_pnl
            from notifications.telegram import notify_startup, start_listener, update_status, update_portfolio_data, set_ib_instance
            set_ib_instance(_ib)
            notify_startup(args.mode, summary)
            start_listener()
            update_status("startup_complete")
            update_portfolio_data(summary, get_open_positions(), get_daily_pnl())

            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
            console.print(f"\nLocal time ({TIMEZONE}): {now.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"Mode: [bold]{args.mode}[/bold] | Markets: [bold]{', '.join(markets)}[/bold]")
        else:
            logger.info("Watchdog reconnected to IBKR after gateway restart")
            _reconcile_and_reattach(_ib)

            from core.portfolio import get_open_positions, get_daily_pnl
            from notifications.telegram import update_status, update_portfolio_data, set_ib_instance
            set_ib_instance(_ib)
            update_status("reconnected", "Gateway restarted — reconnected")
            update_portfolio_data(get_account_summary(_ib), get_open_positions(), get_daily_pnl())

    watchdog.startedEvent += on_connected

    console.print("[cyan]Starting IBC Watchdog — gateway will auto-start and reconnect...[/cyan]")
    watchdog.start()

    # Wait for the first connection — the Watchdog starts the gateway and
    # connects asynchronously.  We pump the event loop in short intervals
    # so that the Watchdog's runAsync task can execute.
    import asyncio
    from ib_insync.util import run as ib_run
    while not ib.isConnected():
        ib_run(asyncio.sleep(1))

    # Now the Watchdog is fully initialized (disconnect/timeout handlers
    # registered).  Run the scheduler in the main thread.
    console.print("\n[cyan]Starting scheduler...[/cyan]")
    from core.scheduler import start_scheduler
    try:
        start_scheduler(ib, markets, mode=args.mode, force=args.force, reconnect=False)
    except KeyboardInterrupt:
        pass
    finally:
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

    # Validate configuration
    from config.settings import validate_settings
    config_errors = validate_settings()
    if config_errors:
        for err in config_errors:
            console.print(f"[bold red]Config error:[/bold red] {err}")
        sys.exit(1)

    # Initialize database
    init_db()
    verify_db()
    logger.info("Portfolio database initialized")

    # Backtest mode doesn't need IBKR connection. The multi-fold rolling
    # walk-forward is a backtest-mode variant — dispatch to it first.
    if args.mode == "backtest":
        if getattr(args, "walk_forward", False):
            run_walk_forward_mode(args)
        else:
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

        # Sync DB with IBKR — close orphaned DB positions, import orphaned IBKR positions
        from core.portfolio import reconcile_positions, get_open_positions, get_daily_pnl
        from core.executor import reattach_exit_handlers, import_ibkr_positions
        ibkr_positions = [
            {"ticker": p.contract.symbol, "quantity": int(p.position)}
            for p in ib.positions()
        ]
        ibkr_fills = _extract_ibkr_fills(ib)
        recon = reconcile_positions(ibkr_positions, auto_fix=True, ibkr_fills=ibkr_fills)
        if recon["auto_closed"]:
            console.print(
                f"[yellow]Auto-closed orphaned DB positions: {recon['auto_closed']} "
                f"(filled at IBKR while bot was offline)[/yellow]"
            )
        if recon["orphaned_ibkr"]:
            imported = import_ibkr_positions(ib, recon["orphaned_ibkr"])
            if imported:
                console.print(
                    f"[green]Imported IBKR positions into DB: {imported}[/green]"
                )
        if recon["in_sync"] and not recon["auto_closed"]:
            logger.info("Position reconciliation OK: %d in sync", recon["db_count"])

        # Reattach exit handlers for existing bracket orders (survives restarts)
        reattached = reattach_exit_handlers(ib)
        if reattached:
            console.print(f"[green]Reattached {reattached} exit handler(s) for existing orders[/green]")

        # Start Telegram notifications + listener
        from notifications.telegram import notify_startup, start_listener, update_status, update_portfolio_data, set_ib_instance
        set_ib_instance(ib)
        notify_startup(args.mode, summary)
        start_listener()
        update_status("startup_complete")
        update_portfolio_data(summary, get_open_positions(), get_daily_pnl())

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
