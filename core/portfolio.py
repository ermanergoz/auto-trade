"""SQLite-backed portfolio tracker."""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

from core.models import (
    Position, Trade, DailySummary, Signal, Action, TradeType,
)
from config.settings import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def _db_connection(db_path: Path = DB_PATH, immediate: bool = False):
    """Context manager for SQLite connections.

    Args:
        immediate: When True, opens a BEGIN IMMEDIATE transaction on entry.
            This acquires the database write lock upfront, serializing
            concurrent writers and preventing SELECT-then-INSERT races
            (e.g., two callers both observing an open position, both
            inserting a Trade row, both DELETE-ing the position).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # busy_timeout lets the second writer wait for the first to commit
    # instead of failing with "database is locked".
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    # sqlite3's default isolation level wraps writes in implicit deferred
    # transactions that acquire the write lock only on first write. For
    # read-then-write critical sections we need the write lock from the
    # start — otherwise two concurrent readers both pass the SELECT check
    # before either writes.
    if immediate:
        conn.isolation_level = None  # enter explicit transaction control
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_TABLES = [
    """CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        exchange TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        entry_price REAL NOT NULL,
        entry_time TEXT NOT NULL,
        stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL,
        trade_type TEXT NOT NULL,
        sector TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    )""",
    """CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        exchange TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT NOT NULL,
        trade_type TEXT NOT NULL,
        sector TEXT DEFAULT '',
        reasoning TEXT DEFAULT '',
        created_at TEXT DEFAULT (datetime('now'))
    )""",
    """CREATE TABLE IF NOT EXISTS daily_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL UNIQUE,
        portfolio_value REAL NOT NULL,
        daily_pnl REAL NOT NULL,
        daily_pnl_pct REAL NOT NULL,
        num_trades INTEGER NOT NULL,
        winning_trades INTEGER DEFAULT 0,
        losing_trades INTEGER DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        action TEXT NOT NULL,
        confidence REAL NOT NULL,
        entry_price REAL NOT NULL,
        stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL,
        reasoning TEXT DEFAULT '',
        source TEXT NOT NULL,
        exchange TEXT DEFAULT '',
        trade_type TEXT DEFAULT 'day',
        timestamp TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    )""",
    """CREATE TABLE IF NOT EXISTS pending_orders (
        perm_id INTEGER PRIMARY KEY,
        ticker TEXT NOT NULL,
        placed_at TEXT NOT NULL,
        confidence REAL
    )""",
    # UNIQUE index enforces one-open-row-per-ticker at the DB level so that
    # concurrent inserts (fill handler thread vs scheduler) cannot both create
    # duplicate rows through the SELECT-then-INSERT check in add_position.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time)",
    "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
]

_REQUIRED_TABLES = {"positions", "trades", "daily_summary", "signals", "pending_orders"}


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables if they don't exist and run in-place migrations."""
    with _db_connection(db_path) as conn:
        for stmt in _TABLES:
            conn.execute(stmt)
        _migrate_pending_orders_confidence(conn)
    logger.info("Database initialized at %s", db_path)


def _migrate_pending_orders_confidence(conn: sqlite3.Connection) -> None:
    """Add the ``confidence`` column to pending_orders for legacy databases.

    Eviction (see core.executor.evict_weakest_pending) ranks pending BUY orders
    by the AI confidence recorded at placement. Databases created before that
    feature need an in-place ALTER TABLE so existing pending rows survive and
    new rows can store the field.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(pending_orders)")}
    if "confidence" not in cols:
        conn.execute("ALTER TABLE pending_orders ADD COLUMN confidence REAL")


def verify_db(db_path: Path = DB_PATH) -> None:
    """Raise RuntimeError if any required table is missing."""
    with _db_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    existing = {r["name"] for r in rows}
    missing = _REQUIRED_TABLES - existing
    if missing:
        raise RuntimeError(
            f"Database is missing tables after init: {sorted(missing)}. "
            f"Delete {db_path} and restart to rebuild."
        )


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def add_position(position: Position, db_path: Path = DB_PATH) -> int:
    """Insert a new open position. Returns the row ID.

    A UNIQUE index on `ticker` enforces one-row-per-ticker at the DB level.
    Two threads (fill handler + scheduler) calling this concurrently will
    not both INSERT — the loser sees an IntegrityError / OR IGNORE no-op.
    """
    with _db_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO positions
               (ticker, exchange, quantity, entry_price, entry_time,
                stop_loss, take_profit, trade_type, sector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                position.ticker, position.exchange, position.quantity,
                position.entry_price, position.entry_time.isoformat(),
                position.stop_loss, position.take_profit,
                position.trade_type.value, position.sector,
            ),
        )
        if cursor.rowcount == 0:
            existing = conn.execute(
                "SELECT id FROM positions WHERE ticker = ?", (position.ticker,)
            ).fetchone()
            existing_id = existing["id"] if existing else 0
            logger.warning(
                "Duplicate position for %s (existing id=%d) — skipping insert",
                position.ticker, existing_id,
            )
            return existing_id
        logger.info("Added position: %s %d @ %.2f",
                     position.ticker, position.quantity, position.entry_price)
        return cursor.lastrowid


def close_position(
    ticker: str,
    exit_price: float,
    exit_time: Optional[datetime] = None,
    reasoning: str = "",
    db_path: Path = DB_PATH,
) -> Optional[Trade]:
    """Close an open position and record it as a trade. Returns the Trade.

    Uses BEGIN IMMEDIATE so the SELECT-INSERT-DELETE is atomic across
    threads. Without this, the exit handler (ib_insync event thread) and
    close_all_day_trades (main thread) can both observe the position and
    both insert duplicate Trade rows before either deletes it.
    """
    exit_time = exit_time or datetime.now(timezone.utc)

    with _db_connection(db_path, immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE ticker = ? ORDER BY id ASC LIMIT 1",
            (ticker,),
        ).fetchone()

        if not row:
            logger.warning("No open position found for %s", ticker)
            return None

        # Insert into trades
        conn.execute(
            """INSERT INTO trades
               (ticker, exchange, quantity, entry_price, exit_price,
                entry_time, exit_time, trade_type, sector, reasoning)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["ticker"], row["exchange"], row["quantity"],
                row["entry_price"], exit_price,
                row["entry_time"], exit_time.isoformat(),
                row["trade_type"], row["sector"], reasoning,
            ),
        )

        # Remove from positions
        conn.execute("DELETE FROM positions WHERE id = ?", (row["id"],))

        trade = Trade(
            ticker=row["ticker"],
            exchange=row["exchange"],
            quantity=row["quantity"],
            entry_price=row["entry_price"],
            exit_price=exit_price,
            entry_time=datetime.fromisoformat(row["entry_time"]),
            exit_time=exit_time,
            trade_type=TradeType(row["trade_type"]),
            sector=row["sector"],
            reasoning=reasoning,
        )
        logger.info("Closed position: %s @ %.2f (P&L: %.2f)",
                     ticker, exit_price, trade.pnl)

        # Log to CSV trade journal
        try:
            from core.logger import log_trade_to_csv
            log_trade_to_csv(trade)
        except Exception as e:
            logger.debug("CSV trade journal write failed: %s", e)

        return trade


def get_open_positions(db_path: Path = DB_PATH) -> list[Position]:
    """Return all open positions."""
    with _db_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM positions").fetchall()
    return [
        Position(
            id=r["id"],
            ticker=r["ticker"],
            exchange=r["exchange"],
            quantity=r["quantity"],
            entry_price=r["entry_price"],
            entry_time=datetime.fromisoformat(r["entry_time"]),
            stop_loss=r["stop_loss"],
            take_profit=r["take_profit"],
            trade_type=TradeType(r["trade_type"]),
            sector=r["sector"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def get_trades(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    ticker: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> list[Trade]:
    """Return completed trades, optionally filtered by date range and/or ticker."""
    query = "SELECT * FROM trades"
    params: list = []
    clauses: list = []

    if start_date:
        clauses.append("date(exit_time) >= ?")
        params.append(start_date.isoformat())
    if end_date:
        clauses.append("date(exit_time) <= ?")
        params.append(end_date.isoformat())
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY exit_time DESC"

    with _db_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    def _ensure_utc(dt_str: str) -> datetime:
        dt = datetime.fromisoformat(dt_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    return [
        Trade(
            id=r["id"],
            ticker=r["ticker"],
            exchange=r["exchange"],
            quantity=r["quantity"],
            entry_price=r["entry_price"],
            exit_price=r["exit_price"],
            entry_time=_ensure_utc(r["entry_time"]),
            exit_time=_ensure_utc(r["exit_time"]),
            trade_type=TradeType(r["trade_type"]),
            sector=r["sector"],
            reasoning=r["reasoning"],
        )
        for r in rows
    ]


def _find_closing_fill(
    ticker: str,
    entry_time: datetime,
    ibkr_fills: Optional[list[dict]],
) -> Optional[dict]:
    """Find the most recent SLD fill for ticker after entry_time.

    Used by reconcile_positions to pick an accurate exit price when a DB
    position was filled at IBKR while the bot was offline. Only SLD
    (sell-to-close) fills are considered — BOT fills belong to entries
    and must not be used as exits.
    """
    if not ibkr_fills:
        return None
    # Normalize entry_time to UTC for comparison against IBKR fill times
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)

    def _aware(dt: datetime) -> datetime:
        # ib_insync sometimes returns naive datetimes for fill.time. Comparing
        # naive to aware datetimes raises TypeError, crashing the whole
        # reconcile path. Treat naive fill times as UTC (IBKR servers report
        # in UTC on the wire) — the failure mode of mis-bucketing by a few
        # hours at most is far less severe than the crash.
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    matches = [
        f for f in ibkr_fills
        if f.get("ticker") == ticker
        and f.get("side") == "SLD"
        and f.get("time") is not None
        and _aware(f["time"]) >= entry_time
    ]
    if not matches:
        return None
    return max(matches, key=lambda f: _aware(f["time"]))


def reconcile_positions(
    ibkr_positions: list[dict],
    auto_fix: bool = False,
    ibkr_fills: Optional[list[dict]] = None,
    db_path: Path = DB_PATH,
) -> dict:
    """Compare IBKR positions with database. Returns reconciliation report.

    Args:
        ibkr_positions: List of dicts with 'ticker' and 'quantity' from ib.positions().
        auto_fix: When True, close DB positions that no longer exist at IBKR
                  (e.g. stop-loss filled while bot was offline).
        ibkr_fills: Optional list of fill dicts from IBKR. Each dict should
                    contain: ticker, side ('SLD'/'BOT'), shares, price, time
                    (timezone-aware datetime). When provided, the most recent
                    SLD fill after position entry_time is used as exit_price
                    — giving accurate P&L. Falls back to stop_loss when no
                    matching fill exists.

    Returns:
        Dict with db_count, ibkr_count, orphaned_db, orphaned_ibkr, in_sync,
        and auto_closed (list of tickers closed when auto_fix=True).
    """
    db_positions = get_open_positions(db_path)
    db_tickers = {p.ticker for p in db_positions}
    ibkr_tickers = {p["ticker"] for p in ibkr_positions}

    orphaned_db = sorted(db_tickers - ibkr_tickers)
    orphaned_ibkr = sorted(ibkr_tickers - db_tickers)

    # Check quantity mismatches for tickers present in both DB and IBKR
    # Sum quantities per ticker to handle multiple positions for the same stock
    common_tickers = db_tickers & ibkr_tickers
    db_qty: dict[str, int] = {}
    for p in db_positions:
        db_qty[p.ticker] = db_qty.get(p.ticker, 0) + p.quantity
    ibkr_qty: dict[str, int] = {}
    for p in ibkr_positions:
        ibkr_qty[p["ticker"]] = ibkr_qty.get(p["ticker"], 0) + p["quantity"]
    qty_mismatches = {}
    for t in common_tickers:
        if db_qty[t] != ibkr_qty[t]:
            is_sign = (db_qty[t] > 0) != (ibkr_qty[t] > 0)
            qty_mismatches[t] = {
                "db": db_qty[t],
                "ibkr": ibkr_qty[t],
                "type": "sign_mismatch" if is_sign else "quantity_mismatch",
            }

    # Auto-fix: close orphaned DB positions that IBKR no longer holds
    auto_closed: list[str] = []
    if auto_fix and orphaned_db:
        for ticker in orphaned_db:
            pos = next((p for p in db_positions if p.ticker == ticker), None)
            if pos is None:
                continue

            # Prefer the actual IBKR fill price when available — that gives
            # accurate P&L for positions filled while the bot was offline.
            # Fallback hierarchy when no fill data is available:
            #   1. midpoint of SL and TP — unbiased expected-value estimator
            #      assuming equal prior probability of hitting either bracket.
            #      Using SL alone (previous behavior) always records a loss,
            #      systematically biasing the circuit breaker and trade journal.
            #   2. stop_loss alone — if no TP was set
            #   3. entry_price — neutral $0 P&L when neither bracket is known
            fill = _find_closing_fill(ticker, pos.entry_time, ibkr_fills)
            if fill is not None:
                exit_price = fill["price"]
                exit_time = fill["time"]
                # Ensure exit_time is tz-aware before downstream use
                if exit_time.tzinfo is None:
                    exit_time = exit_time.replace(tzinfo=timezone.utc)
                reasoning = (
                    f"Auto-reconcile: closed at actual IBKR fill price "
                    f"${exit_price:.4f} (fill recorded at {exit_time.isoformat()})"
                )
                log_detail = f"actual IBKR fill price ${exit_price:.4f}"
            else:
                if pos.stop_loss > 0 and pos.take_profit > 0:
                    exit_price = (pos.stop_loss + pos.take_profit) / 2.0
                    estimate_detail = (
                        f"midpoint estimate ${exit_price:.4f} of "
                        f"SL ${pos.stop_loss:.4f} and TP ${pos.take_profit:.4f}"
                    )
                elif pos.stop_loss > 0:
                    exit_price = pos.stop_loss
                    estimate_detail = f"stop-loss estimate ${exit_price:.4f} (no TP set)"
                else:
                    exit_price = pos.entry_price
                    estimate_detail = f"entry-price estimate ${exit_price:.4f} (no brackets set)"
                exit_time = None
                reasoning = (
                    "Auto-reconcile: position no longer exists at IBKR "
                    "(likely filled while bot was offline). Exit price is an "
                    f"estimate — {estimate_detail}."
                )
                log_detail = f"{estimate_detail} (no IBKR fill data)"

            trade = close_position(
                ticker, exit_price, exit_time=exit_time,
                db_path=db_path, reasoning=reasoning,
            )
            if trade:
                auto_closed.append(ticker)
                logger.warning(
                    "Auto-closed orphaned position %s (entry $%.2f, exit at %s)",
                    ticker, pos.entry_price, log_detail,
                )

    report = {
        "db_count": len(db_positions),
        "ibkr_count": len(ibkr_tickers),
        "orphaned_db": orphaned_db,
        "orphaned_ibkr": orphaned_ibkr,
        "qty_mismatches": qty_mismatches,
        "auto_closed": auto_closed,
        "in_sync": len(orphaned_db) == 0 and len(orphaned_ibkr) == 0 and len(qty_mismatches) == 0,
    }

    if not report["in_sync"]:
        logger.warning(
            "Position mismatch! DB: %d, IBKR: %d. "
            "In DB only: %s. In IBKR only: %s. Qty mismatches: %s",
            report["db_count"], report["ibkr_count"],
            orphaned_db, orphaned_ibkr, qty_mismatches,
        )
        sign_mismatches = [t for t, v in qty_mismatches.items() if v.get("type") == "sign_mismatch"]
        if sign_mismatches:
            logger.critical(
                "DIRECTION MISMATCH for %s — DB and IBKR disagree on long/short! "
                "Manual intervention required.",
                sign_mismatches,
            )

    if auto_closed:
        logger.info("Auto-reconcile closed %d orphaned positions: %s", len(auto_closed), auto_closed)

    return report


def get_daily_pnl(
    day: Optional[date] = None,
    db_path: Path = DB_PATH,
    unrealized_pnl: float = 0.0,
) -> float:
    """Calculate total daily P&L (realized + unrealized).

    Args:
        day: Date to compute realized P&L for (default: today).
        unrealized_pnl: Mark-to-market P&L on open positions (from IBKR).
    """
    day = day or datetime.now(timezone.utc).date()
    day_str = day.isoformat()
    # exit_time is stored as UTC ISO string; extract date portion for comparison
    with _db_connection(db_path) as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM((exit_price - entry_price) * quantity), 0) as pnl
               FROM trades
               WHERE date(exit_time) = ?""",
            (day_str,),
        ).fetchone()
    return row["pnl"] + unrealized_pnl


def get_portfolio_value(ib_account_value: float, db_path: Path = DB_PATH) -> float:
    """Portfolio value = IBKR account value (passed in from account summary)."""
    return ib_account_value


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def record_signal(signal: Signal, db_path: Path = DB_PATH) -> int:
    """Record a generated signal for audit/backtesting. Returns row ID."""
    with _db_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO signals
               (ticker, action, confidence, entry_price, stop_loss, take_profit,
                reasoning, source, exchange, trade_type, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.ticker, signal.action.value, signal.confidence,
                signal.entry_price, signal.stop_loss, signal.take_profit,
                signal.reasoning, signal.source, signal.exchange,
                signal.trade_type.value, signal.timestamp.isoformat(),
            ),
        )
        return cursor.lastrowid


# ---------------------------------------------------------------------------
# Daily Summary
# ---------------------------------------------------------------------------

def record_daily_summary(summary: DailySummary, db_path: Path = DB_PATH) -> None:
    """Insert or update daily summary."""
    with _db_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO daily_summary
               (date, portfolio_value, daily_pnl, daily_pnl_pct,
                num_trades, winning_trades, losing_trades)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                summary.date.isoformat() if isinstance(summary.date, date)
                else summary.date,
                summary.portfolio_value, summary.daily_pnl,
                summary.daily_pnl_pct, summary.num_trades,
                summary.winning_trades, summary.losing_trades,
            ),
        )


def get_daily_summary(
    day: Optional[date] = None, db_path: Path = DB_PATH
) -> Optional[DailySummary]:
    """Get the summary for a specific day."""
    day = day or datetime.now(timezone.utc).date()
    with _db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM daily_summary WHERE date = ?",
            (day.isoformat(),),
        ).fetchone()

    if not row:
        return None

    return DailySummary(
        id=row["id"],
        date=datetime.fromisoformat(row["date"]),
        portfolio_value=row["portfolio_value"],
        daily_pnl=row["daily_pnl"],
        daily_pnl_pct=row["daily_pnl_pct"],
        num_trades=row["num_trades"],
        winning_trades=row["winning_trades"],
        losing_trades=row["losing_trades"],
    )


# ---------------------------------------------------------------------------
# Pending Orders
# ---------------------------------------------------------------------------

def save_pending_order(
    perm_id: int,
    ticker: str,
    db_path: Path = DB_PATH,
    *,
    confidence: Optional[float] = None,
) -> None:
    """Record when a parent order was placed (persists across reconnections).

    ``confidence`` is the AI analyst confidence (0-100) at placement time;
    used later by the eviction logic to rank weakest pending BUYs. Leave it
    unset for non-AI paths (e.g., manual orders) — the column is nullable.
    Keyword-only so the long-standing positional ``db_path`` callers are
    unaffected.
    """
    with _db_connection(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO pending_orders (perm_id, ticker, placed_at, confidence)
               VALUES (?, ?, ?, ?)""",
            (perm_id, ticker, datetime.now(timezone.utc).isoformat(), confidence),
        )


def get_pending_order_time(perm_id: int, db_path: Path = DB_PATH) -> Optional[datetime]:
    """Return the original placement time for a pending order, or None."""
    with _db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT placed_at FROM pending_orders WHERE perm_id = ?",
            (perm_id,),
        ).fetchone()
    if row:
        dt = datetime.fromisoformat(row["placed_at"])
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def get_pending_order_confidence(perm_id: int, db_path: Path = DB_PATH) -> Optional[float]:
    """Return the AI confidence recorded at placement, or None if missing/legacy."""
    with _db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT confidence FROM pending_orders WHERE perm_id = ?",
            (perm_id,),
        ).fetchone()
    if row is None:
        return None
    return row["confidence"]


def remove_pending_order(perm_id: int, db_path: Path = DB_PATH) -> None:
    """Remove a pending order record (after fill or cancellation)."""
    with _db_connection(db_path) as conn:
        conn.execute("DELETE FROM pending_orders WHERE perm_id = ?", (perm_id,))
