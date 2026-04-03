"""SQLite-backed portfolio tracker."""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from core.models import (
    Position, Trade, DailySummary, Signal, Action, TradeType,
)
from config.settings import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def _db_connection(db_path: Path = DB_PATH):
    """Context manager for SQLite connections."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables if they don't exist."""
    with _db_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
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
            );

            CREATE TABLE IF NOT EXISTS trades (
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
            );

            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                portfolio_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_pnl_pct REAL NOT NULL,
                num_trades INTEGER NOT NULL,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS signals (
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
            );

            CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
        """)
    logger.info("Database initialized at %s", db_path)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def add_position(position: Position, db_path: Path = DB_PATH) -> int:
    """Insert a new open position. Returns the row ID."""
    with _db_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO positions
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
    """Close an open position and record it as a trade. Returns the Trade."""
    exit_time = exit_time or datetime.now()

    with _db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE ticker = ? LIMIT 1",
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
    db_path: Path = DB_PATH,
) -> list[Trade]:
    """Return completed trades, optionally filtered by date range."""
    query = "SELECT * FROM trades"
    params: list = []

    if start_date or end_date:
        clauses = []
        if start_date:
            clauses.append("exit_time >= ?")
            params.append(start_date.isoformat())
        if end_date:
            clauses.append("exit_time <= ?")
            params.append(end_date.isoformat() + "T23:59:59")
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY exit_time DESC"

    with _db_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        Trade(
            id=r["id"],
            ticker=r["ticker"],
            exchange=r["exchange"],
            quantity=r["quantity"],
            entry_price=r["entry_price"],
            exit_price=r["exit_price"],
            entry_time=datetime.fromisoformat(r["entry_time"]),
            exit_time=datetime.fromisoformat(r["exit_time"]),
            trade_type=TradeType(r["trade_type"]),
            sector=r["sector"],
            reasoning=r["reasoning"],
        )
        for r in rows
    ]


def get_daily_pnl(day: Optional[date] = None, db_path: Path = DB_PATH) -> float:
    """Calculate realized P&L for a given day (default: today)."""
    day = day or date.today()
    day_str = day.isoformat()
    with _db_connection(db_path) as conn:
        row = conn.execute(
            """SELECT COALESCE(SUM((exit_price - entry_price) * quantity), 0) as pnl
               FROM trades
               WHERE date(exit_time) = ?""",
            (day_str,),
        ).fetchone()
    return row["pnl"]


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
    day = day or date.today()
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
