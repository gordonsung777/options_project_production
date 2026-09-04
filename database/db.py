"""SQLite schema and persistence helpers.

SQLite is intentionally retained from the original project. WAL mode and short
transactions allow one collector to write while FastAPI and Streamlit read.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings

SYMBOL_TABLES = {"option_features", "market_features", "anomaly_results"}


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS tracked_symbols (
                symbol TEXT PRIMARY KEY,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_requested_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS option_snapshots (
                snapshot_time TEXT NOT NULL,
                underlying TEXT NOT NULL,
                underlying_price REAL,
                expiration TEXT NOT NULL,
                option_type TEXT NOT NULL,
                contract_symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                last_price REAL,
                bid REAL,
                ask REAL,
                volume REAL,
                open_interest REAL,
                implied_volatility REAL,
                in_the_money INTEGER,
                contract_size TEXT,
                currency TEXT,
                PRIMARY KEY (snapshot_time, contract_symbol)
            );

            CREATE TABLE IF NOT EXISTS stock_bars (
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (timestamp, symbol)
            );

            CREATE TABLE IF NOT EXISTS option_features (
                snapshot_time TEXT NOT NULL,
                snapshot_minute TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                underlying TEXT NOT NULL,
                underlying_price REAL,
                expiration TEXT,
                option_type TEXT,
                contract_symbol TEXT NOT NULL,
                strike REAL,
                last_price REAL,
                bid REAL,
                ask REAL,
                volume REAL,
                open_interest REAL,
                implied_volatility REAL,
                mid_price REAL,
                spread_pct REAL,
                moneyness REAL,
                days_to_expiration REAL,
                snapshot_gap_minutes REAL,
                volume_delta REAL,
                option_return_1m REAL,
                iv_change REAL,
                volume_oi_ratio REAL,
                volume_mean_20 REAL,
                volume_std_20 REAL,
                volume_zscore REAL,
                cross_section_volume_zscore REAL,
                activity_score REAL,
                PRIMARY KEY (snapshot_time, contract_symbol)
            );

            CREATE TABLE IF NOT EXISTS market_features (
                underlying TEXT NOT NULL,
                snapshot_minute TEXT NOT NULL,
                call_volume_delta REAL,
                put_volume_delta REAL,
                call_total_volume REAL,
                put_total_volume REAL,
                call_open_interest REAL,
                put_open_interest REAL,
                call_iv_mean REAL,
                put_iv_mean REAL,
                call_max_volume_zscore REAL,
                put_max_volume_zscore REAL,
                call_max_activity_score REAL,
                put_max_activity_score REAL,
                put_call_volume_ratio REAL,
                call_volume_oi_ratio REAL,
                put_volume_oi_ratio REAL,
                iv_skew REAL,
                stock_bar_minute TEXT,
                stock_close REAL,
                stock_volume REAL,
                stock_return_1m REAL,
                stock_return_5m REAL,
                stock_volume_zscore REAL,
                future_return_15m REAL,
                target_up_15m REAL,
                stock_bar_lag_seconds REAL,
                PRIMARY KEY (underlying, snapshot_minute)
            );

            CREATE TABLE IF NOT EXISTS anomaly_results (
                snapshot_time TEXT NOT NULL,
                underlying TEXT NOT NULL,
                contract_symbol TEXT NOT NULL,
                option_type TEXT,
                expiration TEXT,
                strike REAL,
                underlying_price REAL,
                last_price REAL,
                bid REAL,
                ask REAL,
                volume REAL,
                open_interest REAL,
                implied_volatility REAL,
                volume_delta REAL,
                volume_zscore REAL,
                cross_section_volume_zscore REAL,
                volume_oi_ratio REAL,
                activity_score REAL,
                anomaly INTEGER NOT NULL,
                anomaly_score REAL NOT NULL,
                method TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (snapshot_time, contract_symbol, method)
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                symbol TEXT NOT NULL,
                model_name TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                trained_at TEXT NOT NULL,
                feature_names_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                training_rows INTEGER NOT NULL,
                PRIMARY KEY (symbol, model_name)
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                message TEXT,
                option_rows INTEGER DEFAULT 0,
                stock_rows INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_options_symbol_time
            ON option_snapshots (underlying, snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_options_contract_time
            ON option_snapshots (contract_symbol, snapshot_time);
            CREATE INDEX IF NOT EXISTS idx_stock_symbol_time
            ON stock_bars (symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_anomaly_symbol_time
            ON anomaly_results (underlying, snapshot_time);
            """
        )
        connection.commit()


def register_symbol(symbol: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO tracked_symbols (symbol, active, created_at, last_requested_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET active=1, last_requested_at=excluded.last_requested_at
            """,
            (symbol, now, now),
        )
        connection.commit()


def get_tracked_symbols() -> list[str]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT symbol FROM tracked_symbols WHERE active=1 ORDER BY symbol"
        ).fetchall()
    return [str(row["symbol"]) for row in rows]


def save_option_snapshots(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    columns = list(frame.columns)
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT OR REPLACE INTO option_snapshots ({','.join(columns)}) VALUES ({placeholders})"
    rows = [
        tuple(_native(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]
    with get_connection() as connection:
        connection.executemany(sql, rows)
        connection.commit()
    return len(rows)


def save_stock_bars(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    rows = [
        tuple(_native(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    with get_connection() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO stock_bars VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        connection.commit()
    return len(rows)


def replace_symbol_frame(table: str, symbol: str, frame: pd.DataFrame) -> None:
    if table not in SYMBOL_TABLES:
        raise ValueError(f"Unsupported table: {table}")
    clean = frame.replace([np.inf, -np.inf], np.nan).copy()
    clean = clean.where(pd.notnull(clean), None)
    with get_connection() as connection:
        connection.execute(f"DELETE FROM {table} WHERE underlying = ?", (symbol,))
        if not clean.empty:
            clean.to_sql(table, connection, if_exists="append", index=False)
        connection.commit()


def query_frame(sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query(sql, connection, params=params)


def upsert_model_registry(
    symbol: str,
    model_name: str,
    artifact_path: str,
    features: list[str],
    metrics: dict[str, float | None],
    training_rows: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO model_registry
            (symbol, model_name, artifact_path, trained_at, feature_names_json, metrics_json, training_rows)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, model_name) DO UPDATE SET
                artifact_path=excluded.artifact_path,
                trained_at=excluded.trained_at,
                feature_names_json=excluded.feature_names_json,
                metrics_json=excluded.metrics_json,
                training_rows=excluded.training_rows
            """,
            (
                symbol,
                model_name,
                artifact_path,
                now,
                json.dumps(features),
                json.dumps(metrics),
                training_rows,
            ),
        )
        connection.commit()


def start_pipeline_run(symbol: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO pipeline_runs (symbol, started_at, status) VALUES (?, ?, 'running')",
            (symbol, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def finish_pipeline_run(
    run_id: int, status: str, message: str, option_rows: int = 0, stock_rows: int = 0
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE pipeline_runs
            SET finished_at=?, status=?, message=?, option_rows=?, stock_rows=?
            WHERE run_id=?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                status,
                message,
                option_rows,
                stock_rows,
                run_id,
            ),
        )
        connection.commit()


def _native(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
