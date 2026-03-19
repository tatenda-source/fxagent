import os
import sqlite3
import json
import time
import logging
from datetime import datetime, timezone

import pandas as pd

from config import DB_PATH

_log = logging.getLogger(__name__)

WRITE_RETRY_ATTEMPTS = 3
WRITE_RETRY_DELAY = 0.5


class Storage:
    """SQLite persistence layer for OHLCV data, signals, predictions, and logs."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = None
        self._init_tables()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _execute_write(self, func):
        for attempt in range(1, WRITE_RETRY_ATTEMPTS + 1):
            try:
                return func()
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < WRITE_RETRY_ATTEMPTS:
                    _log.warning("DB locked, retry %d/%d", attempt, WRITE_RETRY_ATTEMPTS)
                    time.sleep(WRITE_RETRY_DELAY * attempt)
                else:
                    raise

    def health_check(self) -> bool:
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1 FROM ohlcv LIMIT 1")
            return True
        except Exception:
            return False

    def _init_tables(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                interval TEXT DEFAULT '1d',
                UNIQUE(pair, timestamp, interval)
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_type TEXT,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                reasons TEXT,
                predicted_price REAL,
                status TEXT DEFAULT 'OPEN',
                pnl REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                predicted_price REAL,
                actual_price REAL,
                prediction_horizon TEXT,
                model_version TEXT,
                error REAL
            );

            CREATE TABLE IF NOT EXISTS agent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                timestamp TEXT,
                level TEXT,
                message TEXT,
                metadata TEXT
            );
        """)
        conn.commit()

    def save_ohlcv(self, pair: str, df: pd.DataFrame, interval: str = "1d"):
        """Insert or replace OHLCV data from a DataFrame."""
        def _write():
            conn = self._get_conn()
            for ts, row in df.iterrows():
                conn.execute(
                    """INSERT OR REPLACE INTO ohlcv
                       (pair, timestamp, open, high, low, close, volume, interval)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pair, str(ts), row["Open"], row["High"], row["Low"],
                     row["Close"], row.get("Volume", 0), interval),
                )
            conn.commit()
        self._execute_write(_write)

    def get_ohlcv(self, pair: str, interval: str = "1d", limit: int = None) -> pd.DataFrame:
        """Retrieve stored OHLCV data as a DataFrame."""
        conn = self._get_conn()
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE pair=? AND interval=? ORDER BY timestamp"
        params = [pair, interval]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def save_signal(self, signal: dict):
        """Persist a trading signal."""
        def _write():
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO signals
                   (pair, timestamp, signal_type, confidence, entry_price,
                    stop_loss, take_profit, position_size, reasons, predicted_price, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (
                    signal["pair"],
                    datetime.now(timezone.utc).isoformat(),
                    signal["signal_type"],
                    signal.get("confidence", 0),
                    signal["entry_price"],
                    signal["stop_loss"],
                    signal["take_profit"],
                    signal.get("position_size", 0),
                    json.dumps(signal.get("reasons", [])),
                    signal.get("predicted_price", 0),
                ),
            )
            conn.commit()
        self._execute_write(_write)

    def get_open_signals(self) -> pd.DataFrame:
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT * FROM signals WHERE status='OPEN'", conn
        )
        return df

    def get_all_signals(self, limit: int = 100) -> pd.DataFrame:
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
            conn, params=[limit],
        )
        return df

    def update_signal_outcome(self, signal_id: int, status: str, pnl: float):
        def _write():
            conn = self._get_conn()
            conn.execute(
                "UPDATE signals SET status=?, pnl=? WHERE id=?",
                (status, pnl, signal_id),
            )
            conn.commit()
        self._execute_write(_write)

    def save_prediction(self, prediction: dict):
        def _write():
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO predictions
                   (pair, timestamp, predicted_price, prediction_horizon, model_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    prediction["pair"],
                    datetime.now(timezone.utc).isoformat(),
                    prediction["predicted_price"],
                    prediction.get("prediction_horizon", "1d"),
                    prediction.get("model_version", "lstm_v1"),
                ),
            )
            conn.commit()
        self._execute_write(_write)

    def get_predictions(self, pair: str = None, limit: int = 100) -> pd.DataFrame:
        conn = self._get_conn()
        if pair:
            df = pd.read_sql_query(
                "SELECT * FROM predictions WHERE pair=? ORDER BY timestamp DESC LIMIT ?",
                conn, params=[pair, limit],
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
                conn, params=[limit],
            )
        return df

    def log_agent_event(self, agent_name: str, level: str, message: str, metadata: str = ""):
        def _write():
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO agent_logs (agent_name, timestamp, level, message, metadata) VALUES (?, ?, ?, ?, ?)",
                (agent_name, datetime.now(timezone.utc).isoformat(), level, message, metadata),
            )
            conn.commit()
        self._execute_write(_write)

    def get_agent_logs(self, limit: int = 200) -> pd.DataFrame:
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT * FROM agent_logs ORDER BY timestamp DESC LIMIT ?",
            conn, params=[limit],
        )
        return df
