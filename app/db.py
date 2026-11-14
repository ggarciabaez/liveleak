"""SQLite EventHistoryStore adhering strictly to the 3-table ERD (USER, SESSION, RISK_ESTIMATE)."""

from __future__ import annotations

import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    user_id: str
    session_date: str
    sample_count: int
    mean_score: float
    peak_score: float
    mean_cov: float


@dataclass(frozen=True)
class RiskEstimateRow:
    session_id: str
    timestamp: str
    score: float
    covariance: float


class EventHistoryStore:
    """Single-writer, multi-reader SQLite store for retroactive risk analysis."""

    def __init__(self, db_path: str | Path = "risk.db") -> None:
        self.db_path = Path(db_path)
        self._write_queue: queue.Queue[tuple | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._init_schema()
        self._writer = threading.Thread(
            target=self._writer_loop, name="SQLiteWriterThread", daemon=True
        )
        self._writer.start()

    def _get_read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS USER
                (
                    UserID
                    TEXT
                    PRIMARY
                    KEY
                );

                CREATE TABLE IF NOT EXISTS SESSION
                (
                    SessionID
                    TEXT
                    PRIMARY
                    KEY,
                    UserID
                    TEXT
                    NOT
                    NULL,
                    Date
                    TEXT
                    NOT
                    NULL,
                    FOREIGN
                    KEY
                (
                    UserID
                ) REFERENCES USER
                (
                    UserID
                )
                    );

                CREATE TABLE IF NOT EXISTS RISK_ESTIMATE
                (
                    SessionID
                    TEXT
                    NOT
                    NULL,
                    Timestamp
                    TEXT
                    NOT
                    NULL,
                    Score
                    REAL
                    NOT
                    NULL,
                    Covariance
                    REAL
                    NOT
                    NULL,
                    PRIMARY
                    KEY
                (
                    SessionID,
                    Timestamp
                ),
                    FOREIGN KEY
                (
                    SessionID
                ) REFERENCES SESSION
                (
                    SessionID
                )
                    );

                CREATE INDEX IF NOT EXISTS idx_session_date ON SESSION (Date);
                CREATE INDEX IF NOT EXISTS idx_session_user ON SESSION (UserID);
                CREATE INDEX IF NOT EXISTS idx_estimate_score ON RISK_ESTIMATE(Score);
                """
            )

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        batch: list[tuple] = []
        while not self._stop_event.is_set() or not self._write_queue.empty():
            try:
                item = self._write_queue.get(timeout=0.25)
                if item is None:
                    break
                batch.append(item)
                # Drain any immediately available items up to 128
                while len(batch) < 128:
                    try:
                        nxt = self._write_queue.get_nowait()
                        if nxt is None:
                            self._stop_event.set()
                            break
                        batch.append(nxt)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            if batch:
                try:
                    with conn:
                        for op in batch:
                            kind = op[0]
                            if kind == "OPEN_SESSION":
                                _, user_id, session_id, dt_str = op
                                conn.execute(
                                    "INSERT OR IGNORE INTO USER (UserID) VALUES (?)",
                                    (user_id,),
                                )
                                conn.execute(
                                    "INSERT OR IGNORE INTO SESSION (SessionID, UserID, Date) VALUES (?, ?, ?)",
                                    (session_id, user_id, dt_str),
                                )
                            elif kind == "ESTIMATE":
                                _, session_id, ts_str, score, cov = op
                                conn.execute(
                                    "INSERT OR REPLACE INTO RISK_ESTIMATE (SessionID, Timestamp, Score, Covariance) "
                                    "VALUES (?, ?, ?, ?)",
                                    (session_id, ts_str, float(score), float(cov)),
                                )
                except sqlite3.Error as exc:
                    print(f"[EventHistoryStore] Write error: {exc}")
                finally:
                    batch.clear()

        conn.close()

    def open_session(self, user_id: str) -> str:
        """Create a new session for `user_id` and enqueue it for persistence."""
        clean_user = user_id.strip() or "operator_01"
        session_id = f"ses_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        dt_str = date.today().isoformat()
        self._write_queue.put(("OPEN_SESSION", clean_user, session_id, dt_str))
        return session_id

    def record_estimate(
            self, session_id: str, score: float, covariance: float, ts: datetime | None = None
    ) -> None:
        """Non-blocking enqueue of a risk estimate row."""
        ts_str = (ts or datetime.now()).isoformat(timespec="milliseconds")
        self._write_queue.put(("ESTIMATE", session_id, ts_str, float(score), float(covariance)))

    def close(self) -> None:
        self._stop_event.set()
        self._write_queue.put(None)
        if self._writer.is_alive():
            self._writer.join(timeout=2.0)

    # ── Read-Only Analytical Queries for DataTab ──────────────────────────

    def list_users(self) -> list[str]:
        if not self.db_path.exists():
            return []
        with self._get_read_conn() as conn:
            rows = conn.execute("SELECT UserID FROM USER ORDER BY UserID").fetchall()
            return [r["UserID"] for r in rows]

    def query_sessions(
            self,
            user_id: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
    ) -> list[SessionSummary]:
        if not self.db_path.exists():
            return []

        sql = """
              SELECT s.SessionID                      AS session_id,
                     s.UserID                         AS user_id,
                     s.Date                           AS session_date,
                     COUNT(r.Score)                   AS sample_count,
                     COALESCE(AVG(r.Score), 0.0)      AS mean_score,
                     COALESCE(MAX(r.Score), 0.0)      AS peak_score,
                     COALESCE(AVG(r.Covariance), 0.0) AS mean_cov
              FROM SESSION s
                       LEFT JOIN RISK_ESTIMATE r ON s.SessionID = r.SessionID
              WHERE 1 = 1 \
              """
        params: list[object] = []
        if user_id and user_id != "All Users":
            sql += " AND s.UserID = ?"
            params.append(user_id)
        if start_date:
            sql += " AND s.Date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND s.Date <= ?"
            params.append(end_date)

        sql += " GROUP BY s.SessionID ORDER BY s.Date DESC, s.SessionID DESC"

        with self._get_read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                SessionSummary(
                    session_id=r["session_id"],
                    user_id=r["user_id"],
                    session_date=r["session_date"],
                    sample_count=int(r["sample_count"]),
                    mean_score=float(r["mean_score"]),
                    peak_score=float(r["peak_score"]),
                    mean_cov=float(r["mean_cov"]),
                )
                for r in rows
            ]

    def get_session_estimates(self, session_id: str) -> list[RiskEstimateRow]:
        if not self.db_path.exists():
            return []
        with self._get_read_conn() as conn:
            rows = conn.execute(
                """
                SELECT SessionID, Timestamp, Score, Covariance
                FROM RISK_ESTIMATE
                WHERE SessionID = ?
                ORDER BY Timestamp ASC
                """,
                (session_id,),
            ).fetchall()
            return [
                RiskEstimateRow(
                    session_id=r["SessionID"],
                    timestamp=r["Timestamp"],
                    score=float(r["Score"]),
                    covariance=float(r["Covariance"]),
                )
                for r in rows
            ]

    def get_filtered_estimates(
            self,
            user_id: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
            session_id: str | None = None,
            min_score: float = 0.0,
            limit: int = 500,
    ) -> list[RiskEstimateRow]:
        if not self.db_path.exists():
            return []

        sql = """
              SELECT r.SessionID, r.Timestamp, r.Score, r.Covariance
              FROM RISK_ESTIMATE r
                       JOIN SESSION s ON r.SessionID = s.SessionID
              WHERE r.Score >= ? \
              """
        params: list[object] = [min_score]
        if session_id:
            sql += " AND r.SessionID = ?"
            params.append(session_id)
        if user_id and user_id != "All Users":
            sql += " AND s.UserID = ?"
            params.append(user_id)
        if start_date:
            sql += " AND s.Date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND s.Date <= ?"
            params.append(end_date)

        sql += " ORDER BY r.Score DESC, r.Timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                RiskEstimateRow(
                    session_id=r["SessionID"],
                    timestamp=r["Timestamp"],
                    score=float(r["Score"]),
                    covariance=float(r["Covariance"]),
                )
                for r in rows
            ]

    def get_daily_incident_buckets(
            self,
            threshold: float,
            user_id: str | None = None,
            start_date: str | None = None,
            end_date: str | None = None,
    ) -> list[tuple[str, int]]:
        """mirrors ReportAggregator.bucket_by_day: count estimates >= threshold per day."""
        if not self.db_path.exists():
            return []

        sql = """
              SELECT s.Date AS day,
                SUM(CASE WHEN r.Score >= ? THEN 1 ELSE 0 END) AS incident_count
              FROM SESSION s
                  LEFT JOIN RISK_ESTIMATE r
              ON s.SessionID = r.SessionID
              WHERE 1=1 \
              """
        params: list[object] = [threshold]
        if user_id and user_id != "All Users":
            sql += " AND s.UserID = ?"
            params.append(user_id)
        if start_date:
            sql += " AND s.Date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND s.Date <= ?"
            params.append(end_date)

        sql += " GROUP BY s.Date ORDER BY s.Date ASC"

        with self._get_read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [(r["day"], int(r["incident_count"] or 0)) for r in rows]
