"""Where finished traces are kept.

One row per trace: the headline numbers in their own columns, so the run list
never has to read a span, and the whole span tree as one JSON payload, because
a trace is only ever read whole. Traces live in their own SQLite file beside
the main database, so recording them never contends with the analysis records.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    analysis_id TEXT,
    document_id TEXT,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    summary     TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_analysis ON traces(analysis_id, started_at DESC);
"""


class SQLiteTraceStore:
    """Finished traces in a local SQLite file, newest ``retention`` kept."""

    def __init__(self, path: Path, *, retention: int = 200) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention = retention
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(self, trace: dict[str, Any]) -> None:
        summary = {k: v for k, v in trace.items() if k != "spans"}
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO traces
                    (trace_id, analysis_id, document_id, name, status, started_at, ended_at,
                     summary, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace["trace_id"],
                    trace.get("analysis_id"),
                    trace.get("document_id"),
                    trace["name"],
                    trace["status"],
                    trace["started_at"],
                    trace.get("ended_at"),
                    json.dumps(summary, default=str),
                    json.dumps(trace, default=str),
                ),
            )
            if self.retention:
                self._conn.execute(
                    """
                    DELETE FROM traces WHERE trace_id NOT IN (
                        SELECT trace_id FROM traces ORDER BY started_at DESC LIMIT ?
                    )
                    """,
                    (self.retention,),
                )
            self._conn.commit()

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def latest_for_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM traces WHERE analysis_id = ? ORDER BY started_at DESC LIMIT 1",
                (analysis_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(
        self, *, analysis_ids: list[str] | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        if analysis_ids is not None and not analysis_ids:
            return []
        sql = "SELECT summary FROM traces"
        params: list[Any] = []
        if analysis_ids is not None:
            sql += f" WHERE analysis_id IN ({','.join('?' * len(analysis_ids))})"
            params.extend(analysis_ids)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["summary"]) for r in rows]

    def delete(self, trace_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
