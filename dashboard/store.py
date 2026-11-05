"""Reviewer decisions in SQLite. Append-only: the latest row per alert counts."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DECISIONS = ("accept", "dismiss")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id   INTEGER NOT NULL,
    decision   TEXT    NOT NULL CHECK (decision IN ('accept', 'dismiss')),
    comment    TEXT    NOT NULL DEFAULT '',
    decided_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_alert ON decisions (alert_id);
"""


class DecisionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def record(self, alert_id: int, decision: str, comment: str = "") -> None:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:  # parameters, never string formatting (no SQL injection)
            db.execute("INSERT INTO decisions (alert_id, decision, comment, decided_at) "
                       "VALUES (?, ?, ?, ?)", (int(alert_id), decision, comment[:500], now))

    def latest(self) -> dict[int, str]:
        with self._connect() as db:
            rows = db.execute("SELECT alert_id, decision FROM decisions d WHERE id = "
                              "(SELECT MAX(id) FROM decisions WHERE alert_id = d.alert_id)")
            return {r["alert_id"]: r["decision"] for r in rows}

    def history(self, alert_id: int) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT decision, comment, decided_at FROM decisions "
                              "WHERE alert_id = ? ORDER BY id", (int(alert_id),))
            return [dict(r) for r in rows]
