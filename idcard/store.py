"""SQLite log of violations. One connection shared by threads, guarded by a lock."""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class Store:
    def __init__(self, db_path):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS violations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "time TEXT NOT NULL, track_id INTEGER NOT NULL, "
                "score REAL NOT NULL, snapshot TEXT NOT NULL)"
            )
            self._db.commit()

    def add(self, track_id, score, snapshot, time=None) -> None:
        time = time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self._db.execute(
                "INSERT INTO violations (time, track_id, score, snapshot) VALUES (?, ?, ?, ?)",
                (time, int(track_id), float(score), snapshot),
            )
            self._db.commit()

    def list(self, limit=500) -> list[dict]:
        """Newest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, time, track_id, score, snapshot FROM violations "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> list[str]:
        """Delete all rows. Returns the snapshot file names so the caller can delete the files."""
        with self._lock:
            names = [r["snapshot"] for r in self._db.execute("SELECT snapshot FROM violations")]
            self._db.execute("DELETE FROM violations")
            self._db.commit()
        return names

    def to_csv(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["time", "track_id", "score", "snapshot"])
        for r in reversed(self.list(limit=-1)):  # oldest first in the export
            writer.writerow([r["time"], r["track_id"], f"{r['score']:.3f}", r["snapshot"]])
        return out.getvalue()

    def close(self) -> None:
        with self._lock:
            self._db.close()
