"""`state/manifest.sqlite`: one row per archive day plus its state machine (plan §9).

    queued → downloading → downloaded → processing → processed → verified → cleaned
    any → failed (retry_count); failed → queued / downloaded on retry

The manifest is the single source of truth for resuming: on start, half-finished states roll back
to their last safe state (`recover_on_start`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

STATES = ("queued", "downloading", "downloaded", "processing", "processed", "verified",
          "cleaned", "failed")
TRANSITIONS: dict[str, set[str]] = {
    "queued": {"downloading", "downloaded", "failed"},
    "downloading": {"downloaded", "queued", "failed"},
    "downloaded": {"processing", "queued", "failed"},
    "processing": {"processed", "downloaded", "failed"},
    "processed": {"verified", "downloaded", "failed"},
    "verified": {"cleaned", "downloaded", "failed"},
    "cleaned": {"downloaded", "queued"},          # reprocess (retained raw) or re-download
    "failed": {"queued", "downloaded"},
}
DONE = ("verified", "cleaned")
# Raw tars for these states sit in scratch and count toward the buffer high-water mark.
BUFFERED = ("downloading", "downloaded", "processing", "processed", "verified")
TIERS = ("P0", "P1a", "P1b", "P2", "P3a", "P3b", "P4")

SCHEMA = """
CREATE TABLE IF NOT EXISTS days (
    date TEXT PRIMARY KEY,
    repo TEXT, tag TEXT, variant TEXT,
    urls TEXT, part_sizes TEXT, total_bytes INTEGER,
    preferred_tag TEXT, selection_note TEXT,
    tier TEXT, priority INTEGER, retain_raw INTEGER NOT NULL DEFAULT 0, window_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    retry_count INTEGER NOT NULL DEFAULT 0, last_error TEXT,
    bytes_done INTEGER NOT NULL DEFAULT 0,
    raw_dir TEXT,
    t_download_start REAL, t_downloaded REAL, download_sec REAL,
    t_process_start REAL, t_processed REAL, process_sec REAL,
    t_verified REAL, t_cleaned REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS days_status ON days(status);
CREATE INDEX IF NOT EXISTS days_priority ON days(priority);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS history (
    ts REAL, date TEXT, from_status TEXT, to_status TEXT, note TEXT
);
"""

CATALOG_FIELDS = ("repo", "tag", "variant", "urls", "part_sizes", "total_bytes", "preferred_tag",
                  "selection_note")
DAY_FIELDS = {
    "repo", "tag", "variant", "urls", "part_sizes", "total_bytes", "preferred_tag",
    "selection_note", "tier", "priority", "retain_raw", "window_id", "retry_count", "last_error",
    "bytes_done", "raw_dir", "t_download_start", "t_downloaded", "download_sec",
    "t_process_start", "t_processed", "process_sec", "t_verified", "t_cleaned",
}


class InvalidTransition(RuntimeError):
    pass


class Manifest:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        self._lock = threading.RLock()
        if readonly:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---- reads -------------------------------------------------------------------------------
    def get(self, d: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM days WHERE date=?", (d,)).fetchone()
        return _decode(row) if row else None

    def days(self, statuses=None, tiers=None, dates=None, order="priority") -> list[dict]:
        sql, args = "SELECT * FROM days WHERE 1=1", []
        for col, vals in (("status", statuses), ("tier", tiers), ("date", dates)):
            if vals is not None:
                vals = list(vals)
                if not vals:
                    return []
                sql += f" AND {col} IN ({','.join('?' * len(vals))})"
                args += vals
        if order == "priority":
            sql += " ORDER BY priority IS NULL, priority, date"
        elif order:
            sql += f" ORDER BY {order}"
        with self._lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [_decode(r) for r in rows]

    def counts(self) -> dict[str, dict[str, int]]:
        """{tier: {status: n}} including tier None as 'unscheduled'."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT COALESCE(tier,'unscheduled') t, status, COUNT(*) n FROM days GROUP BY t, status"
            ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            out.setdefault(r["t"], {})[r["status"]] = r["n"]
        return out

    def scalar(self, sql: str, args=()) -> object:
        with self._lock:
            row = self.conn.execute(sql, args).fetchone()
        return row[0] if row else None

    def buffer_bytes(self) -> int:
        q = (f"SELECT COALESCE(SUM(CASE WHEN status='downloading' THEN bytes_done ELSE total_bytes END),0) "
             f"FROM days WHERE status IN ({','.join('?' * len(BUFFERED))}) "
             f"AND (raw_dir IS NULL OR raw_dir NOT LIKE '%raw_retained%')")
        return int(self.scalar(q, BUFFERED) or 0)

    # ---- writes ------------------------------------------------------------------------------
    def upsert_catalog(self, rows: list[dict]) -> tuple[int, int]:
        """Insert new days; refresh catalog fields of days that have not started downloading."""
        new = changed = 0
        now = time.time()
        with self._lock, self.conn:
            for r in rows:
                vals = {k: r.get(k) for k in CATALOG_FIELDS}
                for k in ("urls", "part_sizes"):
                    vals[k] = json.dumps(vals[k])
                cur = self.conn.execute("SELECT status, tag, urls FROM days WHERE date=?",
                                        (r["date"],)).fetchone()
                if cur is None:
                    cols = ["date", *CATALOG_FIELDS, "updated_at"]
                    self.conn.execute(
                        f"INSERT INTO days ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [r["date"], *vals.values(), now])
                    new += 1
                elif cur["status"] in ("queued", "failed"):
                    sets = ",".join(f"{k}=?" for k in CATALOG_FIELDS)
                    self.conn.execute(f"UPDATE days SET {sets}, updated_at=? WHERE date=?",
                                      [*vals.values(), now, r["date"]])
                    if cur["urls"] != vals["urls"]:
                        changed += 1
                        if cur["status"] == "failed":
                            # different files now: earlier failures say nothing about them
                            self.conn.execute("UPDATE days SET retry_count=0 WHERE date=?", (r["date"],))
        return new, changed

    def drop_unstarted(self, keep: set[str]) -> int:
        """Delete queued days that are no longer in the catalog. Started days are never deleted."""
        with self._lock, self.conn:
            gone = [r[0] for r in self.conn.execute("SELECT date FROM days WHERE status='queued'")
                    if r[0] not in keep]
            self.conn.executemany("DELETE FROM days WHERE date=?", [(d,) for d in gone])
        return len(gone)

    def set_schedule(self, assignments: list[tuple[str, str, int, bool, str | None]]) -> None:
        """assignments: (date, tier, priority, retain_raw, window_id)."""
        with self._lock, self.conn:
            self.conn.executemany(
                "UPDATE days SET tier=?, priority=?, retain_raw=?, window_id=? WHERE date=?",
                [(t, p, int(r), w, d) for d, t, p, r, w in assignments])

    def update(self, d: str, **fields) -> None:
        bad = set(fields) - DAY_FIELDS
        if bad:
            raise KeyError(f"unknown manifest fields: {bad}")
        if not fields:
            return
        for k in ("urls", "part_sizes"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        sets = ",".join(f"{k}=?" for k in fields)
        with self._lock, self.conn:
            self.conn.execute(f"UPDATE days SET {sets}, updated_at=? WHERE date=?",
                              [*fields.values(), time.time(), d])

    def transition(self, d: str, to: str, note: str | None = None, *, expect: str | None = None,
                   **fields) -> None:
        if to not in STATES:
            raise InvalidTransition(f"unknown state {to}")
        with self._lock, self.conn:
            row = self.conn.execute("SELECT status FROM days WHERE date=?", (d,)).fetchone()
            if row is None:
                raise KeyError(d)
            cur = row["status"]
            if expect is not None and cur != expect:
                raise InvalidTransition(f"{d}: expected {expect}, found {cur}")
            if to != cur and to not in TRANSITIONS[cur]:
                raise InvalidTransition(f"{d}: {cur} -> {to} not allowed")
            for k in ("urls", "part_sizes"):
                if k in fields and not isinstance(fields[k], str):
                    fields[k] = json.dumps(fields[k])
            bad = set(fields) - DAY_FIELDS
            if bad:
                raise KeyError(f"unknown manifest fields: {bad}")
            sets = "".join(f", {k}=?" for k in fields)
            self.conn.execute(f"UPDATE days SET status=?{sets}, updated_at=? WHERE date=?",
                              [to, *fields.values(), time.time(), d])
            self.conn.execute("INSERT INTO history VALUES (?,?,?,?,?)",
                              (time.time(), d, cur, to, note))

    def recover_on_start(self) -> list[tuple[str, str, str]]:
        """Roll interrupted states back. Returns [(date, from, to)]."""
        moves = []
        for row in self.days(statuses=["downloading", "processing"], order=None):
            to = "queued" if row["status"] == "downloading" else "downloaded"
            self.transition(row["date"], to, note="recovered on start")
            moves.append((row["date"], row["status"], to))
        return moves

    # ---- key/value ---------------------------------------------------------------------------
    def kv_get(self, key: str, default=None):
        v = self.scalar("SELECT value FROM kv WHERE key=?", (key,))
        return default if v is None else json.loads(v)

    def kv_set(self, key: str, value) -> None:
        with self._lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value)))


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("urls", "part_sizes"):
        if d.get(k):
            d[k] = json.loads(d[k])
    return d
