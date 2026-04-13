"""
koi/runtime_state.py — Persistent snapshot of Koi's live runtime state.

This is intentionally separate from AgenticMemory:
  - AgenticMemory is append-only learning history.
  - RuntimeStateStore is overwrite-style control-plane state for restart restore.
  - Only deterministic server/monitor/ledger code should write here. LLM outputs
    may influence decisions, but they should not mutate this state directly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RuntimeStateStore:
    """SQLite-backed persistence for Koi's current runtime state."""

    def __init__(self, db_path: str = "data/koi_runtime.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        if db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_tables(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS tracked_jobs (
                    job_id       TEXT PRIMARY KEY,
                    group_id     TEXT,
                    decision_id  TEXT,
                    tracker_json TEXT NOT NULL,
                    updated_at   REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_launches (
                    job_id       TEXT PRIMARY KEY,
                    launch_json  TEXT NOT NULL,
                    updated_at   REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_scale_decisions (
                    group_id      TEXT NOT NULL,
                    queue_index   INTEGER NOT NULL,
                    decision_json TEXT NOT NULL,
                    updated_at    REAL NOT NULL,
                    PRIMARY KEY (group_id, queue_index)
                );

                CREATE TABLE IF NOT EXISTS ledger_reservations (
                    decision_id      TEXT PRIMARY KEY,
                    reservation_json TEXT NOT NULL,
                    expires_at       REAL,
                    updated_at       REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # tracked_jobs
    # ------------------------------------------------------------------

    def upsert_tracked_job(
        self,
        job_id: str,
        tracker: Dict[str, Any],
        group_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tracked_jobs (job_id, group_id, decision_id, tracker_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    group_id=excluded.group_id,
                    decision_id=excluded.decision_id,
                    tracker_json=excluded.tracker_json,
                    updated_at=excluded.updated_at
                """,
                (
                    job_id,
                    group_id if group_id is not None else tracker.get("group_id"),
                    decision_id if decision_id is not None else tracker.get("decision_id"),
                    json.dumps(tracker),
                    now,
                ),
            )
            self._conn.commit()

    def delete_tracked_job(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM tracked_jobs WHERE job_id = ?", (job_id,))
            self._conn.commit()

    def load_tracked_jobs(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id, group_id, decision_id, tracker_json, updated_at FROM tracked_jobs"
            ).fetchall()
        return {
            row["job_id"]: {
                "group_id": row["group_id"],
                "decision_id": row["decision_id"],
                "tracker": json.loads(row["tracker_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    # ------------------------------------------------------------------
    # pending_launches
    # ------------------------------------------------------------------

    def upsert_pending_launch(self, job_id: str, launch_info: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO pending_launches (job_id, launch_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    launch_json=excluded.launch_json,
                    updated_at=excluded.updated_at
                """,
                (job_id, json.dumps(launch_info), now),
            )
            self._conn.commit()

    def delete_pending_launch(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pending_launches WHERE job_id = ?", (job_id,))
            self._conn.commit()

    def load_pending_launches(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id, launch_json, updated_at FROM pending_launches"
            ).fetchall()
        return {
            row["job_id"]: {
                "launch": json.loads(row["launch_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    # ------------------------------------------------------------------
    # pending_scale_decisions
    # ------------------------------------------------------------------

    def replace_pending_scale_group(self, group_id: str, decisions: List[Dict[str, Any]]) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "DELETE FROM pending_scale_decisions WHERE group_id = ?",
                (group_id,),
            )
            if decisions:
                self._conn.executemany(
                    """
                    INSERT INTO pending_scale_decisions (group_id, queue_index, decision_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (group_id, idx, json.dumps(decision), now)
                        for idx, decision in enumerate(decisions)
                    ],
                )
            self._conn.commit()

    def delete_pending_scale_group(self, group_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM pending_scale_decisions WHERE group_id = ?",
                (group_id,),
            )
            self._conn.commit()

    def load_pending_scale_decisions(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT group_id, queue_index, decision_json
                FROM pending_scale_decisions
                ORDER BY group_id, queue_index
                """
            ).fetchall()
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            result.setdefault(row["group_id"], []).append(json.loads(row["decision_json"]))
        return result

    # ------------------------------------------------------------------
    # ledger_reservations
    # ------------------------------------------------------------------

    def upsert_ledger_reservation(
        self,
        decision_id: str,
        reservation: Dict[str, Any],
        expires_at: Optional[float] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ledger_reservations (decision_id, reservation_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    reservation_json=excluded.reservation_json,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at
                """,
                (decision_id, json.dumps(reservation), expires_at, now),
            )
            self._conn.commit()

    def delete_ledger_reservation(self, decision_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM ledger_reservations WHERE decision_id = ?",
                (decision_id,),
            )
            self._conn.commit()

    def load_ledger_reservations(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT decision_id, reservation_json, expires_at, updated_at
                FROM ledger_reservations
                """
            ).fetchall()
        return {
            row["decision_id"]: {
                "reservation": json.loads(row["reservation_json"]),
                "expires_at": row["expires_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }
