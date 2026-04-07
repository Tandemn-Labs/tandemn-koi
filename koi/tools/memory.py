"""
koi/tools/memory.py — Agentic Memory backed by SQLite.

Three tables:
  decisions       — every config Koi proposes (with predictions)
  outcomes        — what actually happened (ground truth)
  launch_attempts — per-attempt launch success/failure tracking
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class AgenticMemory:
    """Structured, persistent memory for the Koi agent."""

    def __init__(self, db_path: str = "data/koi_memory.db"):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persistent_conn = sqlite3.connect(db_path)
        self._persistent_conn.row_factory = sqlite3.Row
        if db_path != ":memory:":
            self._persistent_conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        return self._persistent_conn

    def _init_tables(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id          TEXT PRIMARY KEY,
                job_id               TEXT NOT NULL,
                timestamp            TEXT DEFAULT (datetime('now')),
                model_name           TEXT NOT NULL,
                instance_type        TEXT NOT NULL,
                gpu_type             TEXT NOT NULL,
                tp                   INTEGER NOT NULL,
                pp                   INTEGER NOT NULL,
                dp                   INTEGER NOT NULL,
                num_gpus             INTEGER NOT NULL,
                quantization         TEXT,
                predicted_tps        REAL,
                predicted_cost_per_hour REAL,
                predicted_total_cost REAL,
                predicted_runtime_hours REAL,
                prediction_confidence REAL,
                prediction_source    TEXT,
                slo_deadline_hours   REAL,
                objective            TEXT,
                avg_input_tokens     INTEGER,
                avg_output_tokens    INTEGER,
                num_requests         INTEGER,
                triggered_by         TEXT DEFAULT 'user',
                parent_decision_id   TEXT,
                market               TEXT DEFAULT 'on_demand'
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id           TEXT PRIMARY KEY,
                decision_id          TEXT REFERENCES decisions(decision_id),
                job_id               TEXT NOT NULL,
                timestamp            TEXT DEFAULT (datetime('now')),
                status               TEXT NOT NULL,
                actual_tps           REAL,
                actual_cost_per_hour REAL,
                actual_total_cost    REAL,
                actual_runtime_hours REAL,
                delta_tps_pct        REAL,
                delta_cost_pct       REAL,
                slo_met              INTEGER,
                slo_headroom_pct     REAL,
                failure_category     TEXT,
                diagnosis            TEXT,
                bottleneck           TEXT,
                diff_from_parent     TEXT
            );

            CREATE TABLE IF NOT EXISTS launch_attempts (
                attempt_id           TEXT PRIMARY KEY,
                decision_id          TEXT REFERENCES decisions(decision_id),
                job_id               TEXT NOT NULL,
                timestamp            TEXT DEFAULT (datetime('now')),
                instance_type        TEXT NOT NULL,
                gpu_type             TEXT NOT NULL,
                region               TEXT NOT NULL,
                market               TEXT NOT NULL,
                count                INTEGER NOT NULL,
                launched             INTEGER NOT NULL,
                time_to_launch       REAL,
                failure_reason       TEXT,
                quota_available      INTEGER,
                other_jobs_in_region TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_model ON decisions(model_name);
            CREATE INDEX IF NOT EXISTS idx_outcomes_job ON outcomes(job_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_status ON outcomes(status);
            CREATE INDEX IF NOT EXISTS idx_launch_instance ON launch_attempts(instance_type, region);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_decision(
        self, job_id: str, model_name: str,
        instance_type: str, gpu_type: str, tp: int, pp: int, dp: int,
        num_gpus: int, predicted_tps: float, predicted_cost_per_hour: float,
        slo_deadline_hours: float, objective: str,
        avg_input_tokens: int, avg_output_tokens: int,
        num_requests: Optional[int] = None,
        predicted_total_cost: Optional[float] = None,
        predicted_runtime_hours: Optional[float] = None,
        prediction_confidence: float = 0.5,
        prediction_source: str = "analytical",
        quantization: Optional[str] = None,
        triggered_by: str = "user",
        parent_decision_id: Optional[str] = None,
        market: str = "on_demand",
    ) -> str:
        decision_id = f"dec-{uuid.uuid4().hex[:8]}"
        conn = self._conn()
        conn.execute("""
            INSERT INTO decisions (
                decision_id, job_id, model_name, instance_type, gpu_type,
                tp, pp, dp, num_gpus, quantization,
                predicted_tps, predicted_cost_per_hour, predicted_total_cost,
                predicted_runtime_hours, prediction_confidence, prediction_source,
                slo_deadline_hours, objective, avg_input_tokens, avg_output_tokens,
                num_requests, triggered_by, parent_decision_id, market
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            decision_id, job_id, model_name, instance_type, gpu_type,
            tp, pp, dp, num_gpus, quantization,
            predicted_tps, predicted_cost_per_hour, predicted_total_cost,
            predicted_runtime_hours, prediction_confidence, prediction_source,
            slo_deadline_hours, objective, avg_input_tokens, avg_output_tokens,
            num_requests, triggered_by, parent_decision_id, market,
        ))
        conn.commit()
        return decision_id

    def record_outcome(
        self, decision_id: str, job_id: str, status: str,
        actual_tps: Optional[float] = None,
        actual_cost_per_hour: Optional[float] = None,
        actual_total_cost: Optional[float] = None,
        actual_runtime_hours: Optional[float] = None,
        slo_met: Optional[bool] = None,
        slo_headroom_pct: Optional[float] = None,
        failure_category: Optional[str] = None,
        diagnosis: Optional[str] = None,
        bottleneck: Optional[str] = None,
        diff_from_parent: Optional[str] = None,
    ) -> str:
        outcome_id = f"out-{uuid.uuid4().hex[:8]}"

        # Compute delta from decision's prediction
        delta_tps_pct = None
        delta_cost_pct = None
        conn = self._conn()
        row = conn.execute(
            "SELECT predicted_tps, predicted_cost_per_hour FROM decisions WHERE decision_id = ?",
            (decision_id,)
        ).fetchone()
        if row and actual_tps and row["predicted_tps"]:
            delta_tps_pct = (actual_tps - row["predicted_tps"]) / max(row["predicted_tps"], 1) * 100
        if row and actual_cost_per_hour and row["predicted_cost_per_hour"]:
            delta_cost_pct = (actual_cost_per_hour - row["predicted_cost_per_hour"]) / max(row["predicted_cost_per_hour"], 0.01) * 100

        conn.execute("""
            INSERT INTO outcomes (
                outcome_id, decision_id, job_id, status,
                actual_tps, actual_cost_per_hour, actual_total_cost,
                actual_runtime_hours,
                delta_tps_pct, delta_cost_pct, slo_met, slo_headroom_pct,
                failure_category, diagnosis, bottleneck, diff_from_parent
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            outcome_id, decision_id, job_id, status,
            actual_tps, actual_cost_per_hour, actual_total_cost,
            actual_runtime_hours,
            delta_tps_pct, delta_cost_pct,
            int(slo_met) if slo_met is not None else None,
            slo_headroom_pct,
            failure_category, diagnosis, bottleneck, diff_from_parent,
        ))
        conn.commit()
        return outcome_id

    def record_launch_attempt(
        self, decision_id: str, job_id: str,
        instance_type: str, gpu_type: str, region: str, market: str,
        count: int, launched: bool,
        time_to_launch: Optional[float] = None,
        failure_reason: Optional[str] = None,
        quota_available: Optional[int] = None,
        other_jobs_in_region: Optional[list] = None,
    ) -> str:
        attempt_id = f"att-{uuid.uuid4().hex[:8]}"
        conn = self._conn()
        conn.execute("""
            INSERT INTO launch_attempts (
                attempt_id, decision_id, job_id,
                instance_type, gpu_type, region, market, count,
                launched, time_to_launch, failure_reason,
                quota_available, other_jobs_in_region
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            attempt_id, decision_id, job_id,
            instance_type, gpu_type, region, market, count,
            int(launched), time_to_launch, failure_reason,
            quota_available,
            json.dumps(other_jobs_in_region) if other_jobs_in_region else None,
        ))
        conn.commit()
        return attempt_id

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query_decisions(
        self, model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        conn = self._conn()
        query = """
            SELECT d.*, o.status, o.actual_tps, o.actual_total_cost,
                   o.delta_tps_pct, o.slo_met, o.diagnosis
            FROM decisions d
            LEFT JOIN outcomes o ON d.decision_id = o.decision_id
            WHERE 1=1
        """
        params: list = []
        if model_name:
            query += " AND d.model_name LIKE ?"
            params.append(f"%{model_name}%")
        if gpu_type:
            query += " AND d.gpu_type LIKE ?"
            params.append(f"%{gpu_type}%")
        query += " ORDER BY d.timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def query_outcomes(
        self, model_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        conn = self._conn()
        query = """
            SELECT o.*, d.model_name, d.gpu_type, d.tp, d.pp, d.dp,
                   d.instance_type, d.predicted_tps
            FROM outcomes o
            JOIN decisions d ON o.decision_id = d.decision_id
            WHERE 1=1
        """
        params: list = []
        if model_name:
            query += " AND d.model_name LIKE ?"
            params.append(f"%{model_name}%")
        if status:
            query += " AND o.status = ?"
            params.append(status)
        query += " ORDER BY o.timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_launch_success_rate(
        self, instance_type: str, region: Optional[str] = None, hours: int = 24,
    ) -> Dict[str, Any]:
        conn = self._conn()
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        query = """
            SELECT COUNT(*) as attempts,
                   SUM(launched) as succeeded,
                   AVG(CASE WHEN launched = 1 THEN time_to_launch END) as avg_time
            FROM launch_attempts
            WHERE instance_type = ? AND timestamp > ?
        """
        params: list = [instance_type, cutoff]
        if region:
            query += " AND region = ?"
            params.append(region)

        row = conn.execute(query, params).fetchone()
        attempts = row["attempts"] or 0
        succeeded = row["succeeded"] or 0
        return {
            "instance_type": instance_type,
            "region": region,
            "window_hours": hours,
            "attempts": attempts,
            "succeeded": succeeded,
            "rate": succeeded / max(attempts, 1),
            "avg_time_to_launch": row["avg_time"],
        }

    def decision_count(self) -> int:
        conn = self._conn()
        row = conn.execute("SELECT COUNT(*) as n FROM decisions").fetchone()
        return row["n"]

    def outcome_count(self) -> int:
        conn = self._conn()
        row = conn.execute("SELECT COUNT(*) as n FROM outcomes").fetchone()
        return row["n"]


# ---------------------------------------------------------------------------
# Agent tool functions
# ---------------------------------------------------------------------------

def query_memory(
    memory: AgenticMemory,
    model_name: Optional[str] = None,
    instance_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Query Koi's memory for past decisions and outcomes."""
    lines = []

    # Past outcomes (ground truth from completed jobs)
    outcomes = memory.query_outcomes(model_name=model_name, status=status, limit=limit)
    if outcomes:
        lines.append(f"PAST OUTCOMES ({len(outcomes)} found — ground truth from completed jobs):")
        for o in outcomes:
            slo = "SLO met" if o.get("slo_met") else "SLO missed"
            delta = f"delta={o['delta_tps_pct']:+.1f}%" if o.get("delta_tps_pct") is not None else ""
            if o.get("status") == "failed":
                bottleneck = f"[{o.get('bottleneck', '?')}] " if o.get("bottleneck") else ""
                fail = f" FAILED: {bottleneck}{o.get('diagnosis', '?')}"
            else:
                fail = ""
            lines.append(
                f"  {o.get('model_name','?')} | {o.get('gpu_type','?')} TP={o.get('tp',1)} PP={o.get('pp',1)} | "
                f"TPS={o.get('actual_tps','?')} (pred={o.get('predicted_tps','?')}) {delta} | "
                f"{slo}{fail}"
            )

    # Past decisions (what Koi previously chose — even if no outcome yet)
    decisions = memory.query_decisions(model_name=model_name, gpu_type=instance_type, limit=limit)
    if decisions:
        lines.append(f"\nPAST DECISIONS ({len(decisions)} found — what Koi previously chose):")
        for dec in decisions:
            outcome_status = dec.get("status")
            outcome_tps = dec.get("actual_tps")
            if outcome_status and outcome_tps:
                result = f"→ actual={outcome_tps:.0f} TPS ({outcome_status})"
            elif outcome_status:
                result = f"→ {outcome_status}"
            else:
                result = "→ no outcome yet (job may still be running)"
            triggered = f" [{dec.get('triggered_by', 'user')}]" if dec.get("triggered_by") != "user" else ""
            market = f" {dec.get('market', '')}" if dec.get("market") == "spot" else ""
            lines.append(
                f"  {dec.get('model_name','?')} | {dec.get('gpu_type','?')} TP={dec.get('tp',1)} PP={dec.get('pp',1)} DP={dec.get('dp',1)}{market} | "
                f"predicted={dec.get('predicted_tps','?')} TPS @ ${dec.get('predicted_cost_per_hour','?')}/hr | "
                f"conf={dec.get('prediction_confidence','?')} ({dec.get('prediction_source','?')}){triggered} {result}"
            )

    if not outcomes and not decisions:
        lines.append(f"No memory found for model={model_name or 'any'}. This is the first time Koi has seen this model.")

    return "\n".join(lines)


def record_outcome_tool(
    memory: AgenticMemory,
    decision_id: str,
    job_id: str,
    status: str,
    actual_tps: Optional[float] = None,
    actual_cost_per_hour: Optional[float] = None,
    actual_total_cost: Optional[float] = None,
    actual_runtime_hours: Optional[float] = None,
    failure_category: Optional[str] = None,
    diagnosis: Optional[str] = None,
    bottleneck: Optional[str] = None,
) -> str:
    """Record job outcome in Koi's memory."""
    outcome_id = memory.record_outcome(
        decision_id=decision_id, job_id=job_id, status=status,
        actual_tps=actual_tps, actual_cost_per_hour=actual_cost_per_hour,
        actual_total_cost=actual_total_cost, actual_runtime_hours=actual_runtime_hours,
        failure_category=failure_category, diagnosis=diagnosis, bottleneck=bottleneck,
    )
    return f"Outcome recorded: {outcome_id} (status={status})"
