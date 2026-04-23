"""
koi/tools/memory.py — Agentic Memory backed by SQLite.

Four tables:
  decisions            — every config Koi proposes (with predictions)
  outcomes             — what actually happened (ground truth)
  launch_attempts      — per-attempt launch success/failure tracking
  availability_priors  — Beta(α,β) conjugate priors for GPU availability
"""

import json
import math
import sqlite3
import threading
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
        # check_same_thread=False + Lock — required for FastAPI TestClient
        # (handlers run on a worker thread) and any future background
        # thread that writes outcomes. Matches RuntimeStateStore's pattern.
        self._persistent_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._persistent_conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
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
                cost_roofline_usd    REAL,
                market               TEXT DEFAULT 'unknown'
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
                failure_category     TEXT,
                quota_available      INTEGER,
                other_jobs_in_region TEXT
            );

            CREATE TABLE IF NOT EXISTS availability_priors (
                key              TEXT PRIMARY KEY,
                gpu_type         TEXT NOT NULL,
                region           TEXT,
                market           TEXT,
                alpha            REAL DEFAULT 1.0,
                beta             REAL DEFAULT 1.0,
                last_updated     TEXT DEFAULT (datetime('now')),
                last_decay       TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_decisions_model ON decisions(model_name);
            CREATE INDEX IF NOT EXISTS idx_outcomes_job ON outcomes(job_id);
            CREATE INDEX IF NOT EXISTS idx_outcomes_status ON outcomes(status);
            CREATE INDEX IF NOT EXISTS idx_launch_instance ON launch_attempts(instance_type, region);
            CREATE INDEX IF NOT EXISTS idx_avail_gpu ON availability_priors(gpu_type);
        """)
        conn.commit()

        # The outcomes table is append-only — we never delete historical rows.
        # Dedup is enforced on INSERT via `WHERE NOT EXISTS` in record_outcome.
        # The unique index is created BEST-EFFORT as belt-and-suspenders on
        # fresh DBs; if a legacy DB already contains structural duplicates
        # (from pre-hardening replays), index creation fails and we proceed
        # without it — application-level dedup still prevents new duplicates.
        try:
            conn.execute(
                "ALTER TABLE decisions ADD COLUMN cost_roofline_usd REAL"
            )
            conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()

        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_outcomes_chain "
                "ON outcomes(decision_id, job_id, status)"
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
            # Legacy duplicates detected — leave historical rows untouched.
            # record_outcome's WHERE NOT EXISTS still prevents new ones.

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_decision(
        self,
        job_id: str,
        model_name: str,
        instance_type: str,
        gpu_type: str,
        tp: int,
        pp: int,
        dp: int,
        num_gpus: int,
        predicted_tps: float,
        predicted_cost_per_hour: float,
        slo_deadline_hours: float,
        objective: str,
        avg_input_tokens: int,
        avg_output_tokens: int,
        num_requests: Optional[int] = None,
        predicted_total_cost: Optional[float] = None,
        predicted_runtime_hours: Optional[float] = None,
        prediction_confidence: float = 0.5,
        prediction_source: str = "analytical",
        quantization: Optional[str] = None,
        triggered_by: str = "user",
        parent_decision_id: Optional[str] = None,
        cost_roofline_usd: Optional[float] = None,
        market: str = "unknown",
    ) -> str:
        decision_id = f"dec-{uuid.uuid4().hex[:8]}"
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO decisions (
                decision_id, job_id, model_name, instance_type, gpu_type,
                tp, pp, dp, num_gpus, quantization,
                predicted_tps, predicted_cost_per_hour, predicted_total_cost,
                predicted_runtime_hours, prediction_confidence, prediction_source,
                slo_deadline_hours, objective, avg_input_tokens, avg_output_tokens,
                num_requests, triggered_by, parent_decision_id, cost_roofline_usd, market
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                decision_id,
                job_id,
                model_name,
                instance_type,
                gpu_type,
                tp,
                pp,
                dp,
                num_gpus,
                quantization,
                predicted_tps,
                predicted_cost_per_hour,
                predicted_total_cost,
                predicted_runtime_hours,
                prediction_confidence,
                prediction_source,
                slo_deadline_hours,
                objective,
                avg_input_tokens,
                avg_output_tokens,
                num_requests,
                triggered_by,
                parent_decision_id,
                cost_roofline_usd,
                market,
            ),
        )
        conn.commit()
        return decision_id

    def record_outcome(
        self,
        decision_id: str,
        job_id: str,
        status: str,
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
            (decision_id,),
        ).fetchone()
        if row and actual_tps and row["predicted_tps"]:
            delta_tps_pct = (
                (actual_tps - row["predicted_tps"]) / max(row["predicted_tps"], 1) * 100
            )
        if row and actual_cost_per_hour and row["predicted_cost_per_hour"]:
            delta_cost_pct = (
                (actual_cost_per_hour - row["predicted_cost_per_hour"])
                / max(row["predicted_cost_per_hour"], 0.01)
                * 100
            )

        # Append-only dedup: skip the INSERT if a row with the same
        # (decision_id, job_id, status) already exists. Historical rows
        # are never touched; we just don't add another. This is the second
        # line of defense under inbox event dedup in koi/server.py.
        cur = conn.execute(
            """
            INSERT INTO outcomes (
                outcome_id, decision_id, job_id, status,
                actual_tps, actual_cost_per_hour, actual_total_cost,
                actual_runtime_hours,
                delta_tps_pct, delta_cost_pct, slo_met, slo_headroom_pct,
                failure_category, diagnosis, bottleneck, diff_from_parent
            )
            SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            WHERE NOT EXISTS (
                SELECT 1 FROM outcomes
                WHERE decision_id = ? AND job_id = ? AND status = ?
            )
        """,
            (
                outcome_id,
                decision_id,
                job_id,
                status,
                actual_tps,
                actual_cost_per_hour,
                actual_total_cost,
                actual_runtime_hours,
                delta_tps_pct,
                delta_cost_pct,
                int(slo_met) if slo_met is not None else None,
                slo_headroom_pct,
                failure_category,
                diagnosis,
                bottleneck,
                diff_from_parent,
                # Parameters for the NOT EXISTS sub-select:
                decision_id,
                job_id,
                status,
            ),
        )
        conn.commit()
        if cur.rowcount == 0:
            # Duplicate — return the existing row's outcome_id.
            existing = conn.execute(
                "SELECT outcome_id FROM outcomes "
                "WHERE decision_id = ? AND job_id = ? AND status = ?",
                (decision_id, job_id, status),
            ).fetchone()
            if existing:
                return existing["outcome_id"]
        return outcome_id

    def record_launch_attempt(
        self,
        decision_id: str,
        job_id: str,
        instance_type: str,
        gpu_type: str,
        region: str,
        market: str,
        count: int,
        launched: bool,
        time_to_launch: Optional[float] = None,
        failure_reason: Optional[str] = None,
        failure_category: Optional[str] = None,
        quota_available: Optional[int] = None,
        other_jobs_in_region: Optional[list] = None,
    ) -> str:
        attempt_id = f"att-{uuid.uuid4().hex[:8]}"
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO launch_attempts (
                attempt_id, decision_id, job_id,
                instance_type, gpu_type, region, market, count,
                launched, time_to_launch, failure_reason, failure_category,
                quota_available, other_jobs_in_region
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                attempt_id,
                decision_id,
                job_id,
                instance_type,
                gpu_type,
                region,
                market,
                count,
                int(launched),
                time_to_launch,
                failure_reason,
                failure_category,
                quota_available,
                json.dumps(other_jobs_in_region) if other_jobs_in_region else None,
            ),
        )
        conn.commit()
        return attempt_id

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Look up a single decision by ID."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        return dict(row) if row else None

    def query_decisions(
        self,
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        job_id: Optional[str] = None,
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
        if job_id:
            query += " AND d.job_id LIKE ?"
            params.append(f"%{job_id}%")
        query += " ORDER BY d.timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def query_outcomes(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None,
        job_id: Optional[str] = None,
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
        if job_id:
            query += " AND o.job_id LIKE ?"
            params.append(f"%{job_id}%")
        query += " ORDER BY o.timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Availability Beta priors
    # ------------------------------------------------------------------

    _DECAY_RATE = 0.95  # per-hour decay → ~30% weight after 24h

    def update_availability(
        self,
        gpu_type: str,
        region: str,
        market: str,
        launched: bool,
    ) -> None:
        """Bayesian update: success → α+=1, failure → β+=1 (with time decay)."""
        key = f"{gpu_type}|{region}|{market}"
        conn = self._conn()
        row = conn.execute(
            "SELECT alpha, beta, last_decay FROM availability_priors WHERE key = ?",
            (key,),
        ).fetchone()

        now = datetime.utcnow()
        if row:
            last_decay = datetime.fromisoformat(row["last_decay"])
            hours = (now - last_decay).total_seconds() / 3600
            decay = self._DECAY_RATE**hours
            alpha = row["alpha"] * decay
            beta_val = row["beta"] * decay
        else:
            alpha, beta_val = 1.0, 1.0  # uninformative prior

        if launched:
            alpha += 1
        else:
            beta_val += 1

        conn.execute(
            """
            INSERT INTO availability_priors (key, gpu_type, region, market, alpha, beta, last_updated, last_decay)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                alpha=excluded.alpha, beta=excluded.beta,
                last_updated=excluded.last_updated, last_decay=excluded.last_decay
        """,
            (
                key,
                gpu_type,
                region,
                market,
                alpha,
                beta_val,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        conn.commit()

    def get_failure_summary(
        self,
        gpu_type: str,
        region: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return availability posterior + recent failure context."""
        conn = self._conn()
        now = datetime.utcnow()

        # 1. Beta posterior (apply decay at read time)
        query = (
            "SELECT alpha, beta, last_decay FROM availability_priors WHERE gpu_type = ?"
        )
        params: list = [gpu_type]
        if region:
            query += " AND region = ?"
            params.append(region)
        if market:
            query += " AND market = ?"
            params.append(market)

        rows = conn.execute(query, params).fetchall()
        # Aggregate across matching priors (sum α,β if multiple regions/markets)
        total_alpha, total_beta = 1.0, 1.0  # base prior
        for r in rows:
            last_decay = datetime.fromisoformat(r["last_decay"])
            hours = (now - last_decay).total_seconds() / 3600
            decay = self._DECAY_RATE**hours
            total_alpha += (
                r["alpha"] - 1.0
            ) * decay  # subtract base prior, add decayed
            total_beta += (r["beta"] - 1.0) * decay

        total_alpha = max(total_alpha, 1.0)
        total_beta = max(total_beta, 1.0)
        mean = total_alpha / (total_alpha + total_beta)
        n = total_alpha + total_beta - 2  # effective observations
        variance = (total_alpha * total_beta) / (
            (total_alpha + total_beta) ** 2 * (total_alpha + total_beta + 1)
        )
        uncertainty = math.sqrt(variance)

        # 2. Recent failures (last 6h) for context
        cutoff = (now - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")
        recent_q = """
            SELECT failure_category, failure_reason, timestamp
            FROM launch_attempts
            WHERE gpu_type = ? AND timestamp > ? AND launched = 0
        """
        recent_params: list = [gpu_type, cutoff]
        if region:
            recent_q += " AND region = ?"
            recent_params.append(region)
        if market:
            recent_q += " AND market = ?"
            recent_params.append(market)
        recent_q += " ORDER BY timestamp DESC LIMIT 20"
        recent_launches = conn.execute(recent_q, recent_params).fetchall()

        # Also check outcome failures (replica deaths)
        outcome_q = """
            SELECT o.failure_category, o.diagnosis, o.timestamp
            FROM outcomes o JOIN decisions d ON o.decision_id = d.decision_id
            WHERE d.gpu_type = ? AND o.timestamp > ?
              AND o.status = 'replica_failed'
        """
        outcome_params: list = [gpu_type, cutoff]
        outcome_failures = conn.execute(outcome_q, outcome_params).fetchall()

        all_recent = [dict(r) for r in recent_launches] + [
            dict(r) for r in outcome_failures
        ]
        spot_preemptions = sum(
            1 for r in all_recent if r.get("failure_category") == "spot_preemption"
        )
        no_capacity = sum(
            1 for r in all_recent if r.get("failure_category") == "no_capacity"
        )
        last_failure = max((r.get("timestamp", "") for r in all_recent), default=None)

        return {
            "gpu_type": gpu_type,
            "region": region,
            "market": market,
            "availability_pct": round(mean * 100, 1),
            "uncertainty_pct": round(uncertainty * 100, 1),
            "effective_observations": round(n),
            "spot_preemptions_6h": spot_preemptions,
            "no_capacity_6h": no_capacity,
            "last_failure_at": last_failure,
            "alpha": round(total_alpha, 2),
            "beta": round(total_beta, 2),
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
    job_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Query Koi's memory for past decisions and outcomes."""
    lines = []

    # Past outcomes (ground truth from completed jobs)
    outcomes = memory.query_outcomes(
        model_name=model_name, status=status, job_id=job_id, limit=limit
    )
    if outcomes:
        lines.append(
            f"PAST OUTCOMES ({len(outcomes)} found — ground truth from completed jobs):"
        )
        for o in outcomes:
            slo = "SLO met" if o.get("slo_met") else "SLO missed"
            delta = (
                f"delta={o['delta_tps_pct']:+.1f}%"
                if o.get("delta_tps_pct") is not None
                else ""
            )
            if o.get("status") == "failed":
                bottleneck = (
                    f"[{o.get('bottleneck', '?')}] " if o.get("bottleneck") else ""
                )
                fail = f" FAILED: {bottleneck}{o.get('diagnosis', '?')}"
            else:
                fail = ""
            lines.append(
                f"  {o.get('model_name', '?')} | {o.get('gpu_type', '?')} TP={o.get('tp', 1)} PP={o.get('pp', 1)} | "
                f"TPS={o.get('actual_tps', '?')} (pred={o.get('predicted_tps', '?')}) {delta} | "
                f"{slo}{fail}"
            )

    # Past decisions (what Koi previously chose — even if no outcome yet)
    decisions = memory.query_decisions(
        model_name=model_name, gpu_type=instance_type, job_id=job_id, limit=limit
    )
    if decisions:
        lines.append(
            f"\nPAST DECISIONS ({len(decisions)} found — what Koi previously chose):"
        )
        for dec in decisions:
            outcome_status = dec.get("status")
            outcome_tps = dec.get("actual_tps")
            if outcome_status and outcome_tps:
                result = f"→ actual={outcome_tps:.0f} TPS ({outcome_status})"
            elif outcome_status:
                result = f"→ {outcome_status}"
            else:
                result = "→ no outcome yet (job may still be running)"
            triggered = (
                f" [{dec.get('triggered_by', 'user')}]"
                if dec.get("triggered_by") != "user"
                else ""
            )
            market = f" {dec.get('market', '')}" if dec.get("market") == "spot" else ""
            lines.append(
                f"  {dec.get('model_name', '?')} | {dec.get('gpu_type', '?')} TP={dec.get('tp', 1)} PP={dec.get('pp', 1)} DP={dec.get('dp', 1)}{market} | "
                f"predicted={dec.get('predicted_tps', '?')} TPS @ ${dec.get('predicted_cost_per_hour', '?')}/hr | "
                f"conf={dec.get('prediction_confidence', '?')} ({dec.get('prediction_source', '?')}){triggered} {result}"
            )

    if not outcomes and not decisions:
        lines.append(
            f"No memory found for model={model_name or 'any'}. This is the first time Koi has seen this model."
        )

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
        decision_id=decision_id,
        job_id=job_id,
        status=status,
        actual_tps=actual_tps,
        actual_cost_per_hour=actual_cost_per_hour,
        actual_total_cost=actual_total_cost,
        actual_runtime_hours=actual_runtime_hours,
        failure_category=failure_category,
        diagnosis=diagnosis,
        bottleneck=bottleneck,
    )
    return f"Outcome recorded: {outcome_id} (status={status})"
