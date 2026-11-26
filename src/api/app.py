"""HTTP endpoints for Koi's supervised runner."""

from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse
from psycopg import OperationalError
from pydantic import AwareDatetime
from sqlalchemy import text
from sqlalchemy.exc import OperationalError as StoreOperationalError
from sqlalchemy.exc import TimeoutError as StoreTimeoutError
from tandemn_system_data.clients import PostgresClient

app = FastAPI(title="Koi")
app.state.runner = None
app.state.shutting_down = False
app.state.postgres_pool = None
app.state.telemetry_client = None
app.state.telemetry_user_id = None

JOBS_SELECT = """
    SELECT job_id, kind, status, model_id, target_p99_ttft_ms,
           target_p99_tpot_ms, deadline_hrs, isl_token_avg,
           isl_token_min, isl_token_max, isl_distribution_type,
           osl_token_avg, osl_token_min, osl_token_max,
           osl_distribution_type, request_arrival_rate,
           max_concurrent_streaming, request_arrival_pattern,
           peak_to_mean_ratio, workload_prefix_concentration,
           multi_turn_ratio, shared_prefix_length_avg,
           is_session_affinity, total_token_budget, priority_class,
           scheduling_policy, router_policy
    FROM jobs
"""

RANK_PLACEMENT_COLUMNS = """
    job_id, rank_id, role, cluster_id, pool_id, node_type_id,
    cloud, region, zone, market, accelerator_type,
    tensor_parallelism, pipeline_parallelism, moe_expert_parallelism,
    moe_tensor_parallelism, min_replicas, max_replicas
"""


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    """Report runner process liveness without querying external services."""
    runner = request.app.state.runner
    healthy = (
        runner is not None and runner.returncode is None and not request.app.state.shutting_down
    )
    return JSONResponse(
        content={"status": "ok" if healthy else "unhealthy"},
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.post("/job", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def submit_job() -> None:
    """Reserve job submission until its Store integration is implemented."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@app.get("/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    """Read job records from the existing Koi PostgreSQL database."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            return conn.execute(f"{JOBS_SELECT} ORDER BY job_id LIMIT 500").fetchall()
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """Read one job record by its PostgreSQL primary key."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            job = conn.execute(f"{JOBS_SELECT} WHERE job_id = %s", (job_id,)).fetchone()
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc

    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@app.get("/jobs/{job_id}/configuration")
def get_job_configuration(
    job_id: str, request: Request, rank_id: Annotated[str, Query(min_length=1)]
) -> dict[str, Any]:
    """Read the stored configuration for one rank of an existing job."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            configuration = conn.execute(
                """
                SELECT to_jsonb(jobs) - 'extra_args' - 'misc_kubernetes_labels' AS job,
                       to_jsonb(model_architecture) AS model_architecture,
                       to_jsonb(placement) AS placement,
                       to_jsonb(job_engine_defaults) AS job_engine_defaults
                FROM placement
                JOIN jobs ON jobs.job_id = placement.job_id
                JOIN model_architecture ON model_architecture.model_id = jobs.model_id
                LEFT JOIN job_engine_defaults ON job_engine_defaults.model_id = jobs.model_id
                                             AND job_engine_defaults.engine_name = placement.engine_name
                WHERE placement.job_id = %s AND placement.rank_id = %s
                """,
                (job_id, rank_id),
            ).fetchone()
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc

    if configuration is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job rank not found")
    return configuration


@app.get("/jobs/{job_id}/metrics")
def get_job_metrics(
    job_id: str,
    request: Request,
    rank_id: Annotated[str, Query(min_length=1)],
    chain_index: Annotated[int, Query(ge=0)],
    start: AwareDatetime,
    end: AwareDatetime,
) -> list[dict[str, Any]]:
    """Read one aggregate per replica timestamp in an inclusive time window."""
    if start > end:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "start must be at or before end")

    client: PostgresClient | None = request.app.state.telemetry_client
    user_id = request.app.state.telemetry_user_id
    if client is None or not user_id:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Telemetry Store is unavailable")

    try:
        with client.session() as session:
            rows = session.execute(
                text(
                    """
                    WITH samples AS (
                        SELECT DISTINCT ON (gpu_metrics.ts, gpu_metrics.gpu_uuid) gpu_metrics.*
                        FROM gpu_metrics
                        JOIN jobs ON jobs.job_id = gpu_metrics.job_id
                        WHERE jobs.user_id = :user_id
                          AND gpu_metrics.job_id = :job_id
                          AND gpu_metrics.rank_id = :rank_id
                          AND gpu_metrics.chain_index = :chain_index
                          AND gpu_metrics.ts >= :start AND gpu_metrics.ts <= :end
                        ORDER BY gpu_metrics.ts, gpu_metrics.gpu_uuid,
                                 gpu_metrics.created_at DESC, gpu_metrics.metric_id DESC
                    ), replica_metrics AS (
                        SELECT samples.ts, samples.job_id, samples.rank_id, samples.chain_index,
                               metric.key,
                               CASE WHEN metric.key IN (
                                   'gpu_mem_used_fraction', 'vram_headroom_gb', 'sm_utilization',
                                   'mem_bandwidth_utilization', 'pcie_tput_observed',
                                   'nvlink_tput_observed'
                               ) THEN to_jsonb(avg(
                                   CASE WHEN jsonb_typeof(metric.value) = 'number'
                                        THEN CAST(metric.value AS numeric) END
                               )) ELSE (array_agg(metric.value ORDER BY
                                   samples.created_at DESC, samples.metric_id DESC
                               ) FILTER (WHERE metric.value <> 'null'::jsonb))[1]
                               END AS value
                        FROM samples
                        LEFT JOIN LATERAL jsonb_each(samples.metrics_json) AS metric ON true
                        GROUP BY samples.ts, samples.job_id, samples.rank_id,
                                 samples.chain_index, metric.key
                    )
                    SELECT ts, job_id, rank_id, chain_index,
                           coalesce(jsonb_object_agg(key, value) FILTER (WHERE key IS NOT NULL),
                                    '{}'::jsonb) AS metrics_json
                    FROM replica_metrics
                    GROUP BY ts, job_id, rank_id, chain_index
                    ORDER BY ts
                    """
                ),
                {
                    "user_id": user_id,
                    "job_id": job_id,
                    "rank_id": rank_id,
                    "chain_index": chain_index,
                    "start": start,
                    "end": end,
                },
            ).mappings()
            return [dict(row) for row in rows]
    except (StoreOperationalError, StoreTimeoutError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Telemetry Store is unavailable"
        ) from exc


# this was added to quickly get all ranks + replicas to show in the dropdowns and nav
@app.get("/jobs/{job_id}/placement")
def get_job_placement(job_id: str, request: Request) -> list[dict[str, Any]]:
    """Read the placement and parallelism for every rank of an existing job."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            ranks = conn.execute(
                f"""
                SELECT {RANK_PLACEMENT_COLUMNS}
                FROM placement
                WHERE job_id = %s
                ORDER BY rank_id
                """,
                (job_id,),
            ).fetchall()
            if (
                not ranks
                and conn.execute("SELECT 1 FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
                is None
            ):
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc

    return ranks


@app.get("/jobs/{job_id}/replicas")
def get_job_replicas(job_id: str, request: Request) -> list[dict[str, Any]]:
    """Read configured ranks and the replicas observed in the past hour."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            ranks = conn.execute(
                f"""
                SELECT {RANK_PLACEMENT_COLUMNS}, engine_name, engine_version, inference_framework
                FROM placement
                WHERE job_id = %s
                ORDER BY rank_id
                """,
                (job_id,),
            ).fetchall()
            if not ranks:
                if (
                    conn.execute("SELECT 1 FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
                    is None
                ):
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
                return []
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc

    client: PostgresClient | None = request.app.state.telemetry_client
    user_id = request.app.state.telemetry_user_id
    if client is None or not user_id:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Telemetry Store is unavailable")

    ranks_by_id = {rank["rank_id"]: {**rank, "replicas": []} for rank in ranks}
    try:
        with client.session() as session:
            replicas = session.execute(
                text(
                    """
                    SELECT DISTINCT gpu_metrics.rank_id, gpu_metrics.chain_index
                    FROM gpu_metrics
                    JOIN jobs ON jobs.job_id = gpu_metrics.job_id
                    WHERE jobs.user_id = :user_id
                      AND gpu_metrics.job_id = :job_id
                      AND gpu_metrics.rank_id = ANY(:rank_ids)
                      AND gpu_metrics.chain_index >= 0
                      AND gpu_metrics.ts BETWEEN CURRENT_TIMESTAMP - INTERVAL '1 hour'
                                             AND CURRENT_TIMESTAMP
                    ORDER BY gpu_metrics.rank_id, gpu_metrics.chain_index
                    """
                ),
                {"user_id": user_id, "job_id": job_id, "rank_ids": list(ranks_by_id)},
            ).mappings()
            for replica in replicas:
                ranks_by_id[replica["rank_id"]]["replicas"].append(
                    {"chain_index": replica["chain_index"]}
                )
    except (StoreOperationalError, StoreTimeoutError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Telemetry Store is unavailable"
        ) from exc

    return list(ranks_by_id.values())


@app.get("/jobs/{job_id}/plan/{tick}")
def get_job_plan(
    job_id: str, tick: Annotated[int, Path(ge=0, le=2_147_483_647)], request: Request
) -> dict[str, Any]:
    """Read the job's recorded plan action at the requested tick."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            decision = conn.execute(
                """
                SELECT plan_actions.job_id, plan_actions.tick, plan_actions.action,
                       plan_actions.rationale, plans.status, plans.created_at, plans.tick_rationale,
                       NULL AS placement
                FROM plan_actions
                JOIN plans ON plans.tick = plan_actions.tick
                WHERE plan_actions.tick = %s AND plan_actions.job_id = %s
                """,
                (tick, job_id),
            ).fetchone()
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc

    if decision is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job plan not found")
    return decision



@app.get("/resource_map", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_resource_map() -> None:
    """Reserve resource-map access until its Store integration is implemented."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
