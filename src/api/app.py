"""HTTP endpoints for Koi's supervised runner."""

from math import isfinite
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from src.api.schemas import JobPlacement, JobSummary
from tandemn_system_data.clients import JobStore
from tandemn_system_data.models.enums import JobStatus

app = FastAPI(title="Koi")
app.state.runner = None
app.state.shutting_down = False
app.state.user_id = None
app.state.job_store = None
app.state.postgres_pool = None


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
def list_jobs(request: Request) -> list[JobSummary]:
    """List the configured runner user's newest jobs across all lifecycle states."""
    user_id = request.app.state.user_id
    if not user_id:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job scope is not configured")
    store: JobStore | None = request.app.state.job_store
    if store is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job Store is unavailable")

    try:
        jobs = store.list_jobs(user_id)
        running_ids = {job.job_id for job in jobs if job.status == JobStatus.RUNNING}
        ranks_by_job = {}
        if running_ids:
            ranks_by_job = {
                row.job.job_id: row.ranks
                for row in store.running_jobs(user_id)
                if row.job.job_id in running_ids
            }
    except OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job Store is unavailable"
        ) from exc

    summaries = []
    for job in jobs:
        fields = dict(job.spec_json)
        for key in ("features", "job_features", "job_intent"):
            section = job.spec_json.get(key)
            if isinstance(section, dict):
                fields.update({name: value for name, value in section.items() if value is not None})

        targets: dict[str, float | None] = {}
        for key, measurements in (
            ("target_p99_ttft_ms", (fields.get("target_p99_ttft_ms"),)),
            ("target_p99_tpot_ms", (fields.get("target_p99_tpot_ms"),)),
            ("deadline_hrs", (fields.get("deadline_hrs"), fields.get("deadline_hours"))),
        ):
            targets[key] = None
            for value in measurements:
                if not isinstance(value, (int, float, str)) or isinstance(value, bool):
                    continue
                try:
                    number = float(value)
                except (ValueError, OverflowError):
                    continue
                if isfinite(number):
                    targets[key] = number
                    break

        ranks = ranks_by_job.get(job.job_id)
        places = set()
        for rank in ranks or []:
            env = rank.shape_json.get("env")
            if isinstance(env, str):
                env = env.split("|")
            if isinstance(env, (list, tuple)) and len(env) == 5:
                _, cloud, region, _, _ = env
                if isinstance(cloud, str) and cloud and isinstance(region, str) and region:
                    places.add((cloud, region))

        model_id = fields.get("model_id")
        priority = fields.get("priority_class")
        summaries.append(
            JobSummary(
                job_id=job.job_id,
                kind=job.kind,
                status=job.status,
                owner_user_id=job.user_id,
                model_id=model_id if isinstance(model_id, str) else None,
                priority_class=priority if isinstance(priority, str) else None,
                target_p99_ttft_ms=targets["target_p99_ttft_ms"],
                target_p99_tpot_ms=targets["target_p99_tpot_ms"],
                deadline_hrs=targets["deadline_hrs"],
                allocated_replicas=sum(rank.n_replicas for rank in ranks)
                if ranks is not None
                else None,
                placements=[
                    JobPlacement(cloud=cloud, region=region) for cloud, region in sorted(places)
                ],
                created_at=job.created_at,
                finished_at=job.finished_at,
                finish_reason=job.finish_reason,
            )
        )
    return summaries


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """Read one job record by its PostgreSQL primary key."""
    pool = request.app.state.postgres_pool
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable")

    try:
        with pool.connection() as conn:
            # Leave raw launch arguments and Kubernetes labels out of the dashboard response.
            job = conn.execute(
                """
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
                WHERE job_id = %s
                """,
                (job_id,),
            ).fetchone()
    except psycopg.OperationalError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Job database is unavailable"
        ) from exc

    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@app.get("/resource_map", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def get_resource_map() -> None:
    """Reserve resource-map access until its Store integration is implemented."""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
