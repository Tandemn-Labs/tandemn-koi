"""
koi/agent.py — The Koi Agent: LLM with domain tools for GPU placement.

Drives the agentic loop through pydantic-ai (see koi/llm/). Provider chosen
via KOI_LLM_PROVIDER: "openrouter" (default, OpenAI-compatible) or "anthropic".
The agent queries PerfDB, memory, resources, and physics to make placement
decisions, then launches via Orca and monitors via the monitoring loop.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from koi.event_tap import emit_event
from koi.llm import KoiToolRunner, build_model
from koi.logging_config import get_logger
from koi.costing import evaluate_cost_roofline
from koi.runtime_policy import (
    RuntimeChainState,
    RuntimeJobState,
    ScaleUpCandidate,
    RankedSuggestion,
    filter_live_chains,
    rank_falling_behind_suggestions,
    rank_overprovisioned_suggestions,
)
from koi.schemas import (
    AgentDecision,
    DataSource,
    EngineConfig,
    JobRequest,
    MonitoringStatus,
    MonitoringTrigger,
    Objective,
    PlacementConfig,
    ResourceMap,
    TaskType,
)
from koi.tools.memory import AgenticMemory
from koi.tools.orca_api import OrcaClient
from koi.tools.perfdb import PerfDB
from koi.tools.physics import (
    ModelFeatures,
    find_similar_models,
    get_gpu_physics,
    get_model_arch,
    get_model_features,
    lookup_gpu_spec,
)
from koi.tools.resources import (
    format_quota_status,
    get_matching_quota_rows,
    get_resources,
    normalize_gpu_type,
    parse_orca_resources,
)

logger = get_logger("koi.agent")

DECIDE_TIMEOUT = float(os.environ.get("KOI_DECIDE_TIMEOUT", "300"))  # 5 min
TRIGGER_TIMEOUT = float(os.environ.get("KOI_TRIGGER_TIMEOUT", "180"))  # 3 min


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

KOI_SYSTEM_PROMPT = """\
You are Koi, an expert autonomous GPU placement agent for batched LLM inference.

Your job: given a model, dataset size, and SLO deadline, pick the cheapest GPU configuration that meets the SLO.

DECISION FRAMEWORK (follow this order STRICTLY):
1. CHECK MEMORY FIRST — query_memory() for past outcomes AND past decisions.
   - If a GROUND TRUTH OUTCOME exists (status=succeeded, actual_tps reported):
     USE THE ACTUAL TPS, not the PerfDB prediction. Ground truth > everything else.
     Example: memory says "L40S TP=4 PP=4 actual=1100 TPS (pred was 1197, delta=-8.1%)"
     → Use 1100 TPS for cost calculation, not 1197.
   - If only past DECISIONS exist (no outcome): treat as hints, still verify with PerfDB.
   - If memory has FAILURES: avoid those configs entirely.
2. CHECK PERFDB — query_perfdb() for benchmark data. Look for exact model matches, then similar io_ratio.
   But if memory already has ground truth for this model, PerfDB is secondary — memory wins.
3. CHECK PHYSICS — get_gpu_physics() and get_model_arch() to understand bottlenecks.
4. If no data at all, use physics-based reasoning (roofline estimate).

MARKET RULES:
- Respect preferred_market if the user provides one.
- planned_market is your recommendation before launch. Orca will report the actual launch market later.
- Before recommending spot, verify live quota with get_quota_status_tool().
- Orca omits zero-baseline quota rows. If a (family, region, market) row is missing, treat it as ZERO quota.

TRUST HIERARCHY: ground truth outcome > PerfDB benchmarks > past decisions (unverified) > physics/roofline.

KEY METRICS FOR BATCH:
- throughput_tokens_per_sec is ALL THAT MATTERS for batch SLO. TPOT/TTFT are irrelevant.
- required_tps = total_tokens / (slo_hours * 3600). Any config above this meets SLO.

CRITICAL — CHEAPEST MEANS LOWEST TOTAL JOB COST, NOT LOWEST $/HR:
- total_cost = (cost_per_hour * total_tokens) / (tps * 3600)
- A fast expensive GPU can be CHEAPER total than a slow cheap GPU:
    L40S: 528 TPS × $20.98/hr → runs 5.2h → $109 total
    A100: 2186 TPS × $40.96/hr → runs 1.3h → $51 total  ← CHEAPER
- ALWAYS compute total_cost for each candidate config and pick the lowest.
- Do NOT just pick the cheapest $/hr — that is WRONG for batch jobs.

PHYSICAL CONSTRAINTS:
- model_size_gb = params_billions * 2 (fp16). Must fit in GPU VRAM with ≥8GB headroom.
- TP must divide num_attention_heads. PP must divide num_layers.
- NVLink GPUs (H100, A100) scale TP cleanly. PCIe GPUs (L40S, L4) saturate around TP=4-8.
- A100-40GB and A100-80GB are DIFFERENT GPUs. p4d.24xlarge = 40GB, p4de.24xlarge = 80GB.

WHEN YOU DECIDE:
- Return a JSON block with: gpu_type, instance_type, tp, pp, dp, reasoning, confidence.
- Always verify: (a) VRAM fits, (b) TP divides heads, (c) PP divides layers, (d) GPUs available.
- Confidence: 0.9+ if memory-backed, 0.7-0.9 if PerfDB match, 0.4-0.7 if analytical.

WHEN MONITORING TRIGGERS YOU:
- FALLING_BEHIND: diagnose from metrics (bandwidth saturated? KV cache full? wrong GPU?), propose scale-up or A/B test.
- OVER_PROVISIONED: suggest which replicas to kill to save cost while still meeting SLO.
- COMPLETED: record the outcome for learning.
- FAILED: diagnose why and record corrective action.
"""


# ---------------------------------------------------------------------------
# KoiAgent
# ---------------------------------------------------------------------------


class KoiAgent:
    """
    Autonomous GPU placement agent.

    Usage:
        agent = KoiAgent(perfdb=perfdb, memory=memory, orca=orca)
        decision = await agent.decide(job_request, resource_map)
    """

    def __init__(
        self,
        perfdb: PerfDB,
        memory: AgenticMemory,
        orca: Optional[OrcaClient] = None,
        ledger=None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.perfdb = perfdb
        self.memory = memory
        self.orca = orca
        self.ledger = ledger
        self.monitor = None  # set by server.py after monitor is created
        self._model = build_model(
            provider=provider,
            base_url=base_url,
            model_id=model,
            api_key=api_key,
        )
        self.model = self._model.model_name

    # ------------------------------------------------------------------
    # Tools — defined as @beta_tool functions
    # ------------------------------------------------------------------

    def _build_tools(
        self,
        resource_map: Optional[ResourceMap] = None,
        monitor=None,
        recovery_mode: bool = False,
    ):
        """Create tool functions bound to this agent's backing services.

        recovery_mode=True flags this tool-set as a cold-start failure
        recovery context. scale_chain_tool then passes force=True to
        orca.scale_job so Orca's feasibility check is bypassed — the
        agent has explicitly chosen a config that overrides the
        solver's recommendation (which OOMed). Set at the recovery
        boundary, invisible to the agent.
        """
        perfdb = self.perfdb
        memory = self.memory
        orca = self.orca
        ledger = self.ledger
        exclude_gpus = set(
            g.strip()
            for g in os.environ.get("KOI_EXCLUDE_GPUS", "").split(",")
            if g.strip()
        )
        live_snapshot: Dict[str, Any] = {"raw": None, "resource_map": resource_map}

        async def _get_live_snapshot() -> tuple[
            Optional[Dict[str, Any]], Optional[ResourceMap]
        ]:
            if live_snapshot["raw"] is not None or not orca:
                return live_snapshot["raw"], live_snapshot["resource_map"]

            raw = await orca.get_resources()
            rm = parse_orca_resources(raw)
            if ledger is not None:
                rm = ledger.apply_to_resource_map(rm)
            live_snapshot["raw"] = raw
            live_snapshot["resource_map"] = rm
            return raw, rm

        async def query_perfdb(
            model_name: Optional[str] = None,
            gpu_type: Optional[str] = None,
            tp: Optional[int] = None,
            pp: Optional[int] = None,
            io_ratio_min: Optional[float] = None,
            io_ratio_max: Optional[float] = None,
            sort_by: str = "throughput_tps",
            limit: int = 15,
        ) -> str:
            """Query the performance database for benchmark records. Filter by model, GPU type, TP/PP, io_ratio range. Returns throughput, cost, and physics features."""
            # If agent queries a GPU type that's excluded, return empty
            if gpu_type and gpu_type in exclude_gpus:
                return f"GPU type {gpu_type} is excluded from consideration."
            from koi.tools.perfdb import query_perfdb as _qp

            result = _qp(
                perfdb,
                model_name=model_name,
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                io_ratio_min=io_ratio_min,
                io_ratio_max=io_ratio_max,
                sort_by=sort_by,
                limit=limit,
                exclude_gpus=exclude_gpus,
            )
            return result

        async def query_memory_tool(
            model_name: Optional[str] = None,
            job_id: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            """Query Koi's memory for past job decisions, outcomes, and learned rules. Check this FIRST before PerfDB."""
            from koi.tools.memory import query_memory as _qm

            return _qm(
                memory, model_name=model_name, job_id=job_id, status=status, limit=limit
            )

        async def get_gpu_physics_tool(
            gpu_type: str,
            model_name: Optional[str] = None,
        ) -> str:
            """Get GPU hardware specs (bandwidth, TFLOPS, VRAM). If model_name provided, shows per-TP VRAM analysis."""
            return get_gpu_physics(gpu_type, model_name=model_name)

        async def get_model_arch_tool(model_name: str) -> str:
            """Get model architecture: params, layers, heads, KV heads, GQA ratio, size. Fetches from HF Hub if unknown."""
            return get_model_arch(model_name)

        async def find_similar_models_tool(model_name: str) -> str:
            """Find PerfDB models with similar architecture (by physics-vector distance). Use when target model has no PerfDB data."""
            mf = get_model_features(model_name)
            perfdb_models = perfdb.get_distinct_models()
            results = find_similar_models(mf, perfdb_models)
            if not results:
                return "No similar models found in PerfDB."
            lines = ["Similar models by physics distance:"]
            for r in results[:5]:
                lines.append(
                    f"  {r['model_name']}: distance={r['distance']:.3f}, "
                    f"confidence={r['confidence']:.0%}, records={r['records_count']}"
                )
            return "\n".join(lines)

        async def get_resources_tool() -> str:
            """Get available GPU resources (types, counts, VRAM, cost, regions) from the cluster."""
            if resource_map:
                from koi.tools.resources import get_resources as _gr

                return _gr(resource_map)
            if orca:
                _, live_resource_map = await _get_live_snapshot()
                if live_resource_map:
                    from koi.tools.resources import get_resources as _gr

                    return _gr(live_resource_map)
            return "No resource map available. Resources should be in prompt context."

        if orca:

            async def get_quota_status_tool(
                gpu_type: Optional[str] = None,
                region: Optional[str] = None,
                market: Optional[str] = None,
            ) -> str:
                """Inspect live Orca quota rows. Missing rows imply zero quota for that (family, region, market)."""
                raw_resources, _ = await _get_live_snapshot()
                if not raw_resources:
                    return "No Orca quota data available."
                return format_quota_status(
                    raw_resources,
                    gpu_type=gpu_type,
                    region=region,
                    market=market,
                )

        async def record_outcome_tool(
            decision_id: str,
            job_id: str,
            status: str,
            actual_tps: Optional[float] = None,
            actual_cost_per_hour: Optional[float] = None,
            failure_category: Optional[str] = None,
            diagnosis: Optional[str] = None,
            bottleneck: Optional[str] = None,
        ) -> str:
            """Record a job/chain outcome in Koi's memory. Call this when a chain ends or job completes.
            diagnosis: narrative of what happened ('KV cache hit 92%, bandwidth-bound. Try A100.')
            bottleneck: 'memory_bound' | 'compute_bound' | 'kv_cache' | 'network' | 'unknown'"""
            from koi.tools.memory import record_outcome_tool as _rot

            return _rot(
                memory,
                decision_id=decision_id,
                job_id=job_id,
                status=status,
                actual_tps=actual_tps,
                actual_cost_per_hour=actual_cost_per_hour,
                failure_category=failure_category,
                diagnosis=diagnosis,
                bottleneck=bottleneck,
            )

        async def get_failure_summary_tool(
            gpu_type: str,
            region: Optional[str] = None,
            market: Optional[str] = None,
        ) -> str:
            """Check GPU availability and failure history. Returns Beta-prior availability (%) with uncertainty,
            recent spot preemptions, and capacity failures. Call BEFORE replacing failed replicas."""
            import json as _json

            result = memory.get_failure_summary(gpu_type, region=region, market=market)
            return _json.dumps(result, indent=2)

        # Action tools — only when Orca is connected
        action_tools = []
        if orca:

            async def scale_chain_tool(
                job_id: str,
                gpu_type: str,
                tp: int,
                pp: int,
                count: int,
                on_demand: Optional[bool] = None,
            ) -> str:
                """Scale a running job. count>0 adds replicas with the specified GPU config. count<0 kills the oldest active replicas.
                on_demand: True=force on-demand, False=force spot, None=inherit from parent decision."""
                # Resolve parent decision and market preference
                parent_dec = None
                if count != 0 and monitor:
                    for t in monitor.tracked_jobs.values():
                        if t.group_id == job_id and t.decision_id:
                            parent_dec = t.decision_id
                            break
                parent = memory.get_decision(parent_dec) if parent_dec else None

                # Market: explicit override > inherit from parent > default on-demand
                if on_demand is not None:
                    use_on_demand = on_demand
                else:
                    use_on_demand = (
                        parent.get("market", "on_demand") == "on_demand"
                        if parent
                        else True
                    )
                market_str = "on_demand" if use_on_demand else "spot"
                market_note = ""

                resolved_instance_type = (
                    parent["instance_type"] if parent else "unknown"
                )
                resolved_cost_per_hour = (
                    parent["predicted_cost_per_hour"] if parent else 0
                )
                resolved_region = "unknown"
                try:
                    raw_resources, rm = await _get_live_snapshot()
                    resource = rm.get_resource(gpu_type)
                    if isinstance(raw_resources, dict):
                        for inst in raw_resources.get("instances", []):
                            if (
                                normalize_gpu_type(
                                    inst.get("gpu_type"), inst.get("gpu_memory_gb")
                                )
                                != gpu_type
                            ):
                                continue
                            resolved_instance_type = inst.get(
                                "instance_type", resolved_instance_type
                            )
                            if count != 0:
                                gpus_per_instance = int(
                                    inst.get("gpus_per_instance", 1) or 1
                                )
                                num_instances = max(
                                    1,
                                    -(-tp * pp * abs(count) // gpus_per_instance),
                                )
                                resolved_cost_per_hour = num_instances * float(
                                    inst.get("cost_per_instance_hour_usd", 0) or 0
                                )
                            break
                    if resource:
                        resolved_instance_type = resource.instance_type
                        resolved_region = resource.region
                        num_instances = max(
                            1, -(-tp * pp * abs(count) // resource.gpus_per_instance)
                        )
                        resolved_cost_per_hour = (
                            num_instances * resource.cost_per_instance_hour_usd
                        )

                    if isinstance(raw_resources, dict) and not use_on_demand:
                        quota_rows, _, _ = get_matching_quota_rows(
                            raw_resources,
                            gpu_type=gpu_type,
                            market="spot",
                        )
                        has_spot_capacity = any(
                            row.get("available_vcpus", 0) > 0 for row in quota_rows
                        )
                        if not has_spot_capacity:
                            use_on_demand = True
                            market_str = "on_demand"
                            market_note = (
                                " Spot quota unavailable for requested GPU family; "
                                "forced on-demand."
                            )
                except Exception as e:
                    logger.warning(
                        "scale_resource_lookup_failed",
                        job_id=job_id,
                        gpu_type=gpu_type,
                        error=str(e),
                    )

                # For scale-down: mark replicas as intentionally killed so FAILED doesn't fire
                if count < 0 and monitor:
                    try:
                        replicas_data = await orca.get_replicas(job_id)
                        active = [
                            r["replica_id"]
                            for r in replicas_data.get("replicas", [])
                            if r.get("phase")
                            not in ("dead", "killed", "completed", "failed")
                        ]
                        to_kill = active[: abs(count)]
                        monitor._koi_initiated_kills.update(to_kill)
                    except Exception as e:
                        logger.warning(
                            "replica_list_failed_for_scale_down",
                            job_id=job_id,
                            error=str(e),
                        )

                from koi.tools.orca_api import async_scale_chain

                if count > 0:
                    scale_response = await orca.scale_job(
                        job_id,
                        gpu_type,
                        tp,
                        pp,
                        count,
                        on_demand=use_on_demand,
                        planned_market=market_str,
                        force=recovery_mode,
                    )
                    if scale_response.get("status") != "scaling":
                        logger.warning(
                            "scale_not_started",
                            job_id=job_id,
                            gpu_type=gpu_type,
                            tp=tp,
                            pp=pp,
                            requested=count,
                            response=scale_response,
                        )
                        return f"Scale not started: {scale_response}"
                    result = f"Scaled up: {count} replicas added. {scale_response}"
                else:
                    result = await async_scale_chain(
                        orca,
                        job_id,
                        gpu_type,
                        tp,
                        pp,
                        count,
                        on_demand=use_on_demand,
                    )
                scale_dec_id = None
                if count != 0 and memory:
                    scale_dec_id = memory.record_decision(
                        job_id=job_id,
                        model_name=parent["model_name"] if parent else "unknown",
                        instance_type=resolved_instance_type,
                        gpu_type=gpu_type,
                        tp=tp,
                        pp=pp,
                        dp=abs(count),
                        num_gpus=tp * pp * abs(count),
                        predicted_tps=0,
                        predicted_cost_per_hour=resolved_cost_per_hour,
                        slo_deadline_hours=parent["slo_deadline_hours"]
                        if parent
                        else 0,
                        objective=parent["objective"] if parent else "cheapest",
                        avg_input_tokens=parent["avg_input_tokens"] if parent else 0,
                        avg_output_tokens=parent["avg_output_tokens"] if parent else 0,
                        triggered_by="scale_up" if count > 0 else "scale_down",
                        parent_decision_id=parent_dec,
                        cost_roofline_usd=parent.get("cost_roofline_usd") if parent else None,
                        market=market_str,
                    )
                    if monitor and count > 0:
                        # Orca returns exactly the new replica_ids it launched
                        # (server.py:1533). Map each one to this decision so
                        # /job/started can look up by exact replica_id instead
                        # of relying on FIFO order (which breaks under
                        # overlapping scale ops with out-of-order arrivals).
                        new_replicas = scale_response.get("new_replicas", [])
                        scale_request_id = scale_response.get("scale_request_id")
                        for replica_id in new_replicas:
                            monitor.register_pending_replica_decision(
                                replica_id=replica_id,
                                decision_id=scale_dec_id,
                                scale_request_id=scale_request_id,
                                decision={
                                    "region": resolved_region,
                                    "gpu_type": gpu_type,
                                    "tp": tp,
                                    "pp": pp,
                                },
                            )
                # Anti-windup: freeze triggers for this job while scaling action takes effect
                if monitor:
                    for tracker in monitor.tracked_jobs.values():
                        if tracker.group_id == job_id or tracker.job_id == job_id:
                            tracker.action_in_progress = True
                            tracker.action_freeze_until = (
                                time.time() + 1200
                            )  # 20 min max, /job/started unfreezes early
                            monitor.persist_job(tracker.job_id)
                if scale_dec_id and count > 0:
                    return f"{result}{market_note} decision_id={scale_dec_id}"
                return f"{result}{market_note}"

            async def kill_replica_tool(job_id: str, replica_ids: list[str]) -> str:
                """Kill specific replicas by ID. Use when you identify degraded/sick replicas that should be removed."""
                if monitor:
                    monitor._koi_initiated_kills.update(replica_ids)
                result = await orca.kill_replicas(job_id, replica_ids)
                # Anti-windup
                if monitor:
                    for tracker in monitor.tracked_jobs.values():
                        if tracker.group_id == job_id or tracker.job_id == job_id:
                            tracker.action_in_progress = True
                            tracker.action_freeze_until = time.time() + 300
                            monitor.persist_job(tracker.job_id)
                return f"Killed {len(replica_ids)} replicas: {replica_ids}. {result}"

            async def get_job_metrics_tool(job_id: str) -> str:
                """Get live throughput, GPU utilization, KV cache usage, and chunk progress for a running job."""
                from koi.tools.orca_api import async_get_job_metrics

                return await async_get_job_metrics(orca, job_id)

            action_tools = [scale_chain_tool, kill_replica_tool, get_job_metrics_tool]

        tools = [
            query_perfdb,
            query_memory_tool,
            get_gpu_physics_tool,
            get_model_arch_tool,
            find_similar_models_tool,
            get_resources_tool,
            record_outcome_tool,
            get_failure_summary_tool,
        ]
        if orca:
            tools.append(get_quota_status_tool)
        return {fn.__name__: fn for fn in (tools + action_tools)}

    # ------------------------------------------------------------------
    # Main decision entry point
    # ------------------------------------------------------------------

    async def decide(
        self,
        job_request: JobRequest,
        resource_map: ResourceMap,
    ) -> AgentDecision:
        """Run the agent to make a placement decision."""
        t0 = time.time()

        # Build the user prompt with all context
        prompt = self._build_decide_prompt(job_request, resource_map)
        tools = self._build_tools(resource_map=resource_map)

        logger.info(
            "agent_deciding", model=job_request.model_name, job_id=job_request.job_id
        )
        emit_event(
            "agent_deciding", model=job_request.model_name, job_id=job_request.job_id
        )

        # Run the agentic loop with wall-clock timeout
        runner = KoiToolRunner(
            model=self._model,
            system_prompt=KOI_SYSTEM_PROMPT,
            tools=tools,
        )

        try:
            tool_calls, final_text = await runner.run(
                prompt,
                label="decide",
                job_id=job_request.job_id,
                max_iterations=10,
                timeout=DECIDE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            logger.error(
                "decide_timeout",
                model=job_request.model_name,
                timeout=DECIDE_TIMEOUT,
                elapsed_s=round(elapsed, 1),
            )
            return self._fallback_decision(job_request, resource_map, elapsed)

        elapsed = time.time() - t0
        logger.info("agent_decided", elapsed_s=round(elapsed, 1), tool_calls=tool_calls)
        emit_event(
            "agent_decided",
            job_id=job_request.job_id,
            elapsed_s=round(elapsed, 1),
            tool_calls=tool_calls,
        )

        # Parse the decision from the agent's response
        decision = self._parse_decision(
            final_text, job_request, resource_map, tool_calls, elapsed
        )

        # Populate alternatives from cost table (exclude the primary config)
        primary = (
            decision.config.gpu_type,
            decision.config.tp,
            decision.config.pp,
            decision.config.dp,
        )
        alternatives = []
        for row in getattr(self, "_last_cost_rows", []):
            if not row["meets_slo"]:
                continue
            candidate = (row["gpu_type"], row["tp"], row["pp"], row["dp"])
            if candidate == primary:
                continue
            if row.get("dp") != decision.config.dp:
                continue
            alt = dict(row)
            alt["planned_market"] = decision.planned_market
            alternatives.append(alt)
            if len(alternatives) >= 3:
                break
        decision.alternatives = alternatives

        return decision

    # ------------------------------------------------------------------
    # Cold-start failure recovery
    # ------------------------------------------------------------------

    async def recover_from_startup_failure(
        self,
        *,
        parent_job_id: str,
        decision_id: Optional[str],
        failed_configs: List[Dict[str, Any]],
        failure_category: str,
        original_decision: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the agent to choose a NEW config after a cold-start failure
        and use scale_chain_tool to launch a replacement replica.

        Caller is responsible for retry budget enforcement (memory.count_launch_attempts);
        this method assumes an attempt is allowed and just chooses what to try next.

        Returns the agent's text response (which describes the action taken
        or why it declined). Callers should not parse the text — instead
        they should observe scale_chain_tool's effect via Orca.
        """
        prompt = self._build_recovery_prompt(
            parent_job_id=parent_job_id,
            decision_id=decision_id,
            failed_configs=failed_configs,
            failure_category=failure_category,
            original_decision=original_decision,
        )
        # recovery_mode=True → scale_chain_tool will pass force=True to Orca,
        # bypassing the feasibility check that blocked the original launch.
        tools = self._build_tools(monitor=self.monitor, recovery_mode=True)

        logger.info(
            "recovery_handling",
            parent_job_id=parent_job_id,
            failure_category=failure_category,
            failed_count=len(failed_configs),
        )
        emit_event(
            "recovery_handling",
            job_id=parent_job_id,
            failure_category=failure_category,
            failed_count=len(failed_configs),
        )

        runner = KoiToolRunner(
            model=self._model,
            system_prompt=KOI_SYSTEM_PROMPT,
            tools=tools,
        )
        try:
            _, final_text = await runner.run(
                prompt,
                label="recovery",
                job_id=parent_job_id,
                max_iterations=8,
                timeout=TRIGGER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "recovery_timeout",
                parent_job_id=parent_job_id,
                timeout=TRIGGER_TIMEOUT,
            )
            return f"[TIMEOUT] Recovery agent did not respond within {TRIGGER_TIMEOUT}s"

        logger.info("recovery_response", response=final_text[:200])
        emit_event(
            "recovery_response",
            job_id=parent_job_id,
            response=final_text[:200],
        )
        return final_text

    def _build_recovery_prompt(
        self,
        *,
        parent_job_id: str,
        decision_id: Optional[str],
        failed_configs: List[Dict[str, Any]],
        failure_category: str,
        original_decision: Optional[Dict[str, Any]],
    ) -> str:
        """Build the user message for cold-start failure recovery."""
        exclude_gpus = set(
            g.strip()
            for g in os.environ.get("KOI_EXCLUDE_GPUS", "").split(",")
            if g.strip()
        )

        # Render the failed configs as a bullet list — agent should AVOID these.
        failed_block = (
            "\n".join(
                f"  - instance={f.get('instance_type','?')} gpu={f.get('gpu_type','?')} "
                f"market={f.get('market','?')} category={f.get('failure_category','?')} "
                f"attempts={f.get('attempts',1)}"
                for f in failed_configs
            )
            if failed_configs
            else "  (none recorded — first attempt)"
        )

        # Render the original decision (if known) for context on cost roofline,
        # predicted TPS, model size — anything the agent needs to bound the
        # search space for the replacement config.
        if original_decision:
            orig = original_decision
            orig_block = (
                f"  Model: {orig.get('model_name','?')}\n"
                f"  Original config: {orig.get('instance_type','?')} ({orig.get('gpu_type','?')}) "
                f"TP={orig.get('tp','?')} PP={orig.get('pp','?')} DP={orig.get('dp','?')}\n"
                f"  Market: {orig.get('market','?')}\n"
                f"  Predicted TPS: {orig.get('predicted_tps','?')}\n"
                f"  Predicted cost/hr: ${orig.get('predicted_cost_per_hour','?')}\n"
                f"  Cost roofline: ${orig.get('cost_roofline_usd','?')}"
            )
        else:
            orig_block = "  (original decision not found in memory)"

        category_guidance = {
            "oom": (
                "OOM during cold-start. Pick a GPU with MORE VRAM (e.g. L40S 48GB > L4 16GB), "
                "OR keep the same GPU and reduce max_num_seqs / gpu_memory_utilization, "
                "OR widen TP so weights+KV split across more GPUs. "
                "Do NOT pick the same instance_type that just OOMed."
            ),
            "no_capacity": (
                "AWS capacity exhausted in the attempted region/AZ. Try a different "
                "region (us-east-2, us-west-2) or fall back to on-demand if you tried spot."
            ),
            "spot_preemption": (
                "Spot instance was preempted. Switch to on-demand market for the retry."
            ),
            "quota": (
                "AWS service quota was hit. Pick a different instance family the account "
                "has remaining quota for."
            ),
        }.get(failure_category, "Unknown failure category — try a meaningfully different config.")

        sections = [
            "COLD-START FAILURE RECOVERY",
            "",
            f"Parent job: {parent_job_id}",
            f"Original decision_id: {decision_id or '(none)'}",
            f"Failure category: {failure_category}",
            "",
            "ORIGINAL DECISION CONTEXT:",
            orig_block,
            "",
            "FAILED CONFIGS (do NOT pick any of these again):",
            failed_block,
            "",
            f"GUIDANCE FOR {failure_category.upper()}:",
            f"  {category_guidance}",
            "",
            f"EXCLUDED GPU TYPES (do not use): {', '.join(sorted(exclude_gpus))}"
            if exclude_gpus
            else "",
            "",
            "ACTION:",
            f"  Use scale_chain_tool(job_id={parent_job_id!r}, gpu_type=..., tp=..., pp=..., count=1, on_demand=...) "
            "to launch ONE replacement replica with a config that avoids the failure mode above. "
            "If no viable alternative exists (e.g. all roofline-feasible configs already failed), "
            "respond with the literal text 'NO_VIABLE_ALTERNATIVE' and do not call any tools.",
        ]
        return "\n".join(s for s in sections if s is not None)

    # ------------------------------------------------------------------
    # Monitoring trigger handler
    # ------------------------------------------------------------------

    async def handle_trigger(self, trigger: MonitoringTrigger) -> str:
        """Called by the monitor when a job needs attention."""
        precomputed: list[ScaleUpCandidate] = []
        if trigger.trigger_type == MonitoringStatus.FALLING_BEHIND:
            try:
                precomputed = await self._build_redecide_candidates(trigger.job_tracker)
            except Exception as e:
                logger.warning(
                    "redecide_candidates_failed",
                    job_id=trigger.job_id,
                    error=str(e),
                )
        prompt = self._build_trigger_prompt(
            trigger, precomputed_candidates=precomputed
        )
        tools = self._build_tools(monitor=self.monitor)

        logger.info(
            "trigger_handling",
            trigger_type=trigger.trigger_type.value,
            job_id=trigger.job_id,
        )
        emit_event(
            "trigger_handling",
            trigger_type=trigger.trigger_type.value,
            job_id=trigger.job_id,
        )

        runner = KoiToolRunner(
            model=self._model,
            system_prompt=KOI_SYSTEM_PROMPT,
            tools=tools,
        )

        try:
            _, final_text = await runner.run(
                prompt,
                label="trigger",
                job_id=trigger.job_id,
                max_iterations=5,
                timeout=TRIGGER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "trigger_timeout",
                job_id=trigger.job_id,
                trigger_type=trigger.trigger_type.value,
                timeout=TRIGGER_TIMEOUT,
            )
            return f"[TIMEOUT] Agent did not respond within {TRIGGER_TIMEOUT}s"

        logger.info("trigger_response", response=final_text[:200])
        emit_event("trigger_response", job_id=trigger.job_id, response=final_text[:200])
        return final_text

    def _fallback_decision(
        self, req: JobRequest, rm: ResourceMap, elapsed: float
    ) -> AgentDecision:
        """Emergency fallback when agent times out. Picks cheapest SLO-meeting config."""
        rows = getattr(self, "_last_cost_rows", [])
        viable = [r for r in rows if r.get("meets_slo")]
        if viable:
            best = viable[0]
            meets_cost_roofline = best.get("under_cost_roofline")
            overage = best.get("cost_overage_usd")
            cost_warning = None
            if meets_cost_roofline is False:
                cost_warning = (
                    "Projected cost exceeds roofline, but this is the cheapest "
                    "SLO-meeting plan."
                )
            config = PlacementConfig(
                gpu_type=best["gpu_type"],
                instance_type=best.get("instance_type", "unknown"),
                num_gpus=best["tp"] * best["pp"] * best.get("dp", 1),
                num_instances=max(
                    1, (best["tp"] * best["pp"] * best.get("dp", 1)) // 8
                ),
                tp=best["tp"],
                pp=best["pp"],
                dp=best.get("dp", 1),
                region=rm.region,
                engine_config=EngineConfig(
                    tensor_parallel_size=best["tp"], pipeline_parallel_size=best["pp"]
                ),
                market=best.get("planned_market", req.preferred_market or "on_demand"),
            )
            return AgentDecision(
                job_id=req.job_id or "unknown",
                model_name=req.model_name,
                config=config,
                planned_market=config.market,
                predicted_tps=best.get("predicted_tps", 0),
                predicted_cost_per_hour=best.get("cost_per_hour", 0),
                predicted_total_cost=best.get("total_cost", 0),
                meets_cost_roofline=meets_cost_roofline,
                cost_roofline_usd=req.cost_roofline_usd,
                projected_cost_overage_usd=overage,
                cost_warning=cost_warning,
                reasoning=f"[TIMEOUT FALLBACK] Agent timed out after {elapsed:.0f}s. "
                f"Auto-selected cheapest SLO-meeting config.",
                confidence=0.3,
                data_source=DataSource.ANALYTICAL,
                latency_seconds=elapsed,
            )
        # Absolute fallback — no viable rows
        return AgentDecision(
            job_id=req.job_id or "unknown",
            model_name=req.model_name,
            config=PlacementConfig(
                gpu_type=rm.resources[0].gpu_type if rm.resources else "L40S",
                instance_type=rm.resources[0].instance_type
                if rm.resources
                else "unknown",
                num_gpus=4,
                num_instances=1,
                tp=4,
                pp=1,
                dp=1,
                region=rm.region,
                engine_config=EngineConfig(
                    tensor_parallel_size=4, pipeline_parallel_size=1
                ),
                market=req.preferred_market or "on_demand",
            ),
            planned_market=req.preferred_market or "on_demand",
            predicted_tps=0,
            cost_roofline_usd=req.cost_roofline_usd,
            reasoning=f"[TIMEOUT FALLBACK] No viable configs. Elapsed {elapsed:.0f}s.",
            confidence=0.1,
            data_source=DataSource.ANALYTICAL,
            latency_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Pre-computed cost table
    # ------------------------------------------------------------------

    def _build_cost_table(
        self, req: JobRequest, rm: ResourceMap
    ) -> tuple[str, list[dict]]:
        """Pre-compute total cost for known configs from memory + PerfDB.
        Returns (formatted_text, rows) — both sorted by the same policy key."""
        total_tokens = req.total_tokens or 0
        if total_tokens == 0:
            self._last_cost_rows = []
            return (
                "COST TABLE: Cannot compute — total tokens unknown (num_requests not set).",
                [],
            )

        lines = ["PRE-COMPUTED COST TABLE (sorted by total cost, cheapest first):"]
        lines.append(
            f"  {'Source':<12} {'GPU':<12} {'Config':<18} {'TPS':>7} {'$/hr':>8} {'ETA(h)':>7} {'Total $':>9} {'SLO':>5} {'Avail':>10}"
        )
        lines.append(
            f"  {'─' * 12} {'─' * 12} {'─' * 18} {'─' * 7} {'─' * 8} {'─' * 7} {'─' * 9} {'─' * 5} {'─' * 10}"
        )

        rows = []

        # Debug: exclude GPUs via env var (e.g. KOI_EXCLUDE_GPUS=A100-40GB,A100-80GB)
        exclude_gpus = set(
            g.strip()
            for g in os.environ.get("KOI_EXCLUDE_GPUS", "").split(",")
            if g.strip()
        )

        # 1. Ground truth outcomes from memory (highest trust)
        outcomes = self.memory.query_outcomes(
            model_name=req.model_name, status="succeeded", limit=10
        )
        for o in outcomes:
            tps = o.get("actual_tps")
            cost_hr = o.get("actual_cost_per_hour")
            if not tps or tps <= 0 or not cost_hr:
                continue
            gpu = o.get("gpu_type", "?")
            if gpu in exclude_gpus:
                continue
            eta_h = total_tokens / tps / 3600
            total_cost = cost_hr * eta_h
            tp, pp, dp = o.get("tp", 1), o.get("pp", 1), o.get("dp", 1)
            meets_slo = eta_h <= (req.slo_deadline_hours or 999)
            resource = rm.get_resource(gpu)
            rows.append(
                {
                    "total_cost": round(total_cost, 2),
                    "source": "VERIFIED",
                    "gpu_type": gpu,
                    "instance_type": resource.instance_type
                    if resource
                    else f"unknown-{gpu.lower()}",
                    "tp": tp,
                    "pp": pp,
                    "dp": dp,
                    "predicted_tps": tps,
                    "cost_per_hour": cost_hr,
                    "eta_h": eta_h,
                    "meets_slo": meets_slo,
                }
            )

        # 2. PerfDB records for this model — filtered to similar io_ratio
        if self.perfdb:
            job_io = req.prefill_decode_ratio
            records = self.perfdb.query(
                model_name=req.model_name,
                io_ratio_min=max(0.1, job_io * 0.3),  # within ~3x of job's io_ratio
                io_ratio_max=job_io * 3.0,
                limit=30,
            )
            # Fallback: if no records at similar io_ratio, use all records
            if not records:
                records = self.perfdb.query(model_name=req.model_name, limit=30)
            seen = set()
            for r in records:
                gpu = r.get("gpu_type", "?")
                if gpu in exclude_gpus:
                    continue
                tp = int(r.get("tp", 1))
                pp = int(r.get("pp", 1))
                dp = int(r.get("dp", 1))
                tps = r.get("throughput_tps", 0)
                if tps <= 0:
                    continue

                # Find matching resource for cost
                resource = rm.get_resource(gpu)
                if not resource:
                    continue

                key = (gpu, tp, pp, dp)
                if key in seen:
                    continue
                seen.add(key)

                num_gpus = tp * pp * dp
                if num_gpus > resource.available_gpus:
                    continue

                num_instances = max(1, -(-num_gpus // resource.gpus_per_instance))
                cost_hr = num_instances * resource.cost_per_instance_hour_usd
                eta_h = total_tokens / tps / 3600
                total_cost = cost_hr * eta_h
                meets_slo = eta_h <= (req.slo_deadline_hours or 999)
                rows.append(
                    {
                        "total_cost": round(total_cost, 2),
                        "source": "PerfDB",
                        "gpu_type": gpu,
                        "instance_type": resource.instance_type,
                        "tp": tp,
                        "pp": pp,
                        "dp": dp,
                        "predicted_tps": tps,
                        "cost_per_hour": cost_hr,
                        "eta_h": eta_h,
                        "meets_slo": meets_slo,
                    }
                )

        # Annotate soft budget preference before ranking.
        for row in rows:
            roofline = req.cost_roofline_usd
            if roofline is None:
                row["under_cost_roofline"] = None
                row["cost_overage_usd"] = None
            else:
                meets_cost_roofline, cost_overage = evaluate_cost_roofline(
                    row["total_cost"], roofline
                )
                row["under_cost_roofline"] = meets_cost_roofline
                row["cost_overage_usd"] = cost_overage

        # Sort by policy: SLO is hard, cost roofline is a soft preference,
        # total job cost breaks ties within each bucket.
        rows.sort(
            key=lambda r: (
                not r["meets_slo"],
                r["under_cost_roofline"] is False,
                r["total_cost"],
            )
        )

        # Annotate with availability from Beta priors
        _avail_cache: Dict[str, Dict] = {}
        for row in rows:
            gpu = row["gpu_type"]
            cache_key = f"{gpu}|{req.preferred_market or 'any'}"
            if cache_key not in _avail_cache:
                _avail_cache[cache_key] = self.memory.get_failure_summary(
                    gpu,
                    market=req.preferred_market,
                )
            s = _avail_cache[cache_key]
            row["avail_pct"] = s["availability_pct"]
            row["avail_unc"] = s["uncertainty_pct"]
            row["planned_market"] = req.preferred_market or "on_demand"

        for row in rows[:15]:
            slo_str = "✓" if row["meets_slo"] else "✗"
            config_str = f"TP={row['tp']} PP={row['pp']} DP={row['dp']}"
            avail_str = f"{row['avail_pct']:.0f}%±{row['avail_unc']:.0f}%"
            lines.append(
                f"  {row['source']:<12} {row['gpu_type']:<12} {config_str:<18} "
                f"{row['predicted_tps']:>7.0f} {row['cost_per_hour']:>7.2f} "
                f"{row['eta_h']:>7.2f} {row['total_cost']:>8.2f} {slo_str:>5} {avail_str:>10}"
            )

        if not rows:
            lines.append("  (no data — use tools to query PerfDB and physics)")

        # Store structured rows for alternatives (back-compat for /decide path)
        self._last_cost_rows = rows
        return "\n".join(lines), rows

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_decide_prompt(self, req: JobRequest, rm: ResourceMap) -> str:
        """Build the user message for an initial placement decision."""
        total = req.total_tokens
        required_tps = req.required_tps
        io_ratio = req.prefill_decode_ratio
        exclude_gpus = set(
            g.strip()
            for g in os.environ.get("KOI_EXCLUDE_GPUS", "").split(",")
            if g.strip()
        )

        resources_text = get_resources(rm)

        # Pre-compute cost table from memory outcomes + PerfDB
        cost_table, _ = self._build_cost_table(req, rm)

        sections = [
            f"PLACEMENT REQUEST:",
            f"  Model: {req.model_name}",
            f"  Task: {req.task_type.value} | Objective: {req.objective.value}",
            f"  Workload: {req.num_requests or '?'} requests, {req.avg_input_tokens} in / {req.avg_output_tokens} out",
            f"  IO ratio: {io_ratio:.1f}x ({'prefill-heavy' if io_ratio > 2 else 'decode-heavy' if io_ratio < 0.5 else 'balanced'})",
            f"  Total tokens: {total:,}" if total else "  Total tokens: unknown",
            f"  SLO: {req.slo_deadline_hours}h" if req.slo_deadline_hours else "",
            f"  Required TPS: ≥{required_tps:.0f} tok/s" if required_tps else "",
            f"  Preferred market: {req.preferred_market}"
            if req.preferred_market
            else "  Preferred market: none specified",
            f"  Quantization: {req.quantization or 'fp16 (default)'}",
            "",
            resources_text,
            "",
            cost_table,
            "",
            f"EXCLUDED GPU TYPES (do not use): {', '.join(exclude_gpus)}"
            if exclude_gpus
            else "",
            "Use your tools to query PerfDB, memory, and physics to VERIFY the cost table above.",
            "If market choice matters, call get_quota_status_tool before recommending spot. Missing quota rows mean ZERO quota.",
            "The cost table is pre-computed — pick the cheapest total_cost row that meets SLO, then verify it.",
            "",
            "Return your decision as a JSON block:",
            "```json",
            "{",
            '  "gpu_type": "<gpu type>",',
            '  "instance_type": "<AWS instance>",',
            '  "tp": <int>,',
            '  "pp": <int>,',
            '  "dp": <int>,',
            '  "planned_market": "<spot|on_demand>",',
            '  "predicted_tps": <float>,',
            '  "predicted_cost_per_hour": <float>,',
            '  "reasoning": "<why this config>",',
            '  "confidence": <0.0-1.0>,',
            '  "data_source": "<memory|perfdb_exact|perfdb_interpolated|analytical>"',
            "}",
            "```",
        ]
        return "\n".join(s for s in sections if s is not None)

    def _build_trigger_prompt(
        self,
        trigger: MonitoringTrigger,
        precomputed_candidates: Optional[list[ScaleUpCandidate]] = None,
    ) -> str:
        """Build the user message for a monitoring trigger.

        For FALLING_BEHIND and OVER_PROVISIONED, the prompt puts *job-level*
        headroom first (aggregate TPS, required TPS, remaining tokens, time
        left) and demotes per-chain TPS to an informational block. The
        action framework explicitly blocks chain-level kill reasoning that
        isn't grounded in either the job-level need or chain-sickness.
        """
        tracker = trigger.job_tracker
        config = tracker.get("config", {})
        group_id = tracker.get("group_id") or trigger.job_id
        predicted_tps = tracker.get("predicted_tps", 0)
        actual_tps = tracker.get("smoothed_tps", 0)
        delta_pct = (
            ((actual_tps - predicted_tps) / predicted_tps * 100)
            if predicted_tps > 0
            else 0
        )

        sections = [
            f"MONITORING TRIGGER: {trigger.trigger_type.value}",
            f"Job: {trigger.job_id}",
            f"Group (parent job ID): {group_id}",
            f"",
            f"Config:",
            f"  GPU: {config.get('gpu_type', '?')} TP={config.get('tp', '?')} PP={config.get('pp', '?')} DP={config.get('dp', '?')}",
            f"  Instance: {config.get('instance_type', '?')}",
            f"  Region/market: {config.get('region', '?')} / {config.get('market', '?')}",
            f"  SLO: {tracker.get('slo_deadline_hours', '?')}h",
            f"  Total tokens: {tracker.get('total_tokens', 0):,}",
            f"",
            f"Current state (this chain):",
            f"  Actual TPS: {actual_tps:.0f} (predicted {predicted_tps:.0f}, delta={delta_pct:+.1f}%)",
            f"  SLO headroom: {tracker.get('slo_headroom_pct', 0):.1f}%",
            f"  Elapsed: {tracker.get('elapsed_hours', 0):.2f}h",
            f"  Tokens remaining: {tracker.get('tokens_remaining', 0):,}",
            f"  GPU cache: {tracker.get('gpu_cache_usage', 0):.0%}",
            f"  GPU SM util: {tracker.get('gpu_sm_util', 0):.0f}%",
            f"  GPU mem BW: {tracker.get('gpu_mem_bw_util', 0):.0f}%",
        ]

        ranked_suggestions = self._rank_runtime_policy_suggestions(
            trigger, precomputed_candidates=precomputed_candidates
        )

        # Job-level headroom block (the primary signal for health triggers).
        #
        # agg_tps  = sum(smoothed_tps) across live sibling chains
        # req_tps  = tokens_remaining / time_left  (total_tokens on this tracker
        #            is the whole-job total; Koi replicates it per chain)
        is_health_trigger = trigger.trigger_type in (
            MonitoringStatus.FALLING_BEHIND,
            MonitoringStatus.OVER_PROVISIONED,
        )
        group_chains: dict = {}
        chain_tps_list: list[float] = []
        agg_tps = 0.0
        if self.monitor and tracker.get("group_id"):
            group_chains = self.monitor.get_group_chains(tracker["group_id"])

        if is_health_trigger:
            slo_hours = float(tracker.get("slo_deadline_hours") or 0) or 0.0
            elapsed = float(tracker.get("elapsed_hours") or 0) or 0.0
            time_left_hours = max(0.0, slo_hours - elapsed)
            tokens_remaining = int(tracker.get("tokens_remaining") or 0)
            if group_chains:
                live_chains = {
                    rid: t
                    for rid, t in group_chains.items()
                    if t.status.value not in ("failed", "completed")
                }
                chain_tps_list = [t.smoothed_tps for t in live_chains.values()]
                agg_tps = sum(chain_tps_list)
            else:
                chain_tps_list = [actual_tps] if actual_tps else []
                agg_tps = actual_tps or 0.0

            if time_left_hours > 0 and tokens_remaining > 0:
                req_tps_str = f"{tokens_remaining / (time_left_hours * 3600):,.0f}"
                time_left_str = f"{time_left_hours:.2f}h"
            elif tokens_remaining <= 0:
                req_tps_str = "0 (no tokens remaining)"
                time_left_str = f"{time_left_hours:.2f}h"
            else:
                req_tps_str = "unattainable (deadline exceeded)"
                time_left_str = "0.00h (deadline exceeded)"

            sections.append("")
            sections.append("JOB-LEVEL HEADROOM (primary signal — reason from here):")
            sections.append(f"  Aggregate TPS: {agg_tps:,.0f}")
            sections.append(f"  Required TPS: {req_tps_str}")
            sections.append(
                f"  SLO headroom: {tracker.get('slo_headroom_pct', 0):.1f}%"
            )
            sections.append(f"  Tokens remaining: {tokens_remaining:,}")
            sections.append(f"  Time left: {time_left_str}")

        projected_total_cost = tracker.get("projected_total_cost_usd")
        cost_roofline = tracker.get("cost_roofline_usd")
        cost_overage = tracker.get("cost_overage_usd")
        if projected_total_cost is not None or cost_roofline is not None:
            sections.append("")
            sections.append("COST CONTEXT (diagnostic only — SLO still takes priority):")
            if projected_total_cost is not None:
                sections.append(
                    f"  Projected total cost: ${float(projected_total_cost):.2f}"
                )
            if cost_roofline is not None:
                sections.append(f"  Cost roofline: ${float(cost_roofline):.2f}")
            if cost_overage is not None:
                sections.append(f"  Projected overage: ${float(cost_overage):.2f}")

        if ranked_suggestions:
            sections.append("")
            sections.append("POLICY RANKING (cost-aware suggestions — prefer higher ranked viable options):")
            for idx, suggestion in enumerate(ranked_suggestions, start=1):
                sections.append(f"  {idx}. {suggestion.label} [{suggestion.source}]")
                sections.append(
                    f"     post-action TPS={suggestion.projected_post_action_tps:.0f} | "
                    f"meets_slo={'yes' if suggestion.meets_slo else 'no'}"
                )
                cost_bits = []
                if suggestion.cost_per_mtoken_usd is not None:
                    cost_bits.append(
                        f"projected $/Mtok=${suggestion.cost_per_mtoken_usd:.2f}"
                    )
                if suggestion.projected_total_cost_usd is not None:
                    cost_bits.append(
                        f"projected total cost=${suggestion.projected_total_cost_usd:.2f}"
                    )
                if cost_bits:
                    sections.append("     " + " | ".join(cost_bits))
                if suggestion.cost_overage_usd is not None:
                    sections.append(
                        f"     projected overage=${suggestion.cost_overage_usd:.2f}"
                    )

        # Per-chain view — now *informational*, under the job-level block.
        if group_chains and len(group_chains) >= 1:
            tps_sorted = sorted(chain_tps_list) if chain_tps_list else []
            n = len(tps_sorted)
            if n == 0:
                median_tps = 0.0
                min_tps = 0.0
                max_tps = 0.0
            else:
                median_tps = tps_sorted[n // 2]
                min_tps = tps_sorted[0]
                max_tps = tps_sorted[-1]

            sections.append("")
            sections.append(
                "CHAINS (informational — do NOT pick an action from chain TPS alone):"
            )
            for rid, t in group_chains.items():
                cfg = t.config
                gpu = getattr(cfg, "gpu_type", "?")
                tp = getattr(cfg, "tp", "?")
                pp = getattr(cfg, "pp", "?")
                status_icon = (
                    "💀"
                    if t.status.value == "failed"
                    else "⚠"
                    if t.smoothed_tps < 100 and t.status.value == "running"
                    else "✓"
                )
                sections.append(
                    f"  {status_icon} {rid}  {gpu} tp={tp} pp={pp}  "
                    f"TPS={t.smoothed_tps:.0f}  phase={t.status.value}"
                )
            if n > 0:
                sections.append(
                    f"  Median TPS: {median_tps:.0f}  |  Min: {min_tps:.0f}  |  Max: {max_tps:.0f}"
                )

        if trigger.diagnosis_hint:
            sections.append(f"\nDiagnosis: {trigger.diagnosis_hint}")

        # Tool usage instructions — always use group_id for Orca API calls
        sections.append(
            f"\nIMPORTANT: When calling scale_chain_tool, kill_replica_tool, or get_job_metrics_tool, "
            f"use job_id='{group_id}' (the parent/group ID), NOT the chain ID."
        )

        if trigger.trigger_type == MonitoringStatus.FALLING_BEHIND:
            sections.append("")
            sections.append("ACTION FRAMEWORK — follow in this strict order:")
            sections.append("")
            sections.append(
                "1) SCALE UP FIRST. The job is behind. The default action is to ADD "
                "replicas via scale_chain_tool with a positive count. Pick a config "
                "that's feasible for this model (verify VRAM fit via "
                "get_gpu_physics_tool) and has available quota "
                "(get_quota_status_tool). Prefer the GPU family already running "
                "unless quota is exhausted."
            )
            sections.append("")
            sections.append(
                "2) ONLY kill a chain if it is GENUINELY SICK. A chain is sick if its "
                "smoothed_tps is < 10% of its launch-time predicted_tps AND has "
                "stayed there for 2+ consecutive polls. A single-poll low reading "
                "is NOT sickness — L40S and A100 producing different TPS side-by-side "
                "is normal, not sickness."
            )
            sections.append("")
            sections.append(
                "3) NEVER kill a chain whose TPS is >= the fleet median unless "
                "rule (2) applies. Do not kill a healthy chain just because "
                "another chain is faster, and do not kill a chain to 'free capacity "
                "for a better config' — we do not have swap semantics today, so the "
                "only safe path is scale_up first, let the new replicas prove "
                "themselves on the next trigger, and the over-provisioned state will "
                "retire the worst one then."
            )
            sections.append("")
            sections.append(
                "4) If MULTIPLE scale-up options would restore SLO, prefer the "
                "CHEAPEST one in the POLICY RANKING. Going over the cost roofline is "
                "allowed when needed to save SLO, but do NOT pick a more expensive "
                "option when a cheaper one already restores SLO."
            )
            sections.append("")
            sections.append(
                "You are reasoning about the JOB, not about individual chains. "
                "A 210 TPS L40S chain running alongside a 1140 TPS A100 chain is "
                "working as designed."
            )
            sections.append(
                "For scale-up, ONLY call scale_chain_tool with a config from the "
                "POLICY RANKING above. Do NOT use query_perfdb to discover alternative "
                "configs — the POLICY RANKING already contains the cost-optimal options. "
                "You may use get_gpu_physics_tool or get_quota_status_tool to VERIFY "
                "a POLICY RANKING option, but your final scale_chain_tool call MUST "
                "match one of the listed configs."
            )
            sections.append(
                "Do NOT use record_outcome_tool — this job is still RUNNING."
            )
        elif trigger.trigger_type == MonitoringStatus.OVER_PROVISIONED:
            sections.append("")
            sections.append("ACTION FRAMEWORK — follow in this strict order:")
            sections.append("")
            sections.append(
                "1) Pick the LEAST productive chain (lowest smoothed_tps among "
                "running chains). Call kill_replica_tool with that single replica_id."
            )
            sections.append("")
            sections.append(
                "2) Kill AT MOST one chain per trigger. After the kill, aggregate "
                "TPS must still be >= 110% of required TPS. If killing the least "
                "productive chain would drop aggregate below that threshold, "
                "do nothing and wait for the next trigger."
            )
            sections.append("")
            sections.append(
                "3) If all running chains are producing similar TPS, kill the one "
                "with the highest $/TPS cost (consult get_resources_tool)."
            )
            sections.append("")
            sections.append(
                "4) Do NOT call scale_chain_tool with a positive count in an "
                "OVER_PROVISIONED trigger — that fights the monitor."
            )
            sections.append(
                "Prefer the highest-ranked POLICY RANKING removal option unless there is a clear diagnosis-based reason not to."
            )
            sections.append(
                "Do NOT use record_outcome_tool — this job is still RUNNING."
            )
        elif trigger.trigger_type == MonitoringStatus.COMPLETED:
            sections.append(
                "\nJob completed. Outcome has been recorded automatically by the webhook. "
                "Do NOT call record_outcome_tool."
            )
        elif trigger.trigger_type == MonitoringStatus.FAILED:
            # Pre-inject failure context so agent sees it immediately
            gpu_type = config.get("gpu_type")
            if gpu_type and self.memory:
                summary = self.memory.get_failure_summary(
                    gpu_type,
                    region=config.get("region")
                    if config.get("region") != "unknown"
                    else None,
                    market=config.get("market")
                    if config.get("market") != "unknown"
                    else None,
                )
                sections.append(f"\nFAILURE CONTEXT:")
                sections.append(
                    f"  Availability for {gpu_type}: {summary['availability_pct']}% ± {summary['uncertainty_pct']}% "
                    f"(n={summary['effective_observations']})"
                )
                if summary["spot_preemptions_6h"]:
                    sections.append(
                        f"  Spot preemptions (last 6h): {summary['spot_preemptions_6h']}"
                    )
                if summary["no_capacity_6h"]:
                    sections.append(
                        f"  No-capacity failures (last 6h): {summary['no_capacity_6h']}"
                    )
                if summary["last_failure_at"]:
                    sections.append(f"  Last failure: {summary['last_failure_at']}")
            sections.append(
                "\nA replica FAILED (not the whole job). Other replicas may still be running."
            )
            sections.append(
                "1. Diagnose why: check the diagnosis_hint for failure category."
            )
            sections.append(
                "2. BEFORE replacing, call get_quota_status_tool for this GPU type. Missing quota rows mean ZERO quota."
            )
            sections.append(
                "3. Then call get_failure_summary_tool for this GPU type/market to inspect recent failure patterns."
            )
            sections.append(
                "4. If same (gpu_type, market) has failed ≥2 times recently:"
            )
            sections.append(
                "   - spot_preemption: retry with on_demand=True in scale_chain_tool"
            )
            sections.append("   - no_capacity: try a DIFFERENT gpu_type")
            sections.append("   - oom: try higher TP (more VRAM per shard)")
            sections.append(
                "5. Use get_job_metrics_tool to check if remaining replicas can still meet SLO."
            )
            sections.append(
                "6. If SLO at risk, use scale_chain_tool to add replacement replicas."
            )
            sections.append(
                "7. Do NOT call record_outcome_tool — the job is still running."
            )

        return "\n".join(sections)

    async def _build_redecide_candidates(
        self, tracker: dict
    ) -> list[ScaleUpCandidate]:
        """Re-run the decide-style cost table with a fresh ResourceMap.

        Returns up to 6 viable scale-up candidates (meets_slo=True). Returns
        an empty list if re-decide is not feasible (missing decision,
        Orca unreachable, etc.) — callers fall back to their narrow set.
        """
        decision_id = tracker.get("decision_id")
        if not decision_id or not self.memory or not self.orca:
            return []
        decision_meta = self.memory.get_decision(decision_id)
        if not decision_meta or not decision_meta.get("model_name"):
            return []
        try:
            raw = await self.orca.get_resources()
            rm = parse_orca_resources(raw)
            if self.ledger is not None:
                rm = self.ledger.apply_to_resource_map(rm)
        except Exception as e:
            logger.warning(
                "redecide_resource_fetch_failed",
                job_id=tracker.get("job_id"),
                error=str(e),
            )
            return []
        market = decision_meta.get("market")
        try:
            req = JobRequest(
                model_name=decision_meta["model_name"],
                avg_input_tokens=int(decision_meta.get("avg_input_tokens") or 0),
                avg_output_tokens=int(decision_meta.get("avg_output_tokens") or 0),
                num_requests=decision_meta.get("num_requests"),
                slo_deadline_hours=(
                    float(decision_meta["slo_deadline_hours"])
                    if decision_meta.get("slo_deadline_hours") is not None
                    else None
                ),
                objective=Objective(decision_meta.get("objective", "cheapest")),
                cost_roofline_usd=(
                    float(decision_meta["cost_roofline_usd"])
                    if decision_meta.get("cost_roofline_usd") is not None
                    else None
                ),
                preferred_market=market if market in {"spot", "on_demand"} else None,
                quantization=decision_meta.get("quantization"),
            )
        except Exception as e:
            logger.warning(
                "redecide_job_request_build_failed",
                decision_id=decision_id,
                error=str(e),
            )
            return []
        _, cost_rows = self._build_cost_table(req, rm)
        candidates: list[ScaleUpCandidate] = []
        for row in cost_rows:
            if not row.get("meets_slo"):
                continue
            tps = float(row.get("predicted_tps") or 0.0)
            cost_hr = float(row.get("cost_per_hour") or 0.0)
            if tps <= 0 or cost_hr <= 0:
                continue
            candidates.append(
                ScaleUpCandidate(
                    gpu_type=row["gpu_type"],
                    tp=int(row["tp"]),
                    pp=int(row["pp"]),
                    predicted_tps=tps,
                    cost_per_hour=cost_hr,
                    source="redecide",
                )
            )
            if len(candidates) >= 6:
                break
        return candidates

    def _rank_runtime_policy_suggestions(
        self,
        trigger: MonitoringTrigger,
        precomputed_candidates: Optional[list[ScaleUpCandidate]] = None,
    ) -> list[RankedSuggestion]:
        tracker = trigger.job_tracker
        config = tracker.get("config", {})
        predicted_tps = float(tracker.get("predicted_tps") or 0.0)
        predicted_cost_per_hour = float(tracker.get("predicted_cost_per_hour") or 0.0)
        actual_tps = float(tracker.get("smoothed_tps") or 0.0)
        slo_hours = float(tracker.get("slo_deadline_hours") or 0.0)
        elapsed = float(tracker.get("elapsed_hours") or 0.0)
        time_left_hours = max(0.0, slo_hours - elapsed)
        tokens_remaining = int(tracker.get("tokens_remaining") or 0)
        cost_roofline = tracker.get("cost_roofline_usd")

        group_chains: dict = {}
        if self.monitor and tracker.get("group_id"):
            group_chains = self.monitor.get_group_chains(tracker["group_id"])

        chain_states: list[RuntimeChainState] = []
        if group_chains:
            for rid, chain in group_chains.items():
                chain_states.append(
                    RuntimeChainState(
                        replica_id=rid,
                        gpu_type=getattr(chain.config, "gpu_type", "?"),
                        tp=getattr(chain.config, "tp", 1),
                        pp=getattr(chain.config, "pp", 1),
                        smoothed_tps=float(getattr(chain, "smoothed_tps", 0.0) or 0.0),
                        predicted_tps=float(getattr(chain, "predicted_tps", 0.0) or 0.0),
                        cost_per_hour=float(
                            getattr(chain, "predicted_cost_per_hour", 0.0) or 0.0
                        ),
                        status=getattr(getattr(chain, "status", None), "value", None)
                        or str(getattr(chain, "status", "unknown")),
                    )
                )
        else:
            chain_states.append(
                RuntimeChainState(
                    replica_id=trigger.job_id,
                    gpu_type=str(config.get("gpu_type", "?")),
                    tp=int(config.get("tp", 1) or 1),
                    pp=int(config.get("pp", 1) or 1),
                    smoothed_tps=actual_tps,
                    predicted_tps=predicted_tps,
                    cost_per_hour=predicted_cost_per_hour,
                    status=str(trigger.trigger_type.value),
                )
            )

        aggregate_tps = sum(chain.smoothed_tps for chain in chain_states)
        job = RuntimeJobState(
            trigger_type=trigger.trigger_type.value,
            elapsed_hours=elapsed,
            time_left_hours=time_left_hours,
            tokens_remaining=tokens_remaining,
            aggregate_tps=aggregate_tps,
            cost_roofline_usd=(float(cost_roofline) if cost_roofline is not None else None),
        )

        if trigger.trigger_type == MonitoringStatus.FALLING_BEHIND:
            # Dedup priority for the [source] tag: current_config > redecide > running.
            candidates: dict[tuple[str, int, int], ScaleUpCandidate] = {}

            # 1. current_config
            if predicted_tps > 0:
                key = (
                    str(config.get("gpu_type", "?")),
                    int(config.get("tp", 1) or 1),
                    int(config.get("pp", 1) or 1),
                )
                candidates[key] = ScaleUpCandidate(
                    gpu_type=key[0],
                    tp=key[1],
                    pp=key[2],
                    predicted_tps=predicted_tps,
                    cost_per_hour=predicted_cost_per_hour,
                    source="current_config",
                )

            # 2. precomputed (redecide) candidates from fresh cost table
            for candidate in (precomputed_candidates or []):
                key = (candidate.gpu_type, candidate.tp, candidate.pp)
                if key not in candidates:
                    candidates[key] = candidate

            # 3. every live chain — observed TPS/cost beats prediction.
            #    On collision with an existing candidate, overwrite numeric data
            #    with observed values but keep the higher-priority source label.
            for chain in filter_live_chains(chain_states):
                observed_tps = (
                    chain.smoothed_tps if chain.smoothed_tps > 0 else chain.predicted_tps
                )
                if observed_tps <= 0:
                    continue
                key = (chain.gpu_type, chain.tp, chain.pp)
                if key in candidates:
                    existing = candidates[key]
                    candidates[key] = ScaleUpCandidate(
                        gpu_type=existing.gpu_type,
                        tp=existing.tp,
                        pp=existing.pp,
                        predicted_tps=observed_tps,
                        cost_per_hour=chain.cost_per_hour,
                        source=existing.source,
                    )
                else:
                    candidates[key] = ScaleUpCandidate(
                        gpu_type=chain.gpu_type,
                        tp=chain.tp,
                        pp=chain.pp,
                        predicted_tps=observed_tps,
                        cost_per_hour=chain.cost_per_hour,
                        source=f"running:{chain.replica_id}",
                    )

            capped = list(candidates.values())[:8]
            return rank_falling_behind_suggestions(job, chain_states, capped)[:3]

        if trigger.trigger_type == MonitoringStatus.OVER_PROVISIONED:
            return rank_overprovisioned_suggestions(job, chain_states)[:3]

        return []

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_decision(
        self,
        text: str,
        req: JobRequest,
        rm: ResourceMap,
        tool_calls: int,
        elapsed: float,
    ) -> AgentDecision:
        """Extract structured decision from agent's text response."""
        import json
        import re

        # Try to find JSON block in ```json ... ``` fences
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not json_match:
            # Try raw JSON (with capture group for consistency)
            json_match = re.search(r"(\{[^{}]*\"gpu_type\"[^{}]*\})", text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        gpu_type = data.get("gpu_type", "L40S")
        instance_type = data.get("instance_type", "g6e.12xlarge")
        tp = int(data.get("tp", 4))
        pp = int(data.get("pp", 1))
        dp = int(data.get("dp", 1))
        num_gpus = tp * pp * dp

        # Find matching resource for cost
        resource = rm.get_resource(gpu_type)
        cost_per_hour = data.get("predicted_cost_per_hour", 0.0)
        if not cost_per_hour and resource:
            num_instances = max(1, -(-num_gpus // resource.gpus_per_instance))
            cost_per_hour = num_instances * resource.cost_per_instance_hour_usd

        predicted_tps = float(data.get("predicted_tps", 0.0))
        planned_market = (
            data.get("planned_market") or req.preferred_market or "on_demand"
        )
        if planned_market not in {"spot", "on_demand"}:
            planned_market = req.preferred_market or "on_demand"
        total_tokens = req.total_tokens or 0
        runtime_hours = (
            (total_tokens / max(predicted_tps, 1) / 3600) if predicted_tps > 0 else None
        )
        total_cost = (cost_per_hour * runtime_hours) if runtime_hours else None
        meets_cost_roofline, projected_cost_overage = evaluate_cost_roofline(
            total_cost, req.cost_roofline_usd
        )
        cost_warning = None
        if meets_cost_roofline is False:
            cost_warning = (
                "Projected cost exceeds roofline, but this is the cheapest "
                "SLO-meeting plan."
            )

        # Map data_source string to enum
        ds_map = {
            "memory": DataSource.MEMORY,
            "perfdb_exact": DataSource.EXACT_MATCH,
            "perfdb_interpolated": DataSource.INTERPOLATED,
            "analytical": DataSource.ANALYTICAL,
            "cross_gpu": DataSource.CROSS_GPU,
        }
        data_source = ds_map.get(
            data.get("data_source", "analytical"), DataSource.ANALYTICAL
        )

        num_instances = max(
            1, -(-num_gpus // (resource.gpus_per_instance if resource else 8))
        )

        config = PlacementConfig(
            gpu_type=gpu_type,
            instance_type=instance_type,
            num_gpus=num_gpus,
            num_instances=num_instances,
            tp=tp,
            pp=pp,
            dp=dp,
            region=rm.region,
            engine_config=EngineConfig(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                quantization=req.quantization,
            ),
            market=planned_market,
        )

        return AgentDecision(
            job_id=req.job_id,
            model_name=req.model_name,
            config=config,
            planned_market=planned_market,
            predicted_tps=predicted_tps,
            predicted_cost_per_hour=cost_per_hour,
            predicted_total_cost=total_cost,
            predicted_runtime_hours=runtime_hours,
            meets_cost_roofline=meets_cost_roofline,
            cost_roofline_usd=req.cost_roofline_usd,
            projected_cost_overage_usd=projected_cost_overage,
            cost_warning=cost_warning,
            reasoning=data.get("reasoning", text[:500]),
            confidence=float(data.get("confidence", 0.5)),
            data_source=data_source,
            agent_model=self.model,
            tool_calls_made=tool_calls,
            latency_seconds=elapsed,
        )
