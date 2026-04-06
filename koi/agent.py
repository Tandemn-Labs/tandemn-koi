"""
koi/agent.py — The Koi Agent: Claude with domain tools for GPU placement.

Uses anthropic SDK's @beta_tool + tool_runner for automatic agentic loop.
The agent queries PerfDB, memory, resources, and physics to make placement
decisions, then launches via Orca and monitors via the monitoring loop.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic, beta_tool

from koi.schemas import (
    AgentDecision, DataSource, EngineConfig, JobRequest,
    MonitoringStatus, MonitoringTrigger, PlacementConfig, ResourceMap,
)
from koi.tools.memory import AgenticMemory
from koi.tools.orca_api import OrcaClient
from koi.tools.perfdb import PerfDB
from koi.tools.physics import (
    ModelFeatures, find_similar_models, get_gpu_physics,
    get_model_arch, get_model_features, lookup_gpu_spec,
)
from koi.tools.resources import get_resources, parse_orca_resources

logger = logging.getLogger("koi.agent")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

KOI_SYSTEM_PROMPT = """\
You are Koi, an expert autonomous GPU placement agent for batched LLM inference.

Your job: given a model, dataset size, and SLO deadline, pick the cheapest GPU configuration that meets the SLO.

DECISION FRAMEWORK (follow this order):
1. CHECK MEMORY FIRST — query_memory() for past outcomes on this model. If a past run succeeded, reuse that config.
2. CHECK PERFDB — query_perfdb() for benchmark data. Look for exact model matches, then similar io_ratio.
3. CHECK PHYSICS — get_gpu_physics() and get_model_arch() to understand bottlenecks (bandwidth vs compute, VRAM fit).
4. If no data at all, use physics-based reasoning (roofline estimate).

KEY METRICS FOR BATCH:
- throughput_tokens_per_sec is ALL THAT MATTERS for batch SLO. TPOT/TTFT are irrelevant.
- required_tps = total_tokens / (slo_hours * 3600). Any config above this meets SLO.
- cost_per_m_tokens = (cost_per_hour / tps) * (1e6 / 3600). Minimize this for objective=cheapest.

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
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
    ):
        self.perfdb = perfdb
        self.memory = memory
        self.orca = orca
        self.model = model
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    # ------------------------------------------------------------------
    # Tools — defined as @beta_tool functions
    # ------------------------------------------------------------------

    def _build_tools(self, resource_map: Optional[ResourceMap] = None):
        """Create tool functions bound to this agent's backing services."""
        perfdb = self.perfdb
        memory = self.memory

        @beta_tool
        def query_perfdb(
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
            from koi.tools.perfdb import query_perfdb as _qp
            return _qp(perfdb, model_name=model_name, gpu_type=gpu_type,
                       tp=tp, pp=pp, io_ratio_min=io_ratio_min, io_ratio_max=io_ratio_max,
                       sort_by=sort_by, limit=limit)

        @beta_tool
        def query_memory_tool(
            model_name: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 10,
        ) -> str:
            """Query Koi's memory for past job decisions, outcomes, and learned rules. Check this FIRST before PerfDB."""
            from koi.tools.memory import query_memory as _qm
            return _qm(memory, model_name=model_name, status=status, limit=limit)

        @beta_tool
        def get_gpu_physics_tool(
            gpu_type: str,
            model_name: Optional[str] = None,
        ) -> str:
            """Get GPU hardware specs (bandwidth, TFLOPS, VRAM). If model_name provided, shows per-TP VRAM analysis."""
            return get_gpu_physics(gpu_type, model_name=model_name)

        @beta_tool
        def get_model_arch_tool(model_name: str) -> str:
            """Get model architecture: params, layers, heads, KV heads, GQA ratio, size. Fetches from HF Hub if unknown."""
            return get_model_arch(model_name)

        @beta_tool
        def find_similar_models_tool(model_name: str) -> str:
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

        @beta_tool
        def get_resources_tool() -> str:
            """Get available GPU resources (types, counts, VRAM, cost, regions) from the cluster."""
            if resource_map:
                from koi.tools.resources import get_resources as _gr
                return _gr(resource_map)
            return "No resource map available. Resources should be in prompt context."

        @beta_tool
        def record_outcome_tool(
            decision_id: str,
            job_id: str,
            status: str,
            actual_tps: Optional[float] = None,
            actual_cost_per_hour: Optional[float] = None,
            failure_reason: Optional[str] = None,
            failure_category: Optional[str] = None,
        ) -> str:
            """Record a job/chain outcome in Koi's memory. Call this when a chain ends or job completes."""
            from koi.tools.memory import record_outcome_tool as _rot
            return _rot(memory, decision_id=decision_id, job_id=job_id,
                       status=status, actual_tps=actual_tps,
                       actual_cost_per_hour=actual_cost_per_hour,
                       failure_reason=failure_reason, failure_category=failure_category)

        return [query_perfdb, query_memory_tool, get_gpu_physics_tool,
                get_model_arch_tool, find_similar_models_tool, get_resources_tool,
                record_outcome_tool]

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

        logger.info(f"[Koi] Agent deciding for {job_request.model_name} ({job_request.job_id})")

        # Run the agentic loop
        runner = self._client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=4096,
            system=KOI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            max_iterations=10,
        )

        tool_calls = 0
        final_text = ""
        async for event in runner:
            if hasattr(event, "type"):
                if event.type == "tool_use":
                    tool_calls += 1
                    logger.info(f"[Koi] Tool call #{tool_calls}: {event.name}")

        # Get the final response
        response = await runner.get_final_response()
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        elapsed = time.time() - t0
        logger.info(f"[Koi] Agent decided in {elapsed:.1f}s ({tool_calls} tool calls)")

        # Parse the decision from the agent's response
        decision = self._parse_decision(final_text, job_request, resource_map, tool_calls, elapsed)
        return decision

    # ------------------------------------------------------------------
    # Monitoring trigger handler
    # ------------------------------------------------------------------

    async def handle_trigger(self, trigger: MonitoringTrigger) -> str:
        """Called by the monitor when a job needs attention."""
        prompt = self._build_trigger_prompt(trigger)
        tools = self._build_tools()

        logger.info(f"[Koi] Agent handling {trigger.trigger_type.value} for {trigger.job_id}")

        runner = self._client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=2048,
            system=KOI_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            max_iterations=5,
        )

        final_text = ""
        async for event in runner:
            pass

        response = await runner.get_final_response()
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        logger.info(f"[Koi] Trigger response: {final_text[:200]}")
        return final_text

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_decide_prompt(self, req: JobRequest, rm: ResourceMap) -> str:
        """Build the user message for an initial placement decision."""
        total = req.total_tokens
        required_tps = req.required_tps
        io_ratio = req.prefill_decode_ratio

        resources_text = get_resources(rm)

        sections = [
            f"PLACEMENT REQUEST:",
            f"  Model: {req.model_name}",
            f"  Task: {req.task_type.value} | Objective: {req.objective.value}",
            f"  Workload: {req.num_requests or '?'} requests, {req.avg_input_tokens} in / {req.avg_output_tokens} out",
            f"  IO ratio: {io_ratio:.1f}x ({'prefill-heavy' if io_ratio > 2 else 'decode-heavy' if io_ratio < 0.5 else 'balanced'})",
            f"  Total tokens: {total:,}" if total else "  Total tokens: unknown",
            f"  SLO: {req.slo_deadline_hours}h" if req.slo_deadline_hours else "",
            f"  Required TPS: ≥{required_tps:.0f} tok/s" if required_tps else "",
            f"  Quantization: {req.quantization or 'fp16 (default)'}",
            "",
            resources_text,
            "",
            "Use your tools to query PerfDB, memory, and physics. Then decide.",
            "",
            "Return your decision as a JSON block:",
            "```json",
            "{",
            '  "gpu_type": "<gpu type>",',
            '  "instance_type": "<AWS instance>",',
            '  "tp": <int>,',
            '  "pp": <int>,',
            '  "dp": <int>,',
            '  "predicted_tps": <float>,',
            '  "predicted_cost_per_hour": <float>,',
            '  "reasoning": "<why this config>",',
            '  "confidence": <0.0-1.0>,',
            '  "data_source": "<memory|perfdb_exact|perfdb_interpolated|analytical>"',
            "}",
            "```",
        ]
        return "\n".join(s for s in sections if s is not None)

    def _build_trigger_prompt(self, trigger: MonitoringTrigger) -> str:
        """Build the user message for a monitoring trigger."""
        tracker = trigger.job_tracker
        sections = [
            f"MONITORING TRIGGER: {trigger.trigger_type.value}",
            f"Job: {trigger.job_id}",
            f"",
            f"Current state:",
            f"  Smoothed TPS: {tracker.get('smoothed_tps', 0):.0f}",
            f"  SLO headroom: {tracker.get('slo_headroom_pct', 0):.1f}%",
            f"  Elapsed: {tracker.get('elapsed_hours', 0):.2f}h",
            f"  Tokens remaining: {tracker.get('tokens_remaining', 0):,}",
            f"  GPU cache: {tracker.get('gpu_cache_usage', 0):.0%}",
            f"  GPU SM util: {tracker.get('gpu_sm_util', 0):.0f}%",
            f"  GPU mem BW: {tracker.get('gpu_mem_bw_util', 0):.0f}%",
        ]

        if trigger.diagnosis_hint:
            sections.append(f"\nDiagnosis: {trigger.diagnosis_hint}")

        if trigger.trigger_type == MonitoringStatus.FALLING_BEHIND:
            sections.append("\nWhat should we do? Scale up? A/B test a different GPU? Accept the miss?")
        elif trigger.trigger_type == MonitoringStatus.OVER_PROVISIONED:
            sections.append("\nWe're over-provisioned. Which replicas should we kill to save cost?")
        elif trigger.trigger_type == MonitoringStatus.COMPLETED:
            sections.append("\nJob completed. Record the outcome using record_outcome tool.")
        elif trigger.trigger_type == MonitoringStatus.FAILED:
            sections.append("\nJob failed. Diagnose why and record corrective action.")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_decision(
        self, text: str, req: JobRequest, rm: ResourceMap,
        tool_calls: int, elapsed: float,
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
            num_instances = max(1, num_gpus // resource.gpus_per_instance)
            cost_per_hour = num_instances * resource.cost_per_instance_hour_usd

        predicted_tps = float(data.get("predicted_tps", 0.0))
        total_tokens = req.total_tokens or 0
        runtime_hours = (total_tokens / max(predicted_tps, 1) / 3600) if predicted_tps > 0 else None
        total_cost = (cost_per_hour * runtime_hours) if runtime_hours else None

        # Map data_source string to enum
        ds_map = {
            "memory": DataSource.MEMORY,
            "perfdb_exact": DataSource.EXACT_MATCH,
            "perfdb_interpolated": DataSource.INTERPOLATED,
            "analytical": DataSource.ANALYTICAL,
            "cross_gpu": DataSource.CROSS_GPU,
        }
        data_source = ds_map.get(data.get("data_source", "analytical"), DataSource.ANALYTICAL)

        num_instances = max(1, num_gpus // (resource.gpus_per_instance if resource else 8))

        config = PlacementConfig(
            gpu_type=gpu_type,
            instance_type=instance_type,
            num_gpus=num_gpus,
            num_instances=num_instances,
            tp=tp, pp=pp, dp=dp,
            region=rm.region,
            engine_config=EngineConfig(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                quantization=req.quantization,
            ),
        )

        return AgentDecision(
            job_id=req.job_id,
            model_name=req.model_name,
            config=config,
            predicted_tps=predicted_tps,
            predicted_cost_per_hour=cost_per_hour,
            predicted_total_cost=total_cost,
            predicted_runtime_hours=runtime_hours,
            reasoning=data.get("reasoning", text[:500]),
            confidence=float(data.get("confidence", 0.5)),
            data_source=data_source,
            agent_model=self.model,
            tool_calls_made=tool_calls,
            latency_seconds=elapsed,
        )
