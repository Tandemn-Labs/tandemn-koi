"""
koi/ensemble.py — Multi-LLM proposal engine + judge + diagnosis.

Architecture:
  LLM1, LLM2, LLM3 each PROPOSE a config from scratch (not pick from a list).
  Each proposal includes a causal hypothesis: why this config, what mechanism,
  what evidence, and what would falsify it.

  The judge synthesizes all three proposals and can:
    (a) pick the best individual proposal, OR
    (b) synthesize a novel config by combining reasoning across proposals

  The judge also produces an exploration queue: proposals not deployed now
  but worth testing on future low-priority jobs.

  For monitoring-triggered re-placements, a separate diagnosis prompt
  asks LLMs to diagnose the failure mode first, then propose a repair.

Directives:
  Each LLM gets a different exploration directive drawn from DIRECTIVE_POOL.
  Directives rotate based on what the system currently needs to learn —
  they are NOT fixed personas. No Sagan, no Turing, no Hopper.
  The same LLM gets a different directive depending on system state.

Guardrails (run AFTER the LLM proposes, not before):
  Memory feasibility, parallelism divisibility, GPU availability.
  If a proposal fails guardrails, the rejection reason is fed back
  to the LLM which regenerates. Up to MAX_GUARDRAIL_RETRIES attempts.
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from koi.oracle import Oracle, GPU_SPECS
from koi.schemas import (
    DiagnosisProposal,
    EngineConfig,
    ExplorationQueueEntry,
    GPUResource,
    JobRequest,
    JudgeDecision,
    OracleCandidate,
    PlacementConfig,
    PlacementDecision,
    PredictedMetrics,
    ResourceMap,
    RuntimeMetrics,
    TaskType,
    ThinkerProposal,
)

MAX_GUARDRAIL_RETRIES = 3
DEFAULT_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Directive pool
# ---------------------------------------------------------------------------
# Each directive biases one LLM toward a different region of config space.
# Selected dynamically — not hardcoded per LLM slot.

DIRECTIVE_POOL = [
    {
        "id": "cost_pressure",
        "text": (
            "Your directive: find the CHEAPEST config that has any realistic chance of meeting the SLO. "
            "Push toward fewer GPUs, cheaper GPU types, or lower parallelism if the model fits. "
            "The SLO is a hard floor, not a target — meeting it by 1% is identical to meeting it by 50%."
        ),
    },
    {
        "id": "slo_headroom",
        "text": (
            "Your directive: find the config with the MOST SLO headroom. "
            "Predictions are uncertain — a config that predicts exactly at the SLO will miss it under noise. "
            "Propose a config where you'd expect ≥30% headroom even with a pessimistic prediction."
        ),
    },
    {
        "id": "topology_challenge",
        "text": (
            "Your directive: CHALLENGE the obvious TP/PP split. "
            "If the history shows everyone has tried TP=4 PP=1, ask why not TP=2 PP=2 or TP=8 PP=2. "
            "Pipeline parallelism distributes KV cache across stages — it can help at long sequences. "
            "Tensor parallelism aggregates bandwidth — it helps when bandwidth is the bottleneck. "
            "Propose a topology that hasn't been the obvious choice."
        ),
    },
    {
        "id": "hardware_alternative",
        "text": (
            "Your directive: consider a DIFFERENT GPU type than what's been used most in the history. "
            "Look at the hardware reference: different GPUs have very different bandwidth/compute tradeoffs. "
            "H100 has NVLink — TP scales cleanly. L40S has PCIe — TP=8 saturates. A100 is in between. "
            "Is there a GPU type that fits this workload's bottleneck better than what's been tried?"
        ),
    },
    {
        "id": "frontier_push",
        "text": (
            "Your directive: propose a config that could BEAT the current best known PES for this workload class. "
            "Look at the frontier — what's the best we've seen? What hypothesis could improve on it? "
            "Don't just replicate the frontier — find what it might be missing."
        ),
    },
    {
        "id": "unexplored_region",
        "text": (
            "Your directive: propose a config in a region of the config space that has NOT been tried yet. "
            "Look at the history: what GPU types, TP values, PP values are missing? "
            "Unexplored regions are high-uncertainty — they might be bad, or they might be the best thing we haven't found. "
            "Pick the unexplored region you have the most reason to believe is promising."
        ),
    },
    {
        "id": "quantization_probe",
        "text": (
            "Your directive: consider whether QUANTIZATION could help here. "
            "FP8 on H100 cuts memory footprint in half, enabling larger TP or more KV cache headroom. "
            "INT8 on A100/L40S gives similar memory savings with some accuracy tradeoff. "
            "If the workload is memory-bound (long sequences, large models), quantization might unlock "
            "a fundamentally better config tier. Propose a quantized variant if it's plausible."
        ),
    },
    {
        "id": "scale_rethink",
        "text": (
            "Your directive: rethink the SCALE. "
            "Are we using too many GPUs for too little throughput? Or too few GPUs creating a bottleneck? "
            "Sometimes 4 GPUs at high utilization outperforms 8 GPUs with pipeline inefficiency. "
            "Consider DP (replicas) — for online serving, horizontal scaling via DP can be cheaper "
            "than more TP parallelism. What scale is actually right for this workload?"
        ),
    },
]


def select_directives(
    n: int,
    history_summary: Optional[str] = None,
    monitoring_state: Optional[str] = None,
) -> List[Dict]:
    """
    Select n directives to assign to the n LLMs.
    Tries to pick diverse directives. Can be made smarter over time
    (e.g. weight toward unexplored regions if frontier is stagnant).
    Currently: sample without replacement, prefer diversity.
    """
    # Always include at least one cost and one topology/hardware directive
    priority = ["cost_pressure", "topology_challenge", "frontier_push"]
    pool = DIRECTIVE_POOL.copy()

    selected = []
    for pid in priority:
        if len(selected) >= n:
            break
        match = next((d for d in pool if d["id"] == pid), None)
        if match:
            selected.append(match)
            pool.remove(match)

    # Fill remaining slots randomly
    random.shuffle(pool)
    for d in pool:
        if len(selected) >= n:
            break
        selected.append(d)

    return selected[:n]


# ---------------------------------------------------------------------------
# Hardware reference sheet (injected into every thinker prompt)
# ---------------------------------------------------------------------------

def _build_hardware_reference(resource_map: ResourceMap) -> str:
    lines = ["HARDWARE REFERENCE (memory bandwidth, compute, VRAM, interconnect):"]
    for res in resource_map.resources:
        specs = GPU_SPECS.get(res.gpu_type, {})
        bw = specs.get("bandwidth_gbps", "?")
        tflops = specs.get("fp16_tflops", "?")
        mem = specs.get("mem_gb", res.gpu_memory_gb)
        lines.append(
            f"  {res.gpu_type:10s}: {bw} GB/s bandwidth | {tflops} TFLOPS FP16 | "
            f"{mem}GB VRAM | {res.interconnect} | ${res.cost_per_gpu_hour_usd:.3f}/GPU/hr"
        )
    lines.append(
        "\nKey roofline insight: LLM decode is memory-bandwidth-bound at low batch sizes. "
        "tokens/sec ≈ aggregate_bandwidth / model_weight_bytes. "
        "NVLink scales TP linearly; PCIe saturates around TP=4 for 70B+ models."
    )
    return "\n".join(lines)


def _build_available_resources(resource_map: ResourceMap) -> str:
    lines = [f"AVAILABLE RESOURCES (VPC: {resource_map.vpc_id}, region: {resource_map.region}):"]
    for res in resource_map.resources:
        if res.available_gpus > 0:
            lines.append(
                f"  {res.gpu_type:10s}: {res.available_gpus} GPUs available "
                f"({res.available_gpus // res.gpus_per_instance} instances of {res.instance_type}, "
                f"{res.gpus_per_instance} GPUs each)"
            )
    return "\n".join(lines)


def _build_job_context(request: JobRequest) -> str:
    lines = [
        "JOB REQUEST:",
        f"  model         : {request.model_name}",
        f"  task_type     : {request.task_type.value}",
        f"  avg_input_len : {request.avg_input_tokens} tokens",
        f"  avg_output_len: {request.avg_output_tokens} tokens",
        f"  prefill/decode: {request.prefill_decode_ratio:.1f}x "
        f"({'prefill-heavy' if request.prefill_decode_ratio > 2 else 'decode-heavy'})",
    ]
    if request.num_requests:
        lines.append(f"  num_requests  : {request.num_requests:,}")
        lines.append(f"  total_tokens  : {request.total_tokens:,}")
    if request.expected_concurrency:
        lines.append(f"  concurrency   : {request.expected_concurrency}")
    if request.slo_deadline_hours:
        lines.append(f"  SLO deadline  : {request.slo_deadline_hours}h")
    if request.slo_tpot_ms:
        lines.append(f"  SLO TPOT      : {request.slo_tpot_ms}ms")
    if request.slo_ttft_ms:
        lines.append(f"  SLO TTFT      : {request.slo_ttft_ms}ms")
    lines.append(f"  objective     : {request.objective.value}")
    return "\n".join(lines)


def _build_oracle_reference(candidates: List[OracleCandidate], max_show: int = 8) -> str:
    """
    Show Oracle estimates as REFERENCE — not as constraints.
    The LLM can propose configs not in this list.
    """
    if not candidates:
        return "ORACLE REFERENCE: No benchmark data available — propose based on hardware specs."

    lines = [
        f"ORACLE REFERENCE ESTIMATES (from benchmark database + interpolation):",
        f"These are examples of feasible configs with predicted performance.",
        f"You are NOT limited to these — propose whatever you believe is best.",
        f"",
    ]
    for i, c in enumerate(candidates[:max_show]):
        cfg = c.config
        met = c.metrics
        slo_str = f"+{c.slo_margin_pct:.0f}% headroom" if c.meets_slo else f"{c.slo_margin_pct:.0f}% short"
        tpot_part = f"TPOT~{met.tpot_ms:.0f}ms | " if met.tpot_ms else ""
        lines.append(
            f"  {cfg.gpu_type:8s} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} | "
            f"{cfg.num_gpus}GPUs ({cfg.num_instances}x{cfg.instance_type}) | "
            f"~{met.throughput_tokens_per_sec:.0f} tok/s | "
            f"{tpot_part}"
            f"SLO: {slo_str} | conf={met.confidence:.0%} ({met.data_source.value})"
        )
    lines.append(f"\n  ... {len(candidates)} total feasible configs in Oracle's view.")
    return "\n".join(lines)


def _build_history_context(history: Optional[str]) -> str:
    if not history:
        return "PLACEMENT HISTORY: No prior runs for this workload class yet."
    return f"PLACEMENT HISTORY (recent runs for similar workloads):\n{history}"


def _build_parallelism_constraints(request: JobRequest, oracle: Oracle) -> str:
    spec = oracle.get_model_spec(request.model_name)
    num_heads = spec.get("num_attention_heads", "?")
    num_kv = spec.get("num_kv_heads", "?")
    num_layers = spec.get("num_layers", "?")
    params = spec.get("params_billion", "?")
    is_moe = spec.get("is_moe", False)

    lines = [
        f"HARD CONSTRAINTS FOR {request.model_name}:",
        f"  Parameters    : {params}B {'(MoE — all expert weights must fit in VRAM)' if is_moe else ''}",
        f"  num_layers    : {num_layers}  — PP must evenly divide this",
        f"  attention_heads: {num_heads}  — TP must evenly divide this",
        f"  kv_heads      : {num_kv}  — TP must evenly divide this (or TP ≤ num_kv_heads)",
        f"  Weight memory : ~{params if isinstance(params, (int, float)) else '?'}B params × 2 bytes (FP16) = "
        f"~{params * 2 if isinstance(params, (int, float)) else '?'}GB total",
        f"  Per-GPU weight: weight_GB ÷ TP — must leave ≥8GB for KV cache + activations",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thinker prompts
# ---------------------------------------------------------------------------

THINKER_SYSTEM_PROMPT = """\
You are an expert in distributed LLM inference infrastructure. You understand:
- GPU memory bandwidth and compute roofline models
- Tensor parallelism (TP): splits model weights across GPUs via NVLink/PCIe
- Pipeline parallelism (PP): splits model layers across GPU groups; adds bubble overhead
- Data parallelism (DP): full model replicas for horizontal scaling
- KV cache memory pressure at long sequence lengths
- vLLM internals: chunked prefill, memory utilization, max_num_seqs

Your job is to PROPOSE one specific placement configuration for a given LLM inference job.
You are not picking from a list — you are proposing what you genuinely believe is best.
Your proposal must include a causal hypothesis explaining WHY you think this config works,
what physical mechanism supports it, and what evidence you have.

Think step by step before proposing. Be specific. Be falsifiable."""


def _build_thinker_user_prompt(
    request: JobRequest,
    resource_map: ResourceMap,
    oracle: Oracle,
    candidates: List[OracleCandidate],
    directive: Dict,
    history: Optional[str] = None,
    frontier_summary: Optional[str] = None,
    prior_rejection: Optional[str] = None,
) -> str:
    sections = [
        _build_hardware_reference(resource_map),
        "",
        _build_available_resources(resource_map),
        "",
        _build_job_context(request),
        "",
        _build_parallelism_constraints(request, oracle),
        "",
        _build_oracle_reference(candidates),
        "",
        _build_history_context(history),
    ]

    if frontier_summary:
        sections += ["", f"CURRENT FRONTIER (best known PES per workload class):\n{frontier_summary}"]

    sections += [
        "",
        f"YOUR DIRECTIVE:\n{directive['text']}",
        "",
    ]

    if prior_rejection:
        sections += [
            f"PREVIOUS ATTEMPT REJECTED — reason: {prior_rejection}",
            "Adjust your proposal to fix this constraint violation.",
            "",
        ]

    sections.append(
        'Propose ONE placement configuration. Think through the hardware constraints, '
        'workload characteristics, and your directive before deciding.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "proposed_config": {\n'
        '    "gpu_type": "<e.g. H100, L40S, A100, A10G, L4>",\n'
        '    "tp": <integer, tensor parallel size>,\n'
        '    "pp": <integer, pipeline parallel size>,\n'
        '    "dp": <integer, data parallel replicas>,\n'
        '    "num_gpus": <total GPUs = tp × pp × dp>,\n'
        '    "instance_type": "<AWS instance type>",\n'
        '    "quantization": <null or "fp8" or "int8">\n'
        '  },\n'
        '  "hypothesis": "<why this config should perform well for this job>",\n'
        '  "mechanism": "<physical or architectural principle supporting your hypothesis>",\n'
        '  "evidence": "<past runs, hardware specs, or domain knowledge you are citing>",\n'
        '  "falsification_condition": "<what outcome would prove your hypothesis wrong>",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning": "<full step-by-step reasoning, 4-8 sentences>"\n'
        '}'
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are the Koi placement judge. You receive proposals from three independent LLMs,
each with a different exploration directive. Your job is to:

1. SYNTHESIZE the best deployment config — either pick one proposal or combine
   reasoning from multiple proposals into a novel config that none proposed alone.
   Synthesis is appropriate when: proposals point to the same GPU type but disagree
   on TP/PP, or when two proposals each have a correct insight the other misses.

2. BUILD an exploration queue — proposals not deployed now but worth testing on
   future low-priority jobs. Rank them by novelty and expected information gain.

3. IDENTIFY the most novel hypothesis — even if not the deployment choice,
   which proposal represents the most interesting untested claim?

Weigh proposals by the quality of their causal reasoning, not just their predicted
performance. A well-reasoned hypothesis for an uncertain region beats a confident
proposal for an already-known-good config.

Be decisive. Output one clear deployment config."""


def _build_judge_user_prompt(
    request: JobRequest,
    proposals: List[ThinkerProposal],
    resource_map: ResourceMap,
) -> str:
    job_ctx = _build_job_context(request)

    proposal_text = ""
    for i, p in enumerate(proposals):
        cfg = p.proposed_config
        est = p.oracle_estimate
        tps_str = f"~{est.throughput_tokens_per_sec:.0f} tok/s, conf={est.confidence:.0%}" if est else "no Oracle estimate"
        slo_str = ""
        if est and est.tpot_ms and request.slo_tpot_ms:
            margin = (request.slo_tpot_ms - est.tpot_ms) / request.slo_tpot_ms * 100
            slo_str = f", SLO margin={margin:.0f}%"

        proposal_text += (
            f"\n--- Proposal {i} (LLM{i+1}, directive: {p.directive}) ---\n"
            f"Config: {cfg.gpu_type} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} "
            f"({cfg.num_gpus} GPUs, {cfg.num_instances}x {cfg.instance_type}"
            f"{', quant=' + cfg.engine_config.quantization if cfg.engine_config.quantization else ''})\n"
            f"Oracle estimate: {tps_str}{slo_str}\n"
            f"Hypothesis: {p.hypothesis}\n"
            f"Mechanism: {p.mechanism}\n"
            f"Evidence: {p.evidence}\n"
            f"Falsification: {p.falsification_condition}\n"
            f"Confidence: {p.confidence:.0%}\n"
            f"Reasoning: {p.reasoning}\n"
        )

    return f"""{job_ctx}

THINKER PROPOSALS:
{proposal_text}

Synthesize these proposals. Remember: you CAN propose a novel config that combines
insights from multiple proposals if that's better than any individual proposal.

Return ONLY valid JSON:
{{
  "decision_source": "<proposal_0 | proposal_1 | proposal_2 | synthesis>",
  "deployment_config": {{
    "gpu_type": "<str>",
    "tp": <int>,
    "pp": <int>,
    "dp": <int>,
    "num_gpus": <int>,
    "instance_type": "<str>",
    "quantization": <null or "fp8" or "int8">
  }},
  "synthesis_reasoning": "<if synthesis: what you combined and why — else empty string>",
  "exploration_queue": [
    {{
      "proposal_idx": <0|1|2>,
      "priority": "<high|medium|low>",
      "reason": "<why worth testing on a future job>",
      "suggested_job_constraints": "<e.g. low-priority only, SLO headroom >= 40%>"
    }}
  ],
  "most_novel_hypothesis_thinker": "<LLM1|LLM2|LLM3>",
  "most_novel_hypothesis_summary": "<one sentence on why it's novel>",
  "reasoning": "<your full synthesis reasoning, 4-8 sentences>",
  "confidence": <float 0.0-1.0>,
  "agreement": "<full|partial|split>"
}}"""


# ---------------------------------------------------------------------------
# Diagnosis prompt (monitoring-triggered re-placement)
# ---------------------------------------------------------------------------

DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert LLM infrastructure diagnostician. A running job has triggered
a monitoring alert — its current placement config is underperforming or at risk.

Your job is TWO things:
1. DIAGNOSE: analyze the monitoring trace and identify the root cause of degradation.
   Be specific: is it memory bandwidth saturation? KV cache pressure? Pipeline bubbles?
   GPU memory overflow forcing recomputation? Network interconnect bottleneck?

2. REPAIR: propose a NEW placement config that directly addresses the diagnosed cause.
   Your repair must logically follow from your diagnosis. If you diagnose PCIe bandwidth
   saturation at TP=8, your repair must change TP or change GPU type — not just add GPUs.

Extract a CAUSAL RULE from your diagnosis: a generalizable statement like
"For models >50B on PCIe GPUs: TP > 4 saturates bandwidth at concurrency > 20."
This rule goes into the system's causal library for future placements."""


def _build_diagnosis_user_prompt(
    request: JobRequest,
    current_config: PlacementConfig,
    monitoring_trace: List[RuntimeMetrics],
    resource_map: ResourceMap,
    oracle: Oracle,
    history: Optional[str] = None,
) -> str:
    job_ctx = _build_job_context(request)
    hw_ref = _build_hardware_reference(resource_map)
    constraints = _build_parallelism_constraints(request, oracle)

    cfg = current_config
    trace_lines = [
        f"MONITORING TRACE (last {len(monitoring_trace)} samples for job {request.job_id}):",
        f"Current config: {cfg.gpu_type} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} ({cfg.num_gpus} GPUs)",
        "",
    ]
    for m in monitoring_trace[-12:]:  # show last 12 samples
        tpot_part = f"TPOT={m.tpot_ms:.1f}ms | " if m.tpot_ms else ""
        concurrency_part = f" | concurrency={m.concurrent_requests}" if m.concurrent_requests else ""
        trace_lines.append(
            f"  [{m.timestamp.strftime('%H:%M:%S')}] "
            f"TPS={m.throughput_tokens_per_sec:.0f} | "
            f"{tpot_part}"
            f"GPU_util={m.gpu_utilization_pct:.0f}% | "
            f"GPU_mem={m.gpu_memory_used_gb:.1f}GB"
            f"{concurrency_part}"
        )

    slo_context = []
    if request.slo_tpot_ms:
        slo_context.append(f"SLO TPOT target: {request.slo_tpot_ms}ms")
    if request.slo_deadline_hours:
        slo_context.append(f"SLO deadline: {request.slo_deadline_hours}h")

    hist_section = _build_history_context(history)

    return f"""{job_ctx}

{hw_ref}

{constraints}

{chr(10).join(trace_lines)}

{chr(10).join(slo_context)}

{hist_section}

Diagnose the failure and propose a repair config.

Return ONLY valid JSON:
{{
  "failure_mode": "<specific description of what the monitoring trace shows is failing>",
  "causal_rule": "<generalizable rule extracted — e.g. 'For X models at TP>N on PCIe: ...'>",
  "proposed_config": {{
    "gpu_type": "<str>",
    "tp": <int>,
    "pp": <int>,
    "dp": <int>,
    "num_gpus": <int>,
    "instance_type": "<str>",
    "quantization": <null or "fp8" or "int8">
  }},
  "repair_hypothesis": "<why the new config directly addresses the diagnosed failure>",
  "expected_improvement": "<quantified: e.g. 'GPU mem drops from 94% to ~65%, enabling concurrency=45'>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<full diagnosis chain, 4-8 sentences>"
}}"""


# ---------------------------------------------------------------------------
# Guardrail validation
# ---------------------------------------------------------------------------

def _validate_proposed_config(
    proposed: Dict,
    request: JobRequest,
    resource_map: ResourceMap,
    oracle: Oracle,
) -> Tuple[bool, str, Optional[PlacementConfig]]:
    """
    Validate a raw config dict proposed by the LLM.
    Returns (is_valid, rejection_reason, PlacementConfig_or_None).
    """
    gpu_type = proposed.get("gpu_type", "")
    tp = int(proposed.get("tp", 1))
    pp = int(proposed.get("pp", 1))
    dp = int(proposed.get("dp", 1))
    num_gpus = int(proposed.get("num_gpus", tp * pp * dp))
    instance_type = proposed.get("instance_type", "")
    quantization = proposed.get("quantization")

    # 1. GPU type available
    resource = resource_map.get_resource(gpu_type)
    if resource is None:
        available = resource_map.available_gpu_types()
        return False, f"GPU type '{gpu_type}' not available in this VPC. Available: {available}", None

    # 2. Enough GPUs
    gpus_per_replica = tp * pp
    total_needed = gpus_per_replica * dp
    if num_gpus != total_needed:
        return False, f"num_gpus={num_gpus} inconsistent with TP={tp}×PP={pp}×DP={dp}={total_needed}. Fix num_gpus.", None

    if total_needed > resource.available_gpus:
        return (
            False,
            f"Need {total_needed} {gpu_type} GPUs but only {resource.available_gpus} available.",
            None,
        )

    # 3. Parallelism divisibility
    ok, msg = oracle._check_parallelism(request.model_name, tp, pp)
    if not ok:
        return False, msg, None

    # 4. Memory feasibility (per TP group, not per DP replica)
    mem_ok, headroom, note = oracle._check_memory(request.model_name, resource, tp)
    if not mem_ok:
        return (
            False,
            f"Memory insufficient: {note}. Model weights require {oracle.get_model_spec(request.model_name)['params_billion']}B params "
            f"÷ TP={tp}. Try higher TP or a GPU with more VRAM.",
            None,
        )

    # 5. Instance type sanity
    from koi.oracle import INSTANCE_TO_GPU
    if instance_type and instance_type in INSTANCE_TO_GPU:
        mapped = INSTANCE_TO_GPU[instance_type]
        if mapped != gpu_type:
            return (
                False,
                f"instance_type '{instance_type}' maps to {mapped}, not {gpu_type}. Fix the instance_type.",
                None,
            )

    # Infer instance_type if not given or wrong
    if not instance_type or instance_type not in INSTANCE_TO_GPU:
        instance_type = resource.instance_type

    num_instances = max(1, total_needed // resource.gpus_per_instance)

    engine = EngineConfig(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        quantization=quantization,
    )

    config = PlacementConfig(
        gpu_type=gpu_type,
        instance_type=instance_type,
        num_gpus=total_needed,
        num_instances=num_instances,
        tp=tp,
        pp=pp,
        dp=dp,
        region=resource_map.region,
        engine_config=engine,
    )

    return True, "", config


# ---------------------------------------------------------------------------
# Core LLM calls
# ---------------------------------------------------------------------------

async def _call_one_thinker(
    client: AsyncAnthropic,
    thinker_id: str,
    directive: Dict,
    request: JobRequest,
    resource_map: ResourceMap,
    oracle: Oracle,
    candidates: List[OracleCandidate],
    history: Optional[str] = None,
    frontier_summary: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> ThinkerProposal:
    """
    One LLM proposes a config + causal hypothesis.
    Retries up to MAX_GUARDRAIL_RETRIES if guardrails reject the proposal.
    """
    rejections: List[str] = []
    prior_rejection = None

    for attempt in range(MAX_GUARDRAIL_RETRIES):
        user_prompt = _build_thinker_user_prompt(
            request=request,
            resource_map=resource_map,
            oracle=oracle,
            candidates=candidates,
            directive=directive,
            history=history,
            frontier_summary=frontier_summary,
            prior_rejection=prior_rejection,
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=THINKER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()

            data = json.loads(raw)
            proposed_raw = data["proposed_config"]

            # Run guardrails
            valid, rejection_msg, config = _validate_proposed_config(
                proposed_raw, request, resource_map, oracle
            )

            if not valid:
                prior_rejection = rejection_msg
                rejections.append(f"Attempt {attempt+1}: {rejection_msg}")
                print(f"[Ensemble] {thinker_id} attempt {attempt+1} rejected: {rejection_msg}")
                continue

            # Get Oracle estimate for the proposed config
            oracle_estimate = oracle.estimate_for_config(
                request=request,
                resource=resource_map.get_resource(config.gpu_type),
                tp=config.tp,
                pp=config.pp,
                dp=config.dp,
            )

            print(
                f"[Ensemble] {thinker_id} → {config.gpu_type} TP={config.tp} PP={config.pp} DP={config.dp} "
                f"({config.num_gpus} GPUs) | directive={directive['id']}"
            )

            return ThinkerProposal(
                thinker_id=thinker_id,
                directive=directive["id"],
                proposed_config=config,
                oracle_estimate=oracle_estimate,
                hypothesis=data.get("hypothesis", ""),
                mechanism=data.get("mechanism", ""),
                evidence=data.get("evidence", ""),
                falsification_condition=data.get("falsification_condition", ""),
                confidence=float(data.get("confidence", 0.6)),
                reasoning=data.get("reasoning", ""),
                guardrail_rejections=rejections,
            )

        except Exception as e:
            print(f"[Ensemble] {thinker_id} attempt {attempt+1} error: {e}")
            prior_rejection = f"JSON parse or API error: {e}"
            rejections.append(f"Attempt {attempt+1}: {prior_rejection}")

    # All retries exhausted — fall back to cheapest feasible candidate
    print(f"[Ensemble] {thinker_id} all retries failed, using fallback candidate")
    fallback = next((c for c in candidates if c.meets_slo), candidates[0])
    return ThinkerProposal(
        thinker_id=thinker_id,
        directive=directive["id"],
        proposed_config=fallback.config,
        oracle_estimate=fallback.metrics,
        hypothesis="Fallback: could not generate valid proposal after retries.",
        mechanism="N/A",
        evidence="N/A",
        falsification_condition="N/A",
        confidence=0.3,
        reasoning=f"Fallback after {MAX_GUARDRAIL_RETRIES} failed attempts. Rejections: {rejections}",
        guardrail_rejections=rejections,
    )


async def _call_judge(
    client: AsyncAnthropic,
    proposals: List[ThinkerProposal],
    request: JobRequest,
    resource_map: ResourceMap,
    oracle: Oracle,
    model: str = DEFAULT_MODEL,
) -> JudgeDecision:
    """
    Judge synthesizes proposals → deployment config + exploration queue.
    """
    user_prompt = _build_judge_user_prompt(request, proposals, resource_map)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1280,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()

        data = json.loads(raw)
        source = data.get("decision_source", "proposal_0")
        cfg_raw = data.get("deployment_config", {})

        # Validate the judge's chosen/synthesized config
        valid, rejection, config = _validate_proposed_config(cfg_raw, request, resource_map, oracle)
        if not valid:
            # Fall back to the first valid proposal
            print(f"[Ensemble] Judge synthesis invalid ({rejection}), using proposal_0")
            config = proposals[0].proposed_config
            source = "proposal_0"

        oracle_estimate = oracle.estimate_for_config(
            request=request,
            resource=resource_map.get_resource(config.gpu_type),
            tp=config.tp, pp=config.pp, dp=config.dp,
        )

        # Build exploration queue
        eq_raw = data.get("exploration_queue", [])
        exploration_queue = []
        for entry in eq_raw:
            idx = entry.get("proposal_idx")
            if idx is not None and 0 <= idx < len(proposals):
                exploration_queue.append(ExplorationQueueEntry(
                    proposal=proposals[idx],
                    priority=entry.get("priority", "medium"),
                    reason=entry.get("reason", ""),
                    suggested_job_constraints=entry.get("suggested_job_constraints", ""),
                ))

        print(
            f"[Ensemble] Judge → {source}: {config.gpu_type} TP={config.tp} PP={config.pp} DP={config.dp} "
            f"({config.num_gpus} GPUs) | conf={data.get('confidence', 0.7):.0%} | "
            f"agreement={data.get('agreement', 'partial')} | "
            f"exploration_queue={len(exploration_queue)} items"
        )

        return JudgeDecision(
            decision_source=source,
            deployment_config=config,
            deployment_oracle_estimate=oracle_estimate,
            synthesis_reasoning=data.get("synthesis_reasoning", ""),
            exploration_queue=exploration_queue,
            most_novel_hypothesis_thinker=data.get("most_novel_hypothesis_thinker", ""),
            most_novel_hypothesis_summary=data.get("most_novel_hypothesis_summary", ""),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.7)),
            agreement=data.get("agreement", "partial"),
        )

    except Exception as e:
        print(f"[Ensemble] Judge failed: {e}. Falling back to proposal_0.")
        p = proposals[0]
        return JudgeDecision(
            decision_source="proposal_0",
            deployment_config=p.proposed_config,
            deployment_oracle_estimate=p.oracle_estimate,
            reasoning=f"Judge fallback due to error: {e}",
            confidence=0.4,
            agreement="split",
        )


async def _call_diagnosis(
    client: AsyncAnthropic,
    thinker_id: str,
    request: JobRequest,
    current_config: PlacementConfig,
    monitoring_trace: List[RuntimeMetrics],
    resource_map: ResourceMap,
    oracle: Oracle,
    history: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> DiagnosisProposal:
    """
    One LLM diagnoses a monitoring failure and proposes a repair config.
    """
    user_prompt = _build_diagnosis_user_prompt(
        request, current_config, monitoring_trace, resource_map, oracle, history
    )

    for attempt in range(MAX_GUARDRAIL_RETRIES):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=DIAGNOSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()

            data = json.loads(raw)
            cfg_raw = data["proposed_config"]

            valid, rejection, config = _validate_proposed_config(cfg_raw, request, resource_map, oracle)
            if not valid:
                print(f"[Ensemble] {thinker_id} diagnosis attempt {attempt+1} rejected: {rejection}")
                # Append rejection to message and retry
                user_prompt += f"\n\nPREVIOUS PROPOSAL REJECTED: {rejection}\nAdjust your repair config."
                continue

            oracle_estimate = oracle.estimate_for_config(
                request=request,
                resource=resource_map.get_resource(config.gpu_type),
                tp=config.tp, pp=config.pp, dp=config.dp,
            )

            print(
                f"[Ensemble] {thinker_id} diagnosis → {config.gpu_type} TP={config.tp} PP={config.pp} "
                f"DP={config.dp} | failure_mode: {data.get('failure_mode', '')[:60]}..."
            )

            return DiagnosisProposal(
                thinker_id=thinker_id,
                failure_mode=data.get("failure_mode", ""),
                causal_rule=data.get("causal_rule", ""),
                proposed_config=config,
                oracle_estimate=oracle_estimate,
                repair_hypothesis=data.get("repair_hypothesis", ""),
                expected_improvement=data.get("expected_improvement", ""),
                confidence=float(data.get("confidence", 0.6)),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            print(f"[Ensemble] {thinker_id} diagnosis attempt {attempt+1} error: {e}")

    # Fallback: current config unchanged
    return DiagnosisProposal(
        thinker_id=thinker_id,
        failure_mode="Could not diagnose",
        causal_rule="",
        proposed_config=current_config,
        repair_hypothesis="Fallback — keeping current config",
        expected_improvement="Unknown",
        confidence=0.1,
        reasoning="All diagnosis attempts failed.",
    )


# ---------------------------------------------------------------------------
# Main ensemble class
# ---------------------------------------------------------------------------

class KoiEnsemble:
    """
    Runs the full LLM proposal pipeline:
      - Initial placement: 3 LLMs propose configs → judge synthesizes
      - Re-placement: 3 LLMs diagnose failure → judge picks best repair

    Usage:
        ensemble = KoiEnsemble(oracle=oracle, api_key="sk-ant-...")
        decision = await ensemble.run(request, resource_map, candidates)
        # or for monitoring-triggered:
        decision = await ensemble.run_diagnosis(request, current_config, trace, resource_map)
    """

    def __init__(
        self,
        oracle: Oracle,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        n_thinkers: int = 3,
    ):
        self.oracle = oracle
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.n_thinkers = n_thinkers

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    async def run(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
        candidates: List[OracleCandidate],
        history: Optional[str] = None,
        frontier_summary: Optional[str] = None,
    ) -> PlacementDecision:
        """
        Initial placement: n LLMs propose → judge synthesizes → PlacementDecision.
        candidates: Oracle reference estimates (shown to LLMs, not used as constraint).
        history: DeltaStore/PolicyMemory summary string (optional).
        frontier_summary: Current efficiency frontier summary (optional).
        """
        if not candidates and not resource_map.available_gpu_types():
            raise ValueError("No available GPU resources in VPC")

        client = AsyncAnthropic(api_key=self.api_key)

        # Select directives for this round
        directives = select_directives(self.n_thinkers)

        print(f"[Ensemble] Running {self.n_thinkers} thinkers in parallel...")
        print(f"[Ensemble] Directives: {[d['id'] for d in directives]}")

        # Run all thinkers concurrently
        tasks = [
            _call_one_thinker(
                client=client,
                thinker_id=f"LLM{i+1}",
                directive=directives[i],
                request=request,
                resource_map=resource_map,
                oracle=self.oracle,
                candidates=candidates,
                history=history,
                frontier_summary=frontier_summary,
                model=self.model,
            )
            for i in range(self.n_thinkers)
        ]
        proposals = await asyncio.gather(*tasks)
        proposals = list(proposals)

        # Run judge
        print(f"[Ensemble] Running judge...")
        judge = await _call_judge(
            client=client,
            proposals=proposals,
            request=request,
            resource_map=resource_map,
            oracle=self.oracle,
            model=self.model,
        )

        chosen_config = judge.deployment_config
        chosen_metrics = judge.deployment_oracle_estimate or (
            proposals[0].oracle_estimate if proposals[0].oracle_estimate else
            # last resort: get a fresh estimate
            self.oracle.estimate_for_config(
                request=request,
                resource=resource_map.get_resource(chosen_config.gpu_type),
                tp=chosen_config.tp, pp=chosen_config.pp, dp=chosen_config.dp,
            )
        )

        return PlacementDecision(
            job_id=request.job_id,
            model_name=request.model_name,
            recommendation=chosen_config,
            predicted_metrics=chosen_metrics,
            reasoning=judge.reasoning + (
                f"\n\nSynthesis note: {judge.synthesis_reasoning}" if judge.synthesis_reasoning else ""
            ),
            confidence=judge.confidence,
            thinker_proposals=proposals,
            judge_decision=judge,
            exploration_queue=judge.exploration_queue,
            total_llm_calls=self.n_thinkers + 1,
            oracle_candidates_evaluated=len(candidates),
            is_reconfig=False,
            triggered_by="initial",
        )

    async def run_diagnosis(
        self,
        request: JobRequest,
        current_config: PlacementConfig,
        monitoring_trace: List[RuntimeMetrics],
        resource_map: ResourceMap,
        history: Optional[str] = None,
    ) -> PlacementDecision:
        """
        Monitoring-triggered re-placement.
        3 LLMs diagnose independently → judge picks best repair.
        """
        client = AsyncAnthropic(api_key=self.api_key)

        print(f"[Ensemble] Running diagnosis ({len(monitoring_trace)} trace samples)...")

        tasks = [
            _call_diagnosis(
                client=client,
                thinker_id=f"LLM{i+1}",
                request=request,
                current_config=current_config,
                monitoring_trace=monitoring_trace,
                resource_map=resource_map,
                oracle=self.oracle,
                history=history,
                model=self.model,
            )
            for i in range(self.n_thinkers)
        ]
        diagnoses = await asyncio.gather(*tasks)
        diagnoses = list(diagnoses)

        # Convert diagnosis proposals to thinker proposals for the judge
        thinker_proposals = [
            ThinkerProposal(
                thinker_id=d.thinker_id,
                directive="diagnosis",
                proposed_config=d.proposed_config,
                oracle_estimate=d.oracle_estimate,
                hypothesis=d.repair_hypothesis,
                mechanism=d.failure_mode,
                evidence=d.causal_rule,
                falsification_condition="If repair config shows same failure mode within 30 min",
                confidence=d.confidence,
                reasoning=d.reasoning,
            )
            for d in diagnoses
        ]

        # Judge picks best repair
        judge = await _call_judge(
            client=client,
            proposals=thinker_proposals,
            request=request,
            resource_map=resource_map,
            oracle=self.oracle,
            model=self.model,
        )

        chosen_config = judge.deployment_config
        chosen_metrics = judge.deployment_oracle_estimate

        # Build combined causal rule from the winning diagnosis
        winning_idx = int(judge.decision_source.split("_")[-1]) if judge.decision_source.startswith("proposal") else 0
        winning_diagnosis = diagnoses[min(winning_idx, len(diagnoses) - 1)]

        return PlacementDecision(
            job_id=request.job_id,
            model_name=request.model_name,
            recommendation=chosen_config,
            predicted_metrics=chosen_metrics or thinker_proposals[0].oracle_estimate,
            reasoning=(
                f"Re-placement triggered by monitoring.\n"
                f"Failure mode: {winning_diagnosis.failure_mode}\n"
                f"Causal rule: {winning_diagnosis.causal_rule}\n"
                f"Repair: {judge.reasoning}"
            ),
            confidence=judge.confidence,
            thinker_proposals=thinker_proposals,
            judge_decision=judge,
            exploration_queue=judge.exploration_queue,
            total_llm_calls=self.n_thinkers + 1,
            is_reconfig=True,
            triggered_by="monitoring_hard",
        )

    def run_sync(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
        candidates: List[OracleCandidate],
        history: Optional[str] = None,
        frontier_summary: Optional[str] = None,
    ) -> PlacementDecision:
        """Synchronous wrapper for non-async callers."""
        return asyncio.run(self.run(request, resource_map, candidates, history, frontier_summary))

    def run_diagnosis_sync(
        self,
        request: JobRequest,
        current_config: PlacementConfig,
        monitoring_trace: List[RuntimeMetrics],
        resource_map: ResourceMap,
        history: Optional[str] = None,
    ) -> PlacementDecision:
        """Synchronous wrapper for diagnosis."""
        return asyncio.run(self.run_diagnosis(request, current_config, monitoring_trace, resource_map, history))
