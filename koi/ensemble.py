"""
koi/ensemble.py — Multi-LLM proposal engine + judge.

Architecture (post-RAG refactor):
  Each of 3 LLMs receives:
    - Top-10 RAG records (real observed performance data)
    - Full model architecture features (30-70 variables)
    - Resource map with hardware specs and availability
    - Exploration directive (cost / headroom / topology / etc.)
    - Evolutionary history (past runs for this workload class)

  Each LLM proposes 5 ranked configs grounded in the RAG evidence.
  The proposals are NOT invented from scratch — they are selected and adapted
  from what the performance DB shows can work, constrained to available hardware.

  Judge receives all 15 proposals (5 × 3), ranks to top 5 for Orca.
  The top_placements list in PlacementDecision is the final Orca input.

Guardrails:
  Memory feasibility, parallelism divisibility, GPU availability.
  Applied per-config within the thinker's 5-proposal JSON.
  Invalid proposals within a batch are flagged but don't retry the whole call.
"""

import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from koi.model_features import (
    ModelFeatures,
    compute_config_features,
    config_features_to_llm_context,
    get_model_features,
)
from koi.oracle import GPU_SPECS, INSTANCE_TO_GPU, Oracle
from koi.perf_rag import PerfRAG
from koi.schemas import (
    DiagnosisProposal,
    EngineConfig,
    ExplorationQueueEntry,
    GPUResource,
    JobRequest,
    JudgeDecision,
    OracleCandidate,
    OracleResult,
    PlacementConfig,
    PlacementDecision,
    PredictedMetrics,
    RankedPlacement,
    ResourceMap,
    RuntimeMetrics,
    TaskType,
    ThinkerProposal,
    ThinkerResult,
)

MAX_GUARDRAIL_RETRIES = 2
DEFAULT_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Directive pool
# ---------------------------------------------------------------------------

DIRECTIVE_POOL = [
    {
        "id": "cost_pressure",
        "text": (
            "DIRECTIVE: Find the CHEAPEST config that realistically meets the SLO. "
            "Push toward fewer GPUs, cheaper GPU types, or lower parallelism if memory allows. "
            "Meeting the SLO by 1% is identical to meeting it by 50%. "
            "Consider whether DP (replicas) is cheaper than more TP for throughput targets."
        ),
    },
    {
        "id": "slo_headroom",
        "text": (
            "DIRECTIVE: Find the config with the MOST SLO headroom. "
            "Predictions are uncertain — a config at exactly the SLO will miss it under noise. "
            "Target ≥30% headroom even under pessimistic prediction. "
            "Higher confidence RAG records should be weighted more heavily."
        ),
    },
    {
        "id": "topology_challenge",
        "text": (
            "DIRECTIVE: CHALLENGE the obvious TP/PP split. "
            "If the history shows TP=4 PP=1, ask why not TP=2 PP=2 or TP=8 PP=2. "
            "PP distributes KV cache across stages — helps at long sequences, adds bubble overhead. "
            "TP aggregates bandwidth — helps when bandwidth is the bottleneck. "
            "Consider kv_heads_per_tp_shard: replication happens when TP > num_kv_heads."
        ),
    },
    {
        "id": "hardware_alternative",
        "text": (
            "DIRECTIVE: Consider a DIFFERENT GPU type than the obvious choice. "
            "H100/H200 have NVLink — TP scales linearly. L40S has PCIe — TP=8 saturates. "
            "H200 has 140GB VRAM — enables larger TP or avoids quantization. "
            "Look at bandwidth_per_param and flops_per_param in the RAG records to identify "
            "which bottleneck dominates for this model×workload."
        ),
    },
    {
        "id": "frontier_push",
        "text": (
            "DIRECTIVE: Propose configs that BEAT the current frontier. "
            "Look at what the RAG records show for the best per-GPU throughput. "
            "What combination of TP, PP, DP, and GPU type maximizes throughput/GPU? "
            "Consider whether the top RAG record's config can be adapted to available hardware."
        ),
    },
    {
        "id": "moe_aware",
        "text": (
            "DIRECTIVE: Reason carefully about MoE-specific constraints if this is an MoE model. "
            "ALL expert weights must fit in VRAM even though only active_experts fire per token. "
            "Expert parallelism (EP) is not yet in scope but affects memory calculations. "
            "For dense models: focus on GQA ratio — high GQA means KV cache is small, "
            "enabling more aggressive TP without KV replication overhead."
        ),
    },
    {
        "id": "quantization_probe",
        "text": (
            "DIRECTIVE: Consider whether QUANTIZATION changes the picture. "
            "FP8 on H100 halves weight memory → larger TP headroom or smaller GPU count. "
            "INT8 on A100/L40S gives similar savings with accuracy tradeoff. "
            "If the model barely fits at FP16, FP8 might unlock a fundamentally better tier. "
            "Check vram_headroom in the RAG records for quantized vs non-quantized entries."
        ),
    },
    {
        "id": "scale_rethink",
        "text": (
            "DIRECTIVE: Rethink SCALE. Are we over- or under-provisioned? "
            "Sometimes 4 GPUs at high utilization beats 8 GPUs with pipeline inefficiency. "
            "DP (replicas) can be cheaper than more TP for online serving throughput. "
            "roofline_tps in the RAG records shows theoretical ceiling — "
            "compare to actual TPS to estimate how much headroom the config has."
        ),
    },
]


def select_directives(n: int) -> List[Dict]:
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
    random.shuffle(pool)
    for d in pool:
        if len(selected) >= n:
            break
        selected.append(d)
    return selected[:n]


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_hardware_reference(resource_map: ResourceMap) -> str:
    lines = ["HARDWARE REFERENCE (bandwidth, compute, VRAM, interconnect, cost):"]
    for res in resource_map.resources:
        specs = GPU_SPECS.get(res.gpu_type, {})
        bw = specs.get("bandwidth_gbps", "?")
        tflops = specs.get("fp16_tflops", "?")
        mem = specs.get("mem_gb", res.gpu_memory_gb)
        gen = specs.get("generation", "?")
        lines.append(
            f"  {res.gpu_type:12s}: {bw} GB/s bw | {tflops} TFLOPS FP16 | "
            f"{mem}GB VRAM | {res.interconnect} | {gen} | ${res.cost_per_gpu_hour_usd:.3f}/GPU/hr"
        )
    lines += [
        "",
        "Roofline insight: LLM DECODE is memory-bandwidth-bound at typical batch sizes.",
        "  tokens/sec ≈ aggregate_bandwidth / model_weight_bytes",
        "  NVLink scales TP cleanly; PCIe saturates around TP=4-8 for large models.",
        "  PREFILL is compute-bound; larger TFLOPS = faster TTFT.",
    ]
    return "\n".join(lines)


def _build_available_resources(resource_map: ResourceMap) -> str:
    lines = [f"AVAILABLE RESOURCES (VPC: {resource_map.vpc_id}, region: {resource_map.region}):"]
    for res in resource_map.resources:
        if res.available_gpus > 0:
            lines.append(
                f"  {res.gpu_type:12s}: {res.available_gpus} GPUs available "
                f"({res.available_gpus // res.gpus_per_instance} instances × "
                f"{res.instance_type}, {res.gpus_per_instance} GPUs/instance)"
            )
    lines.append("You MUST only propose configs using GPU types and counts shown above.")
    return "\n".join(lines)


def _build_job_context(request: JobRequest) -> str:
    lines = [
        "JOB REQUEST:",
        f"  model_name      : {request.model_name}",
        f"  task_type       : {request.task_type.value}",
        f"  avg_input_len   : {request.avg_input_tokens} tokens",
        f"  avg_output_len  : {request.avg_output_tokens} tokens",
        f"  io_ratio        : {request.prefill_decode_ratio:.2f}x "
        f"({'prefill-heavy (compute-bound)' if request.prefill_decode_ratio > 2 else 'decode-heavy (bandwidth-bound)'})",
    ]
    if request.num_requests:
        lines.append(f"  num_requests    : {request.num_requests:,}")
    if request.expected_concurrency:
        lines.append(f"  concurrency     : {request.expected_concurrency} concurrent users")
    if request.slo_deadline_hours:
        lines.append(f"  SLO deadline    : {request.slo_deadline_hours}h")
    if request.slo_tpot_ms:
        lines.append(f"  SLO TPOT        : {request.slo_tpot_ms} ms")
    if request.slo_ttft_ms:
        lines.append(f"  SLO TTFT        : {request.slo_ttft_ms} ms")
    lines.append(f"  objective       : {request.objective.value}")
    return "\n".join(lines)


def _build_model_features_context(mf: ModelFeatures) -> str:
    """Full model architecture context with ALL placement-relevant variables."""
    return mf.to_llm_context()


def _build_parallelism_constraints(mf: ModelFeatures) -> str:
    lines = [
        f"HARD PARALLELISM CONSTRAINTS for {mf.model_name}:",
        f"  TP must evenly divide num_attention_heads = {mf.num_attention_heads}",
        f"  TP must evenly divide num_kv_heads = {mf.num_kv_heads} "
        f"(or TP ≤ num_kv_heads; beyond that KV heads are replicated)",
        f"  PP must evenly divide num_layers = {mf.num_layers}",
        f"  Valid TP values: {[t for t in [1,2,4,8,16,32] if mf.num_attention_heads % t == 0 and t <= mf.num_attention_heads]}",
        f"  Valid PP values: {[p for p in [1,2,4,8] if mf.num_layers % p == 0]}",
        f"  KV replication at TP > {mf.num_kv_heads} "
        f"(adds memory overhead but usually acceptable for GQA models)",
    ]
    return "\n".join(lines)


def _build_rag_context(rag_records: List[Dict], rag: PerfRAG) -> str:
    """Format RAG records for LLM consumption."""
    return rag.format_records_for_llm(rag_records, max_show=10)


def _build_oracle_reference(candidates: List[OracleCandidate], max_show: int = 6) -> str:
    if not candidates:
        return "ORACLE FEASIBILITY ESTIMATES: No data — propose based on hardware specs and RAG records."

    lines = [
        "ORACLE FEASIBILITY ESTIMATES (interpolated from RAG — for reference only):",
        "These show which configs are memory+parallelism feasible and their estimated metrics.",
        "",
    ]
    for c in candidates[:max_show]:
        cfg, met = c.config, c.metrics
        slo_str = f"+{c.slo_margin_pct:.0f}% headroom" if c.meets_slo else f"{c.slo_margin_pct:.0f}% short"
        tpot = f"TPOT~{met.tpot_ms:.0f}ms " if met.tpot_ms else ""
        lines.append(
            f"  {cfg.gpu_type:10s} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} | "
            f"{cfg.num_gpus}GPUs | ~{met.throughput_tokens_per_sec:.0f} tok/s | "
            f"{tpot}${met.cost_per_hour_usd:.2f}/hr | SLO:{slo_str} | "
            f"conf={met.confidence:.0%} ({met.data_source.value})"
        )
    return "\n".join(lines)


def _build_history_context(history: Optional[str]) -> str:
    if not history:
        return "PLACEMENT HISTORY: No prior runs for this workload class yet."
    return f"PLACEMENT HISTORY (prior runs for similar workloads):\n{history}"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

THINKER_SYSTEM_PROMPT = """\
You are an expert in distributed LLM inference infrastructure. You deeply understand:

MODEL ARCHITECTURE VARIABLES (all affect placement):
  - num_params_billions: total parameter count (all experts for MoE)
  - num_layers: depth; determines valid PP values (PP must divide num_layers)
  - hidden_dim: embedding dimension; determines GEMM sizes and compute intensity
  - num_attention_heads: attention heads; TP must divide this
  - num_kv_heads: KV cache heads; with GQA this is << num_attention_heads
  - gqa_ratio = attention_heads / kv_heads: high ratio = small KV cache (good for memory)
  - vocab_size: affects embedding layer size and logit computation
  - is_moe: Mixture of Experts; ALL expert weights load into VRAM even though only
    active_experts fire per token
  - num_experts / active_experts: determines expert routing; power-law load imbalance
    means hot experts dominate latency
  - active_expert_ratio = active/total: effective compute fraction per token
  - dtype_bytes: 2=FP16/BF16, 1=FP8/INT8, 0.5=INT4
  - model_size_gb = num_params × dtype_bytes / 1e9
  - architecture_family: llama/qwen/deepseek/mistral/phi affect hidden dim ratios

HARDWARE VARIABLES:
  - gpu_type: H100/H200/A100/L40S/A10G/L4 — each has distinct bandwidth/compute profile
  - gpu_vram_gb: hard constraint for weight + KV cache + activations
  - gpu_bandwidth_gbps: decode bottleneck; more bandwidth = faster token generation
  - gpu_tflops_fp16: prefill bottleneck; more TFLOPS = faster TTFT
  - gpu_generation: Hopper (H100/H200) has FP8 native, better NVLink; Ampere (A100) does not
  - num_gpus_total = TP × PP × DP: total GPU count
  - price_per_gpu_hour: cost driver for batch jobs

CONFIG VARIABLES:
  - tp (tensor parallelism): shards weight matrices across GPUs via NVLink/PCIe
    → aggregates bandwidth: tps ∝ tp × bandwidth_per_gpu
    → reduces per-GPU memory: weight_per_gpu = model_size_gb / tp
    → NVLink (H100/H200): scales linearly; PCIe (L40S/A100): saturates ~TP=4-8
  - pp (pipeline parallelism): assigns layers to pipeline stages
    → reduces per-GPU memory further: each stage holds num_layers/pp layers
    → adds bubble overhead: efficiency ≈ 1 - (pp-1)/(pp×microbatch_count)
    → helps at very long sequences where KV cache doesn't fit
  - dp (data parallelism): full model replicas for horizontal scaling
    → linearly scales throughput: total_tps = replica_tps × dp
    → useful when tp=max already and more throughput is needed
  - quantization_level: FP8 halves memory vs FP16 with ~1% accuracy loss on H100

DERIVED PHYSICS FEATURES (critical for placement accuracy):
  - params_per_gpu = num_params / tp: how much model each GPU holds
  - model_fits_single_gpu = (model_size_gb < gpu_vram_gb): forces TP if False
  - vram_headroom = (vram - weight_per_gpu) / vram: KV cache space fraction
    → must be > 0.10 (need ≥8GB free); higher is better for long contexts
  - bandwidth_per_param = (bw × tp) / params: decode speed proxy
    → larger = faster generation at low batch; decode is BW-bound
  - flops_per_param = (tflops × tp) / params: prefill speed proxy
    → larger = faster TTFT; prefill is compute-bound
  - crosses_node_boundary = (tp > gpus_per_node): inter-node latency penalty
    → 8×NVLink intra-node is fast; inter-node NVSwitch adds ~50μs/allreduce
  - kv_heads_per_tp_shard = num_kv_heads / tp:
    → < 1.0: KV heads replicated per TP shard (memory waste, but OK for GQA)
    → 0: all TP shards use full KV (avoid if MHA)
  - total_cost_per_hour = num_gpus × price_per_gpu_hour

WORKLOAD VARIABLES:
  - max_input_length / max_output_length: context window usage
  - total_context = input + output: KV cache sizing
  - io_ratio = input / output: prefill-heavy (>2) vs decode-heavy (<0.5)
    → prefill-heavy: TFLOPS matters more, TP scales differently
    → decode-heavy: bandwidth matters more, TP bandwidth aggregation is key

INFERENCE SERVING MODES:
  - Aggregated (continuous batching): prefill + decode mixed; standard vLLM mode
  - Disaggregated: separate prefill/decode pools; better for large ISL/OSL ratios
    (not currently in scope but affects how you think about compute vs memory balance)

Your job: study the RAG performance records and resource map, then propose 5 ranked
configs that are GROUNDED in the observed data. You are not guessing — you are
selecting and adapting the best observed configs to the available hardware.
Think through ALL the variables above before proposing. Be specific and physical."""


JUDGE_SYSTEM_PROMPT = """\
You are the Koi placement judge. You receive 15 proposals from 3 independent LLMs
(5 proposals each), each with different exploration directives.

Your job:
1. RANK all 15 proposals, producing a TOP 5 list for the Orca scheduler.
   Orca will use these in order (rank 1 if available, else rank 2, etc.).
2. EVALUATE each proposal by:
   - Is it grounded in the RAG evidence? (high weight)
   - Is the physics reasoning sound? (high weight)
   - Does it respect resource map constraints?
   - Does it meet the SLO?
   - Is the cost reasonable for the objective?
3. SYNTHESIZE when appropriate: if two proposals each have a correct insight
   the other misses, you may combine them into a novel rank-1 config.
4. BUILD an exploration queue: proposals not in top 5 but worth testing later.
5. IDENTIFY the most novel hypothesis: even if not deployed, which proposal
   represents the most interesting untested claim?

Weigh quality of physical reasoning over predicted performance numbers — the
predictions are uncertain, but the reasoning structure reveals whether the
proposer actually understands what limits this workload.

Be decisive. Output exactly 5 ranked placements."""

DIAGNOSIS_SYSTEM_PROMPT = """\
You are an expert LLM infrastructure diagnostician. A running job has triggered
a monitoring alert — its current placement config is underperforming or at risk.

Your job is TWO things:
1. DIAGNOSE: analyze the monitoring trace and identify the root cause.
   Be specific: memory bandwidth saturation? KV cache pressure? Pipeline bubbles?
   PCIe saturation at high TP? GPU memory overflow? Inter-node bottleneck?

2. REPAIR: propose a NEW config that directly addresses the diagnosed cause.
   Your repair must logically follow your diagnosis. If you diagnose PCIe
   saturation at TP=8, your repair must change TP or change GPU type.

Extract a CAUSAL RULE: a generalizable statement like
"For models >50B on PCIe GPUs: TP > 4 saturates bandwidth at concurrency > 20."
This rule goes into the causal library for future placements."""


# ---------------------------------------------------------------------------
# Thinker user prompt
# ---------------------------------------------------------------------------

def _build_thinker_user_prompt(
    request: JobRequest,
    resource_map: ResourceMap,
    mf: ModelFeatures,
    candidates: List[OracleCandidate],
    rag_records: List[Dict],
    rag: PerfRAG,
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
        _build_model_features_context(mf),
        "",
        _build_parallelism_constraints(mf),
        "",
        _build_rag_context(rag_records, rag),
        "",
        _build_oracle_reference(candidates),
        "",
        _build_history_context(history),
    ]

    if frontier_summary:
        sections += ["", f"CURRENT FRONTIER:\n{frontier_summary}"]

    sections += ["", f"YOUR DIRECTIVE:\n{directive['text']}", ""]

    if prior_rejection:
        sections += [
            f"PREVIOUS PROPOSALS PARTIALLY REJECTED — issues: {prior_rejection}",
            "Fix the constraint violations in your retry.",
            "",
        ]

    sections.append(
        "TASK: Propose EXACTLY 5 placement configurations, ranked 1 (best) to 5.\n"
        "Ground each proposal in the RAG records shown above.\n"
        "For each config, verify: (a) GPU type exists in available resources, "
        "(b) TP divides num_attention_heads, (c) PP divides num_layers, "
        "(d) model weights fit in VRAM with ≥8GB headroom, "
        "(e) num_gpus ≤ available GPUs for that type.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "proposals": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "proposed_config": {\n'
        '        "gpu_type": "<H100|H200|A100|L40S|A10G|L4>",\n'
        '        "tp": <int>,\n'
        '        "pp": <int>,\n'
        '        "dp": <int>,\n'
        '        "num_gpus": <tp × pp × dp>,\n'
        '        "instance_type": "<AWS instance type>",\n'
        '        "quantization": <null|"fp8"|"int8">\n'
        "      },\n"
        '      "hypothesis": "<why this config fits this workload>",\n'
        '      "mechanism": "<physical principle: bandwidth-bound, VRAM-limited, etc.>",\n'
        '      "evidence": "<which RAG record(s) support this — cite sim score and TPS>",\n'
        '      "falsification_condition": "<what result would prove this wrong>",\n'
        '      "confidence": <0.0-1.0>,\n'
        '      "reasoning": "<step-by-step, cite variables: vram_headroom, bw_per_param, etc.>"\n'
        "    },\n"
        "    ... 4 more proposals\n"
        "  ],\n"
        '  "overall_reasoning": "<summary of your ranking logic across all 5>"\n'
        "}"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Judge user prompt
# ---------------------------------------------------------------------------

def _build_judge_user_prompt(
    request: JobRequest,
    thinker_results: List[ThinkerResult],
    resource_map: ResourceMap,
) -> str:
    job_ctx = _build_job_context(request)
    avail = _build_available_resources(resource_map)

    proposals_text = ""
    for tr in thinker_results:
        proposals_text += f"\n=== {tr.thinker_id} (directive: {tr.directive}) ===\n"
        for p in tr.proposals:
            cfg = p.proposed_config
            est = p.oracle_estimate
            tps_str = f"~{est.throughput_tokens_per_sec:.0f} tok/s conf={est.confidence:.0%}" if est else "no estimate"
            slo_str = ""
            if est and est.tpot_ms and request.slo_tpot_ms:
                margin = (request.slo_tpot_ms - est.tpot_ms) / request.slo_tpot_ms * 100
                slo_str = f" SLO_margin={margin:.0f}%"
            proposals_text += (
                f"  Rank {p.rank}: {cfg.gpu_type} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} "
                f"({cfg.num_gpus}GPUs{', quant=' + cfg.engine_config.quantization if cfg.engine_config.quantization else ''}) | "
                f"{tps_str}{slo_str}\n"
                f"    Hyp: {p.hypothesis}\n"
                f"    Mech: {p.mechanism}\n"
                f"    Evidence: {p.evidence}\n"
                f"    Conf: {p.confidence:.0%} | Reasoning: {p.reasoning[:200]}\n"
            )
        proposals_text += f"  Overall: {tr.overall_reasoning[:200]}\n"

    return f"""{job_ctx}

{avail}

ALL PROPOSALS (15 total — 5 per LLM):
{proposals_text}

Rank these 15 proposals into the TOP 5 best placements for Orca.
You may synthesize a novel config that combines insights from multiple proposals.
Consider: RAG grounding, physical reasoning quality, SLO compliance, cost.

Return ONLY valid JSON:
{{
  "top_placements": [
    {{
      "rank": 1,
      "config": {{
        "gpu_type": "<str>",
        "tp": <int>,
        "pp": <int>,
        "dp": <int>,
        "num_gpus": <int>,
        "instance_type": "<str>",
        "quantization": <null|"fp8"|"int8">
      }},
      "source": "<e.g. LLM1_rank1 or LLM2_rank3 or synthesis>",
      "reasoning": "<why this ranks here — cite specific physics variables>",
      "confidence": <0.0-1.0>,
      "meets_slo": <true|false>
    }},
    ... 4 more (ranks 2-5)
  ],
  "decision_source": "<source of rank-1 config>",
  "synthesis_reasoning": "<if any rank is synthesis: what you combined and why>",
  "exploration_queue": [
    {{
      "thinker_id": "<LLM1|LLM2|LLM3>",
      "proposal_rank": <int 1-5>,
      "priority": "<high|medium|low>",
      "reason": "<why worth testing on a future job>"
    }}
  ],
  "most_novel_hypothesis_thinker": "<LLM1|LLM2|LLM3>",
  "most_novel_hypothesis_summary": "<one sentence>",
  "reasoning": "<full synthesis reasoning — 4-8 sentences>",
  "confidence": <0.0-1.0>,
  "agreement": "<full|partial|split>"
}}"""


# ---------------------------------------------------------------------------
# Diagnosis user prompt
# ---------------------------------------------------------------------------

def _build_diagnosis_user_prompt(
    request: JobRequest,
    current_config: PlacementConfig,
    monitoring_trace: List[RuntimeMetrics],
    resource_map: ResourceMap,
    mf: ModelFeatures,
    history: Optional[str] = None,
) -> str:
    job_ctx = _build_job_context(request)
    hw_ref = _build_hardware_reference(resource_map)
    constraints = _build_parallelism_constraints(mf)

    cfg = current_config
    trace_lines = [
        f"MONITORING TRACE (last {len(monitoring_trace)} samples):",
        f"Current config: {cfg.gpu_type} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} ({cfg.num_gpus} GPUs)",
        "",
    ]
    for m in monitoring_trace[-12:]:
        tpot_part = f"TPOT={m.tpot_ms:.1f}ms " if m.tpot_ms else ""
        conc_part = f"conc={m.concurrent_requests} " if m.concurrent_requests else ""
        trace_lines.append(
            f"  [{m.timestamp.strftime('%H:%M:%S')}] "
            f"TPS={m.throughput_tokens_per_sec:.0f} {tpot_part}"
            f"GPU_util={m.gpu_utilization_pct:.0f}% "
            f"GPU_mem={m.gpu_memory_used_gb:.1f}GB {conc_part}"
        )

    slo_ctx = []
    if request.slo_tpot_ms:
        slo_ctx.append(f"SLO TPOT: {request.slo_tpot_ms}ms")
    if request.slo_deadline_hours:
        slo_ctx.append(f"SLO deadline: {request.slo_deadline_hours}h")

    hist = _build_history_context(history)

    return f"""{job_ctx}

{hw_ref}

{_build_model_features_context(mf)}

{constraints}

{chr(10).join(trace_lines)}

{chr(10).join(slo_ctx)}

{hist}

Diagnose and propose a repair config.

Return ONLY valid JSON:
{{
  "failure_mode": "<specific description of what the trace shows is failing>",
  "causal_rule": "<generalizable rule — e.g. 'For X models at TP>N on PCIe: ...'>",
  "proposed_config": {{
    "gpu_type": "<str>", "tp": <int>, "pp": <int>, "dp": <int>,
    "num_gpus": <int>, "instance_type": "<str>", "quantization": <null|"fp8"|"int8">
  }},
  "repair_hypothesis": "<why the new config addresses the diagnosed failure>",
  "expected_improvement": "<quantified: e.g. 'GPU mem drops from 94% to ~65%'>",
  "confidence": <0.0-1.0>,
  "reasoning": "<full diagnosis chain, 4-8 sentences>"
}}"""


# ---------------------------------------------------------------------------
# Guardrail validation
# ---------------------------------------------------------------------------

def _validate_proposed_config(
    proposed: Dict,
    request: JobRequest,
    resource_map: ResourceMap,
    mf: ModelFeatures,
) -> Tuple[bool, str, Optional[PlacementConfig]]:
    gpu_type = proposed.get("gpu_type", "")
    try:
        tp = int(proposed.get("tp", 1))
        pp = int(proposed.get("pp", 1))
        dp = int(proposed.get("dp", 1))
    except (TypeError, ValueError):
        return False, "tp/pp/dp must be integers", None

    num_gpus_claimed = int(proposed.get("num_gpus", tp * pp * dp))
    quantization = proposed.get("quantization")
    instance_type = proposed.get("instance_type", "")

    resource = resource_map.get_resource(gpu_type)
    if resource is None:
        return False, f"GPU '{gpu_type}' not in resource map. Available: {resource_map.available_gpu_types()}", None

    total_needed = tp * pp * dp
    if num_gpus_claimed != total_needed:
        return False, f"num_gpus={num_gpus_claimed} ≠ TP×PP×DP={total_needed}", None
    if total_needed > resource.available_gpus:
        return False, f"Need {total_needed} {gpu_type} GPUs but only {resource.available_gpus} available", None

    # Parallelism
    if mf.num_attention_heads % tp != 0:
        return False, f"TP={tp} does not divide num_attention_heads={mf.num_attention_heads}", None
    if mf.num_layers % pp != 0:
        return False, f"PP={pp} does not divide num_layers={mf.num_layers}", None

    # Memory
    weight_per_gpu = mf.model_size_gb / max(tp, 1)
    headroom = resource.gpu_memory_gb - weight_per_gpu
    if headroom < 8.0:
        return False, (
            f"Only {headroom:.1f}GB headroom after weights ({weight_per_gpu:.1f}GB/GPU, "
            f"{resource.gpu_memory_gb}GB VRAM). Increase TP or use quantization."
        ), None

    # Instance type sanity
    if instance_type and instance_type in INSTANCE_TO_GPU:
        if INSTANCE_TO_GPU[instance_type] != gpu_type:
            return False, f"instance_type '{instance_type}' maps to {INSTANCE_TO_GPU[instance_type]}, not {gpu_type}", None
    if not instance_type or instance_type not in INSTANCE_TO_GPU:
        instance_type = resource.instance_type

    num_instances = max(1, total_needed // resource.gpus_per_instance)

    config = PlacementConfig(
        gpu_type=gpu_type,
        instance_type=instance_type,
        num_gpus=total_needed,
        num_instances=num_instances,
        tp=tp, pp=pp, dp=dp,
        region=resource_map.region,
        engine_config=EngineConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            quantization=quantization,
        ),
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
    rag_records: List[Dict],
    rag: PerfRAG,
    mf: ModelFeatures,
    history: Optional[str] = None,
    frontier_summary: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> ThinkerResult:
    """
    One LLM call → 5 ranked placement proposals.
    Guardrails are applied per-proposal; invalid ones are noted but don't retry the whole call.
    """
    prior_rejection: Optional[str] = None
    rejections: List[str] = []

    for attempt in range(MAX_GUARDRAIL_RETRIES + 1):
        user_prompt = _build_thinker_user_prompt(
            request=request,
            resource_map=resource_map,
            mf=mf,
            candidates=candidates,
            rag_records=rag_records,
            rag=rag,
            directive=directive,
            history=history,
            frontier_summary=frontier_summary,
            prior_rejection=prior_rejection,
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=2500,
                system=THINKER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
                # also strip trailing ```
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            data = json.loads(raw)
            raw_proposals = data.get("proposals", [])

            valid_proposals: List[ThinkerProposal] = []
            invalid_notes: List[str] = []

            for raw_p in raw_proposals[:5]:
                rank = int(raw_p.get("rank", len(valid_proposals) + 1))
                cfg_raw = raw_p.get("proposed_config", {})
                ok, rejection, config = _validate_proposed_config(cfg_raw, request, resource_map, mf)

                if not ok:
                    invalid_notes.append(f"rank={rank}: {rejection}")
                    continue

                oracle_est = oracle.estimate_for_config(
                    request=request,
                    resource=resource_map.get_resource(config.gpu_type),
                    tp=config.tp, pp=config.pp, dp=config.dp,
                    rag_records=rag_records,
                )

                valid_proposals.append(ThinkerProposal(
                    thinker_id=thinker_id,
                    directive=directive["id"],
                    rank=rank,
                    proposed_config=config,
                    oracle_estimate=oracle_est,
                    hypothesis=raw_p.get("hypothesis", ""),
                    mechanism=raw_p.get("mechanism", ""),
                    evidence=raw_p.get("evidence", ""),
                    falsification_condition=raw_p.get("falsification_condition", ""),
                    confidence=float(raw_p.get("confidence", 0.6)),
                    reasoning=raw_p.get("reasoning", ""),
                ))

            if valid_proposals:
                # Re-rank by original rank field
                valid_proposals.sort(key=lambda p: p.rank)
                for i, p in enumerate(valid_proposals):
                    p.rank = i + 1

                if invalid_notes:
                    prior_rejection = "; ".join(invalid_notes)
                    rejections += invalid_notes
                    print(f"[Ensemble] {thinker_id} attempt {attempt+1}: {len(valid_proposals)}/5 valid, invalid: {invalid_notes}")

                print(
                    f"[Ensemble] {thinker_id} ({directive['id']}): "
                    f"{len(valid_proposals)} proposals — "
                    + ", ".join(f"[{p.rank}]{p.proposed_config.gpu_type}TP{p.proposed_config.tp}PP{p.proposed_config.pp}"
                                for p in valid_proposals)
                )
                return ThinkerResult(
                    thinker_id=thinker_id,
                    directive=directive["id"],
                    proposals=valid_proposals,
                    overall_reasoning=data.get("overall_reasoning", ""),
                    guardrail_rejections=rejections,
                )

            # All 5 invalid — retry with rejection feedback
            prior_rejection = "; ".join(invalid_notes) if invalid_notes else "All proposals failed guardrails"
            rejections.append(f"Attempt {attempt+1}: all proposals invalid — {prior_rejection}")
            print(f"[Ensemble] {thinker_id} attempt {attempt+1}: all 5 invalid, retrying")

        except Exception as e:
            prior_rejection = f"Error: {e}"
            rejections.append(f"Attempt {attempt+1}: {e}")
            print(f"[Ensemble] {thinker_id} attempt {attempt+1} error: {e}")

    # Fallback: use top oracle candidates
    print(f"[Ensemble] {thinker_id}: all retries failed, using oracle fallback")
    fallback_proposals = []
    for i, c in enumerate(candidates[:5]):
        fallback_proposals.append(ThinkerProposal(
            thinker_id=thinker_id,
            directive=directive["id"],
            rank=i + 1,
            proposed_config=c.config,
            oracle_estimate=c.metrics,
            hypothesis="Fallback: guardrail retries exhausted",
            mechanism="N/A",
            evidence="Oracle candidate",
            falsification_condition="N/A",
            confidence=0.25,
            reasoning=f"Fallback after {MAX_GUARDRAIL_RETRIES + 1} failed attempts. {rejections}",
            guardrail_rejections=rejections,
        ))
    return ThinkerResult(
        thinker_id=thinker_id,
        directive=directive["id"],
        proposals=fallback_proposals,
        overall_reasoning="Fallback to oracle candidates",
        guardrail_rejections=rejections,
    )


async def _call_judge(
    client: AsyncAnthropic,
    thinker_results: List[ThinkerResult],
    request: JobRequest,
    resource_map: ResourceMap,
    oracle: Oracle,
    mf: ModelFeatures,
    rag_records: List[Dict],
    model: str = DEFAULT_MODEL,
) -> JudgeDecision:
    """
    Judge ranks all 15 proposals → top 5 for Orca.
    """
    user_prompt = _build_judge_user_prompt(request, thinker_results, resource_map)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=3000,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        data = json.loads(raw)

        # Parse top 5 placements
        top_placements: List[RankedPlacement] = []
        for rp in data.get("top_placements", [])[:5]:
            cfg_raw = rp.get("config", {})
            ok, rejection, config = _validate_proposed_config(cfg_raw, request, resource_map, mf)
            if not ok:
                print(f"[Ensemble] Judge rank {rp.get('rank')} invalid: {rejection}")
                continue

            oracle_est = oracle.estimate_for_config(
                request=request,
                resource=resource_map.get_resource(config.gpu_type),
                tp=config.tp, pp=config.pp, dp=config.dp,
                rag_records=rag_records,
            )

            top_placements.append(RankedPlacement(
                rank=int(rp.get("rank", len(top_placements) + 1)),
                config=config,
                oracle_estimate=oracle_est,
                source=str(rp.get("source", "unknown")),
                reasoning=str(rp.get("reasoning", "")),
                confidence=float(rp.get("confidence", 0.6)),
                meets_slo=bool(rp.get("meets_slo", True)),
            ))

        # Sort and re-rank
        top_placements.sort(key=lambda p: p.rank)
        for i, p in enumerate(top_placements):
            p.rank = i + 1

        # Fallback if judge returned nothing valid
        if not top_placements:
            print("[Ensemble] Judge produced no valid placements, using thinker proposals")
            all_proposals = [p for tr in thinker_results for p in tr.proposals]
            all_proposals.sort(key=lambda p: (-p.confidence, p.rank))
            for i, p in enumerate(all_proposals[:5]):
                top_placements.append(RankedPlacement(
                    rank=i + 1,
                    config=p.proposed_config,
                    oracle_estimate=p.oracle_estimate,
                    source=f"{p.thinker_id}_rank{p.rank}",
                    reasoning=p.reasoning[:300],
                    confidence=p.confidence,
                    meets_slo=True,
                ))

        # Exploration queue
        exploration_queue: List[ExplorationQueueEntry] = []
        for entry in data.get("exploration_queue", [])[:5]:
            tid = entry.get("thinker_id", "")
            prank = int(entry.get("proposal_rank", 1))
            matching_tr = next((tr for tr in thinker_results if tr.thinker_id == tid), None)
            if matching_tr:
                matching_p = next((p for p in matching_tr.proposals if p.rank == prank), None)
                if matching_p:
                    exploration_queue.append(ExplorationQueueEntry(
                        proposal=matching_p,
                        priority=entry.get("priority", "medium"),
                        reason=entry.get("reason", ""),
                    ))

        deploy_config = top_placements[0].config if top_placements else thinker_results[0].proposals[0].proposed_config
        deploy_est = top_placements[0].oracle_estimate if top_placements else None

        print(
            f"[Ensemble] Judge → top 5: "
            + " | ".join(
                f"[{p.rank}]{p.config.gpu_type}TP{p.config.tp}PP{p.config.pp} conf={p.confidence:.0%}"
                for p in top_placements
            )
        )

        return JudgeDecision(
            decision_source=data.get("decision_source", "proposal"),
            deployment_config=deploy_config,
            deployment_oracle_estimate=deploy_est,
            top_placements=top_placements,
            synthesis_reasoning=data.get("synthesis_reasoning", ""),
            exploration_queue=exploration_queue,
            most_novel_hypothesis_thinker=data.get("most_novel_hypothesis_thinker", ""),
            most_novel_hypothesis_summary=data.get("most_novel_hypothesis_summary", ""),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.7)),
            agreement=data.get("agreement", "partial"),
        )

    except Exception as e:
        print(f"[Ensemble] Judge failed: {e}. Using top proposals as fallback.")
        all_proposals = [p for tr in thinker_results for p in tr.proposals]
        all_proposals.sort(key=lambda p: (-p.confidence, p.rank))
        top_placements = []
        for i, p in enumerate(all_proposals[:5]):
            top_placements.append(RankedPlacement(
                rank=i + 1,
                config=p.proposed_config,
                oracle_estimate=p.oracle_estimate,
                source=f"{p.thinker_id}_fallback",
                reasoning=p.reasoning[:200],
                confidence=p.confidence * 0.8,
                meets_slo=True,
            ))
        fallback_config = top_placements[0].config
        return JudgeDecision(
            decision_source="fallback",
            deployment_config=fallback_config,
            top_placements=top_placements,
            reasoning=f"Judge fallback due to error: {e}",
            confidence=0.3,
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
    mf: ModelFeatures,
    history: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> DiagnosisProposal:
    user_prompt = _build_diagnosis_user_prompt(
        request, current_config, monitoring_trace, resource_map, mf, history
    )

    for attempt in range(MAX_GUARDRAIL_RETRIES + 1):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=1200,
                system=DIAGNOSIS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
                if raw.endswith("```"):
                    raw = raw[:-3].strip()

            data = json.loads(raw)
            cfg_raw = data["proposed_config"]

            ok, rejection, config = _validate_proposed_config(cfg_raw, request, resource_map, mf)
            if not ok:
                user_prompt += f"\n\nPREVIOUS PROPOSAL REJECTED: {rejection}\nAdjust repair config."
                continue

            oracle_est = oracle.estimate_for_config(
                request=request,
                resource=resource_map.get_resource(config.gpu_type),
                tp=config.tp, pp=config.pp, dp=config.dp,
            )

            print(f"[Ensemble] {thinker_id} diagnosis → {config.gpu_type} TP={config.tp} PP={config.pp}")
            return DiagnosisProposal(
                thinker_id=thinker_id,
                failure_mode=data.get("failure_mode", ""),
                causal_rule=data.get("causal_rule", ""),
                proposed_config=config,
                oracle_estimate=oracle_est,
                repair_hypothesis=data.get("repair_hypothesis", ""),
                expected_improvement=data.get("expected_improvement", ""),
                confidence=float(data.get("confidence", 0.6)),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            print(f"[Ensemble] {thinker_id} diagnosis attempt {attempt+1} error: {e}")

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
# KoiEnsemble
# ---------------------------------------------------------------------------

class KoiEnsemble:
    """
    Runs the full LLM proposal pipeline:
      Initial: 3 LLMs × 5 proposals → judge → top 5 for Orca
      Reconfig: 3 LLMs diagnose failure → judge picks best repair
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
        oracle_result: OracleResult,
        history: Optional[str] = None,
        frontier_summary: Optional[str] = None,
    ) -> PlacementDecision:
        """
        Initial placement: 3 LLMs propose 5 each → judge → top 5 in PlacementDecision.
        """
        candidates = oracle_result.candidates
        rag_records = oracle_result.rag_records
        mf: ModelFeatures = oracle_result.model_features or get_model_features(request.model_name)
        rag = self.oracle.rag

        if not candidates and not resource_map.available_gpu_types():
            raise ValueError("No available GPU resources in VPC")

        client = AsyncAnthropic(api_key=self.api_key)
        directives = select_directives(self.n_thinkers)

        print(f"[Ensemble] {self.n_thinkers} thinkers | directives: {[d['id'] for d in directives]}")
        print(f"[Ensemble] {len(rag_records)} RAG records, {len(candidates)} oracle candidates")

        tasks = [
            _call_one_thinker(
                client=client,
                thinker_id=f"LLM{i+1}",
                directive=directives[i],
                request=request,
                resource_map=resource_map,
                oracle=self.oracle,
                candidates=candidates,
                rag_records=rag_records,
                rag=rag,
                mf=mf,
                history=history,
                frontier_summary=frontier_summary,
                model=self.model,
            )
            for i in range(self.n_thinkers)
        ]
        thinker_results: List[ThinkerResult] = list(await asyncio.gather(*tasks))

        print(f"[Ensemble] Running judge over {sum(len(tr.proposals) for tr in thinker_results)} proposals...")
        judge = await _call_judge(
            client=client,
            thinker_results=thinker_results,
            request=request,
            resource_map=resource_map,
            oracle=self.oracle,
            mf=mf,
            rag_records=rag_records,
            model=self.model,
        )

        deploy_config = judge.deployment_config
        deploy_metrics = judge.deployment_oracle_estimate or self.oracle.estimate_for_config(
            request=request,
            resource=resource_map.get_resource(deploy_config.gpu_type),
            tp=deploy_config.tp, pp=deploy_config.pp, dp=deploy_config.dp,
            rag_records=rag_records,
        )

        flat_proposals = [p for tr in thinker_results for p in tr.proposals]

        return PlacementDecision(
            job_id=request.job_id,
            model_name=request.model_name,
            recommendation=deploy_config,
            predicted_metrics=deploy_metrics,
            top_placements=judge.top_placements,
            reasoning=judge.reasoning + (
                f"\n\nSynthesis: {judge.synthesis_reasoning}" if judge.synthesis_reasoning else ""
            ),
            confidence=judge.confidence,
            thinker_results=thinker_results,
            thinker_proposals=flat_proposals,
            judge_decision=judge,
            exploration_queue=judge.exploration_queue,
            total_llm_calls=self.n_thinkers + 1,
            oracle_candidates_evaluated=len(candidates),
            rag_records_retrieved=len(rag_records),
            is_reconfig=False,
            triggered_by="initial",
        )

    def run_sync(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
        oracle_result: OracleResult,
        history: Optional[str] = None,
        frontier_summary: Optional[str] = None,
    ) -> PlacementDecision:
        return asyncio.run(self.run(
            request, resource_map, oracle_result,
            history=history, frontier_summary=frontier_summary,
        ))

    async def run_diagnosis(
        self,
        request: JobRequest,
        current_config: PlacementConfig,
        monitoring_trace: List[RuntimeMetrics],
        resource_map: ResourceMap,
        history: Optional[str] = None,
    ) -> PlacementDecision:
        mf = get_model_features(request.model_name)
        client = AsyncAnthropic(api_key=self.api_key)

        print(f"[Ensemble] Diagnosis ({len(monitoring_trace)} trace samples)...")
        tasks = [
            _call_diagnosis(
                client=client,
                thinker_id=f"LLM{i+1}",
                request=request,
                current_config=current_config,
                monitoring_trace=monitoring_trace,
                resource_map=resource_map,
                oracle=self.oracle,
                mf=mf,
                history=history,
                model=self.model,
            )
            for i in range(self.n_thinkers)
        ]
        diagnoses = list(await asyncio.gather(*tasks))

        # Convert to ThinkerProposals for the judge
        thinker_results = [
            ThinkerResult(
                thinker_id=d.thinker_id,
                directive="diagnosis",
                proposals=[ThinkerProposal(
                    thinker_id=d.thinker_id,
                    directive="diagnosis",
                    rank=1,
                    proposed_config=d.proposed_config,
                    oracle_estimate=d.oracle_estimate,
                    hypothesis=d.repair_hypothesis,
                    mechanism=d.failure_mode,
                    evidence=d.causal_rule,
                    falsification_condition="Same failure reappears within 30 min",
                    confidence=d.confidence,
                    reasoning=d.reasoning,
                )],
                overall_reasoning=d.reasoning,
            )
            for d in diagnoses
        ]

        judge = await _call_judge(
            client=client,
            thinker_results=thinker_results,
            request=request,
            resource_map=resource_map,
            oracle=self.oracle,
            mf=mf,
            rag_records=[],
            model=self.model,
        )

        deploy_config = judge.deployment_config
        deploy_metrics = judge.deployment_oracle_estimate or self.oracle.estimate_for_config(
            request=request,
            resource=resource_map.get_resource(deploy_config.gpu_type),
            tp=deploy_config.tp, pp=deploy_config.pp, dp=deploy_config.dp,
        )

        flat_proposals = [p for tr in thinker_results for p in tr.proposals]

        return PlacementDecision(
            job_id=request.job_id,
            model_name=request.model_name,
            recommendation=deploy_config,
            predicted_metrics=deploy_metrics,
            top_placements=judge.top_placements,
            reasoning=judge.reasoning,
            confidence=judge.confidence,
            thinker_results=thinker_results,
            thinker_proposals=flat_proposals,
            judge_decision=judge,
            total_llm_calls=self.n_thinkers + 1,
            is_reconfig=True,
            triggered_by="monitoring_hard",
        )

    def run_diagnosis_sync(
        self,
        request: JobRequest,
        current_config: PlacementConfig,
        monitoring_trace: List[RuntimeMetrics],
        resource_map: ResourceMap,
        history: Optional[str] = None,
    ) -> PlacementDecision:
        return asyncio.run(self.run_diagnosis(
            request, current_config, monitoring_trace, resource_map, history=history
        ))
