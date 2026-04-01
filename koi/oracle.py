"""
koi/oracle.py — Oracle: feasibility checks + RAG-based performance estimation.

The Oracle is the pure-numerical layer. Zero LLM calls here.
It runs BEFORE the ensemble so LLMs receive pre-pruned, pre-scored
candidate configs alongside the raw RAG evidence that informed those scores.

Architecture post-RAG refactor:
  1. get_candidates() calls PerfRAG.retrieve_multi_query() to pull the top-k
     most physically similar records from the performance DB.
  2. For each available GPU in the ResourceMap, it enumerates valid (TP, PP, DP)
     combos (memory + parallelism feasibility), estimates performance from the
     RAG records via estimate_for_config(), and returns OracleCandidate objects.
  3. The raw RAG records are also returned (in OracleResult) so the ensemble
     LLMs can read the actual observed data, not just the Oracle's interpolation.

estimate_for_config() uses a 3-tier approach:
  Tier 1: Direct RAG hit  — same GPU + similar TP/PP + similar I/O → scale only
  Tier 2: Cross-GPU RAG   — different GPU, scale by bandwidth/compute ratio
  Tier 3: Analytical      — pure roofline, no data needed

Feasibility checks (unchanged from prior version):
  - Memory: model weights ÷ TP must leave ≥ 8 GB headroom for KV cache
  - Parallelism: TP must divide num_attention_heads; PP must divide num_layers
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from koi.model_features import (
    ModelFeatures,
    _GPU_SPECS,
    compute_config_features,
    get_model_features,
)
from koi.perf_rag import PerfRAG, _safe_float
from koi.schemas import (
    DataSource,
    EngineConfig,
    GPUResource,
    JobRequest,
    OracleCandidate,
    OracleResult,
    PlacementConfig,
    PredictedMetrics,
    ResourceMap,
    TaskType,
)

# AWS instance → GPU type (kept for guardrail use in ensemble.py)
INSTANCE_TO_GPU: Dict[str, str] = {
    "g6e.xlarge":    "L40S",  "g6e.2xlarge":   "L40S",  "g6e.4xlarge":   "L40S",
    "g6e.8xlarge":   "L40S",  "g6e.12xlarge":  "L40S",  "g6e.16xlarge":  "L40S",
    "g6e.24xlarge":  "L40S",  "g6e.48xlarge":  "L40S",
    "p4d.24xlarge":  "A100-80GB",  "p4de.24xlarge": "A100-80GB",
    "p5.48xlarge":   "H100",  "p5e.48xlarge":  "H100",
    "p5en.48xlarge": "H200",
    "g5.xlarge":     "A10G",  "g5.2xlarge":    "A10G",  "g5.4xlarge":    "A10G",
    "g5.8xlarge":    "A10G",  "g5.12xlarge":   "A10G",  "g5.16xlarge":   "A10G",
    "g5.48xlarge":   "A10G",
    "g6.xlarge":     "L4",    "g6.2xlarge":    "L4",    "g6.4xlarge":    "L4",
    "g6.8xlarge":    "L4",    "g6.16xlarge":   "L4",
}

GPU_SPECS = _GPU_SPECS  # re-export for ensemble.py compatibility

BYTES_PER_PARAM = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

# TP values to enumerate during candidate generation
_TP_CANDIDATES = [1, 2, 4, 8, 16]
# PP values to enumerate
_PP_CANDIDATES = [1, 2, 4]


class Oracle:
    """
    Performance estimation + feasibility pruning engine.

    Usage:
        oracle = Oracle(perfdb_path="./perfdb")
        result = oracle.get_candidates(request, resource_map)
        # result.candidates: List[OracleCandidate] (feasible, estimated)
        # result.rag_records: List[Dict] (raw retrieved perf DB records)
        # result.model_features: ModelFeatures
    """

    def __init__(self, perfdb_path: str = "./perfdb"):
        self.perfdb_path = Path(perfdb_path)
        self.rag = PerfRAG(csv_dir=str(self.perfdb_path))

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
    ) -> "OracleResult":
        """
        Generate feasible placement candidates + raw RAG evidence.

        Steps:
          1. Get model architecture features
          2. Retrieve top-10 RAG records across available GPU types
          3. Enumerate (GPU × TP × PP × DP) combos, filter by memory + parallelism
          4. Estimate metrics for each feasible combo via RAG-based interpolation
          5. Return OracleResult bundling candidates + raw records
        """
        mf = get_model_features(request.model_name)

        available_gpus = resource_map.available_gpu_types()
        if not available_gpus:
            return OracleResult(candidates=[], rag_records=[], model_features=mf)

        tp_options = _TP_CANDIDATES
        pp_options = _PP_CANDIDATES

        # Step 2: retrieve RAG evidence
        rag_records = self.rag.retrieve_multi_query(
            model_name=request.model_name,
            num_params_billions=mf.num_params_billions,
            is_moe=mf.is_moe,
            available_gpu_types=available_gpus,
            tp_options=tp_options,
            pp_options=pp_options,
            input_len=request.avg_input_tokens,
            output_len=request.avg_output_tokens,
            dtype=mf.dtype,
            k=10,
        )

        # Step 3+4: enumerate feasible configs + estimate metrics
        candidates: List[OracleCandidate] = []

        for gpu_res in resource_map.resources:
            if gpu_res.available_gpus <= 0:
                continue

            for tp in tp_options:
                for pp in _PP_CANDIDATES:
                    for dp in [1, 2]:
                        total_gpus = tp * pp * dp
                        if total_gpus > gpu_res.available_gpus:
                            continue
                        if request.max_total_gpus and total_gpus > request.max_total_gpus:
                            continue

                        # Memory feasibility
                        mem_ok, headroom, note = self._check_memory(
                            mf, gpu_res, tp
                        )
                        if not mem_ok:
                            continue

                        # Parallelism feasibility
                        par_ok, par_msg = self._check_parallelism(mf, tp, pp)
                        if not par_ok:
                            continue

                        # Estimate metrics
                        metrics = self._estimate_metrics(
                            request, mf, gpu_res, tp, pp, dp, rag_records
                        )
                        if metrics is None:
                            continue

                        # SLO check
                        meets_slo, slo_margin = self._check_slo(request, metrics)

                        num_instances = max(1, total_gpus // gpu_res.gpus_per_instance)
                        config = PlacementConfig(
                            gpu_type=gpu_res.gpu_type,
                            instance_type=gpu_res.instance_type,
                            num_gpus=total_gpus,
                            num_instances=num_instances,
                            tp=tp,
                            pp=pp,
                            dp=dp,
                            region=resource_map.region,
                            engine_config=EngineConfig(
                                tensor_parallel_size=tp,
                                pipeline_parallel_size=pp,
                                max_num_seqs=min(256, max(8, int(headroom / 2))),
                            ),
                        )

                        candidates.append(OracleCandidate(
                            config=config,
                            metrics=metrics,
                            meets_slo=meets_slo,
                            slo_margin_pct=slo_margin,
                            feasibility_notes=[note],
                        ))

        # Sort: SLO-meeting first, then by cost
        candidates.sort(key=lambda c: (not c.meets_slo, c.metrics.cost_per_hour_usd))

        print(
            f"[Oracle] {len(candidates)} feasible candidates "
            f"({sum(1 for c in candidates if c.meets_slo)} meet SLO) | "
            f"{len(rag_records)} RAG records retrieved"
        )
        return OracleResult(
            candidates=candidates,
            rag_records=rag_records,
            model_features=mf,
        )

    # ------------------------------------------------------------------
    # Post-proposal estimation (called by ensemble after guardrail check)
    # ------------------------------------------------------------------

    def estimate_for_config(
        self,
        request: JobRequest,
        resource: Optional[GPUResource],
        tp: int,
        pp: int,
        dp: int,
        rag_records: Optional[List[Dict]] = None,
    ) -> PredictedMetrics:
        """
        Estimate metrics for a specific config proposed by an LLM.
        Called AFTER guardrail validation — resource is known to be available.
        If rag_records is None, runs a fresh retrieval.
        """
        if resource is None:
            # Return minimal analytical estimate
            return self._analytical_metrics(request, "unknown", tp, pp, dp)

        mf = get_model_features(request.model_name)

        if rag_records is None:
            rag_records = self.rag.retrieve(
                self.rag.build_query_text(
                    model_name=request.model_name,
                    num_params_billions=mf.num_params_billions,
                    is_moe=mf.is_moe,
                    gpu_type=resource.gpu_type,
                    tp=tp, pp=pp, dp=dp,
                    input_len=request.avg_input_tokens,
                    output_len=request.avg_output_tokens,
                    dtype=mf.dtype,
                ),
                k=5,
            )

        result = self._estimate_metrics(request, mf, resource, tp, pp, dp, rag_records)
        if result is None:
            result = self._analytical_metrics(request, resource.gpu_type, tp, pp, dp)
        return result

    # ------------------------------------------------------------------
    # Model / arch helpers (called by ensemble.py for prompt building)
    # ------------------------------------------------------------------

    def get_model_spec(self, model_name: str) -> Dict[str, Any]:
        """Return model architectural spec dict (backward compat with ensemble.py)."""
        mf = get_model_features(model_name)
        return {
            "params_billion": mf.num_params_billions,
            "num_layers": mf.num_layers,
            "num_attention_heads": mf.num_attention_heads,
            "num_kv_heads": mf.num_kv_heads,
            "is_moe": mf.is_moe,
            "num_experts": mf.num_experts,
            "active_experts": mf.active_experts,
            "hidden_dim": mf.hidden_dim,
            "vocab_size": mf.vocab_size,
            "architecture_family": mf.architecture_family,
        }

    # ------------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------------

    def _check_memory(
        self,
        mf: ModelFeatures,
        gpu: GPUResource,
        tp: int,
        min_headroom_gb: float = 8.0,
    ) -> Tuple[bool, float, str]:
        weight_per_gpu = mf.model_size_gb / max(tp, 1)
        headroom = gpu.gpu_memory_gb - weight_per_gpu
        if headroom < min_headroom_gb:
            return (
                False, headroom,
                f"Only {headroom:.1f}GB headroom ({weight_per_gpu:.1f}GB weights, {gpu.gpu_memory_gb}GB VRAM)"
            )
        return True, headroom, f"{headroom:.1f}GB headroom"

    def _check_parallelism(
        self, mf: ModelFeatures, tp: int, pp: int
    ) -> Tuple[bool, str]:
        if mf.num_attention_heads % tp != 0:
            return False, f"TP={tp} does not divide num_attention_heads={mf.num_attention_heads}"
        if mf.num_layers % pp != 0:
            return False, f"PP={pp} does not divide num_layers={mf.num_layers}"
        return True, "OK"

    # ------------------------------------------------------------------
    # Metric estimation (RAG-based)
    # ------------------------------------------------------------------

    def _estimate_metrics(
        self,
        request: JobRequest,
        mf: ModelFeatures,
        gpu: GPUResource,
        tp: int,
        pp: int,
        dp: int,
        rag_records: List[Dict],
    ) -> Optional[PredictedMetrics]:
        """
        Estimate throughput + latency using RAG records.

        Tier 1: Same GPU family + same TP + similar I/O → direct scale by I/O ratio
        Tier 2: Different GPU → scale by bandwidth ratio (decode) + TFLOPS (prefill)
        Tier 3: Analytical roofline (lowest confidence)
        """
        num_gpus = tp * pp * dp
        cost_per_hour = gpu.cost_per_gpu_hour_usd * num_gpus
        gpu_spec = _GPU_SPECS.get(gpu.gpu_type, {"bandwidth_gbps": 400, "fp16_tflops": 300, "mem_gb": 40})

        # --- Tier 1: same GPU + same TP ---
        tier1 = [
            r for r in rag_records
            if _gpu_matches(str(r.get("gpu_type", "")), gpu.gpu_type)
            and int(r.get("tp", 1) or 1) == tp
            and int(r.get("pp", 1) or 1) == pp
        ]
        if tier1:
            best = min(tier1, key=lambda r: abs(
                _safe_float(r.get("input_len"), 512) - request.avg_input_tokens
            ) + abs(_safe_float(r.get("output_len"), 128) - request.avg_output_tokens))
            tps, conf, source = self._scale_from_record(
                best, request, mf, dp, gpu_spec, DataSource.INTERPOLATED, 0.85
            )
            return self._build_metrics(
                tps, conf, source, cost_per_hour, num_gpus, gpu,
                mf, tp, pp, dp, request, best
            )

        # --- Tier 2: different GPU, same TP+PP ---
        tier2 = [
            r for r in rag_records
            if int(r.get("tp", 1) or 1) == tp
            and int(r.get("pp", 1) or 1) == pp
        ]
        if tier2:
            best = tier2[0]  # already sorted by RAG similarity
            src_gpu = str(best.get("gpu_type", ""))
            src_spec = _GPU_SPECS.get(src_gpu, gpu_spec)
            bw_scale = gpu_spec["bandwidth_gbps"] / max(src_spec["bandwidth_gbps"], 1)
            fl_scale = gpu_spec["fp16_tflops"] / max(src_spec["fp16_tflops"], 1)
            pdr = request.prefill_decode_ratio
            prefill_w = min(1.0, pdr / 4.0)
            gpu_scale = (1 - prefill_w) * bw_scale + prefill_w * fl_scale

            tps, conf, source = self._scale_from_record(
                best, request, mf, dp, gpu_spec, DataSource.CROSS_GPU, 0.60
            )
            tps *= gpu_scale
            return self._build_metrics(
                tps, conf, source, cost_per_hour, num_gpus, gpu,
                mf, tp, pp, dp, request, best
            )

        # --- Tier 3: closest RAG record regardless of TP/PP ---
        if rag_records:
            best = rag_records[0]
            src_gpu = str(best.get("gpu_type", ""))
            src_spec = _GPU_SPECS.get(src_gpu, gpu_spec)
            bw_scale = gpu_spec["bandwidth_gbps"] / max(src_spec["bandwidth_gbps"], 1)
            tps, conf, _ = self._scale_from_record(
                best, request, mf, dp, gpu_spec, DataSource.CROSS_GPU, 0.40
            )
            tps *= bw_scale
            return self._build_metrics(
                tps, conf, DataSource.CROSS_GPU, cost_per_hour, num_gpus, gpu,
                mf, tp, pp, dp, request, best
            )

        # --- Analytical fallback ---
        return self._analytical_metrics(request, gpu.gpu_type, tp, pp, dp,
                                        cost_per_hour, num_gpus, gpu)

    def _scale_from_record(
        self,
        rec: Dict,
        request: JobRequest,
        mf: ModelFeatures,
        dp: int,
        gpu_spec: Dict,
        source: DataSource,
        base_conf: float,
    ) -> Tuple[float, float, DataSource]:
        """Scale a RAG record's throughput to the target request."""
        base_tps = _safe_float(rec.get("throughput_tps"), 0.0)
        rec_dp = int(rec.get("dp", 1) or 1)

        # Scale for DP change
        if rec_dp != dp:
            base_tps = base_tps * (dp / max(rec_dp, 1))

        # Scale for I/O length difference
        rec_in = _safe_float(rec.get("input_len"), 512)
        rec_out = _safe_float(rec.get("output_len"), 128)
        base_work = 0.3 * rec_in + 1.0 * rec_out
        tgt_work = 0.3 * request.avg_input_tokens + 1.0 * request.avg_output_tokens
        if base_work > 0:
            io_scale = math.pow(base_work / tgt_work, 0.5)
            io_scale = max(0.5, min(1.4, io_scale))
            base_tps *= io_scale

        # Scale for model param count difference
        rec_params = _safe_float(rec.get("num_params_billions"), mf.num_params_billions)
        if rec_params and abs(rec_params - mf.num_params_billions) / max(rec_params, 1) > 0.1:
            param_scale = rec_params / max(mf.num_params_billions, 0.1)
            param_scale = max(0.3, min(3.0, param_scale))
            base_tps *= param_scale
            base_conf *= 0.85

        return base_tps, base_conf, source

    def _build_metrics(
        self,
        tps: float,
        confidence: float,
        source: DataSource,
        cost_per_hour: float,
        num_gpus: int,
        gpu: GPUResource,
        mf: ModelFeatures,
        tp: int,
        pp: int,
        dp: int,
        request: JobRequest,
        best_rec: Dict,
    ) -> PredictedMetrics:
        tpot_ms = _safe_float(best_rec.get("tpot_ms"))
        ttft_ms = _safe_float(best_rec.get("ttft_ms"))

        # Estimate TPOT if missing
        if tpot_ms is None and tps > 0:
            concurrency = max(1, (request.expected_concurrency or 10) / dp)
            replica_tps = tps / dp
            if replica_tps > 0:
                tpot_ms = min(5000, (concurrency / replica_tps) * request.avg_output_tokens * 1000)

        # Batch runtime + cost
        runtime_h = total_cost = None
        if request.task_type == TaskType.BATCH and request.total_tokens and tps > 0:
            runtime_h = (request.total_tokens / tps) / 3600.0
            total_cost = runtime_h * cost_per_hour

        mem_feats = compute_config_features(mf, gpu.gpu_type, tp, pp, dp)
        used_gb = mf.model_size_gb / max(tp, 1)
        used_pct = (used_gb / gpu.gpu_memory_gb) * 100 if gpu.gpu_memory_gb else None

        cost_per_1m = (cost_per_hour / max(tps * 3600, 1)) * 1e6 if tps > 0 else None

        nearest = (
            f"{best_rec.get('model_name', '?')} "
            f"{best_rec.get('gpu_type', '?')} "
            f"TP={best_rec.get('tp', '?')} PP={best_rec.get('pp', '?')} "
            f"(in={best_rec.get('input_len', '?')}/out={best_rec.get('output_len', '?')})"
        )

        return PredictedMetrics(
            throughput_tokens_per_sec=max(0.0, tps),
            throughput_per_gpu_tokens_per_sec=max(0.0, tps / max(num_gpus, 1)),
            estimated_runtime_hours=runtime_h,
            total_cost_usd=total_cost,
            tpot_ms=tpot_ms,
            ttft_ms=ttft_ms,
            cost_per_hour_usd=cost_per_hour,
            cost_per_1m_tokens_usd=cost_per_1m,
            estimated_gpu_mem_used_gb=used_gb,
            estimated_gpu_mem_pct=used_pct,
            confidence=min(1.0, max(0.0, confidence)),
            data_source=source,
            nearest_db_entry=nearest,
        )

    def _analytical_metrics(
        self,
        request: JobRequest,
        gpu_type: str,
        tp: int,
        pp: int,
        dp: int,
        cost_per_hour: float = 0.0,
        num_gpus: int = 1,
        gpu: Optional[GPUResource] = None,
    ) -> PredictedMetrics:
        mf = get_model_features(request.model_name)
        gpu_spec = _GPU_SPECS.get(gpu_type, {"bandwidth_gbps": 400, "fp16_tflops": 300, "mem_gb": 40})

        agg_bw = gpu_spec["bandwidth_gbps"] * tp
        tps = (agg_bw / max(mf.model_size_gb, 0.1)) * 0.65
        pp_eff = 1.0 - (pp - 1) / (pp * 8)
        tps *= pp_eff * dp

        if gpu is None:
            cost_per_hour = 0.0
            num_gpus = tp * pp * dp
            used_gb = mf.model_size_gb / max(tp, 1)
            used_pct = None
        else:
            used_gb = mf.model_size_gb / max(tp, 1)
            used_pct = (used_gb / gpu.gpu_memory_gb) * 100 if gpu.gpu_memory_gb else None

        return PredictedMetrics(
            throughput_tokens_per_sec=max(0.0, tps),
            throughput_per_gpu_tokens_per_sec=max(0.0, tps / max(num_gpus, 1)),
            cost_per_hour_usd=cost_per_hour,
            estimated_gpu_mem_used_gb=used_gb,
            estimated_gpu_mem_pct=used_pct,
            confidence=0.30,
            data_source=DataSource.ANALYTICAL,
            nearest_db_entry="analytical roofline",
        )

    # ------------------------------------------------------------------
    # SLO evaluation
    # ------------------------------------------------------------------

    def _check_slo(
        self, request: JobRequest, metrics: PredictedMetrics
    ) -> Tuple[bool, Optional[float]]:
        meets = True
        margins = []

        if request.slo_tpot_ms and metrics.tpot_ms:
            m = (request.slo_tpot_ms - metrics.tpot_ms) / request.slo_tpot_ms * 100
            margins.append(m)
            if m < 0:
                meets = False

        if request.slo_deadline_hours and metrics.estimated_runtime_hours:
            m = (request.slo_deadline_hours - metrics.estimated_runtime_hours) / request.slo_deadline_hours * 100
            margins.append(m)
            if m < 0:
                meets = False

        margin = min(margins) if margins else None
        return meets, margin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_matches(a: str, b: str) -> bool:
    """Loose GPU type match (H100_SXM ≈ H100)."""
    a, b = a.upper().strip(), b.upper().strip()
    return a == b or a.startswith(b) or b.startswith(a)
