"""
koi/placement.py — KoiPlacement: main orchestrator.

End-to-end flow:

  PLACEMENT (every job):
    JobRequest + ResourceMap
      → Oracle.get_candidates()               feasibility prune + metric predictions
      → KoiRefinement._build_history()        pull delta history + policy memory from DB
      → KoiRefinement._build_frontier()       pull Pareto frontier for this workload class
      → KoiEnsemble.run()                     3 LLMs freely propose configs + judge synthesizes
      → PlacementDecision                     config + exploration queue

  JOB COMPLETION (every job finish):
      → KoiPlacement.on_job_complete()
          → DeltaStore.insert()               log prediction vs actual delta
          → EfficiencyFrontier.update()       update Pareto frontier
          → PolicyMemory.add_outcome()        log natural-language outcome summary
          (future: HypothesisLedger.validate() — validate/falsify LLM hypotheses)

  RECONFIG (monitoring-triggered):
      → KoiPlacement.reconfig()
          → KoiEnsemble.run_diagnosis()       3 LLMs diagnose failure + propose repair
          → on_job_complete()                 record what broke and why

The evolutionary loop closes when on_job_complete() writes to the DB and the next
decide() call reads richer context than the previous one did.
"""

import os
import time
from typing import List, Optional

from koi.ensemble import KoiEnsemble
from koi.oracle import Oracle
from koi.refinement import KoiRefinement
from koi.schemas import (
    DeltaRecord,
    JobRequest,
    PlacementConfig,
    PlacementDecision,
    ResourceMap,
    RuntimeMetrics,
)


class KoiPlacement:
    """
    Main entry point for the Koi placement system.

    Usage:
        koi = KoiPlacement(api_key="sk-ant-...")
        decision = koi.decide(request, resource_map)
        print(decision.display_summary())

        # after job finishes:
        pes = koi.on_job_complete(decision, request, actual_tps=820, slo_met=True, ...)

        # if monitoring triggers reconfiguration:
        new_decision = koi.reconfig(request, current_config, monitoring_trace, resource_map)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        perfdb_path: str = "./perfdb",
        data_dir: str = "./data",
        llm_model: str = "claude-opus-4-6",
        n_thinkers: int = 3,
    ):
        self.oracle = Oracle(perfdb_path=perfdb_path)
        self.refinement = KoiRefinement(data_dir=data_dir)
        self.ensemble = KoiEnsemble(
            oracle=self.oracle,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            model=llm_model,
            n_thinkers=n_thinkers,
        )

    # ------------------------------------------------------------------
    # Primary placement path
    # ------------------------------------------------------------------

    def decide(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
    ) -> PlacementDecision:
        """
        Main synchronous placement entry point.

        Steps:
          1. Oracle generates reference candidates (feasibility + metric estimates)
          2. Evolutionary DB queried: delta history + policy memory + frontier
          3. 3 LLMs each propose a config + causal hypothesis (not pick from list)
          4. Judge synthesizes → deployment config + exploration queue
        """
        t0 = time.time()
        print(f"\n[Koi] Placement: {request.job_id} — {request.model_name}")
        print(f"[Koi] Task: {request.task_type.value} | Objective: {request.objective.value}")

        # Step 1 — Oracle reference candidates
        t_oracle = time.time()
        all_candidates = self.oracle.get_candidates(request, resource_map)
        if not all_candidates:
            raise RuntimeError(
                f"[Koi] Oracle found 0 feasible candidates for {request.model_name}. "
                "Check ResourceMap GPU availability and model memory constraints."
            )
        slo_candidates = [c for c in all_candidates if c.meets_slo]
        reference_candidates = slo_candidates if slo_candidates else all_candidates
        print(
            f"[Koi] Oracle: {len(all_candidates)} candidates, "
            f"{len(slo_candidates)} meet SLO ({time.time() - t_oracle:.2f}s)"
        )

        # Step 2 — Evolutionary context from DB
        history_str = self._build_history(request, reference_candidates)
        frontier_str = self._build_frontier_summary(request)
        if history_str:
            print(f"[Koi] Evolutionary context: {len(history_str)} chars from DB")
        else:
            print("[Koi] Evolutionary context: empty (first run for this workload class)")

        # Step 3+4 — LLM proposals + judge synthesis
        t_llm = time.time()
        decision = self.ensemble.run_sync(
            request, resource_map, reference_candidates,
            history=history_str,
            frontier_summary=frontier_str,
        )
        rec = decision.recommendation
        src = decision.judge_decision.decision_source if decision.judge_decision else "unknown"
        print(
            f"[Koi] Decision ({time.time() - t_llm:.2f}s): "
            f"{rec.gpu_type} TP={rec.tp} PP={rec.pp} DP={rec.dp} "
            f"({rec.num_gpus} GPUs) | conf={decision.confidence:.0%} | source={src}"
        )
        print(f"[Koi] Exploration queue: {len(decision.exploration_queue)} proposals queued")
        print(f"[Koi] Total: {time.time() - t0:.2f}s")
        return decision

    async def decide_async(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
    ) -> PlacementDecision:
        """Async version — same flow, non-blocking."""
        t0 = time.time()
        print(f"\n[Koi] Placement (async): {request.job_id} — {request.model_name}")

        all_candidates = self.oracle.get_candidates(request, resource_map)
        if not all_candidates:
            raise RuntimeError("Oracle found 0 feasible candidates.")

        slo_candidates = [c for c in all_candidates if c.meets_slo]
        reference_candidates = slo_candidates if slo_candidates else all_candidates

        history_str = self._build_history(request, reference_candidates)
        frontier_str = self._build_frontier_summary(request)

        decision = await self.ensemble.run(
            request, resource_map, reference_candidates,
            history=history_str,
            frontier_summary=frontier_str,
        )
        print(f"[Koi] Async placement done: {time.time() - t0:.2f}s")
        return decision

    # ------------------------------------------------------------------
    # Job completion — write to evolutionary DB
    # ------------------------------------------------------------------

    def on_job_complete(
        self,
        decision: PlacementDecision,
        request: JobRequest,
        actual_throughput_tps: float,
        slo_met: bool,
        total_hours: float,
        time_in_final_config_hours: float,
        roofline_peak_tps: float,
        actual_tpot_ms: Optional[float] = None,
        vpc_id: str = "unknown",
    ):
        """
        Call this when a job finishes (or is about to be reconfigured).

        Writes to three evolutionary DB stores:
          1. DeltaStore     — prediction error for this config (RAG corpus for Oracle correction)
          2. EfficiencyFrontier — update Pareto frontier if this config beat the current best
          3. PolicyMemory   — add natural-language outcome summary for LLM few-shot context

        Returns PESComponents (CER, PER, SS, composite score).

        Future (once HypothesisLedger exists):
          → validate/falsify the ThinkerProposal hypotheses attached to this decision
          → extract causal rules from confirmed mechanisms
          → run counterfactual estimates for unchosen proposals
        """
        cfg = decision.recommendation
        pred = decision.predicted_metrics

        delta_tpot = None
        if actual_tpot_ms is not None and pred.tpot_ms is not None:
            delta_tpot = actual_tpot_ms - pred.tpot_ms

        delta_record = DeltaRecord(
            vpc_id=vpc_id,
            job_id=decision.job_id,
            model_name=request.model_name,
            gpu_type=cfg.gpu_type,
            tp=cfg.tp,
            pp=cfg.pp,
            dp=cfg.dp,
            avg_input_tokens=request.avg_input_tokens,
            avg_output_tokens=request.avg_output_tokens,
            task_type=request.task_type.value,
            predicted_throughput_tps=pred.throughput_tokens_per_sec,
            actual_throughput_tps=actual_throughput_tps,
            predicted_tpot_ms=pred.tpot_ms,
            actual_tpot_ms=actual_tpot_ms,
            delta_throughput_pct=(
                (actual_throughput_tps - pred.throughput_tokens_per_sec)
                / max(pred.throughput_tokens_per_sec, 1.0) * 100
            ),
            delta_tpot_ms=delta_tpot,
            prediction_data_source=pred.data_source.value,
        )

        pes = self.refinement.record_completion(
            decision=decision,
            request=request,
            delta_record=delta_record,
            actual_throughput_tps=actual_throughput_tps,
            slo_met=slo_met,
            total_hours=total_hours,
            time_in_final_config=time_in_final_config_hours,
            roofline_peak_tps=roofline_peak_tps,
        )

        print(
            f"[Koi] Job {decision.job_id} recorded: PES={pes.composite:.3f} | "
            f"delta_tps={delta_record.delta_throughput_pct:+.1f}% | "
            f"slo_met={slo_met}"
        )
        return pes

    # ------------------------------------------------------------------
    # Monitoring-triggered reconfiguration
    # ------------------------------------------------------------------

    def reconfig(
        self,
        request: JobRequest,
        current_config: PlacementConfig,
        monitoring_trace: List[RuntimeMetrics],
        resource_map: ResourceMap,
    ) -> PlacementDecision:
        """
        Called when KoiMonitor fires a hard alert (SLO at risk).

        3 LLMs each:
          1. Diagnose the failure mode from the monitoring trace
          2. Extract a generalizable causal rule (stored in future HypothesisLedger)
          3. Propose a repair config that directly addresses the diagnosed cause

        The judge picks the best-reasoned repair.
        """
        print(
            f"\n[Koi] RECONFIG triggered for {request.job_id} | "
            f"current: {current_config.gpu_type} TP={current_config.tp} PP={current_config.pp} | "
            f"{len(monitoring_trace)} monitoring samples"
        )
        history_str = self._build_history(request, [])
        decision = self.ensemble.run_diagnosis_sync(
            request, current_config, monitoring_trace, resource_map,
            history=history_str,
        )
        print(
            f"[Koi] Repair → {decision.recommendation.gpu_type} "
            f"TP={decision.recommendation.tp} PP={decision.recommendation.pp} "
            f"DP={decision.recommendation.dp}"
        )
        return decision

    # ------------------------------------------------------------------
    # Evolutionary context builders
    # ------------------------------------------------------------------

    def _build_history(
        self,
        request: JobRequest,
        reference_candidates: list,
    ) -> Optional[str]:
        """
        Query DeltaStore + PolicyMemory for this workload class.
        Returns a formatted string injected into every LLM thinker's prompt.

        The context grows richer with each completed job — this is the core
        of the evolutionary feedback loop. LLMs that placed job #50 see
        far more relevant history than the LLM that placed job #1.
        """
        # Use the top Oracle candidate as the query anchor for delta lookup
        if reference_candidates:
            top = reference_candidates[0]
            return self.refinement.get_context_for_ensemble(
                request=request,
                gpu_type=top.config.gpu_type,
                tp=top.config.tp,
                pp=top.config.pp,
                oracle_tps=top.metrics.throughput_tokens_per_sec,
                oracle_tpot=top.metrics.tpot_ms,
            )

        # No Oracle candidates available — fall back to model-name-only policy memory query
        past = self.refinement.policy_memory.retrieve_similar(
            model_name=request.model_name,
            task_type=request.task_type.value,
            k=3,
        )
        if past:
            return "SIMILAR PAST DECISIONS:\n" + "\n".join(f"  {d}" for d in past)
        return None

    def _build_frontier_summary(self, request: JobRequest) -> Optional[str]:
        """
        Format the current Pareto frontier for this workload class.
        Injected into every LLM thinker prompt as "here's the bar to beat."

        An empty frontier means this is the first time Koi has seen this
        workload class — LLMs go in with no reference point beyond the Oracle.
        """
        wc = self.refinement.frontier._workload_class(
            request.model_name,
            request.task_type.value,
            request.avg_input_tokens,
            request.avg_output_tokens,
        )
        configs = self.refinement.frontier.get_frontier_configs(wc)
        if not configs:
            return None

        lines = ["CURRENT FRONTIER — best known SLO-meeting configs for this workload class:"]
        for c in configs[:5]:
            lines.append(
                f"  {c['gpu_type']:8s} TP={c['tp']} PP={c['pp']} DP={c['dp']} | "
                f"${c['cost_per_hour_usd']:.2f}/hr | "
                f"PES={c['pes_composite']:.2f} | "
                f"{c['throughput_tps']:.0f} tok/s"
            )
        lines.append("Beat the top entry to expand the frontier.")
        return "\n".join(lines)
