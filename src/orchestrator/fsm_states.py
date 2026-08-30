"""Per-tick state machine: the deterministic spine that drives S0 -> S7.

The tick is Koi's single front door. Runtime events (chain deaths, launch
failures, degradations, completions) are recorded as facts in durable state
when they happen; no event triggers an LLM. The next tick observes them and
the one root planner reasons over the whole cluster with those facts
included. This prevents local reflex prompts from racing the global planner.

One tick is a closed learning loop:

    S0 ENTER_TICK     Freeze a consistent snapshot; reset per-tick caches
                      (user envelopes, validated BudgetBook).
    S1 OBSERVE        Pull per-rank telemetry bundles for [t-1, t].
    S2 VALIDATE       Per rank: residuals -> applicable mechanisms ->
                      per-mechanism (V-CUSUM, Y-CUSUM) -> per-mechanism Q ->
                      ICP per edge -> one EvidenceRow appended.
    S3 SLOW_UPDATE    1) Decision-band coverage, then DRO residual ingestion.
                      2) Beta(alpha, beta) fan-out: every decided
                         (row, mechanism) pair updates that mechanism and
                         its edges via ConfidenceService.
                      3) SlowLoop.slow_update_all: w_t, z_star_t,
                         lambda_swit, beta_t, B_t, epsilon_dro.
                      4) Meta cadence: CUSUM (delta, h) recalibration every
                         `recalibrate_every` ticks.
    S4 AGENTIC_PLAN   One KoiAgentHarness.run_agent_loop call -> plan.
                      The harness owns K_P sampling, budget-first specialist
                      protocol, and best-of-K selection.
    S5 VALIDATE_PLAN  PlanValidator (C0..C7). One repair iteration back to
                      S4 with violations; second failure -> keep-all.
    S6 DEPLOY         Executor submission (A/B canary semantics); record
                      swap bookkeeping for next tick's observed_swap_rate;
                      persist trace.
    S7 EXIT_TICK      Sleep the REMAINDER of the tick interval (interval
                      minus elapsed), so ticks do not drift.

Evidence semantics decided here (and why):

    Observation is rank-scoped; verdicts are mechanism-scoped. One rank
    produces one set of V/Y trajectories; every applicable mechanism filters
    those trajectories through ITS bundle to get ITS (v_verdict, y_verdict)
    and Q label. One row therefore feeds N Beta updates - evidence compounds
    across mechanisms.

    Q comes from the two CUSUM axes ONLY. ICP never nulls a Q label: the
    EDGE_BETA_UPDATE table has an explicit "undecided" row (small-magnitude
    deltas), so undecided invariance modulates EDGE updates rather than
    gating learning. Nulling Q on undecided ICP would freeze every Beta
    update until each edge had n_env_min envs with n_b samples - a bootstrap
    deadlock on a young cluster. q_label_per_mechanism[mid] is None only
    when the mechanism's bundle was not observable in this rank's telemetry
    (a bundle variable missing from the trajectories).

    CUSUM (delta, h) resolution: the slow loop's recalibrated tables first
    (meta timescale), self-calibration from this rank's own residuals as the
    cold-start fallback (0.5 sigma, 4 sigma) - never fixed unit-blind
    defaults, which would misfire on raw-unit objectives.

Failure policy: any unhandled exception in S0-S6 aborts the tick into the
keep-all fallback (keep active jobs, defer pending). The running cluster is
the safe state; a half-planned tick is not.

This module is wiring only. The math lives in cusum / icp / quadrants /
regret / slow_loop / dro / eig / tchebecheff / switchcost; the planning
lives in agent.py; the tools live in agent_tools.py.
"""

import copy
import logging
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from src.config.hyperparameters import PARTIAL_DIVERGENCE_BETA
from src.core.models import (
    SWAP_BUDGET_ACTIONS,
    ActionType,
    EvidenceRow,
    Plan,
    PlanAction,
    deployment_ladder_identity,
    deployment_rank_identity,
    env_gpu_type,
)
from src.infra.deployment_x import build_deployment_x_index
from src.validation.cusum import CusumResult
from src.validation.icp import ICPResult

log = logging.getLogger("koi.fsm")

_DEPLOYMENT_GRACE_TICKS = 1
_DEPLOYMENT_RETRY_BACKOFF_TICKS = 1
_DEPLOYMENT_MAX_ATTEMPTS = 3


class FSMState(Enum):
    S0_ENTER_TICK = "S0_ENTER_TICK"
    S1_OBSERVE = "S1_OBSERVE"
    S2_VALIDATE = "S2_VALIDATE"
    S3_SLOW_UPDATE = "S3_SLOW_UPDATE"
    S4_AGENTIC_PLAN = "S4_AGENTIC_PLAN"
    S5_VALIDATE_PLAN = "S5_VALIDATE_PLAN"
    S6_DEPLOY = "S6_DEPLOY"
    S7_EXIT_TICK = "S7_EXIT_TICK"
    ABORT = "ABORT"


@dataclass
class TickContext:
    """Per-tick scratch space carrying every artifact a tick produces.

    Everything needed to trace, replay, and debug one tick lives here:
    the snapshot, the telemetry bundle, the evidence rows written, the
    candidate and validated plans, deploy acks, per-state durations, and
    the failure record when a tick aborts.
    """

    tick: int
    tick_started_at: float = 0.0

    cluster_snapshot: Any = None
    telemetry: Any = None
    telemetry_diagnostics: dict[str, Any] = field(default_factory=dict)
    deployment_x: Any = None
    evidence_rows: list[Any] = field(default_factory=list)
    mechanism_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    confidence_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    slow_update_diagnostics: dict[str, Any] = field(default_factory=dict)
    new_slow_state: Any = None
    candidate_plan: Any = None
    validated_plan: Any = None
    deploy_acks: list[Any] = field(default_factory=list)
    deployment_reconciliation: list[dict[str, Any]] = field(default_factory=list)
    active_health: dict[str, dict[str, Any]] = field(default_factory=dict)

    s5_repair_count: int = 0
    max_s5_repairs: int = 1

    state_durations_ms: dict[str, float] = field(default_factory=dict)
    state_history: list[FSMState] = field(default_factory=list)

    error: Exception | None = None
    aborted_from_state: FSMState | None = None


class _MechanismBundle:
    """Cusum-facing view of a mechanism: its V and Y variable bundles.

    Mechanism stores edge_ids only; the bundles are derived through the
    CandidateGraph. Mechanisms are immutable once admitted, so TickRunner
    caches one bundle per mechanism_id for the runner's lifetime.
    """

    def __init__(self, mechanism, candidate_graph):
        self.mechanism_id = mechanism.mechanism_id
        self.edge_ids = list(mechanism.edge_ids)
        edges = [
            candidate_graph.edge_table[eid]
            for eid in mechanism.edge_ids
            if eid in candidate_graph.edge_table
        ]
        v_names = {e.dst for e in edges if e.dst_type == "V"}
        v_names |= {e.src for e in edges if e.src_type == "V"}
        self.bundle_v_variables = sorted(v_names)
        self.bundle_y_outcomes = sorted({e.dst for e in edges if e.dst_type == "Y"})


class TickRunner:
    """One TickRunner per cluster; run_tick(tick_id) drives one full tick.

    Construction wires every component. The telemetry adapter must yield
        per-rank bundles via iter_per_rank(telemetry); each bundle exposes:
        job_id, rank_id, observed V/Y trajectories, committed_mechanism_id,
        and optionally deploy_timestamp_utc. Deploy-time X and predictions come
        from Store/catalog snapshots, not telemetry.

    Args:
        evidence_store: Append-only EvidenceRow ledger (EvidenceService).
        telemetry: Adapter with collect_telemetry(tick_start, tick_end, snapshot)
            and iter_per_rank(bundle).
        cusum: Cusum instance (V and Y trajectory drift).
        icp: ICP instance (per-edge invariance).
        quadrant_validator: QuadrantValidator (two-verdict classify).
        confidence_service: The single Beta(alpha, beta) writer.
        slow_loop: SlowLoop instance.
        dro: DRO instance.
        mechanism_registry: MechanismRegistry.
        resource_map: Cluster state service with
            snapshot_cluster_state(tick) and build_keep_all_plan(snapshot).
        agent: KoiAgentHarness (run_agent_loop / receive_validator_feedback).
        plan_validator: Validator with val_plan(plan, cluster_snapshot,
            slow_state) -> result(.feasible, .violations).
        executor: Deterministic deployer with send_to_executor(plan).
        candidate_graph: CandidateGraph; defaults to the one inside
            confidence_service.
        tchebycheff: Optional module exposing compute_tchebycheff, used to
            stamp J_realized on evidence rows. None -> J_realized = 0.0.
        trace_logger: Optional sink with persist_tick(ctx).
        tick_interval_sec: Tick period; S7 sleeps the remainder.
        recalibrate_every: Meta-cadence (ticks) for CUSUM (delta, h)
            recalibration. 0 disables.
        on_tick_start: Optional zero-arg hook run in S0. Boot wires
            agent_tools.reset_tick_caches here so user envelopes and the
            validated BudgetBook cannot leak across ticks.
        typical_ranges: Per-objective scale; defaults to slow_loop's.
    """

    def __init__(
        self,
        *,
        evidence_store,
        telemetry,
        cusum,
        icp,
        quadrant_validator,
        confidence_service,
        slow_loop,
        dro,
        mechanism_registry,
        resource_map,
        agent,
        plan_validator,
        executor,
        candidate_graph=None,
        tchebycheff=None,
        trace_logger=None,
        tick_interval_sec: int = 300,
        recalibrate_every: int = 100,
        on_tick_start=None,
        typical_ranges: dict[str, float] | None = None,
    ):
        self.evidence_store = evidence_store
        self.telemetry = telemetry
        self.cusum = cusum
        self.icp = icp
        self.qv = quadrant_validator
        self.confidence_service = confidence_service
        self.slow_loop = slow_loop
        self.dro = dro
        self.mechanism_registry = mechanism_registry
        self.resource_map = resource_map
        self.agent = agent
        self.plan_validator = plan_validator
        self.executor = executor
        self.candidate_graph = candidate_graph or getattr(
            confidence_service, "candidate_graph", None
        )
        self.tchebycheff = tchebycheff
        self.trace = trace_logger
        self.tick_interval_sec = int(tick_interval_sec)
        self.recalibrate_every = int(recalibrate_every)
        self.on_tick_start = on_tick_start
        self.typical_ranges = typical_ranges or getattr(slow_loop, "typical_ranges", {})

        self._bundle_cache: dict[str, _MechanismBundle] = {}
        # Swap bookkeeping recorded in S6, consumed by next tick's S3.
        self._last_swap_count: int = 0
        self._last_active_count: int = 0
        self._deployment_ledger: dict[str, dict[str, Any]] = {}
        self._prediction_ledger: dict[tuple[str, str], dict[str, Any]] = {}
        self._health_state: dict[str, dict[str, Any]] = {}
        # Run-lifetime memory of shapes that launched and served nothing under
        # load, keyed (model_id, gpu_type, tp, pp). Surfaced on job descriptors
        # as observed_dead_shapes so neither the candidate builder nor the LLM
        # proposes them again this run.
        self._dead_shapes: dict[tuple[str, str, int, int], dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run_tick(self, tick_id: int) -> TickContext:
        """Drive the FSM from S0 to S7 (or ABORT) for one tick.

        Every state is duration-instrumented; any unhandled exception
        captures the error, aborts into the keep-all fallback, and still
        persists the trace.

        Args:
            tick_id: Monotonic tick id.

        Returns:
            The TickContext with all per-state artifacts.
        """
        ctx = TickContext(tick=tick_id, tick_started_at=time.time())
        state = FSMState.S0_ENTER_TICK

        while state not in (FSMState.S7_EXIT_TICK, FSMState.ABORT):
            ctx.state_history.append(state)
            t0 = time.time()
            try:
                next_state = self._dispatch(state, ctx)
            except Exception as exc:
                log.exception("FSM error in %s at tick %d", state.value, tick_id)
                ctx.error = exc
                ctx.aborted_from_state = state
                next_state = FSMState.ABORT
            finally:
                ctx.state_durations_ms[state.value] = (time.time() - t0) * 1000.0
                self._persist_state(state, ctx)
            state = next_state

        ctx.state_history.append(state)
        if state == FSMState.ABORT:
            self._handle_abort(ctx)
        else:
            self._handle_s7(ctx)
        return ctx

    def _dispatch(self, state: FSMState, ctx: TickContext) -> FSMState:
        """Route to the handler for the current state."""
        handlers = {
            FSMState.S0_ENTER_TICK: self.S0,
            FSMState.S1_OBSERVE: self.S1,
            FSMState.S2_VALIDATE: self.S2,
            FSMState.S3_SLOW_UPDATE: self.S3,
            FSMState.S4_AGENTIC_PLAN: self.S4,
            FSMState.S5_VALIDATE_PLAN: self.S5,
            FSMState.S6_DEPLOY: self.S6,
        }
        return handlers[state](ctx)

    def _persist_state(self, state: FSMState, ctx: TickContext) -> None:
        """Persist optional per-state debug events without affecting the FSM."""
        sink = getattr(self.trace, "persist_state", None)
        if not callable(sink):
            return
        try:
            sink(state, ctx)
        except Exception:
            log.exception("state trace persist failed for %s at tick %d", state.value, ctx.tick)

    # ------------------------------------------------------------------
    # S0 - S6
    # ------------------------------------------------------------------

    def S0(self, ctx: TickContext) -> FSMState:
        """Freeze the tick's view of the world and reset per-tick caches.

        The snapshot is the single consistent input every later state
        references. The on_tick_start hook clears agent-tool tick caches
        (user envelopes, validated BudgetBook) so nothing budget-shaped
        survives from the previous tick's capacity.
        """
        if self.on_tick_start is not None:
            self.on_tick_start()
        ctx.cluster_snapshot = self.resource_map.snapshot_cluster_state(ctx.tick)
        ctx.deployment_reconciliation = self._reconcile_deployments(ctx)
        self._annotate_dead_shapes(ctx)
        return FSMState.S1_OBSERVE

    def S1(self, ctx: TickContext) -> FSMState:
        """Pull telemetry and build deploy-time X for the [t-1, t] window.

        Telemetry owns runtime V/Y. Store/catalog snapshots own deployment X.
        """
        ctx.telemetry = self.telemetry.collect_telemetry(
            tick_start=ctx.tick - 1,
            tick_end=ctx.tick,
            snapshot=ctx.cluster_snapshot,
        )
        ctx.deployment_x = self._build_deployment_x_index(ctx)
        ctx.telemetry_diagnostics = self._telemetry_diagnostics(ctx)
        return FSMState.S2_VALIDATE

    def S2(self, ctx: TickContext) -> FSMState:
        """Validate every deployed rank and write the evidence backbone.

        Per rank: compute residuals; resolve applicable mechanisms
        (committed + scope matches, restricted to bundles fully observable
        in this rank's telemetry); run V-CUSUM and Y-CUSUM per mechanism;
        classify a Q per mechanism; run ICP once per edge in the union of
        applicable bundles; append one EvidenceRow. S3 evaluates the
        immutable decision-time DRO band before feeding this row's residual.

        No Beta updates here - S2 writes evidence, S3 reads it. The
        separation keeps S2 idempotent for replay.
        """
        w_t_snapshot = self.slow_loop.get_sss_wt()
        z_star_snapshot = self.slow_loop.get_sss_z_star_t()
        cached_v_params = self.slow_loop.get_sss_cusum_params_v()
        cached_y_params = self.slow_loop.get_sss_cusum_params_y()
        snapshot = ctx.cluster_snapshot
        jobs: list[dict[str, Any]] = []
        if snapshot is not None:
            jobs = (
                snapshot.active_jobs_summary()
                if hasattr(snapshot, "active_jobs_summary")
                else getattr(snapshot, "active_jobs", [])
            )
        job_features = {
            str(job.get("job_id", job.get("id"))): dict(job.get("job_features") or {})
            for job in jobs or []
        }
        health_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        tick_icp_details: dict[str, dict[str, Any]] = {}

        # if ctx.deployment_x is None:
        #     ctx.deployment_x = self._build_deployment_x_index(ctx)

        for rank_telem in self.telemetry.iter_per_rank(ctx.telemetry):
            job_id = str(rank_telem.job_id)
            raw_rank_id = getattr(rank_telem, "rank_id", None)
            deployment = ctx.deployment_x.resolve(job_id, raw_rank_id)
            rank_id = deployment.rank_id
            x = dict(deployment.x)
            env_label = deployment.env_label
            v_obs = dict(rank_telem.v_observed)
            v_pred = dict(deployment.v_predicted)
            y_obs = dict(rank_telem.y_observed)
            y_pred = dict(deployment.y_predicted)

            residuals_per_v = self._residuals(v_obs, v_pred)
            comparable_y_obs = dict(y_obs)
            mode = str(job_features.get(job_id, {}).get("type") or "online").lower()
            queue_depth = max(
                (float(value) for value in v_obs.get("depth_req_q", [])),
                default=0.0,
            )
            if mode == "online":
                if queue_depth <= 0:
                    comparable_y_obs.pop("throughput_token_per_sec", None)
                else:
                    comparable_y_obs.pop("p99_ttft_ms", None)
                    comparable_y_obs.pop("p99_tpot_ms", None)
            residuals_per_y = self._residuals(comparable_y_obs, y_pred)
            y_observed_mean = {name: float(np.mean(arr)) for name, arr in y_obs.items() if len(arr)}
            # Zero throughput with a backlog is the strongest observation a rank can
            # make: the shape ran and served nothing. Remember it for the run and let
            # it classify below even though no latency could be measured.
            observed_tps = y_observed_mean.get("throughput_token_per_sec")
            dead_rank = observed_tps is not None and observed_tps <= 0.0 and queue_depth > 0
            if dead_rank:
                self._record_dead_shape(
                    ctx, job_id, rank_id, job_features.get(job_id, {}), env_label, x, queue_depth
                )
            health_sample = dict(y_observed_mean)
            if len(v_obs.get("depth_req_q", [])):
                health_sample["depth_req_q"] = max(float(value) for value in v_obs["depth_req_q"])
            health_sample["telemetry_complete"] = bool(
                getattr(rank_telem, "health_observed_replicas", 1)
                >= getattr(rank_telem, "expected_replicas", 1)
            )
            health_samples[job_id].append(health_sample)

            committed = self._committed_mechanism_id(rank_telem) or (
                self._prediction_ledger.get((job_id, rank_id), {}).get("mechanism_id")
            )
            mechanism_context = {
                key: value
                for key, value in {**job_features.get(job_id, {}), **x}.items()
                if value is not None and value != "NA"
            }
            applicable = self._applicable_mechanisms(
                mechanism_context,
                committed,
                diagnostics=ctx.mechanism_diagnostics,
                job_id=job_id,
                rank_id=rank_id,
            )

            v_params = self._resolve_cusum_params(residuals_per_v, cached_v_params)
            y_params = self._resolve_cusum_params(residuals_per_y, cached_y_params)

            cusum_per_mech: dict[str, Any] = {}
            q_per_mech: dict[str, Any] = {}
            touched_edge_ids: set = set()
            rank_diagnostics: list[dict[str, Any]] = []

            for mech in applicable:
                bundle = self._bundle(mech)
                missing = self._missing_bundle_inputs(bundle, v_obs, v_pred, y_obs, y_pred)
                available_v = [
                    name for name in bundle.bundle_v_variables if name in residuals_per_v
                ]
                available_y = [name for name in bundle.bundle_y_outcomes if name in residuals_per_y]
                missing["v_unusable"] = [
                    name for name in bundle.bundle_v_variables if name not in residuals_per_v
                ]
                missing["y_unusable"] = [
                    name for name in bundle.bundle_y_outcomes if name not in residuals_per_y
                ]
                v_verdict = (
                    self.cusum.cusum_per_bundle(available_v, v_obs, v_pred, v_params)
                    if available_v
                    else CusumResult.MATCHED
                    if not bundle.bundle_v_variables
                    else None
                )
                y_verdict = (
                    self.cusum.cusum_per_bundle(available_y, y_obs, y_pred, y_params)
                    if available_y
                    else CusumResult.MATCHED
                    if not bundle.bundle_y_outcomes
                    else None
                )
                if v_verdict is not None or y_verdict is not None:
                    cusum_per_mech[mech.mechanism_id] = (v_verdict, y_verdict)
                # A dead rank completes no requests, so its latency outcomes cannot
                # be measured. That absence is the divergence, not missing data:
                # let the diverged throughput classify the mechanism instead of
                # leaving it unlabelled tick after tick.
                y_keys = {"y_observed", "y_predicted", "y_unusable"}
                v_complete = not any(value for key, value in missing.items() if key not in y_keys)
                y_complete = not any(missing.get(key) for key in y_keys) or (
                    dead_rank and y_verdict is CusumResult.DIVERGED
                )
                fully_observable = v_complete and y_complete
                q_per_mech[mech.mechanism_id] = (
                    self.qv.classify_quadrant(v_verdict, y_verdict) if fully_observable else None
                )
                observable_edges = self._observable_edge_ids(
                    bundle, set(available_v), set(available_y)
                )
                touched_edge_ids.update(observable_edges)
                diagnostic_status = (
                    "evaluated"
                    if fully_observable
                    else "partially_evaluated"
                    if v_verdict is not None or y_verdict is not None
                    else "unobservable"
                )
                rank_diagnostics.append(
                    {
                        "job_id": job_id,
                        "rank_id": rank_id,
                        "mechanism_id": mech.mechanism_id,
                        "status": diagnostic_status,
                        "missing": missing,
                        "cusum": {
                            "V": self._cusum_axis_diagnostics(available_v, v_obs, v_pred, v_params),
                            "Y": self._cusum_axis_diagnostics(available_y, y_obs, y_pred, y_params),
                        },
                        "v_verdict": getattr(v_verdict, "value", v_verdict),
                        "y_verdict": getattr(y_verdict, "value", y_verdict),
                        "q_label": q_per_mech[mech.mechanism_id],
                        "_edge_ids": observable_edges,
                    }
                )

            icp_per_edge: dict[str, Any] = {}
            icp_diagnostics: dict[str, dict[str, Any]] = {}
            for edge_id in sorted(touched_edge_ids):
                edge = self._resolve_edge(edge_id)
                if edge is None:
                    continue
                if edge_id not in tick_icp_details:
                    tick_icp_details[edge_id] = self.icp.compute_icp_details_per_edge(
                        edge=edge,
                        evidence_store=self.evidence_store,
                        before_tick=ctx.tick,
                    )
                details = tick_icp_details[edge_id]
                icp_per_edge[edge_id] = details["result"]
                icp_diagnostics[edge_id] = details

            for diagnostic in rank_diagnostics:
                edge_ids = diagnostic.pop("_edge_ids", [])
                if edge_ids:
                    diagnostic["icp"] = [icp_diagnostics[edge_id] for edge_id in edge_ids]
            ctx.mechanism_diagnostics.extend(rank_diagnostics)

            j_realized = self._j_realized(y_observed_mean, w_t_snapshot, z_star_snapshot)
            deploy_timestamp = getattr(rank_telem, "deploy_timestamp_utc", None)
            if deploy_timestamp is None:
                deploy_timestamp = time.time()
            row = EvidenceRow(
                row_id=f"{ctx.tick}_{job_id}_{rank_id}",
                tick=ctx.tick,
                deploy_timestamp_utc=float(deploy_timestamp),
                job_id=job_id,
                rank_id=rank_id,
                env_label=env_label,
                X=x,
                V_observed_trajectory=v_obs,
                V_predicted_trajectory=v_pred,
                y_observed_trajectory=y_obs,
                y_predicted=y_pred,
                y_observed_mean=y_observed_mean,
                residuals_per_v=residuals_per_v,
                residuals_per_y=residuals_per_y,
                mechanism_ids=[m.mechanism_id for m in applicable],
                cusum_per_mechanism=cusum_per_mech,
                q_label_per_mechanism=q_per_mech,
                icp_result_per_edge=icp_per_edge,
                w_t_snapshot=w_t_snapshot,
                z_star_snapshot=z_star_snapshot,
                J_realized=j_realized,
                # forward-looking sigma terms (EIG, switch cost) are zero at
                # observation time; v0 stamps the realized exploit term only
                sigma_realized=j_realized,
                deployment_id=(deployment.prediction_lineage or {}).get("deployment_id"),
                evidence_available_timestamp_utc=time.time(),
                prediction_lineage=deployment.prediction_lineage,
            )
            self.evidence_store.append_row(row)
            ctx.evidence_rows.append(row)

        ctx.active_health = self._update_active_health(ctx, job_features, health_samples)

        return FSMState.S3_SLOW_UPDATE

    def S3(self, ctx: TickContext) -> FSMState:
        """Apply the learning updates, then refresh the slow-loop knobs.

        Part 1 - Evaluate decision-time DRO coverage, then append every
        current residual to DRO history.

        Part 2 - Beta fan-out: every decided (row, mechanism) pair updates
        that mechanism's Beta and the Betas of ITS edges, with the edge
        delta modulated by that edge's ICP result. An edge shared by
        several applicable mechanisms receives one update per mechanism
        context - each context is independent evidence about the edge.
        ConfidenceService also records env coverage and recency (single
        writer for all confidence state).

        Part 3 - SlowLoop.slow_update_all receives coverage, the observed
        swap rate (recorded by last tick's S6), the R2 gradient (v0 stub),
        and annealed targets.

        Part 4 - meta cadence: CUSUM (delta, h) recalibration from
        accumulated residual history every recalibrate_every ticks.
        """
        ctx.confidence_diagnostics = []
        slow_before = self._slow_state_snapshot()
        coverage = self._observed_coverage_details(ctx)
        for row in ctx.evidence_rows:
            comparable_y = set(row.residuals_per_y)
            self.dro.append_residual_history(
                pred_y={
                    name: value for name, value in row.y_predicted.items() if name in comparable_y
                },
                obs_y={
                    name: value
                    for name, value in row.y_observed_mean.items()
                    if name in comparable_y
                },
            )

        did_confidence_update = False
        for row in ctx.evidence_rows:
            for mid, q in row.q_label_per_mechanism.items():
                if q is None:
                    verdicts = row.cusum_per_mechanism.get(mid) or ()
                    if any(
                        getattr(verdict, "value", verdict) == "diverged" for verdict in verdicts
                    ):
                        try:
                            mechanism = self.mechanism_registry.get_mechanism(mid)
                        except KeyError:
                            continue
                        observable_edges = self._observable_edge_ids(
                            self._bundle(mechanism),
                            set(row.residuals_per_v),
                            set(row.residuals_per_y),
                            include_v=any(
                                getattr(verdict, "value", verdict) == "diverged"
                                for verdict in verdicts[:1]
                            ),
                            include_y=any(
                                getattr(verdict, "value", verdict) == "diverged"
                                for verdict in verdicts[1:2]
                            ),
                        )
                        apply_partial = getattr(
                            self.confidence_service, "apply_partial_divergence", None
                        )
                        if not callable(apply_partial):
                            continue
                        apply_partial(
                            mid,
                            observable_edges,
                            PARTIAL_DIVERGENCE_BETA,
                            icp_results=row.icp_result_per_edge,
                            env_label=row.env_label,
                            tick=ctx.tick,
                        )
                        did_confidence_update = True
                        ctx.confidence_diagnostics.append(
                            {
                                "evidence_row_id": row.row_id,
                                "job_id": row.job_id,
                                "rank_id": row.rank_id,
                                "mechanism_id": mid,
                                "q_label": None,
                                "partial_divergence": True,
                                "observable_edges": observable_edges,
                                "beta_delta": PARTIAL_DIVERGENCE_BETA,
                            }
                        )
                    continue
                mechanism_before = self._mechanism_confidence_snapshot(mid)
                mechanism_delta = self.confidence_service.get_delta_c_mechanism(q)
                self.confidence_service.apply_delta_c_mechanism(
                    mid, q, env_label=row.env_label, tick=ctx.tick
                )
                did_confidence_update = True
                diagnostic = {
                    "evidence_row_id": row.row_id,
                    "job_id": row.job_id,
                    "rank_id": row.rank_id,
                    "mechanism_id": mid,
                    "q_label": getattr(q, "value", q),
                    "mechanism": {
                        "before": mechanism_before,
                        "delta": self._confidence_delta(mechanism_delta),
                        "after": self._mechanism_confidence_snapshot(mid),
                    },
                    "edges": [],
                }
                try:
                    mech = self.mechanism_registry.get_mechanism(mid)
                except KeyError:
                    log.warning("row %s references unknown mechanism %s", row.row_id, mid)
                    ctx.confidence_diagnostics.append(diagnostic)
                    continue
                for edge_id in mech.edge_ids:
                    icp_result = row.icp_result_per_edge.get(edge_id, ICPResult.UNDECIDED)
                    edge_before = self._edge_confidence_snapshot(edge_id)
                    edge_delta = self.confidence_service.get_delta_c_edge(q, icp_result)
                    self.confidence_service.apply_delta_c_edge(
                        edge_id, q, icp_result, env_label=row.env_label, tick=ctx.tick
                    )
                    diagnostic["edges"].append(
                        {
                            "edge_id": edge_id,
                            "icp_result": getattr(icp_result, "value", icp_result),
                            "before": edge_before,
                            "delta": self._confidence_delta(edge_delta),
                            "after": self._edge_confidence_snapshot(edge_id),
                        }
                    )
                ctx.confidence_diagnostics.append(diagnostic)

        flush_confidence = getattr(self.confidence_service, "flush", None)
        if did_confidence_update and callable(flush_confidence):
            flush_confidence()

        observed_swap_rate = self._observed_swap_rate(ctx)
        r2_gradient = self._r2_gradient(ctx)
        target_overrides = self.slow_loop.anneal_targets(ctx.tick)
        dro_before = self._dro_parameters()
        ctx.new_slow_state = self.slow_loop.slow_update_all(
            tick=ctx.tick,
            observed_swap_rate=observed_swap_rate,
            observed_coverage=coverage["value"],
            r2_gradient=r2_gradient,
            target_overrides=target_overrides,
        )

        recalibration = {"ran": False}
        if (
            self.recalibrate_every > 0
            and ctx.tick > 0
            and ctx.tick % self.recalibrate_every == 0
            and hasattr(self.slow_loop, "recalibrate_cusum_params")
        ):
            recalibration = {
                "ran": True,
                "before_v": slow_before.get("cusum_params_v", {}),
                "before_y": slow_before.get("cusum_params_y", {}),
            }
            params_v, params_y = self.slow_loop.recalibrate_cusum_params()
            recalibration["after_v"] = params_v
            recalibration["after_y"] = params_y
            log.info("CUSUM (delta, h) recalibrated at tick %d", ctx.tick)

        ctx.slow_update_diagnostics = {
            "inputs": {
                "observed_swap_rate": observed_swap_rate,
                "observed_coverage": coverage["value"],
                "r2_gradient": r2_gradient,
                "target_overrides": target_overrides,
            },
            "before": slow_before,
            "after": self._slow_state_snapshot(),
            "dro": {
                "before": dro_before,
                "after": self._dro_parameters(),
                "coverage": coverage,
            },
            "cusum_recalibration": recalibration,
        }

        return FSMState.S4_AGENTIC_PLAN

    def S4(self, ctx: TickContext) -> FSMState:
        """Run the root RLM planner once; it returns the candidate plan.

        The harness owns K_P sampling, the budget-first specialist
        protocol, best-of-K selection, and its own bounded-trajectory
        safety. On a repair iteration (from S5) the harness already holds
        the violations via receive_validator_feedback.
        """
        ctx.candidate_plan = self.agent.run_agent_loop(
            cluster_snapshot=ctx.cluster_snapshot,
            slow_state=ctx.new_slow_state,
            evidence_store=self.evidence_store,
            mechanism_registry=self.mechanism_registry,
            tick=ctx.tick,
        )
        return FSMState.S5_VALIDATE_PLAN

    def S5(self, ctx: TickContext) -> FSMState:
        """Validate the candidate plan; repair once; fall back to keep-all.

        A None candidate (harness produced nothing usable) skips straight
        to the fallback. Violations from the first failure go back to the
        agent for one repair iteration; a second failure keeps the running
        cluster untouched and defers pending jobs.
        """
        if ctx.candidate_plan is None:
            log.warning("S4 returned no plan at tick %d; keep-all fallback", ctx.tick)
            ctx.validated_plan = self._fallback_keep_all(ctx)
            return FSMState.S6_DEPLOY

        result = self.plan_validator.val_plan(
            plan=ctx.candidate_plan,
            cluster_snapshot=ctx.cluster_snapshot,
            slow_state=ctx.new_slow_state,
        )
        if result.feasible:
            ctx.validated_plan = ctx.candidate_plan
            return FSMState.S6_DEPLOY

        if ctx.s5_repair_count < ctx.max_s5_repairs:
            ctx.s5_repair_count += 1
            log.info(
                "S5 infeasible at tick %d; repair %d: %s",
                ctx.tick,
                ctx.s5_repair_count,
                result.violations,
            )
            self.agent.receive_validator_feedback(result.violations)
            return FSMState.S4_AGENTIC_PLAN

        log.warning(
            "S5 still infeasible after %d repair(s) at tick %d: %s; keep-all",
            ctx.max_s5_repairs,
            ctx.tick,
            result.violations,
        )
        ctx.validated_plan = self._fallback_keep_all(ctx)
        return FSMState.S6_DEPLOY

    def S6(self, ctx: TickContext) -> FSMState:
        """Submit the validated plan and record swap bookkeeping.

        The executor owns A/B canary semantics and submits only changed or
        new ladders (keep / defer / diagnose are bookkeeping).
        The swap count recorded here feeds next tick's observed_swap_rate,
        which drives lambda_swit.
        """
        new_attempts = self._record_deployment_requests(ctx)
        self._record_prediction_ledger(ctx)
        try:
            ctx.deploy_acks = self.executor.send_to_executor(ctx.validated_plan)
        except Exception as exc:
            self._record_deployment_acks(
                new_attempts,
                [{"status": "error", "error": str(exc)}],
                ctx.tick,
            )
            raise
        self._record_deployment_acks(new_attempts, ctx.deploy_acks, ctx.tick)

        self._last_swap_count = self._count_plan_swaps(ctx)
        self._last_active_count = self._active_job_count(ctx)

        if self.trace is not None:
            persist_tick = getattr(self.trace, "persist_tick", None)
            if not callable(persist_tick):
                return FSMState.S7_EXIT_TICK
            try:
                persist_tick(ctx)
            except Exception:
                log.exception("trace persist failed at tick %d", ctx.tick)
        return FSMState.S7_EXIT_TICK

    def _reconcile_deployments(self, ctx: TickContext) -> list[dict[str, Any]]:
        """Compare prior PLACE/SWAP requests with the next frozen active snapshot."""
        snapshot = ctx.cluster_snapshot
        active_jobs = (
            list(snapshot.active_jobs_summary() or [])
            if snapshot is not None and hasattr(snapshot, "active_jobs_summary")
            else []
        )
        pending_jobs = (
            list(snapshot.pending_jobs_summary() or [])
            if snapshot is not None and hasattr(snapshot, "pending_jobs_summary")
            else []
        )
        active_ranks: dict[str, set[str]] = defaultdict(set)
        active_chains: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for job in active_jobs:
            jid = str(job.get("job_id", job.get("id")))
            for chain in job.get("active_chains") or job.get("current_ladder") or []:
                rank_id = (chain.get("shape_json") or {}).get("rank_id") or chain.get("rank_id")
                if rank_id:
                    active_ranks[jid].add(str(rank_id))
                    active_chains[jid][str(rank_id)].append(chain)
        active_shapes = {
            jid: {rank_id: self._rank_group_signature(chains) for rank_id, chains in groups.items()}
            for jid, groups in active_chains.items()
        }
        pending_by_id = {str(job.get("job_id", job.get("id"))): job for job in pending_jobs}
        active_by_id = {str(job.get("job_id", job.get("id"))): job for job in active_jobs}
        reconciled = []
        completed = []
        for jid, request in self._deployment_ledger.items():
            attempts = request.get("attempt_details") or [
                {
                    "attempt_number": index + 1,
                    "request_tick": request.get("request_tick", ctx.tick),
                    "rank_ids": rank_ids,
                    "rank_shapes": request.get("rank_shapes") or {},
                    "state": "submitted",
                }
                for index, rank_ids in enumerate(
                    request.get("rank_id_attempts") or [request.get("rank_ids") or []]
                )
            ]
            matching_attempt = next(
                (
                    attempt
                    for attempt in attempts
                    if self._deployment_attempt_active(
                        attempt,
                        active_ranks.get(jid, set()),
                        active_shapes.get(jid, {}),
                    )
                ),
                None,
            )
            latest_attempt = attempts[-1]
            if matching_attempt is not None:
                matching_attempt["state"] = "materialized"
                matching_attempt["terminal_tick"] = ctx.tick
                if (
                    matching_attempt.get("action_type") or request.get("action_type")
                ) == ActionType.SWAP.value:
                    self._health_state.setdefault(jid, {})["last_swap_tick"] = ctx.tick
                status = "active"
                completed.append(jid)
            else:
                attempt_state = str(latest_attempt.get("state") or "submitted")
                request_tick = int(latest_attempt.get("request_tick", request["request_tick"]))
                if attempt_state in {"timed_out", "executor_rejected"}:
                    status = "deployment_not_materialized"
                elif ctx.tick - request_tick <= _DEPLOYMENT_GRACE_TICKS:
                    status = "deployment_pending"
                else:
                    latest_attempt["state"] = "timed_out"
                    latest_attempt["terminal_tick"] = ctx.tick
                    latest_attempt["retry_after_tick"] = ctx.tick + _DEPLOYMENT_RETRY_BACKOFF_TICKS
                    status = "deployment_not_materialized"
            request["attempt_details"] = attempts
            row = {
                **request,
                "job_id": jid,
                "status": status,
                "observed_rank_ids": sorted(active_ranks.get(jid, set())),
            }
            reconciled.append(row)
            descriptor = pending_by_id.get(jid) or active_by_id.get(jid)
            if descriptor is not None:
                descriptor["deployment_status"] = status
                descriptor["deployment_attempts"] = request.get("attempts", 1)
                descriptor["deployment_action_type"] = request.get("action_type")
                descriptor["last_requested_shapes"] = copy.deepcopy(request.get("shapes") or [])
                if status == "deployment_not_materialized":
                    terminal_attempts = [
                        attempt
                        for attempt in attempts
                        if attempt.get("state") in {"timed_out", "executor_rejected"}
                    ]
                    retry_after_tick = latest_attempt.get("retry_after_tick")
                    attempt_count = int(request.get("attempts") or len(attempts))
                    retry_exhausted = attempt_count >= _DEPLOYMENT_MAX_ATTEMPTS
                    descriptor["deployment_retry_after_tick"] = retry_after_tick
                    descriptor["deployment_retry_exhausted"] = retry_exhausted
                    # Exhaustion reports how many attempts a shape series has cost;
                    # it does NOT retire the job. Retrying is paced by the backoff
                    # alone, so a job whose placements keep failing stays plannable
                    # against untried hardware instead of being stranded for the
                    # rest of the run while capacity sits idle.
                    descriptor["deployment_retry_allowed"] = bool(
                        retry_after_tick is not None and ctx.tick >= int(retry_after_tick)
                    )
                    descriptor["attempted_deployment_identities"] = [
                        copy.deepcopy(attempt.get("deployment_identity"))
                        for attempt in terminal_attempts
                        if attempt.get("deployment_identity")
                    ]
                    descriptor["recent_failures"] = max(
                        1, int(descriptor.get("recent_failures") or 0)
                    )
        for jid in completed:
            self._deployment_ledger.pop(jid, None)
        return reconciled

    def _record_deployment_requests(self, ctx: TickContext) -> list[tuple[str, int]]:
        new_attempts: list[tuple[str, int]] = []
        for action in getattr(ctx.validated_plan, "actions", []) or []:
            if action.type not in {ActionType.PLACE, ActionType.SWAP} or not action.ladder:
                continue
            shapes = [
                {
                    "env": list(rank.env or []),
                    "instance_type": rank.config.get("instance_type"),
                    "gpu_count": rank.config.get("gpu_count", rank.config.get("count")),
                    "tp": rank.config.get("tp"),
                    "pp": rank.config.get("pp"),
                    "sp": rank.config.get("sp"),
                    "ep": rank.config.get("ep"),
                    "cp": rank.config.get("cp"),
                    "num_nodes_per_chain": rank.config.get("num_nodes_per_chain"),
                    "interconnect_type": rank.config.get("interconnect_type"),
                    "n_replicas": rank.n_replicas,
                }
                for rank in action.ladder
            ]
            previous = self._deployment_ledger.get(action.job_id)
            previous_record = previous or {}
            current_rank_ids = {str(rank.rank_id) for rank in action.ladder if rank.rank_id}
            rank_shapes = {
                str(rank.rank_id): self._deployment_shape_signature(rank.to_dict())
                for rank in action.ladder
                if rank.rank_id
            }
            attempt_number = int(previous_record.get("attempts", 0)) + 1
            attempt = {
                "attempt_number": attempt_number,
                "request_tick": ctx.tick,
                "action_type": action.type.value,
                "rank_ids": sorted(current_rank_ids),
                "rank_shapes": rank_shapes,
                "deployment_identity": deployment_ladder_identity(
                    rank.to_dict() for rank in action.ladder
                ),
                "executor_ack": [],
                "state": "submitted",
                "terminal_tick": None,
                "retry_after_tick": None,
            }
            attempt_details = [*list(previous_record.get("attempt_details") or []), attempt]
            self._deployment_ledger[action.job_id] = {
                "request_tick": ctx.tick,
                "first_tick": previous_record.get("first_tick", ctx.tick),
                "attempts": attempt_number,
                "action_type": action.type.value,
                "rank_ids": sorted(current_rank_ids),
                "rank_shapes": rank_shapes,
                "attempt_details": attempt_details,
                "shapes": shapes,
            }
            new_attempts.append((action.job_id, attempt_number))
        return new_attempts

    def _record_deployment_acks(
        self,
        attempt_refs: list[tuple[str, int]],
        acks: list[Any],
        tick: int,
    ) -> None:
        copied_acks = copy.deepcopy(list(acks or []))
        rejected = any(
            str((ack or {}).get("status") or "").lower() in {"failed", "rejected", "error"}
            for ack in copied_acks
            if isinstance(ack, dict)
        )
        for job_id, attempt_number in attempt_refs:
            request = self._deployment_ledger.get(job_id) or {}
            attempt = next(
                (
                    item
                    for item in request.get("attempt_details") or []
                    if int(item.get("attempt_number") or 0) == attempt_number
                ),
                None,
            )
            if attempt is None:
                continue
            attempt["executor_ack"] = copied_acks
            attempt["state"] = "executor_rejected" if rejected else "acknowledged"
            if rejected:
                attempt["terminal_tick"] = tick
                attempt["retry_after_tick"] = tick + _DEPLOYMENT_RETRY_BACKOFF_TICKS

    @staticmethod
    def _deployment_shape_signature(raw: dict[str, Any], replicas: int | None = None) -> tuple:
        return deployment_rank_identity(raw, replicas=replicas)

    @staticmethod
    def _deployment_attempt_active(
        attempt: dict[str, Any],
        observed_rank_ids: set[str],
        observed_shapes: dict[str, tuple],
    ) -> bool:
        expected_ids = set(attempt.get("rank_ids") or [])
        expected_shapes = dict(attempt.get("rank_shapes") or {})
        ids_active = bool(expected_ids) and expected_ids <= observed_rank_ids
        if not expected_shapes:
            return ids_active
        return ids_active and all(
            observed_shapes.get(rank_id) == tuple(expected_shapes.get(rank_id) or ())
            for rank_id in expected_ids
        )

    @classmethod
    def _rank_group_signature(cls, chains: list[dict[str, Any]]) -> tuple:
        signatures = {cls._deployment_shape_signature(chain, len(chains)) for chain in chains}
        return next(iter(signatures)) if len(signatures) == 1 else ("mixed_replica_shapes",)

    @staticmethod
    def _prediction_shape_signature(raw: dict[str, Any], replicas: int | None = None) -> tuple:
        """Identity for safely restoring predictions across Store round trips."""
        shape = dict(raw.get("shape_json") or raw)
        config = dict(shape.get("config") or shape)
        return (
            deployment_rank_identity(raw, replicas=replicas),
            config.get("weight_dtype"),
            config.get("activation_dtype"),
            config.get("kvcache_dtype"),
            config.get("weight_quantization_bits"),
            config.get("weight_quantization_method"),
            config.get("engine_name"),
            config.get("engine_version"),
            config.get("router_policy"),
            config.get("scheduling_policy"),
            config.get("preemption_policy"),
            config.get("prefix_cache_enabled"),
            config.get("chunked_prefill_enable"),
            config.get("interconnect_type"),
        )

    @classmethod
    def _prediction_rank_group_signature(cls, chains: list[dict[str, Any]]) -> tuple:
        signatures = {cls._prediction_shape_signature(chain, len(chains)) for chain in chains}
        return next(iter(signatures)) if len(signatures) == 1 else ("mixed_replica_shapes",)

    def _record_prediction_ledger(self, ctx: TickContext) -> None:
        for action in getattr(ctx.validated_plan, "actions", []) or []:
            for rank in action.ladder or []:
                if not rank.rank_id:
                    continue
                self._prediction_ledger[(action.job_id, str(rank.rank_id))] = {
                    "predicted_y": copy.deepcopy(rank.predicted_y or {}),
                    "predicted_v": copy.deepcopy(rank.predicted_v or {}),
                    "prediction_lineage": copy.deepcopy(rank.prediction_lineage or {}),
                    "mechanism_id": rank.mechanism_id or action.mechanism_id,
                    "shape_signature": self._prediction_shape_signature(rank.to_dict()),
                }

    # Rank failure reasons that are a property of the shape, not of the moment.
    _DETERMINISTIC_RANK_FAILURES = frozenset(
        {"SOFTWARE_STACK_RANK_FAILURE", "OOM", "MODEL_CATALOG_INVALID"}
    )

    def _record_dead_shape(
        self,
        ctx: TickContext,
        job_id: str,
        rank_id: str,
        features: dict[str, Any],
        env_label: Any,
        x: dict[str, Any],
        queue_depth: float,
    ) -> None:
        model_id = features.get("model_id") or x.get("model_id")
        gpu_type = env_gpu_type(env_label) or x.get("gpu_type")
        if not model_id or not gpu_type:
            return
        try:
            tp, pp = int(x.get("tp") or 1), int(x.get("pp") or 1)
        except (TypeError, ValueError):
            return
        key = (str(model_id), str(gpu_type), tp, pp)
        entry = self._dead_shapes.setdefault(
            key,
            {
                "model_id": str(model_id),
                "gpu_type": str(gpu_type),
                "tp": tp,
                "pp": pp,
                "reason": "zero_throughput_under_load",
                "first_tick": ctx.tick,
                "ticks": 0,
            },
        )
        entry.update(
            last_tick=ctx.tick,
            ticks=int(entry.get("ticks") or 0) + 1,
            job_id=job_id,
            rank_id=rank_id,
            queue_depth=float(queue_depth),
        )

    def _annotate_dead_shapes(self, ctx: TickContext) -> None:
        """Attach observed_dead_shapes to every job descriptor for its model.

        Two sources: ranks S2 saw serve nothing under load this run, and rank
        launch failures the Store reports with a deterministic reason. Keyed on
        (gpu_type, tp, pp) - never replicas - so a dead shape is not retried
        wider.
        """
        snapshot = ctx.cluster_snapshot
        if snapshot is None:
            return
        descriptors: list[dict[str, Any]] = []
        for accessor in ("active_jobs_summary", "pending_jobs_summary"):
            if hasattr(snapshot, accessor):
                descriptors.extend(job for job in (getattr(snapshot, accessor)() or []) if job)
        for job in descriptors:
            model_id = str((job.get("job_features") or {}).get("model_id") or "")
            dead: dict[tuple[str, int, int], dict[str, Any]] = {}
            for entry in self._dead_shapes.values():
                if entry.get("model_id") == model_id:
                    dead[(entry["gpu_type"], entry["tp"], entry["pp"])] = dict(entry)
            for failure in job.get("recent_rank_failures") or []:
                if not isinstance(failure, dict):
                    continue
                if str(failure.get("reason_code") or "") not in self._DETERMINISTIC_RANK_FAILURES:
                    continue
                shape = dict(failure.get("shape_json") or {})
                config = dict(shape.get("config") or shape)
                gpu_type = env_gpu_type(shape.get("env")) or config.get("gpu_type")
                if not gpu_type:
                    continue
                try:
                    tp, pp = int(config.get("tp") or 1), int(config.get("pp") or 1)
                except (TypeError, ValueError):
                    continue
                dead.setdefault(
                    (str(gpu_type), tp, pp),
                    {
                        "model_id": model_id,
                        "gpu_type": str(gpu_type),
                        "tp": tp,
                        "pp": pp,
                        "reason": str(failure.get("reason_code")),
                        "rank_id": failure.get("rank_id"),
                    },
                )
            if dead:
                job["observed_dead_shapes"] = sorted(
                    dead.values(), key=lambda item: (item["gpu_type"], item["tp"], item["pp"])
                )

    def _update_active_health(
        self,
        ctx: TickContext,
        job_features: dict[str, dict[str, Any]],
        samples: dict[str, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        health: dict[str, dict[str, Any]] = {}
        active_jobs = list(ctx.cluster_snapshot.active_jobs_summary() or [])
        incomplete_jobs = {
            str(item.get("job_id"))
            for item in (ctx.telemetry_diagnostics.get("missing_ranks") or [])
            if item.get("job_id")
        }
        for job in active_jobs:
            jid = str(job.get("job_id", job.get("id")))
            rows = samples.get(jid, [])
            features = job_features.get(jid, {})
            observed = {
                name: max(
                    (float(row[name]) for row in rows if row.get(name) is not None),
                    default=None,
                )
                for name in ("p99_ttft_ms", "p99_tpot_ms", "depth_req_q")
            }
            throughput_values = [
                float(row["throughput_token_per_sec"])
                for row in rows
                if row.get("throughput_token_per_sec") is not None
            ]
            observed_throughput = sum(throughput_values) if throughput_values else None
            observed["throughput_token_per_sec"] = observed_throughput
            reasons = []
            ttft_target = features.get("target_p99_ttft_ms")
            tpot_target = features.get("target_p99_tpot_ms")
            previous = self._health_state.get(jid, {})
            previous_depth = float(previous.get("last_queue_depth") or 0.0)
            current_depth = float(observed["depth_req_q"] or 0.0)
            try:
                job_type = str(
                    features.get("type")
                    or features.get("workload_type")
                    or job.get("kind")
                    or "online"
                ).lower()
                if job_type == "batch":
                    deadline_hours = features.get("deadline_hours")
                    if deadline_hours is None:
                        deadline_hours = features.get("deadline_hrs", 24.0)
                    required_throughput = float(features.get("total_token_budget") or 0.0) / max(
                        1.0,
                        float(deadline_hours) * 3600.0,
                    )
                else:
                    required_throughput = (
                        float(features.get("request_arrival_rate") or 0.0)
                        * float(
                            features.get("osl_token_avg")
                            or features.get("output_len_tokens_avg")
                            or 0.0
                        )
                        * float(features.get("headroom_factor") or 1.5)
                    )
            except (TypeError, ValueError, OverflowError):
                required_throughput = 0.0
            if job_type == "batch":
                if observed_throughput is not None and observed_throughput <= 0:
                    reasons.append("zero_throughput")
                elif (
                    observed_throughput is not None
                    and required_throughput > 0
                    and observed_throughput < required_throughput
                ):
                    reasons.append("throughput_shortfall")
            else:
                has_backlog = current_depth > 0
                if has_backlog and observed_throughput is not None and observed_throughput <= 0:
                    reasons.append("zero_throughput")
                elif (
                    has_backlog
                    and observed_throughput is not None
                    and required_throughput > 0
                    and observed_throughput < 0.5 * required_throughput
                ):
                    reasons.append("throughput_shortfall")
                if (
                    ttft_target
                    and observed["p99_ttft_ms"] is not None
                    and observed["p99_ttft_ms"] > float(ttft_target)
                ):
                    reasons.append("ttft_breach")
                if (
                    tpot_target
                    and observed["p99_tpot_ms"] is not None
                    and observed["p99_tpot_ms"] > float(tpot_target)
                ):
                    reasons.append("tpot_breach")
                if current_depth >= 10 and current_depth > max(10.0, previous_depth * 1.1):
                    reasons.append("queue_growing")
                if current_depth >= 100:
                    reasons.append("queue_critical")
            complete_samples = bool(rows) and all(
                sample.get("telemetry_complete") is True for sample in rows
            )
            telemetry_incomplete = jid in incomplete_jobs or not complete_samples
            critical = (
                "zero_throughput" in reasons
                or (
                    observed_throughput is not None
                    and required_throughput > 0
                    and observed_throughput < 0.1 * required_throughput
                    and (job_type == "batch" or current_depth > 0)
                )
                or (job_type != "batch" and current_depth >= 100)
                or (
                    job_type != "batch"
                    and ttft_target
                    and observed["p99_ttft_ms"] is not None
                    and observed["p99_ttft_ms"] >= 3 * float(ttft_target)
                )
            )
            if telemetry_incomplete and not critical:
                entry = {
                    "status": "unknown",
                    "reasons": ["telemetry_incomplete"],
                    "unhealthy_ticks": int(previous.get("unhealthy_ticks", 0)),
                    "observed": observed,
                    "observed_slo_met": None,
                    "rehabilitation_eligible": False,
                    "last_swap_tick": int(previous.get("last_swap_tick", -10_000)),
                    "last_queue_depth": float(previous.get("last_queue_depth") or 0.0),
                }
                self._health_state[jid] = copy.deepcopy(entry)
                job["health"] = copy.deepcopy(entry)
                job["recent_failures"] = entry["unhealthy_ticks"]
                health[jid] = copy.deepcopy(entry)
                continue
            if telemetry_incomplete:
                reasons.append("telemetry_incomplete")
            streak = int(previous.get("unhealthy_ticks", 0)) + 1 if reasons else 0
            last_swap_tick = int(previous.get("last_swap_tick", -10_000))
            status = (
                "critical"
                if critical
                else "degraded"
                if reasons
                else "healthy"
                if rows
                else "unknown"
            )
            entry = {
                "status": status,
                "reasons": reasons,
                "unhealthy_ticks": streak,
                "observed": observed,
                "observed_slo_met": None
                if telemetry_incomplete
                else bool(rows)
                and not any(
                    reason in reasons
                    for reason in (
                        "zero_throughput",
                        "throughput_shortfall",
                        "ttft_breach",
                        "tpot_breach",
                    )
                ),
                "rehabilitation_eligible": bool(
                    (critical or streak >= 2) and ctx.tick - last_swap_tick >= 2
                ),
                "last_swap_tick": last_swap_tick,
                "last_queue_depth": current_depth,
            }
            self._health_state[jid] = copy.deepcopy(entry)
            job["health"] = copy.deepcopy(entry)
            job["recent_failures"] = streak
            health[jid] = copy.deepcopy(entry)
        return health

    # ------------------------------------------------------------------
    # Terminal handlers
    # ------------------------------------------------------------------

    def _handle_s7(self, ctx: TickContext) -> None:
        """Sleep the remainder of the tick interval (no drift)."""
        if self.tick_interval_sec <= 0:
            return
        elapsed = time.time() - ctx.tick_started_at
        remaining = self.tick_interval_sec - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            log.warning("tick %d overran its interval by %.1fs", ctx.tick, -remaining)

    def _handle_abort(self, ctx: TickContext) -> None:
        """Deploy the keep-all fallback after an unrecoverable error."""
        try:
            ctx.validated_plan = self._fallback_keep_all(ctx)
            ctx.deploy_acks = self.executor.send_to_executor(ctx.validated_plan)
        except Exception:
            log.exception("keep-all fallback deploy failed; cluster held this tick")
        if self.trace is not None:
            persist_tick = getattr(self.trace, "persist_tick", None)
            if not callable(persist_tick):
                return
            try:
                persist_tick(ctx)
            except Exception:
                log.exception("trace persist failed at tick %d (abort)", ctx.tick)

    # ------------------------------------------------------------------
    # S2 helpers
    # ------------------------------------------------------------------

    def _build_deployment_x_index(self, ctx: TickContext):
        """Rank-level deployment X, built from Store/catalog state once per tick."""
        x_fields = getattr(self.candidate_graph, "x", None)
        if not x_fields:
            raise ValueError("candidate graph X fields are required for deployment X")
        if not self._has_active_chains(ctx):
            return build_deployment_x_index(
                ctx.cluster_snapshot,
                hardware_catalog={},
                model_catalogs={},
                x_fields=x_fields,
            )
        index = build_deployment_x_index(
            ctx.cluster_snapshot,
            hardware_catalog=self._hardware_catalog(),
            model_catalogs=self._model_catalogs(ctx),
            x_fields=x_fields,
        )
        observed_shapes: dict[tuple[str, str], tuple] = {}
        for job in ctx.cluster_snapshot.active_jobs_summary() or []:
            jid = str(job.get("job_id", job.get("id")))
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for chain in job.get("active_chains") or job.get("current_ladder") or []:
                shape = dict(chain.get("shape_json") or chain)
                rank_id = shape.get("rank_id")
                if rank_id:
                    groups[str(rank_id)].append(chain)
            for rank_id, chains in groups.items():
                observed_shapes[(jid, rank_id)] = self._prediction_rank_group_signature(chains)
        for key, deployment in index.by_rank.items():
            cached = self._prediction_ledger.get(key)
            if not cached or cached.get("shape_signature") != observed_shapes.get(key):
                continue
            if not deployment.prediction_lineage:
                deployment.prediction_lineage = cached.get("prediction_lineage")
            deployment.y_predicted = {
                **dict(cached.get("predicted_y") or {}),
                **deployment.y_predicted,
            }
            deployment.v_predicted = {
                **dict(cached.get("predicted_v") or {}),
                **deployment.v_predicted,
            }
        return index

    @staticmethod
    def _has_active_chains(ctx: TickContext) -> bool:
        snapshot = ctx.cluster_snapshot
        if snapshot is None:
            return False
        jobs = (
            snapshot.active_jobs_summary()
            if hasattr(snapshot, "active_jobs_summary")
            else getattr(snapshot, "active_jobs", [])
        )
        return any(job.get("active_chains") for job in jobs or [])

    def _telemetry_diagnostics(self, ctx: TickContext) -> dict[str, Any]:
        """Summarize S1 coverage without embedding raw telemetry rows."""
        by_rank = getattr(ctx.deployment_x, "by_rank", {}) or {}
        expected = {(str(job_id), str(rank_id)) for job_id, rank_id in by_rank}
        observed = list(self.telemetry.iter_per_rank(ctx.telemetry))
        observed_by_rank = {(str(rank.job_id), str(rank.rank_id)): rank for rank in observed}
        return {
            "window": {
                "start": getattr(ctx.telemetry, "start", None),
                "end": getattr(ctx.telemetry, "end", None),
            },
            "expected_rank_count": len(expected),
            "observed_rank_count": len(observed_by_rank),
            "missing_ranks": [
                {"job_id": job_id, "rank_id": rank_id}
                for job_id, rank_id in sorted(expected - set(observed_by_rank))
            ],
            "unexpected_ranks": [
                {"job_id": job_id, "rank_id": rank_id}
                for job_id, rank_id in sorted(set(observed_by_rank) - expected)
            ],
            "observed_ranks": [
                {
                    "job_id": job_id,
                    "rank_id": rank_id,
                    "v_sample_counts": {
                        name: len(values) for name, values in rank.v_observed.items()
                    },
                    "y_sample_counts": {
                        name: len(values) for name, values in rank.y_observed.items()
                    },
                }
                for (job_id, rank_id), rank in sorted(observed_by_rank.items())
            ],
        }

    def _hardware_catalog(self) -> dict[str, Any]:
        getter = getattr(self.resource_map, "hardware_catalog", None)
        if not callable(getter):
            raise ValueError("resource_map must expose hardware_catalog()")
        catalog = dict(getter() or {})
        if not catalog:
            raise ValueError("hardware catalog is required to build deployment X")
        return catalog

    def _model_catalogs(self, ctx: TickContext) -> dict[str, Any]:
        getter = getattr(self.resource_map, "model_catalog", None)
        if not callable(getter):
            raise ValueError("resource_map must expose model_catalog(model_id)")
        snapshot = ctx.cluster_snapshot
        if snapshot is None:
            return {}
        jobs = (
            snapshot.active_jobs_summary()
            if hasattr(snapshot, "active_jobs_summary")
            else getattr(snapshot, "active_jobs", [])
        )
        model_ids: set[str] = set()
        for job in jobs or []:
            spec = dict(job.get("spec_json") or {})
            features = dict(job.get("job_features") or {})
            job_model = spec.get("model_id") or features.get("model_id")
            for chain in job.get("active_chains") or []:
                shape = dict(chain.get("shape_json") or {})
                model_id = shape.get("model_id") or job_model
                if model_id:
                    model_ids.add(str(model_id))
        return {model_id: dict(getter(model_id) or {}) for model_id in model_ids}

    @staticmethod
    def _residuals(observed: dict[str, Any], predicted: dict[str, Any]) -> dict[str, np.ndarray]:
        """Per-variable residual arrays; scalar predictions broadcast."""
        out: dict[str, np.ndarray] = {}
        for name, obs in observed.items():
            pred = predicted.get(name)
            if pred is None:
                continue
            obs_arr = np.asarray(obs, dtype=float)
            if isinstance(pred, (int, float)):
                pred_arr = np.full_like(obs_arr, float(pred))
            else:
                pred_arr = np.asarray(pred, dtype=float)
                if pred_arr.shape != obs_arr.shape:
                    log.warning("residual shape mismatch for %s; skipped", name)
                    continue
            out[name] = obs_arr - pred_arr
        return out

    @staticmethod
    def _committed_mechanism_id(rank_telem) -> str | None:
        """The mechanism the agent committed to at deploy time."""
        committed = getattr(rank_telem, "committed_mechanism_id", None)
        if committed is not None:
            return committed
        mech = getattr(rank_telem, "mechanism", None)
        if mech is not None:
            return getattr(mech, "mechanism_id", None)
        return getattr(rank_telem, "mechanism_id", None)

    def _applicable_mechanisms(
        self,
        context: dict[str, Any],
        committed_id: str | None,
        *,
        diagnostics: list[dict[str, Any]] | None = None,
        job_id: str | None = None,
        rank_id: str | None = None,
    ) -> list[Any]:
        """Resolve the mechanisms this rank's evidence speaks to.

        Active structured matches on deployed X and workload values, plus the
        committed mechanism regardless of status. Every returned mechanism has
        all X values required by its edges, scope, and conditions.
        """
        if self.candidate_graph is None:
            raise ValueError("candidate graph is required for mechanism X validation")
        candidates = {
            mechanism.mechanism_id: mechanism
            for mechanism, _ in self.mechanism_registry.find_applicable(context)
        }
        if committed_id is not None and committed_id not in candidates:
            try:
                candidates[committed_id] = self.mechanism_registry.get_mechanism(committed_id)
            except KeyError:
                log.warning("committed mechanism %s not in registry", committed_id)

        applicable = []
        for mechanism in candidates.values():
            required_x = self.candidate_graph.required_x_for_mechanism(mechanism)
            missing_x = [name for name in required_x if name not in context]
            if missing_x:
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "job_id": job_id,
                            "rank_id": rank_id,
                            "mechanism_id": mechanism.mechanism_id,
                            "status": "skipped",
                            "reason": "missing_x",
                            "missing_x": missing_x,
                            "icp": [],
                        }
                    )
                continue
            applicable.append(mechanism)
        return applicable

    def _bundle(self, mechanism) -> _MechanismBundle:
        """Cached V/Y bundle view for one mechanism."""
        mid = mechanism.mechanism_id
        if mid not in self._bundle_cache:
            self._bundle_cache[mid] = _MechanismBundle(mechanism, self.candidate_graph)
        return self._bundle_cache[mid]

    @staticmethod
    def _bundle_observable(bundle, v_obs, v_pred, y_obs, y_pred) -> bool:
        """True iff every bundle variable has observed AND predicted data."""
        return not any(
            TickRunner._missing_bundle_inputs(bundle, v_obs, v_pred, y_obs, y_pred).values()
        )

    @staticmethod
    def _missing_bundle_inputs(bundle, v_obs, v_pred, y_obs, y_pred) -> dict[str, list[str]]:
        """Return missing CUSUM inputs grouped by trajectory type."""
        return {
            "v_observed": [name for name in bundle.bundle_v_variables if v_obs.get(name) is None],
            "v_predicted": [name for name in bundle.bundle_v_variables if v_pred.get(name) is None],
            "y_observed": [name for name in bundle.bundle_y_outcomes if y_obs.get(name) is None],
            "y_predicted": [name for name in bundle.bundle_y_outcomes if y_pred.get(name) is None],
        }

    def _observable_edge_ids(
        self,
        bundle: _MechanismBundle,
        available_v: set[str],
        available_y: set[str],
        *,
        include_v: bool = True,
        include_y: bool = True,
    ) -> list[str]:
        """Return mechanism edges whose measured endpoints can be evaluated."""
        observable = []
        for edge_id in bundle.edge_ids:
            edge = self._resolve_edge(edge_id)
            if edge is None:
                continue
            if (
                include_v
                and edge.src_type == "X"
                and edge.dst_type == "V"
                and edge.dst in available_v
            ) or (
                include_y
                and edge.src_type == "V"
                and edge.dst_type == "Y"
                and edge.src in available_v
                and edge.dst in available_y
            ):
                observable.append(edge_id)
        return observable

    def _cusum_diagnostics(self, bundle, v_obs, v_pred, y_obs, y_pred, v_params, y_params):
        """Return the exact per-variable CUSUM calculations for one mechanism."""
        return {
            "V": self._cusum_axis_diagnostics(bundle.bundle_v_variables, v_obs, v_pred, v_params),
            "Y": self._cusum_axis_diagnostics(bundle.bundle_y_outcomes, y_obs, y_pred, y_params),
        }

    def _cusum_axis_diagnostics(self, names, observed, predicted, params):
        """Return one CUSUM trace per named variable."""
        traces = []
        for name in sorted(names):
            obs = np.asarray(observed[name], dtype=float)
            prediction = predicted[name]
            if np.isscalar(prediction):
                pred = np.full_like(obs, float(prediction))
                logged_prediction: float | list[float] = float(prediction)
            else:
                pred = np.asarray(prediction, dtype=float)
                if pred.shape != obs.shape:
                    raise ValueError(
                        f"shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
                    )
                logged_prediction = pred.tolist()
            delta, h = params[name]
            residuals = obs - pred
            s_plus, s_minus, (direction, fired, fire_tick) = self.cusum.compute_cusum_statistic(
                residuals, delta, h
            )
            traces.append(
                {
                    "name": name,
                    "observed": obs.tolist(),
                    "predicted": logged_prediction,
                    "residual": residuals.tolist(),
                    "delta": float(delta),
                    "h": float(h),
                    "s_plus": s_plus.tolist(),
                    "s_minus": s_minus.tolist(),
                    "direction": direction.value,
                    "fired": fired,
                    "first_fire_sample": fire_tick,
                }
            )
        return traces

    def _resolve_cusum_params(
        self,
        residuals: dict[str, np.ndarray],
        cached: dict[str, Any],
    ) -> dict[str, Any]:
        """(delta, h) per variable: recalibrated table, else self-calibrate.

        The fallback derives (0.5 sigma, 4 sigma) from this rank's own
        residuals, so cold-start thresholds are scaled to each variable's
        actual units instead of unit-blind constants.
        """
        params: dict[str, Any] = {}
        for name, res in residuals.items():
            if name in cached:
                params[name] = cached[name]
            else:
                params[name] = self.cusum.cusum_params_per_v(name, res)
        return params

    def _resolve_edge(self, edge_id: str):
        """Edge object by id, or None when not in the CandidateGraph."""
        if self.candidate_graph is None:
            return None
        return self.candidate_graph.edge_table.get(edge_id)

    def _j_realized(
        self,
        y_observed_mean: dict[str, float],
        w_t: dict[str, float],
        z_star: dict[str, float],
    ) -> float:
        """Realized Tchebycheff J over objectives present everywhere.

        Stamped on the row for outcome-regret dashboards. 0.0 when no
        tchebycheff module was wired or no objective is jointly covered.
        """
        if self.tchebycheff is None or not y_observed_mean:
            return 0.0
        keys = [k for k in y_observed_mean if k in w_t and k in z_star and k in self.typical_ranges]
        if not keys:
            return 0.0
        try:
            return float(
                self.tchebycheff.compute_tchebycheff(
                    y_hat={k: y_observed_mean[k] for k in keys},
                    w_t=w_t,
                    z_star_t=z_star,
                    normalization_range=self.typical_ranges,
                )
            )
        except Exception:
            log.exception("J_realized computation failed")
            return 0.0

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _mechanism_confidence_snapshot(self, mechanism_id: str) -> dict[str, float | int]:
        """Return the current Beta posterior for one mechanism."""
        alpha, beta = self.confidence_service.get_mechanism_alpha_beta(mechanism_id)
        return {
            "alpha": alpha,
            "beta": beta,
            "confidence": self.confidence_service.get_mechanism_confidence(mechanism_id),
            "visit_count": self.confidence_service.get_mechanism_visit_count(mechanism_id),
        }

    def _edge_confidence_snapshot(self, edge_id: str) -> dict[str, float | int]:
        """Return the current Beta posterior for one edge."""
        alpha, beta = self.confidence_service.get_edge_alpha_beta(edge_id)
        return {
            "alpha": alpha,
            "beta": beta,
            "confidence": self.confidence_service.get_edge_confidence(edge_id),
            "visit_count": self.confidence_service.get_edge_visit_count(edge_id),
        }

    @staticmethod
    def _confidence_delta(delta: tuple[float, float]) -> dict[str, float]:
        """Name the configured Beta increments for a log record."""
        return {"alpha": float(delta[0]), "beta": float(delta[1])}

    def _slow_state_snapshot(self) -> dict[str, Any]:
        """Copy the slow-loop state before or after an S3 mutation."""
        state = getattr(self.slow_loop, "state", None)
        if state is None:
            return {}
        return dict(vars(state))

    def _dro_parameters(self) -> dict[str, float | None]:
        """Return the DRO controller values that affect epsilon updates."""
        return {
            "epsilon": self._float_or_none(getattr(self.dro, "epsilon", None)),
            "target_coverage": self._float_or_none(getattr(self.dro, "target", None)),
            "dead_band": self._float_or_none(getattr(self.dro, "dead_band", None)),
        }

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        """Convert an optional numeric debug value without affecting control flow."""
        return None if value is None else float(value)

    def _observed_swap_rate(self, ctx: TickContext) -> float:
        """Fraction of active jobs swapped by LAST tick's deployed plan.

        Recorded in S6 (swap count and active count); falls back to a
        snapshot-provided rate, then 0.0. Drives lambda_swit.
        """
        if self._last_active_count > 0:
            return self._last_swap_count / self._last_active_count
        snapshot = ctx.cluster_snapshot
        if snapshot is not None and hasattr(snapshot, "observed_swap_rate"):
            return float(snapshot.observed_swap_rate())
        return 0.0

    def _observed_coverage_details(self, ctx: TickContext) -> dict[str, Any]:
        """Evaluate outcomes against immutable DRO bands persisted at decision time.

        The target coverage is the existing float API's neutral no-signal
        value. A persisted but unusable band is measured as uncovered; only
        a missing band or missing required observation is non-evaluable.
        """
        target = float(getattr(self.dro, "target", 0.90))
        rows = [r for r in ctx.evidence_rows if r.y_observed_mean]
        if not rows:
            return {
                "value": target,
                "inside_rows": 0,
                "row_count": 0,
                "evaluable_row_count": 0,
                "has_signal": False,
                "reason": "no_evidence",
                "rows": [],
            }
        inside = 0
        evaluable = 0
        row_details = []
        for row in rows:
            lineage = getattr(row, "prediction_lineage", None)
            has_band = isinstance(lineage, Mapping) and "decision_dro_band" in lineage
            band = (
                lineage.get("decision_dro_band")
                if isinstance(lineage, Mapping) and has_band
                else None
            )
            required = (
                lineage.get("decision_required_objectives")
                if isinstance(lineage, Mapping) and has_band
                else None
            )
            comparable_names = set(row.residuals_per_y)
            if required is None:
                requested_names = list(band) if isinstance(band, Mapping) else []
            elif isinstance(required, (list, tuple, set, frozenset)):
                requested_names = [name for name in required if isinstance(name, str)]
            else:
                requested_names = []
            required_names = [name for name in requested_names if name in comparable_names]
            comparable_observed = {
                name: value
                for name, value in row.y_observed_mean.items()
                if name in comparable_names
            }
            status = (
                self.dro._coverage_status(comparable_observed, band, required_names)
                if required_names
                else None
            )
            row_inside = status is True
            if status is not None:
                evaluable += 1
            if row_inside:
                inside += 1
            objectives = {}
            for name in required_names:
                observed = row.y_observed_mean.get(name)
                objective_band = band.get(name) if isinstance(band, Mapping) else None
                if not isinstance(objective_band, Mapping):
                    objectives[name] = {
                        "observed": self._float_or_none(observed),
                        "band": None,
                        "inside": False,
                    }
                    continue
                objectives[name] = {
                    "observed": self._float_or_none(observed),
                    "point": objective_band.get("point"),
                    "lower": objective_band.get("lower"),
                    "upper": objective_band.get("upper"),
                    "inside": self.dro._all_objectives_inside(
                        {name: observed}, {name: objective_band}, [name]
                    ),
                }
            row_details.append(
                {
                    "row_id": row.row_id,
                    "inside": row_inside,
                    "evaluable": status is not None,
                    "objectives": objectives,
                }
            )
        if not evaluable:
            return {
                "value": target,
                "inside_rows": 0,
                "row_count": len(rows),
                "evaluable_row_count": 0,
                "has_signal": False,
                "reason": "no_evaluable_decision_bands",
                "rows": row_details,
            }
        return {
            "value": inside / evaluable,
            "inside_rows": inside,
            "row_count": len(rows),
            "evaluable_row_count": evaluable,
            "has_signal": True,
            "reason": "measured",
            "rows": row_details,
        }

    def _r2_gradient(self, ctx: TickContext) -> dict[str, float] | None:
        """R2 Pareto-coverage gradient. None until implemented (v0).

        None makes compute_wt a no-op, so w_t holds steady. The production
        implementation finite-differences the R2 indicator over recent
        y_observed_mean rows against z_star.
        """
        return None

    # ------------------------------------------------------------------
    # S6 / fallback helpers
    # ------------------------------------------------------------------

    def _count_plan_swaps(self, ctx: TickContext) -> int:
        """Active jobs whose deployed action churns running workload.

        Counts actions in SWAP_BUDGET_ACTIONS (swap) for
        jobs that were active - matching the C4 definition that only
        active-job churn is budgeted. PLACE/DEFER are admission, not churn.
        """
        plan = ctx.validated_plan
        if not isinstance(plan, Plan):
            return 0
        active_ids = self._active_job_ids(ctx)
        return sum(
            1 for a in plan.actions if a.type in SWAP_BUDGET_ACTIONS and a.job_id in active_ids
        )

    def _active_job_ids(self, ctx: TickContext) -> set:
        snapshot = ctx.cluster_snapshot
        if snapshot is None or not hasattr(snapshot, "active_jobs_summary"):
            return set()
        return {j.get("job_id", j.get("id")) for j in snapshot.active_jobs_summary()}

    def _active_job_count(self, ctx: TickContext) -> int:
        return len(self._active_job_ids(ctx))

    def _fallback_keep_all(self, ctx: TickContext) -> Plan:
        """Keep every active job; defer every pending. Typed, tick-correct.

        Feasible by construction - the running cluster is the safe state.
        Built here (not via the agent) so the abort path stays correct
        even when the tick aborted before S4 ever ran, and so a mock agent
        in tests cannot break recovery. Prefers the resource map's typed
        builder when present.
        """
        snapshot = ctx.cluster_snapshot
        if self.resource_map is not None and hasattr(self.resource_map, "build_keep_all_plan"):
            try:
                return Plan.from_raw(self.resource_map.build_keep_all_plan(snapshot), tick=ctx.tick)
            except (ValueError, TypeError):
                log.exception("resource_map keep-all plan malformed; synthesizing")

        actions: list[PlanAction] = []
        if snapshot is not None:
            for j in getattr(snapshot, "active_jobs_summary", list)() or []:
                actions.append(
                    PlanAction(job_id=j.get("job_id", j.get("id")), type=ActionType.KEEP)
                )
            for j in getattr(snapshot, "pending_jobs_summary", list)() or []:
                actions.append(
                    PlanAction(job_id=j.get("job_id", j.get("id")), type=ActionType.DEFER)
                )
        return Plan(tick=ctx.tick, actions=actions, tick_rationale="safe keep-all fallback")


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------

_DEFAULT_RUNNER: TickRunner | None = None


def bind_runner(runner: TickRunner) -> None:
    """Bind the cluster's TickRunner to the module-level run_tick."""
    global _DEFAULT_RUNNER
    _DEFAULT_RUNNER = runner


def run_tick(tick_id: int) -> TickContext:
    """Run one tick through the bound TickRunner."""
    if _DEFAULT_RUNNER is None:
        raise RuntimeError("No TickRunner bound. Call fsm_states.bind_runner(runner) at boot.")
    return _DEFAULT_RUNNER.run_tick(tick_id)
