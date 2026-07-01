"""Per-tick state machine: the deterministic spine that drives S0 -> S7.

The tick is Koi's single front door. Runtime events (chain deaths, launch
failures, degradations, completions) are recorded as facts in durable state
when they happen; no event triggers an LLM. The next tick observes them and
the one root planner reasons over the whole cluster with those facts
included. This prevents local reflex prompts from racing the global planner.

One tick is a closed learning loop:

    S0 ENTER_TICK     Freeze a consistent snapshot; reset per-tick caches
                      (tenant envelopes, validated BudgetBook).
    S1 OBSERVE        Pull per-rank telemetry bundles for [t-1, t].
    S2 VALIDATE       Per rank: residuals -> applicable mechanisms ->
                      per-mechanism (V-CUSUM, Y-CUSUM) -> per-mechanism Q ->
                      ICP per edge -> one EvidenceRow appended; DRO fed.
    S3 SLOW_UPDATE    1) Beta(alpha, beta) fan-out: every decided
                         (row, mechanism) pair updates that mechanism and
                         its edges via ConfidenceService.
                      2) SlowLoop.slow_update_all: w_t, z_star_t,
                         lambda_swit, beta_t, B_t, epsilon_dro.
                      3) Meta cadence: CUSUM (delta, h) recalibration every
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

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from src.core.models import (
    SWAP_BUDGET_ACTIONS,
    ActionType,
    EvidenceRow,
    Plan,
    PlanAction,
)
from src.infra.deployment_x import build_deployment_x_index
from src.validation.icp import ICPResult

log = logging.getLogger("koi.fsm")


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
    deployment_x: Any = None
    evidence_rows: list[Any] = field(default_factory=list)
    new_slow_state: Any = None
    candidate_plan: Any = None
    validated_plan: Any = None
    deploy_acks: list[Any] = field(default_factory=list)

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
    job_id, rank_id, W_observed, v_observed / v_predicted (dict name ->
    trajectory; predicted may be a scalar), y_observed (dict name ->
    trajectory), y_predicted (dict name -> scalar), committed_mechanism_id,
    and optionally deploy_timestamp_utc. Deploy-time X comes from
    Store/catalog snapshots, not telemetry.

    Args:
        evidence_store: Append-only EvidenceRow ledger (EvidenceService).
        telemetry: Adapter with collect_telemetry(tick_start, tick_end)
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
            agent_tools.reset_tick_caches here so tenant envelopes and the
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

    # ------------------------------------------------------------------
    # S0 - S6
    # ------------------------------------------------------------------

    def S0(self, ctx: TickContext) -> FSMState:
        """Freeze the tick's view of the world and reset per-tick caches.

        The snapshot is the single consistent input every later state
        references. The on_tick_start hook clears agent-tool tick caches
        (tenant envelopes, validated BudgetBook) so nothing budget-shaped
        survives from the previous tick's capacity.
        """
        if self.on_tick_start is not None:
            self.on_tick_start()
        ctx.cluster_snapshot = self.resource_map.snapshot_cluster_state(ctx.tick)
        return FSMState.S1_OBSERVE

    def S1(self, ctx: TickContext) -> FSMState:
        """Pull telemetry and build deploy-time X for the [t-1, t] window.

        Telemetry owns runtime V/Y. Store/catalog snapshots own deployment X.
        """
        ctx.telemetry = self.telemetry.collect_telemetry(tick_start=ctx.tick - 1, tick_end=ctx.tick)
        ctx.deployment_x = self._build_deployment_x_index(ctx)
        return FSMState.S2_VALIDATE

    def S2(self, ctx: TickContext) -> FSMState:
        """Validate every deployed rank and write the evidence backbone.

        Per rank: compute residuals; resolve applicable mechanisms
        (committed + scope matches, restricted to bundles fully observable
        in this rank's telemetry); run V-CUSUM and Y-CUSUM per mechanism;
        classify a Q per mechanism; run ICP once per edge in the union of
        applicable bundles; append one EvidenceRow; feed DRO's residual
        ring.

        No Beta updates here - S2 writes evidence, S3 reads it. The
        separation keeps S2 idempotent for replay.
        """
        w_t_snapshot = self.slow_loop.get_sss_wt()
        z_star_snapshot = self.slow_loop.get_sss_z_star_t()
        cached_v_params = self.slow_loop.get_sss_cusum_params_v()
        cached_y_params = self.slow_loop.get_sss_cusum_params_y()

        if ctx.deployment_x is None:
            ctx.deployment_x = self._build_deployment_x_index(ctx)

        for rank_telem in self.telemetry.iter_per_rank(ctx.telemetry):
            job_id = str(rank_telem.job_id)
            raw_rank_id = getattr(rank_telem, "rank_id", None)
            deployment = ctx.deployment_x.resolve(job_id, raw_rank_id)
            rank_id = deployment.rank_id
            x = dict(deployment.x)
            env_label = deployment.env_label
            v_obs = dict(rank_telem.v_observed)
            v_pred = dict(rank_telem.v_predicted)
            y_obs = dict(rank_telem.y_observed)
            y_pred = dict(rank_telem.y_predicted)

            residuals_per_v = self._residuals(v_obs, v_pred)
            residuals_per_y = self._residuals(y_obs, y_pred)
            y_observed_mean = {name: float(np.mean(arr)) for name, arr in y_obs.items() if len(arr)}

            committed = self._committed_mechanism_id(rank_telem)
            applicable = self._applicable_mechanisms(x.keys(), v_obs.keys(), committed)

            v_params = self._resolve_cusum_params(residuals_per_v, cached_v_params)
            y_params = self._resolve_cusum_params(residuals_per_y, cached_y_params)

            cusum_per_mech: dict[str, Any] = {}
            q_per_mech: dict[str, Any] = {}
            touched_edge_ids: set = set()

            for mech in applicable:
                bundle = self._bundle(mech)
                touched_edge_ids.update(bundle.edge_ids)
                if not self._bundle_observable(bundle, v_obs, v_pred, y_obs, y_pred):
                    q_per_mech[mech.mechanism_id] = None
                    continue
                v_verdict, y_verdict = self.cusum.cusum_per_mechanism(
                    mechanism=mech,
                    candidate_graph=self.candidate_graph,
                    v_obs_traj=v_obs,
                    v_hat_traj=v_pred,
                    y_obs_traj=y_obs,
                    y_hat_traj=y_pred,
                    v_params=v_params,
                    y_params=y_params,
                )
                cusum_per_mech[mech.mechanism_id] = (v_verdict, y_verdict)
                q_per_mech[mech.mechanism_id] = self.qv.classify_quadrant(v_verdict, y_verdict)

            icp_per_edge: dict[str, Any] = {}
            for edge_id in sorted(touched_edge_ids):
                edge = self._resolve_edge(edge_id)
                if edge is None:
                    continue
                icp_per_edge[edge_id] = self.icp.compute_icp_per_edge(
                    edge=edge, evidence_store=self.evidence_store
                )

            j_realized = self._j_realized(y_observed_mean, w_t_snapshot, z_star_snapshot)
            row = EvidenceRow(
                row_id=f"{ctx.tick}_{job_id}_{rank_id}",
                tick=ctx.tick,
                deploy_timestamp_utc=float(
                    getattr(rank_telem, "deploy_timestamp_utc", time.time())
                ),
                job_id=job_id,
                rank_id=rank_id,
                env_label=env_label,
                X=x,
                W_observed=dict(getattr(rank_telem, "W_observed", {})),
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
            )
            self.evidence_store.append_row(row)
            ctx.evidence_rows.append(row)

            self.dro.append_residual_history(pred_y=y_pred, obs_y=y_observed_mean)

        return FSMState.S3_SLOW_UPDATE

    def S3(self, ctx: TickContext) -> FSMState:
        """Apply the learning updates, then refresh the slow-loop knobs.

        Part 1 - Beta fan-out: every decided (row, mechanism) pair updates
        that mechanism's Beta and the Betas of ITS edges, with the edge
        delta modulated by that edge's ICP result. An edge shared by
        several applicable mechanisms receives one update per mechanism
        context - each context is independent evidence about the edge.
        ConfidenceService also records env coverage and recency (single
        writer for all confidence state).

        Part 2 - SlowLoop.slow_update_all with the observed swap rate
        (recorded by last tick's S6), the observed DRO coverage (this
        tick's rows vs their predicted bands), the R2 gradient (v0 stub),
        and annealed targets.

        Part 3 - meta cadence: CUSUM (delta, h) recalibration from
        accumulated residual history every recalibrate_every ticks.
        """
        did_confidence_update = False
        for row in ctx.evidence_rows:
            for mid, q in row.q_label_per_mechanism.items():
                if q is None:
                    continue
                self.confidence_service.apply_delta_c_mechanism(
                    mid, q, env_label=row.env_label, tick=ctx.tick
                )
                did_confidence_update = True
                try:
                    mech = self.mechanism_registry.get_mechanism(mid)
                except KeyError:
                    log.warning("row %s references unknown mechanism %s", row.row_id, mid)
                    continue
                for edge_id in mech.edge_ids:
                    icp_result = row.icp_result_per_edge.get(edge_id, ICPResult.UNDECIDED)
                    self.confidence_service.apply_delta_c_edge(
                        edge_id, q, icp_result, env_label=row.env_label, tick=ctx.tick
                    )

        flush_confidence = getattr(self.confidence_service, "flush", None)
        if did_confidence_update and callable(flush_confidence):
            flush_confidence()

        ctx.new_slow_state = self.slow_loop.slow_update_all(
            tick=ctx.tick,
            observed_swap_rate=self._observed_swap_rate(ctx),
            observed_coverage=self._observed_coverage(ctx),
            r2_gradient=self._r2_gradient(ctx),
            target_overrides=self.slow_loop.anneal_targets(ctx.tick),
        )

        if (
            self.recalibrate_every > 0
            and ctx.tick > 0
            and ctx.tick % self.recalibrate_every == 0
            and hasattr(self.slow_loop, "recalibrate_cusum_params")
        ):
            self.slow_loop.recalibrate_cusum_params()
            log.info("CUSUM (delta, h) recalibrated at tick %d", ctx.tick)

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
        ctx.deploy_acks = self.executor.send_to_executor(ctx.validated_plan)

        self._last_swap_count = self._count_plan_swaps(ctx)
        self._last_active_count = self._active_job_count(ctx)

        if self.trace is not None:
            try:
                self.trace.persist_tick(ctx)
            except Exception:
                log.exception("trace persist failed at tick %d", ctx.tick)
        return FSMState.S7_EXIT_TICK

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
            try:
                self.trace.persist_tick(ctx)
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
        return build_deployment_x_index(
            ctx.cluster_snapshot,
            hardware_catalog=self._hardware_catalog(),
            x_fields=x_fields,
        )

    def _hardware_catalog(self) -> dict[str, Any]:
        getter = getattr(self.resource_map, "hardware_catalog", None)
        if not callable(getter):
            raise ValueError("resource_map must expose hardware_catalog()")
        catalog = dict(getter() or {})
        if not catalog:
            raise ValueError("hardware catalog is required to build deployment X")
        return catalog

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
        x_keys,
        v_keys,
        committed_id: str | None,
    ) -> list[Any]:
        """Resolve the mechanisms this rank's evidence speaks to.

        Active scope matches on (X keys, observed V names), plus the
        committed mechanism regardless of status - the agent's bet always
        receives its verdict, even if the mechanism was archived
        mid-flight.
        """
        applicable: dict[str, Any] = {}
        matches = self.mechanism_registry.filter_by_scope(sorted(x_keys), sorted(v_keys))
        for mech in matches:
            if mech.status == "active":
                applicable[mech.mechanism_id] = mech
        if committed_id is not None and committed_id not in applicable:
            try:
                applicable[committed_id] = self.mechanism_registry.get_mechanism(committed_id)
            except KeyError:
                log.warning("committed mechanism %s not in registry", committed_id)
        return list(applicable.values())

    def _bundle(self, mechanism) -> _MechanismBundle:
        """Cached V/Y bundle view for one mechanism."""
        mid = mechanism.mechanism_id
        if mid not in self._bundle_cache:
            self._bundle_cache[mid] = _MechanismBundle(mechanism, self.candidate_graph)
        return self._bundle_cache[mid]

    @staticmethod
    def _bundle_observable(bundle, v_obs, v_pred, y_obs, y_pred) -> bool:
        """True iff every bundle variable has observed AND predicted data."""
        for name in bundle.bundle_v_variables:
            if name not in v_obs or name not in v_pred:
                return False
        for name in bundle.bundle_y_outcomes:
            if name not in y_obs or name not in y_pred:
                return False
        return True

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

    def _observed_coverage(self, ctx: TickContext) -> float:
        """Fraction of this tick's outcomes inside their predicted DRO band.

        Bands are recomputed from each row's stored y_predicted with the
        current epsilon - one tick of epsilon drift versus the band that
        existed at deploy time, accepted as the honest v0 approximation.
        Returns the DRO target (no-signal) when the tick produced no rows.
        """
        rows = [r for r in ctx.evidence_rows if r.y_observed_mean]
        if not rows:
            return float(getattr(self.dro, "target", 0.90))
        inside = 0
        for row in rows:
            band = self.dro.compute_dro_band(row.y_predicted)
            if self.dro._all_objectives_inside(row.y_observed_mean, band):
                inside += 1
        return inside / len(rows)

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
