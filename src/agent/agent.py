"""S4 root RLM harness: the cluster planner agent.

Implements the design in realactualopencodeagentic.md: one root planner
owns cluster-level tradeoffs, allocates budgets before specialists run,
and never performs side effects. The LLM writes Python in a REPL whose
namespace holds the full cluster state and the agent_tools registry; the
prompt carries only compact metadata.

Components:
    RLMRuntime        REPL sandbox: variable bindings, code execution with
                      truncated stdout, FINAL_VAR extraction, trace capture.
    SpecialistRunner  Bounded per-job specialist calls under BudgetSlices.
                      Validates outputs against the slice; retries once;
                      falls back to keep/defer.
    KoiAgentHarness   The S4 entry point. run_agent_loop matches the call
                      signature in fsm_states.TickRunner.S4 exactly:
                      (cluster_snapshot, slow_state, evidence_store,
                      mechanism_registry, tick) -> plan.
                      receive_validator_feedback feeds S5 violations back
                      for the single repair iteration.

The LLM client contract is one method:
    llm_client.complete(messages: list[dict]) -> str
where messages are {"role": "system" | "user" | "assistant", "content": str}.
Scripted mocks satisfy this for tests.

Safety invariants enforced here, not in the prompt:
    - No tool in the REPL performs side effects (agent_tools exposes
      read/compute tools plus validated mechanism admission only).
    - Specialists cannot run without a BudgetBook validated by
      agent_tools.validate_budget_book (the anti-split-brain order).
    - The returned plan is materialized and shape-checked before S5 sees
      it; malformed plans become None so the FSM falls back to keep-all.
    - Trajectories are bounded by K_MAX turns, a wall-clock timeout, and
      a consecutive REPL-error limit.

v0 scope: reserved market only; K_P = 1 (code supports more).
"""

import contextlib
import io
import json
import logging
import re
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import src.agent.tools.agent_tools as agent_tools
from src.config.hyperparameters import K_MAX, K_P
from src.core.models import (
    LADDER_ACTIONS,
    REQUIRED_JOB_STATE,
    ActionType,
    Plan,
    PlanAction,
)

log = logging.getLogger("koi.agent")

# A specialist optimizes ONE job inside its BudgetSlice, so it may only
# propose place/keep/swap/defer. Root-only actions remain terminate and
# diagnose. TODO(v0): restore preempt/resume/retry with lifecycle support.
SPECIALIST_ACTIONS = frozenset(
    {
        ActionType.PLACE.value,
        ActionType.KEEP.value,
        ActionType.SWAP.value,
        ActionType.DEFER.value,
    }
)

ALLOWED_FITNESS = frozenset({"starved", "happy", "overprovisioned", "blocked"})

DEFAULT_WALL_CLOCK_SEC = 180.0
DEFAULT_STDOUT_LIMIT = 2000
DEFAULT_CONSECUTIVE_ERROR_LIMIT = 5

_CODE_BLOCK_RE = re.compile(r"```(?:repl|python|py)?\s*\n(.*?)```", re.DOTALL)
_FINAL_VAR_TEXT_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_]\w*)\s*\)")


class PlanMaterializationError(ValueError):
    """Raised when a candidate plan fails shape validation."""


class AgentTrace:
    # TODO - Comment this for production
    """Append-only event log for one agent invocation.

    Every trajectory turn, REPL execution, specialist call, and fallback
    is recorded so a tick can be replayed and audited.
    """

    def __init__(self, live_sink=None):
        self.events: list[dict[str, Any]] = []
        self.live_sink = live_sink

    def add(self, kind: str, **payload) -> None:
        """Record one event with its wall-clock timestamp."""
        event = {"kind": kind, "ts": time.time(), **payload}
        self.events.append(event)
        if self.live_sink is not None:
            self.live_sink(event)


class RLMRuntime:
    """REPL sandbox the root LLM writes Python into.

    The namespace persists across turns within one trajectory and is
    discarded afterward. FINAL_VAR is injected as a callable: the LLM
    calls FINAL_VAR(plan) to commit its assembled plan, which the
    harness then materializes and validates. stdout is truncated in the
    transcript but kept whole in the trace.
    """

    def __init__(
        self,
        stdout_limit: int = DEFAULT_STDOUT_LIMIT,
        trace: AgentTrace | None = None,
    ):
        self.stdout_limit = int(stdout_limit)
        self.trace = trace or AgentTrace()
        self.namespace: dict[str, Any] = {}
        self._final: Any | None = None
        self.namespace["FINAL_VAR"] = self._final_var

    def _final_var(self, value):
        """Record the LLM's committed final value and echo it back."""
        self._final = value
        return value

    @property
    def final_value(self) -> Any | None:
        """The value committed via FINAL_VAR, or None."""
        return self._final

    def bind(self, **variables) -> None:
        """Bind variables into the REPL namespace."""
        self.namespace.update(variables)

    def extract_code_blocks(self, response: str) -> list[str]:
        """Pull fenced code blocks (repl/python/bare) from an LLM response."""
        return [block.strip() for block in _CODE_BLOCK_RE.findall(response) if block.strip()]

    def extract_final_from_text(self, response: str) -> Any | None:
        """Resolve a textual FINAL_VAR(name) reference against the namespace.

        Supports the pattern where the LLM writes FINAL_VAR(plan) in prose
        after assembling `plan` in an earlier REPL turn.
        """
        match = _FINAL_VAR_TEXT_RE.search(response)
        if match is None:
            return None
        return self.namespace.get(match.group(1))

    def exec_code(self, code: str) -> str:
        """Execute one code block in the shared namespace.

        Returns captured stdout truncated to stdout_limit characters.
        Exceptions are caught and returned as error text so the LLM can
        self-correct; the full output and traceback go to the trace.
        """
        self.trace.add("repl_exec_started", code=code)
        buffer = io.StringIO()
        error_text = ""
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(code, "<repl>", "exec"), self.namespace)
        except Exception:
            error_text = traceback.format_exc(limit=3)
        full_output = buffer.getvalue()
        self.trace.add(
            "repl_exec",
            code=code,
            stdout=full_output,
            error=error_text,
        )
        shown = full_output
        if error_text:
            shown = shown + ("\n" if shown else "") + f"ERROR:\n{error_text}"
        if len(shown) > self.stdout_limit:
            shown = shown[: self.stdout_limit] + f"\n... [truncated at {self.stdout_limit} chars]"
        return shown


class SpecialistRunner:
    """Run bounded per-job specialists under validated BudgetSlices.

    A specialist is one LLM call that optimizes a single job inside its
    BudgetSlice and returns a JobSpecialistResult dict. Results are
    validated deterministically: schema, capacity within slice, legal
    fitness value. One retry with the violation message; then a safe
    keep/defer fallback for that job.

    The harness binds an instance into agent_tools at construction so
    the root can call run_job_specialists from its REPL.
    """

    def __init__(
        self,
        llm_client,
        prompt_builder: Callable[[str, dict[str, Any], dict[str, Any]], str] | None = None,
        trace: AgentTrace | None = None,
    ):
        self.llm = llm_client
        self.prompt_builder = prompt_builder or self._default_prompt
        self.trace = trace or AgentTrace()

    @staticmethod
    def _default_prompt(job_id: str, slice_: dict[str, Any], brief: dict[str, Any]) -> str:
        return (
            f"You are the bounded specialist for job {job_id}.\n"
            "You are not the cluster planner. You do not own resource "
            "allocation. You must stay inside the BudgetSlice.\n\n"
            f"BudgetSlice:\n{json.dumps(slice_, indent=2, default=str)}\n\n"
            f"Job brief:\n{json.dumps(brief, indent=2, default=str)}\n\n"
            "Task:\n"
            "1. Explore candidate ladders internally, then return the single best one inside the budget.\n"
            "2. Choose existing mechanisms or propose a new valid mechanism.\n"
            "3. Estimate predicted_y and predicted_sigma.\n"
            "4. Report exact budget utilization.\n"
            "5. Report fitness as starved, happy, overprovisioned, or blocked.\n"
            "6. If starved, give marginal value of more capacity by env.\n"
            "7. If overprovisioned, give unused capacity by env.\n\n"
            "Do not exceed budget. If you want more, report it as fitness, "
            "not as allocated ladder demand.\n\n"
            "Mechanism IDs are opaque Store IDs: prefer exact entries in "
            "mechanism_candidates, then partial entries. "
            "If none fits, submit new_mechanism_proposals without inventing an ID.\n\n"
            "Your output is a PROPOSAL to the root cluster planner, not a "
            "decision. The root may accept, modify, or discard it during "
            "cluster-level reconciliation. Do not reason about other jobs "
            "or cluster tradeoffs - that is the root's job, and you cannot "
            "see the information needed to do it.\n\n"
            "type must be one of: place, keep, swap, defer. You CANNOT "
            "terminate or diagnose - those are root-only. "
            "Preempt/resume/retry are disabled in MVP v0. If no "
            "safe ladder fits the budget, return keep (running job) or "
            "defer (waiting job) and report fitness=starved or blocked.\n\n"
            "Output a single JSON object with keys: job_id, user_id, "
            "type, ladder, predicted_y, predicted_sigma, "
            "budget_utilization, used_capacity, fitness, "
            "marginal_value_of_more, unused_capacity, mechanism_ids (multiple mechanisms are allowed), "
            "new_mechanism_proposals, reasoning. No prose outside the JSON. "
            "Env keys MUST be canonical 5-part labels including market: "
            "market|cloud|region|zone|gpu_type, e.g. "
            "reserved|aws|us-east-1|us-east-1b|L40S. Never omit market. "
            "Ladder entries MUST be canonical rank dicts: "
            "{'role':'aggregate','env':[market,cloud,region,zone,gpu_type],"
            "'config':{'instance_type':str,'gpu_count':int,'tp':int,'pp':int,"
            "'engine_name':'vllm'},'n_replicas':int,'mechanism_id':'M_...'}. "
            "Never set engine_version or AIC/engine-autotuned values. "
            "Do not emit shorthand ladder entries like {'env': ..., 'count': ...}."
        )

    def run_many(
        self,
        jobs: list[str],
        budget_book: dict[str, Any],
        max_workers: int = 8,
    ) -> list[dict[str, Any]]:
        """Run specialists for several jobs concurrently.

        Args:
            jobs: Job ids; each must have a slice in budget_book.
            budget_book: The validated BudgetBook.
            max_workers: Thread pool width.

        Returns:
            One result dict per job, fallbacks included.
        """
        slices = budget_book.get("job_budgets") or {}
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {job_id: pool.submit(self.run_one, job_id, slices[job_id]) for job_id in jobs}
            return [futures[job_id].result() for job_id in jobs]

    def run_one(self, job_id: str, slice_: dict[str, Any]) -> dict[str, Any]:
        """Run one specialist with single-retry validation.

        Args:
            job_id: The job to optimize.
            slice_: The job's BudgetSlice.

        Returns:
            A validated JobSpecialistResult dict, or the keep/defer
            fallback with fitness "blocked" after two failures.
        """
        try:
            brief = agent_tools.get_job_brief(job_id)
        except Exception:
            log.exception("specialist brief failed for %s", job_id)
            brief = {"job_id": job_id}

        prompt = self.prompt_builder(job_id, slice_, brief)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            try:
                response = self.llm.complete(messages)
            except Exception:
                log.exception("specialist LLM call failed for %s", job_id)
                break
            result = self._parse_json(response)
            violations = self._validate(result, job_id, slice_)
            self.trace.add(
                "specialist_result",
                job_id=job_id,
                attempt=attempt,
                violations=violations,
            )
            if not violations:
                assert result is not None
                return result
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your output was rejected: "
                        + "; ".join(violations)
                        + ". Emit a corrected JSON object only."
                    ),
                }
            )

        is_active = brief.get("current_ladder") is not None
        return {
            "job_id": job_id,
            "user_id": slice_.get("user_id"),
            "type": "keep" if is_active else "defer",
            "ladder": None,
            "predicted_y": {},
            "predicted_sigma": 0.0,
            "budget_utilization": {},
            "used_capacity": {},
            "fitness": "blocked",
            "marginal_value_of_more": {},
            "unused_capacity": {},
            "mechanism_ids": [],
            "new_mechanism_proposals": [],
            "reasoning": "specialist fallback after validation failure",
        }

    @staticmethod
    def _parse_json(response: str) -> dict[str, Any] | None:
        """Parse a JSON object from a raw or fenced LLM response."""
        text = response.strip()
        fenced = _CODE_BLOCK_RE.search(text)
        if fenced:
            text = fenced.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _validate(
        result: dict[str, Any] | None,
        job_id: str,
        slice_: dict[str, Any],
    ) -> list[str]:
        """Deterministic checks on a specialist result."""
        if result is None:
            return ["output was not parseable JSON"]
        violations: list[str] = []
        if result.get("job_id") != job_id:
            violations.append(f"job_id mismatch: expected {job_id}")
        if result.get("fitness") not in ALLOWED_FITNESS:
            violations.append(f"fitness must be one of {sorted(ALLOWED_FITNESS)}")
        action = result.get("type")
        if action is None or str(action).lower() not in SPECIALIST_ACTIONS:
            violations.append(f"type must be one of {sorted(SPECIALIST_ACTIONS)}")
        action_name = str(action).lower() if action is not None else ""

        for field in (
            "used_capacity",
            "unused_capacity",
            "marginal_value_of_more",
            "budget_utilization",
        ):
            value = result.get(field) or {}
            if not isinstance(value, dict):
                violations.append(f"{field} must be a dict keyed by canonical env")
                continue
            for env in value:
                if len(agent_tools._env_key(env).split("|")) != 5:
                    violations.append(
                        f"{field} env {env!r} must be market|cloud|region|zone|gpu_type"
                    )

        ladder = result.get("ladder")
        mechanism_ids = result.get("mechanism_ids") or []
        if action_name in (ActionType.PLACE.value, ActionType.SWAP.value):
            if not isinstance(mechanism_ids, list) or not mechanism_ids:
                violations.append(f"{action_name} requires non-empty mechanism_ids")
            if not isinstance(ladder, list) or not ladder:
                violations.append(f"{action_name} requires non-empty canonical ladder")
            else:
                for i, rank in enumerate(ladder):
                    violations.extend(
                        SpecialistRunner._validate_rank_schema(
                            rank,
                            i,
                            mechanism_ids,
                        )
                    )
        elif ladder not in (None, []):
            violations.append(f"{action_name} must not include ladder")

        if action_name in (ActionType.PLACE.value, ActionType.SWAP.value) and isinstance(
            ladder, list
        ):
            try:
                action_obj = PlanAction.from_dict(result)
                violations.extend(agent_tools._budget_violations(action_obj, slice_))
            except (TypeError, ValueError):
                pass  # Schema violations above already describe malformed ladders.
        return violations

    @staticmethod
    def _validate_rank_schema(rank: Any, index: int, mechanism_ids: list[Any]) -> list[str]:
        violations: list[str] = []
        prefix = f"ladder[{index}]"
        if not isinstance(rank, dict):
            return [f"{prefix} must be a dict"]
        if rank.get("role") != "aggregate":
            violations.append(f"{prefix}.role must be 'aggregate'")

        env = rank.get("env")
        if not isinstance(env, (list, tuple)) or len(env) != 5:
            violations.append(f"{prefix}.env must be [market, cloud, region, zone, gpu_type]")
        elif any(not isinstance(part, str) or not part for part in env):
            violations.append(f"{prefix}.env entries must be non-empty strings")

        config = rank.get("config")
        if not isinstance(config, dict):
            violations.append(f"{prefix}.config must be a dict")
            config = {}
        for key in ("model_id", "instance_type", "gpu_type"):
            if not config.get(key):
                violations.append(f"{prefix}.config.{key} is required")
        if config.get("engine_name") not in {"vllm", "sglang"}:
            violations.append(f"{prefix}.config.engine_name must be 'vllm' or 'sglang'")
        for key in ("gpu_count", "tp", "pp"):
            value = config.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                violations.append(f"{prefix}.config.{key} must be a positive int")

        replicas = rank.get("n_replicas")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas <= 0:
            violations.append(f"{prefix}.n_replicas must be a positive int")
        mechanism_id = rank.get("mechanism_id")
        if not mechanism_id:
            violations.append(f"{prefix}.mechanism_id is required")
        elif mechanism_ids and mechanism_id not in mechanism_ids:
            violations.append(f"{prefix}.mechanism_id must be present in mechanism_ids")
        return violations


class KoiAgentHarness:
    """The S4 root planner. One instance per cluster, built at boot.

    Construction wires the LLM client and the deterministic services the
    REPL exposes. Per tick, fsm_states.TickRunner.S4 calls
    run_agent_loop(cluster_snapshot, slow_state, evidence_store,
    mechanism_registry, tick); the harness runs K_P trajectories, scores
    feasible plans by aggregate sigma, and returns the best one (or the
    keep-all fallback).

    Works with any model behind a complete(messages) -> str client:
    frontier APIs, or open models (Gemma, Qwen, Llama, DeepSeek) served
    through vLLM / Ollama / llama.cpp OpenAI-compatible endpoints. The
    harness never relies on native function-calling - tools are invoked
    by the model writing Python - so weak tool-calling support in open
    models does not matter. See llm_clients.OpenAICompatClient for the
    adapter (including the no-system-role fold Gemma needs).

    Args:
        llm_client: Object with complete(messages) -> str. The root
            planner model.
        specialist_llm_client: Optional cheaper/smaller model for the
            per-job specialist calls. Defaults to llm_client. With ~50
            specialist calls per tick and one root call, a small local
            model here cuts cost where quality matters least.
        resource_map: Cluster resource service (snapshot, simulation,
            keep-all plan builder). Also reachable via agent_tools.
        user_registry: Optional user policy service for envelopes. None
            means Store user owns all visible capacity in v0.
        plan_validator: Optional validator with val_plan(...) used to
            pre-screen K_P candidates before scoring. S5 still runs the
            authoritative validation.
        tool_dependencies: Optional dict of the shared math/state singletons
            the tools need (slow_loop, dro, evidence_store,
            mechanism_registry, confidence_service, candidate_graph,
            eig_module, tchebycheff_module, switchcost_module, surrogate).
            Forwarded verbatim to agent_tools.bind_tools. Pass it here for a
            self-contained harness (tests, simple boot); omit it if a boot
            script already bound them - bind_tools is additive, and
            run_agent_loop asserts the full surface is wired before planning
            either way.
        config: Optional overrides: k_p, k_max, wall_clock_sec,
            stdout_limit, consecutive_error_limit, max_history_messages.
            max_history_messages bounds the transcript for small-context
            models (0 = unlimited; 20-30 suits an 8K-context model).
    """

    def __init__(
        self,
        llm_client,
        specialist_llm_client=None,
        resource_map=None,
        user_registry=None,
        plan_validator=None,
        tool_dependencies: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        cfg = config or {}
        self.llm = llm_client
        self.resource_map = resource_map
        self.user_registry = user_registry
        self.plan_validator = plan_validator
        self.k_p = int(cfg.get("k_p", K_P))
        self.k_max = int(cfg.get("k_max", K_MAX))
        self.wall_clock_sec = float(cfg.get("wall_clock_sec", DEFAULT_WALL_CLOCK_SEC))
        self.stdout_limit = int(cfg.get("stdout_limit", DEFAULT_STDOUT_LIMIT))
        self.consecutive_error_limit = int(
            cfg.get("consecutive_error_limit", DEFAULT_CONSECUTIVE_ERROR_LIMIT)
        )
        self.max_history_messages = int(cfg.get("max_history_messages", 0))
        self.trace = AgentTrace()
        self.specialist_runner = SpecialistRunner(
            specialist_llm_client or llm_client, trace=self.trace
        )
        self._pending_violations: list[str] = []
        self._current_tick: int = 0

        # Bind in two layers, both additive. Shared singletons come from the
        # boot script OR tool_dependencies here; the harness owns binding the
        # specialist_runner it just created plus the agent-flow services it
        # received. assert_planning_ready() in run_agent_loop catches any gap.
        if tool_dependencies:
            agent_tools.bind_tools(**tool_dependencies)
        agent_tools.bind_tools(
            specialist_runner=self.specialist_runner,
            user_registry=user_registry,
            resource_map=resource_map,
            plan_validator=plan_validator,
        )

    # ------------------------------------------------------------------
    # FSM-facing API
    # ------------------------------------------------------------------

    def run_agent_loop(
        self,
        cluster_snapshot,
        slow_state,
        evidence_store,
        mechanism_registry,
        tick: int,
    ):
        """Produce a candidate cluster plan for this tick.

        Runs K_P independent trajectories, pre-screens each plan with the
        bound validator when available, scores survivors with
        agent_tools.compute_sigma, and returns the best plan. Returns the
        keep-all/defer-pending fallback when no trajectory produces a
        usable plan.

        Matches the call in fsm_states.TickRunner.S4 exactly.

        Args:
            cluster_snapshot: State snapshot from S0.
            slow_state: SlowState from S3.
            evidence_store: The evidence ledger.
            mechanism_registry: The mechanism registry.
            tick: Current tick id.

        Returns:
            A materialized plan dict {job_id: action}, or the fallback.
        """
        # Rebind this tick's evidence_store and mechanism_registry so the
        # tools (which read _CTX) and the REPL namespace (bound in
        # one_trajectory) see the SAME objects - never a boot-bound instance
        # diverging from the one the FSM passes. Then fail fast if any
        # planning dependency is still unbound, before burning trajectory turns.
        agent_tools.bind_tools(
            evidence_store=evidence_store,
            mechanism_registry=mechanism_registry,
            cluster_snapshot=cluster_snapshot,
        )
        agent_tools.assert_planning_ready()
        self._current_tick = tick

        violations = self._pending_violations
        self._pending_violations = []

        candidates = []
        for k_idx in range(self.k_p):
            self.trace.add("trajectory_started", tick=tick, k_idx=k_idx)
            try:
                plan = self.one_trajectory(
                    cluster_snapshot=cluster_snapshot,
                    slow_state=slow_state,
                    evidence_store=evidence_store,
                    mechanism_registry=mechanism_registry,
                    tick=tick,
                    k_idx=k_idx,
                    repair_violations=violations,
                )
            except Exception:
                log.exception("trajectory %d failed at tick %d", k_idx, tick)
                continue
            if plan is None:
                continue
            if self.plan_validator is not None:
                try:
                    result = self.plan_validator.val_plan(
                        plan=plan,
                        cluster_snapshot=cluster_snapshot,
                        slow_state=slow_state,
                    )
                    if not getattr(result, "feasible", False):
                        self.trace.add(
                            "kp_candidate_infeasible",
                            k_idx=k_idx,
                            violations=list(getattr(result, "violations", [])),
                        )
                        continue
                except Exception:
                    log.exception("pre-screen validation failed; passing plan to S5")
            score = self._score_plan(plan)
            self.trace.add("kp_candidate_scored", k_idx=k_idx, score=score)
            candidates.append((score, plan))

        if candidates:
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            self.trace.add("kp_winner_selected", score=candidates[0][0])
            return agent_tools.stamp_plan_predictions(candidates[0][1], cluster_snapshot)

        self.trace.add("safe_fallback_used", tick=tick)
        return self._fallback_plan(cluster_snapshot)

    def receive_validator_feedback(self, violations: list[str]) -> None:
        """Store S5 violations for the next repair-mode run_agent_loop call."""
        self._pending_violations = list(violations or [])
        self.trace.add("validator_feedback", violations=self._pending_violations)

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def one_trajectory(
        self,
        cluster_snapshot,
        slow_state,
        evidence_store,
        mechanism_registry,
        tick: int,
        k_idx: int = 0,
        repair_violations: list[str] | None = None,
    ):
        """Run one bounded REPL trajectory and return a materialized plan.

        The trajectory ends when the LLM commits FINAL_VAR(plan), when
        K_MAX turns or the wall clock are exhausted (the REPL's `plan`
        variable is extracted if well formed), or when the consecutive
        error limit trips.

        Args:
            cluster_snapshot: State snapshot bound into the REPL.
            slow_state: SlowState bound into the REPL.
            evidence_store: Bound into the REPL.
            mechanism_registry: Bound into the REPL.
            tick: Current tick id.
            k_idx: Trajectory index inside the K_P loop.
            repair_violations: S5 violations when in repair mode.

        Returns:
            A materialized Plan, or None.
        """
        self._current_tick = tick
        runtime = RLMRuntime(stdout_limit=self.stdout_limit, trace=self.trace)
        runtime.bind(
            cluster_snapshot=cluster_snapshot,
            state=cluster_snapshot,
            slow_state=slow_state,
            evidence_store=evidence_store,
            mechanism_registry=mechanism_registry,
            resource_map=self.resource_map,
            user_registry=self.user_registry,
            plan=None,
            tick=tick,
            **agent_tools.all_callables(),
        )

        history = [
            {"role": "system", "content": self.build_root_prompt(tick, repair_violations)},
            {
                "role": "user",
                "content": (
                    f"Tick {tick}, trajectory {k_idx}. Plan the cluster. "
                    "Write Python in ```repl blocks to inspect state and "
                    "build your plan. Call FINAL_VAR(plan) when done."
                ),
            },
        ]

        started = time.time()
        consecutive_errors = 0

        for turn in range(self.k_max):
            if time.time() - started > self.wall_clock_sec:
                self.trace.add("trajectory_timeout", turn=turn)
                break

            try:
                response = self.llm.complete(history)
            except Exception:
                log.exception("root LLM call failed at turn %d", turn)
                break
            history.append({"role": "assistant", "content": response})

            blocks = runtime.extract_code_blocks(response)
            if not blocks:
                final = runtime.extract_final_from_text(response)
                if final is not None:
                    return self._try_materialize(final, cluster_snapshot)
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "Emit Python in ```repl blocks to act, or call "
                            "FINAL_VAR(plan) to commit your assembled plan."
                        ),
                    }
                )
                continue

            outputs = []
            turn_had_error = False
            for code in blocks:
                shown = runtime.exec_code(code)
                outputs.append(shown)
                if "ERROR:" in shown:
                    turn_had_error = True
                if runtime.final_value is not None:
                    return self._try_materialize(runtime.final_value, cluster_snapshot)

            consecutive_errors = consecutive_errors + 1 if turn_had_error else 0
            if consecutive_errors >= self.consecutive_error_limit:
                self.trace.add("trajectory_error_limit", turn=turn)
                break

            history.append(
                {
                    "role": "user",
                    "content": "[REPL output]\n" + "\n---\n".join(outputs),
                }
            )
            history = self._compact_history(history)

        leftover = runtime.namespace.get("plan")
        if leftover is not None:
            return self._try_materialize(leftover, cluster_snapshot)
        return None

    def _compact_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        """Bound the transcript for small-context models.

        Keeps the system prompt, the initial user instruction, and the
        most recent turns; elides the middle with one marker message.
        Safe because the REPL namespace persists across turns - the
        model can re-print any value an elided turn produced. Disabled
        when max_history_messages is 0.
        """
        limit = self.max_history_messages
        if limit <= 0 or len(history) <= limit:
            return history
        head = history[:2]
        tail = history[-(limit - 3) :]
        marker = {
            "role": "user",
            "content": (
                "[earlier turns elided to fit context. The REPL namespace "
                "still holds every variable you defined - print anything "
                "you need to see again.]"
            ),
        }
        return [*head, marker, *tail]

    # ------------------------------------------------------------------
    # Plan materialization and fallback
    # ------------------------------------------------------------------

    def _try_materialize(self, raw_plan, cluster_snapshot) -> Plan | None:
        """Materialize a raw plan into a typed Plan, None when malformed."""
        try:
            return self.materialize_plan(raw_plan, cluster_snapshot)
        except (PlanMaterializationError, ValueError) as exc:
            self.trace.add("plan_materialization_failed", reason=str(exc))
            log.warning("plan materialization failed: %s", exc)
            return None

    def materialize_plan(self, raw_plan, cluster_snapshot) -> Plan:
        """Parse and validate the LLM's committed plan into a typed Plan.

        Parsing (Plan.from_raw) accepts the Plan-shaped dict, a plain
        actions list, or a job_id->action dict. This method then adds the
        contextual checks that need the snapshot:

            - No duplicate job_ids (one action per job).
            - Job exists in the snapshot (when the snapshot exposes ids).
            - Action is legal for the job's current state (PLACE only on
              waiting, SWAP only on running, ...).
            - Ladder actions carry a non-empty ladder; every rank has a
              5-tuple env (launch target + ICP key), >= 1 replica, and a
              mechanism_id (rank-level or inherited from the action).
            - budget_ref present on ladder actions when a BudgetBook was
              validated this tick.
            - Coverage: jobs in the snapshot with no action are auto-filled
              (active -> KEEP, pending -> DEFER) with a warning, so a
              partial plan from a weak model still covers the cluster.

        Args:
            raw_plan: Whatever the LLM committed via FINAL_VAR(plan).
            cluster_snapshot: This tick's snapshot, for existence/state.

        Returns:
            A validated Plan.

        Raises:
            PlanMaterializationError: On any unrecoverable shape/semantic
                violation.
        """
        try:
            plan = Plan.from_raw(raw_plan, tick=self._current_tick)
        except ValueError as exc:
            raise PlanMaterializationError(str(exc)) from exc
        states = self._job_states(cluster_snapshot)
        book = agent_tools._CTX.validated_budget_book

        seen: set = set()
        for action in plan.actions:
            jid = action.job_id
            if jid in seen:
                raise PlanMaterializationError(f"duplicate action for job {jid}")
            seen.add(jid)

            if states is not None and jid not in states:
                raise PlanMaterializationError(f"job {jid} not in this tick's snapshot")

            required = REQUIRED_JOB_STATE.get(action.type)
            if states is not None and required is not None:
                actual = states.get(jid)
                if actual is not None and actual != required:
                    raise PlanMaterializationError(
                        f"job {jid}: {action.type.value} needs state "
                        f"{required!r}, job is {actual!r}"
                    )

            if action.type in LADDER_ACTIONS:
                self._validate_ladder(action, book, cluster_snapshot)

        if states is not None:
            self._autofill_coverage(plan, states)

        # Empty is valid only when the snapshot explicitly exposes an empty
        # job inventory. If inventory is unavailable, an empty commit gives us
        # no way to distinguish "no work" from an incomplete plan.
        if not plan.actions and states is None:
            raise PlanMaterializationError("plan has no actions and no job inventory to cover")

        return plan

    def _validate_ladder(self, action: PlanAction, book, cluster_snapshot=None) -> None:
        """Validate a ladder-bearing action; raise on hard violations."""
        jid = action.job_id
        if action.ladder is None:
            raise PlanMaterializationError(f"job {jid}: {action.type.value} requires a ladder")
        if not action.ladder:
            raise PlanMaterializationError(f"job {jid}: ladder is empty")

        try:
            PlanAction.assign_rank_ids(jid, action.ladder)
        except ValueError as exc:
            raise PlanMaterializationError(str(exc)) from exc

        job_features = agent_tools._job_features_for(cluster_snapshot, jid)
        for i, rank in enumerate(action.ladder):
            if rank.env is None or len(rank.env) != 5:
                raise PlanMaterializationError(
                    f"job {jid} rank {i}: env must be a 5-tuple "
                    "(market, cloud, region, zone, gpu_type) to be launchable"
                )
            if rank.n_replicas < 1:
                raise PlanMaterializationError(f"job {jid} rank {i}: n_replicas must be >= 1")
            if rank.mechanism_id is None:
                rank.mechanism_id = action.mechanism_id
            if rank.mechanism_id is None:
                raise PlanMaterializationError(f"job {jid} rank {i}: mechanism_id is required")
            registry = agent_tools._CTX.mechanism_registry
            if registry is None:
                raise PlanMaterializationError("mechanism registry is unavailable")
            try:
                mechanism = registry.get_mechanism(rank.mechanism_id)
            except KeyError:
                raise PlanMaterializationError(
                    f"job {jid} rank {i}: unknown mechanism_id {rank.mechanism_id!r}"
                ) from None
            context = agent_tools._rank_mechanism_context(rank, job_features)
            match = registry.match_scope(mechanism, context)
            if match["quality"] == "reject":
                raise PlanMaterializationError(
                    f"job {jid} rank {i}: mechanism {rank.mechanism_id!r} does not apply "
                    f"({'; '.join(match['reasons'])})"
                )

        if book is not None:
            slice_ = (book.get("job_budgets") or {}).get(jid)
            if slice_ is None or action.budget_ref != slice_.get("slice_id"):
                raise PlanMaterializationError(f"job {jid}: invalid BudgetSlice reference")
            resources = (
                cluster_snapshot.resources_summary()
                if cluster_snapshot is not None and hasattr(cluster_snapshot, "resources_summary")
                else None
            )
            violations = agent_tools._budget_violations(action, slice_, resources)
            if violations:
                raise PlanMaterializationError(f"job {jid}: {'; '.join(violations)}")

    def _autofill_coverage(self, plan: Plan, states: dict[str, str]) -> None:
        """Add conservative no-op actions for jobs the plan omitted.

        Active/running -> KEEP, waiting -> DEFER. Lets a weak model emit
        only the jobs it wants to change while the cluster stays fully
        covered. Logged so silent omissions are visible.
        """
        covered = plan.job_ids()
        for job_id, state in states.items():
            if job_id in covered:
                continue
            if state == "waiting":
                plan.actions.append(PlanAction(job_id=job_id, type=ActionType.DEFER))
            else:
                plan.actions.append(PlanAction(job_id=job_id, type=ActionType.KEEP))
            log.warning(
                "job %s omitted from plan; auto-filled %s",
                job_id,
                "DEFER" if state == "waiting" else "KEEP",
            )

    @staticmethod
    def _job_states(cluster_snapshot) -> dict[str, str] | None:
        """Map job_id -> state from the snapshot, or None if unavailable.

        Prefers an explicit job_states() method; else infers from the
        active/pending summaries (active -> running, pending -> waiting).
        None means the snapshot exposes no job inventory, so existence,
        state, and coverage checks are skipped.
        """
        if cluster_snapshot is None:
            return None
        if hasattr(cluster_snapshot, "job_states"):
            return dict(cluster_snapshot.job_states())
        states: dict[str, str] = {}
        has_any = False
        if hasattr(cluster_snapshot, "active_jobs_summary"):
            has_any = True
            for j in cluster_snapshot.active_jobs_summary():
                states[j.get("job_id", j.get("id"))] = j.get("state", "running")
        if hasattr(cluster_snapshot, "pending_jobs_summary"):
            has_any = True
            for j in cluster_snapshot.pending_jobs_summary():
                states[j.get("job_id", j.get("id"))] = j.get("state", "waiting")
        return states if has_any else None

    def _score_plan(self, plan: Plan) -> float:
        """Score a plan by aggregate sigma; 0.0 when scoring fails."""
        try:
            return float(agent_tools.compute_sigma(plan)["aggregate_sigma"])
        except Exception:
            log.exception("compute_sigma failed during K_P scoring")
            return 0.0

    def _fallback_plan(self, cluster_snapshot) -> Plan:
        """Build the typed keep-all / defer-pending fallback Plan.

        Uses the resource map's builder when present (normalized through
        Plan.from_raw); otherwise synthesizes from the snapshot summaries.
        Keeping the running cluster untouched is feasible by construction.
        """
        if self.resource_map is not None and hasattr(self.resource_map, "build_keep_all_plan"):
            raw = self.resource_map.build_keep_all_plan(cluster_snapshot)
            try:
                return Plan.from_raw(raw, tick=self._current_tick)
            except ValueError:
                log.exception("resource_map keep-all plan was malformed")

        actions: list[PlanAction] = []
        states = self._job_states(cluster_snapshot) or {}
        for job_id, state in states.items():
            kind = ActionType.DEFER if state == "waiting" else ActionType.KEEP
            actions.append(PlanAction(job_id=job_id, type=kind))
        return Plan(
            tick=self._current_tick, actions=actions, tick_rationale="safe keep-all fallback"
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_root_prompt(
        self,
        tick: int,
        repair_violations: list[str] | None = None,
    ) -> str:
        """Build the compact root system prompt for one tick.

        Metadata and contracts only - the full state lives in REPL
        variables. In repair mode the S5 violations are prepended so the
        root fixes the specific failures.
        """
        repair_section = ""
        if repair_violations:
            repair_section = (
                "REPAIR MODE. Your previous plan failed validation:\n- "
                + "\n- ".join(str(v) for v in repair_violations)
                + "\nFix these specific violations and re-commit.\n\n"
            )

        return (
            f"{repair_section}"
            # ---------- WHO YOU ARE ----------
            f"You are Koi's root cluster planner for tick {tick}.\n\n"
            "Once per 5-minute tick you observe the ENTIRE cluster, decide "
            "what runs on which GPUs, and commit exactly ONE plan P_t. The "
            "executor deploys it, telemetry measures the real outcomes, and "
            "NEXT tick you see whether you were right and correct course. You "
            "are one step in a closed control loop, not a one-shot optimizer: "
            "evidence, mechanism confidence, and the ideal point z* all carry "
            "across ticks, so a config that was wrong last tick is "
            "self-correcting.\n\n"
            # ---------- REPL STATE ----------
            "Full cluster state is loaded in the REPL - do NOT ask for it in "
            "the prompt; inspect it with code and print summaries only. REPL "
            "variables: cluster_snapshot, slow_state, evidence_store, "
            "mechanism_registry, resource_map, user_registry, plan, tick. "
            "Every agent tool is bound as a function (get_cluster_state, "
            "get_priority, build_user_envelopes, allocate_budget_book, "
            "validate_budget_book, run_job_specialists, predict_outcome, "
            "compute_sigma, get_applicable_mechanisms, get_influencing_knobs, "
            "optimize_config, "
            "get_z_star, size_ladder, check_feasibility, ...).\n\n"
            # ---------- YOUR JOB THIS TICK ----------
            "YOUR JOB: produce one cluster-wide plan that maximizes aggregate "
            "sigma, subject to user policy/quota, reserved GPU capacity, "
            "physical chain feasibility, SLO chance under worst-case demand "
            "(DRO), the swap budget B_t, and admission control. Only the "
            "reserved market exists this version - never plan spot or "
            "on-demand capacity.\n\n"
            # ---------- THE OBJECTIVE, EXACTLY ----------
            "THE OBJECTIVE, EXACTLY. Only PLACE and SWAP actions score. "
            "keep / defer / terminate / diagnose deploy nothing new and score "
            "EXACTLY 0. Aggregate sigma = sum over your place/swap actions of "
            "sigma_i = J + beta*EIG - gamma*Pr_DRO - lambda*SwitchCost, "
            "where:\n"
            "  - J (augmented Tchebycheff, higher is better): how close the "
            "job's predicted y_hat is to the ideal point z*. This is NOT a "
            "weighted sum - J is dominated by your WORST weighted objective "
            "gap (max_j w_j*gap_j), so you CANNOT hide bad latency behind "
            "great cost. To raise J, lift the WEAKEST objective. get_z_star() "
            "shows z* (what 'good' means right now); w_t are the objective "
            "weights.\n"
            "  - EIG (weighted by beta, exploration): expected information "
            "gain from actually trying this ladder. High when the mechanism is "
            "low-confidence - an uncertain-but-promising config is rewarded "
            "because Koi LEARNS from deploying it.\n"
            "  - Pr_DRO (weighted by gamma, risk): probability ANY SLO is "
            "violated under the worst-case demand band (Wasserstein-DRO). "
            "Placements sitting on the SLO edge cost you here.\n"
            "  - SwitchCost (weighted by lambda, churn): cost of moving a job "
            "off its current ladder (migration, reprice, disruption). Keeping "
            "a good-enough config beats churning for a tiny gain.\n\n"
            # ---------- WHEN TO DEFER ----------
            "WHEN TO DEFER - READ THIS. Deferring a waiting job scores 0. That "
            "is NOT 'harmless': the job stays UNSERVED and you earned no "
            "credit for it. defer / keep-all is a LAST RESORT, valid ONLY when "
            "either (a) no feasible placement exists given reserved capacity, "
            "or (b) EVERY feasible placement remains negative sigma AFTER you "
            "tried to fix its config with get_influencing_knobs + "
            "optimize_config and re-sized with size_ladder. Free capacity alone "
            "does not make a placement worthwhile; use its computed sigma. "
            "Never defer a job you did not attempt to place. If you are about "
            "to defer most jobs, stop and re-check z*, your configs, and your "
            "sizing before committing.\n\n"
            # ---------- HOW TO REASON ----------
            "HOW TO REASON. Before committing P_t, sketch several allocation "
            "frames psi_t - internal scratch reasoning only, not physical "
            "cluster state, not deployable objects, not part of the final "
            "schema. For non-trivial ticks compare several angles: "
            "feasibility/SLO-first, aggregate-sigma-first, "
            "churn/B_t-minimizing, and scarce-resource/user-fair. For each "
            "serious candidate, use tools to size ladders, simulate resources, "
            "compute_sigma, and check_feasibility. Independent config evaluations "
            "must run concurrently, never in a serial loop: define evaluate_config, "
            "then use `from concurrent.futures import ThreadPoolExecutor` and "
            "`with ThreadPoolExecutor(max_workers=5) as pool: results = "
            "list(pool.map(evaluate_config, configs))`. Keep compact summaries, "
            "then choose exactly one final Plan P_t and call "
            "FINAL_VAR(plan).\n\n"
            # ---------- MANDATORY ORDER ----------
            "MANDATORY ORDER (budgets before specialists - anti split-brain). "
            "Use this exact sequence, with no extra arguments: "
            "build_user_envelopes(); priority = get_priority(); "
            "budget_book = allocate_budget_book(); "
            "validation = validate_budget_book(budget_book); "
            "if validation['ok']: specialist_results = run_job_specialists(). "
            "budget_book is NOT a pre-existing REPL variable and NOT an "
            "object - you CREATE it by calling allocate_budget_book(); do not "
            "call budget_book.allocate(). It returns a dict shaped: "
            "{'tick': int, 'job_budgets': {job_id: {'slice_id': str, "
            "'job_id': str, 'user_id': str, 'env_budget': {env_key: gpus}, "
            "'pool_budget': {env_key: {instance_type: units}}, "
            "'allowed_actions': [...], 'strategy_hint': str, 'canary_cap': "
            "int, 'priority_score': float}}, 'reserves': {env_key: int}, "
            "'rationale': str}. Read a job's slice id as "
            "budget_book['job_budgets'][job_id]['slice_id'] and set that "
            "action's budget_ref to it. Do not pass cluster_snapshot into "
            "get_priority, allocate_budget_book, validate_budget_book, or "
            "run_job_specialists. Specialists optimize inside their BudgetSlice "
            "and report fitness; they never compete for resources. Their "
            "results are PROPOSALS - you own every final per-job decision: "
            "accept, modify, or discard each one, and never copy one into the "
            "plan unexamined. Do NOT trust specialist predicted_sigma - "
            "rescore with compute_sigma. Reallocate from fitness signals when "
            "the sigma gain is positive, rerun only affected specialists, then "
            "commit. Mechanism IDs are opaque Store IDs: never author one. Use "
            "an applicable ID, or call set_new_mechanisms and use its returned ID. "
            "After proposing a rank, call get_applicable_mechanisms(rank, "
            "job_features); prefer exact matches, then partial matches.\n\n"
            # ---------- SHARPEN BEFORE YOU SCORE ----------
            "SHARPEN EACH JOB BEFORE YOU SCORE IT. predict_outcome returns "
            "predictions already CALIBRATED against the evidence database "
            "(plus y_hat_raw and the calibration_offsets applied); trust the "
            "calibrated y_hat - it is what compute_sigma and size_ladder use, "
            "so a surrogate that was wrong last tick is self-correcting. To "
            "improve a config: get_influencing_knobs(job_features, objective) "
            "ranks which X knobs move that objective and how confident we are, "
            "then optimize_config(base_config, candidates, job_features) runs a "
            "calibrated coordinate descent over candidate values YOU choose and "
            "returns the best config by Tchebycheff J. These are optional "
            "refinement aids - use them on a config you reasoned to; you stay "
            "free to propose configs directly, and they never replace "
            "cluster-level reconciliation.\n\n"
            f"{self._plan_schema_section()}"
            # ---------- MECHANICS ----------
            "Write Python in ```repl blocks. Print what you need to see. Think "
            "and inspect as much as the turn budget allows; you have "
            f"{self.k_max} turns and {int(self.wall_clock_sec)} seconds. You "
            "never deploy anything - the executor runs only after validation. "
            "Call FINAL_VAR(plan) exactly once, when the plan is coherent and "
            "feasible."
        )

    @staticmethod
    def _plan_schema_section() -> str:
        """The exact plan schema the LLM must build, with field-by-field shape.

        Spelling out the dict shape (not just naming fields) is what lets a
        weak open model emit a parseable plan on the first try. The
        materializer accepts this Plan-shaped dict, a bare actions list, or
        a job_id->action dict, and auto-fills any omitted job with KEEP
        (active) or DEFER (waiting) - so a partial plan is safe.
        """
        return (
            "Commit `plan` as a dict with this shape:\n"
            "  plan = {\n"
            "    'tick_rationale': '<1-3 paragraphs of cluster-wide reasoning>',\n"
            "    'actions': [ <one action dict per job you decide> ],\n"
            "  }\n"
            "Action dict:\n"
            "  {'job_id': str, 'type': <action>, 'user_id': str,\n"
            "   'ladder': [<rank>, ...],            # only for place/swap\n"
            "   'target_tps': float,                # required throughput for place/swap\n"
            "   'target_p99_ttft_ms': float,        # online SLA, copied from job_features\n"
            "   'target_p99_tpot_ms': float,        # online SLA, copied from job_features\n"
            "   'mechanism_id': 'M_...',            # committed mechanism for the job\n"
            "   'swap_reason': 'scale_up|scale_down|migrate|replace|retune',  # swap only\n"
            "   'budget_ref': '<BudgetSlice id>',   # required if a BudgetBook was validated\n"
            "   'rationale': str}\n"
            "Rank dict (each entry of ladder):\n"
            "  {'role': 'aggregate',     # v0: AGGREGATE ONLY - one engine does prefill+decode\n"
            "   'rank_id': 'rank_0',      # omit rank_id; Koi auto-fills rank_0, rank_1, ...\n"
            "   'env': [market, cloud, region, zone, gpu_type],   # REQUIRED - launch target + ICP key\n"
            "   'config': {instance_type, gpu_count, tp, pp, sp, ep, cp, engine_name,\n"
            "              weight_dtype, kvcache_dtype,\n"
            "              weight_quantization_bits, prefix_cache_enabled, router_policy},\n"
            "   # Koi supplies max_num_seq, max_num_batched_tokens, and block_size;\n"
            "   # do not put engine-managed, workload, or hardware facts in config.\n"
            "   'n_replicas': int,       # rank DP / max endpoint count; do NOT put dp in config\n"
            "   'mechanism_id': 'M_...'}            # defaults to the action's mechanism_id\n"
            "v0 is AGGREGATE-ONLY per rank: every rank is one full "
            "prefill+decode engine (role 'aggregate'); do NOT split prefill/"
            "decode or set pd_enabled / prefill_worker_count / "
            "decode_worker_count. But a ladder MAY mix multiple HETEROGENEOUS "
            "ranks - different gpu_type, tp/pp, quant, even different clouds - "
            "whenever that improves the job's sigma or cost (e.g. spill a job "
            "across H100 and A100 pools, or pair a low-latency rank with a "
            "cheap high-throughput one). The only hard rule per rank: it must "
            "hold one whole model copy (tp*pp must fit the model). Pick "
            "whatever mix of ranks maximizes sigma - scoring and sizing compose "
            "your ranks for you (see size_ladder below).\n"
            "Action types and the job state each needs:\n"
            "  place    waiting->running   (needs ladder, target_tps; online needs p99 TTFT/TPOT targets)\n"
            "  keep     running->running   (no ladder)\n"
            "  swap     running->running   (needs ladder; scale/migrate/retune/replace)\n"
            "  defer    waiting->waiting    (no ladder)\n"
            "  terminate any->stopped       (no ladder; give up after budget/policy exhaustion)\n"
            "  diagnose  no change          (no ladder; record a theory only)\n"
            "Every ladder rank MUST carry a unique rank_id (or let Koi auto-fill one), "
            "MUST carry a 5-element env, and MUST resolve a "
            "mechanism_id, either on the rank or inherited from the action. "
            "config.instance_type, gpu_count, tp, and pp are required; sp/ep/cp "
            "default to 1 when omitted. For cloud instance pools, "
            "config.instance_type is required when the env has multiple pools. "
            "gpu_count is engine GPU demand, not "
            "reserved capacity; Koi reserves and charges one full instance per "
            "n_replicas. n_replicas is the rank's DP/max endpoint count; do not "
            "set dp separately. Discrete on-prem GPU pools may remain GPU-granular. "
            "Jobs you omit are auto-kept (running) or auto-deferred (waiting), "
            "so list only the jobs you actually decide.\n"
            "Do NOT guess n_replicas. Call size_ladder(ranks, job_features): "
            "it SHARES one throughput target across your ranks (filling them in "
            "the order you list, each covering the remaining target) and sizes "
            "each rank's n_replicas; achieved_tps is the SUM across ranks, so "
            "heterogeneous ranks add capacity. For online it EXCLUDES any rank "
            "whose predicted p99 TTFT/TPOT exceeds target (latency is "
            "per-replica - replicas cannot fix it) and spills its share to the "
            "others, and it derates per-replica throughput so queue wait stays "
            "bounded. size_ladder sums achieved throughput across ranks to "
            "meet target_tps. compute_sigma scores the composed job-level "
            "y_hat using worst-case latency, throughput/cost rollups, DRO "
            "risk, EIG, and switch cost, so a heterogeneous ladder is scored "
            "as the whole job, not one rank. Use it as: sized = size_ladder(ranks, "
            "job_features); action['ladder'] = sized['ranks']. The other fields "
            "(meets_target, unmet_tps, per-rank slo_violations, marginal_value) "
            "are diagnostics.\n\n"
        )
