"""Smokes for the paper ablation modes (mechanism inert / learning frozen)."""

import pytest

from src.config import ablation
from src.orchestrator.fsm_states import FSMState, TickContext, TickRunner

try:
    import src.agent.tools.agent_tools as agent_tools
    from src.agent.agent import KoiAgentHarness, SpecialistRunner

    _AGENT_STACK_AVAILABLE = True
except ModuleNotFoundError:
    # aiconfigurator / tandemn_system_data are absent on machines that cannot
    # run the full stack; the agent-layer smokes only run where they exist.
    _AGENT_STACK_AVAILABLE = False

needs_agent_stack = pytest.mark.skipif(
    not _AGENT_STACK_AVAILABLE, reason="agent stack dependencies unavailable"
)


@pytest.fixture(autouse=True)
def _reset_ablation():
    yield
    ablation.configure_mechanism_mode("full")
    ablation.configure_learning_mode("online")
    ablation._passthrough_mechanism_id = None


class _Untouchable:
    """Stub that fails the test on any attribute access."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, item):
        raise AssertionError(f"{self._name}.{item} must not be touched in this mode")


class _StubSlowState:
    def __init__(self):
        self.tick = 0
        self.B_t = 10
        self.beta_t = 0.5


class _StubSlowLoop:
    def __init__(self):
        self.state = _StubSlowState()

    def slow_update_all(self, *args, **kwargs):
        raise AssertionError("slow_update_all must not run when learning is frozen")

    def anneal_targets(self, tick):
        raise AssertionError("anneal_targets must not run when learning is frozen")


class _StubSnapshot:
    def __init__(self, active=None, pending=None):
        self._active = active or []
        self._pending = pending or []

    def active_jobs_summary(self):
        return self._active

    def pending_jobs_summary(self):
        return self._pending


# ----------------------------------------------------------------------
# Ablation config module
# ----------------------------------------------------------------------


def test_default_modes_are_full_and_online():
    assert ablation.mechanism_mode() == "full"
    assert ablation.learning_mode() == "online"
    assert not ablation.mechanism_inert()
    assert not ablation.learning_frozen()


def test_configure_rejects_unknown_modes():
    with pytest.raises(ValueError):
        ablation.configure_mechanism_mode("off")
    with pytest.raises(ValueError):
        ablation.configure_learning_mode("disabled")
    with pytest.raises(TypeError):
        ablation.configure_mechanism_mode(None)


def test_passthrough_id_requires_registration():
    ablation.configure_mechanism_mode("inert")
    with pytest.raises(RuntimeError):
        ablation.passthrough_mechanism_id()
    ablation.set_passthrough_mechanism_id("M_test1234")
    assert ablation.passthrough_mechanism_id() == "M_test1234"


def test_ablation_status_reports_effective_modes():
    ablation.configure_mechanism_mode("inert")
    ablation.configure_learning_mode("frozen")
    ablation.set_passthrough_mechanism_id("M_test1234")
    assert ablation.ablation_status() == {
        "mechanism_mode": "inert",
        "learning_mode": "frozen",
        "passthrough_mechanism_id": "M_test1234",
    }


# ----------------------------------------------------------------------
# Frozen learning: FSM behavior
# ----------------------------------------------------------------------


def _stub_runner():
    runner = TickRunner.__new__(TickRunner)
    runner._deployment_ledger = {}
    runner._prediction_ledger = {}
    runner._health_state = {}
    runner._dead_shapes = {}
    runner._starved_streaks = {}
    return runner


def test_frozen_s3_skips_all_learning_and_hands_forward_frozen_state():
    ablation.configure_learning_mode("frozen")
    runner = _stub_runner()
    runner.slow_loop = _StubSlowLoop()
    runner.confidence_service = _Untouchable("confidence_service")
    runner.dro = _Untouchable("dro")
    runner.mechanism_registry = _Untouchable("mechanism_registry")
    ctx = TickContext(tick=7)
    ctx.evidence_rows = [object()]  # rows exist; frozen S3 must not consume them

    next_state = runner.S3(ctx)

    assert next_state == FSMState.S4_AGENTIC_PLAN
    assert ctx.new_slow_state is runner.slow_loop.state
    assert ctx.new_slow_state.tick == 7
    assert ctx.slow_update_diagnostics == {"learning_frozen": True}
    assert ctx.confidence_diagnostics == []


def test_frozen_s0_does_not_annotate_dead_shapes():
    runner = _stub_runner()
    runner.on_tick_start = None
    runner._dead_shapes[("m1", "H100", 2, 1)] = {
        "model_id": "m1",
        "gpu_type": "H100",
        "tp": 2,
        "pp": 1,
        "reason": "zero_throughput_under_load",
        "first_tick": 1,
        "ticks": 2,
        "last_tick": 2,
        "retry_after_tick": 99,
    }
    pending = {"job_id": "job-1", "job_features": {"model_id": "m1"}}

    class _Map:
        def snapshot_cluster_state(self, tick):
            return _StubSnapshot(pending=[pending])

    runner.resource_map = _Map()

    ablation.configure_learning_mode("frozen")
    runner.S0(TickContext(tick=3))
    assert "observed_dead_shapes" not in pending

    ablation.configure_learning_mode("online")
    runner.S0(TickContext(tick=3))
    assert pending["observed_dead_shapes"][0]["gpu_type"] == "H100"


# ----------------------------------------------------------------------
# Inert mechanism mode: agent tooling
# ----------------------------------------------------------------------


@needs_agent_stack
def test_inert_compute_eig_is_exactly_zero_without_bindings():
    ablation.configure_mechanism_mode("inert")
    assert agent_tools.compute_eig({"ranks": []}) == 0.0


@needs_agent_stack
def test_inert_mechanism_selection_returns_passthrough_sentinel():
    ablation.configure_mechanism_mode("inert")
    ablation.set_passthrough_mechanism_id("M_passthru")
    assert agent_tools._applicable_mechanism_id({}, {}) == "M_passthru"


@needs_agent_stack
def test_inert_mode_hides_dag_tools_from_the_llm():
    baseline = set(agent_tools.all_callables())
    ablation.configure_mechanism_mode("inert")
    tools = set(agent_tools.all_callables())
    for name in (
        "get_edge_confidence",
        "get_mechanism_confidence",
        "get_influencing_knobs",
        "get_scope",
        "get_applicable_mechanisms",
        "compute_eig",
        "set_new_mechanisms",
        "val_new_mechanisms",
    ):
        assert name in baseline
        assert name not in tools
    # Non-DAG tools stay exposed.
    assert "predict_outcome" in tools
    assert "build_scored_candidates" in tools


@needs_agent_stack
def test_frozen_mode_hides_mechanism_admission_tools():
    ablation.configure_learning_mode("frozen")
    tools = set(agent_tools.all_callables())
    assert "set_new_mechanisms" not in tools
    assert "val_new_mechanisms" not in tools
    assert "get_edge_confidence" in tools


@needs_agent_stack
def test_mechanism_admission_is_refused_in_both_ablations():
    for configure in (
        lambda: ablation.configure_mechanism_mode("inert"),
        lambda: ablation.configure_learning_mode("frozen"),
    ):
        ablation.configure_mechanism_mode("full")
        ablation.configure_learning_mode("online")
        configure()
        result = agent_tools.set_new_mechanisms([], {}, "blurb")
        assert result["ok"] is False
        assert result["mechanism_id"] is None
        assert "disabled" in result["violations"][0]


# ----------------------------------------------------------------------
# Inert mechanism mode: agent prompts and validation
# ----------------------------------------------------------------------


@needs_agent_stack
def test_inert_specialist_rank_schema_drops_mechanism_requirement():
    rank = {
        "role": "aggregate",
        "env": ["reserved", "aws", "r1", "z1", "H100"],
        "config": {"instance_type": "p5", "gpu_count": 2, "tp": 2, "pp": 1},
        "n_replicas": 1,
    }
    full_violations = SpecialistRunner._validate_rank_schema(dict(rank), 0, [])
    assert any("mechanism_id" in v for v in full_violations)

    ablation.configure_mechanism_mode("inert")
    inert_violations = SpecialistRunner._validate_rank_schema(dict(rank), 0, [])
    assert not any("mechanism_id" in v for v in inert_violations)


@needs_agent_stack
def test_inert_prompts_never_mention_mechanisms():
    ablation.configure_mechanism_mode("inert")
    ablation.set_passthrough_mechanism_id("M_passthru")
    specialist = SpecialistRunner._default_prompt("job-1", {}, {})
    assert "mechanism_candidates" not in specialist
    assert "mechanism_ids" not in specialist

    harness = KoiAgentHarness.__new__(KoiAgentHarness)
    harness.k_max = 4
    harness.wall_clock_sec = 240.0
    root = harness.build_root_prompt(tick=1)
    assert "EIG" not in root
    assert "mechanism confidence" not in root
    assert "get_applicable_mechanisms" not in root
    assert "pass-through" in root


@needs_agent_stack
def test_frozen_root_prompt_declares_fixed_knobs():
    ablation.configure_learning_mode("frozen")
    harness = KoiAgentHarness.__new__(KoiAgentHarness)
    harness.k_max = 4
    harness.wall_clock_sec = 240.0
    root = harness.build_root_prompt(tick=1)
    assert "no online learning" in root
    assert "FIXED seed priors" in root
    assert "self-correcting" not in root
