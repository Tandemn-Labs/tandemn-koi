"""Koi runner wired against Tandemn Store."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.agent.agent import KoiAgentHarness
from src.agent.llm_clients import OpenAICompatClient, RecordingLLMClient
from src.agent.tools import agent_tools
from src.bootstrap.initialization import (
    ensure_passthrough_mechanism,
    init_causal_graph,
    init_surrogate_stack,
)
from src.config import ablation
from src.core.evidence_service import EvidenceService
from src.cost import switch_cost as switchcost_module
from src.cost.dro import DRO
from src.executor.executor import StorePlanExecutor
from src.exploration import eig as eig_module
from src.infra.resource_map import RANK_FAILURE_HISTORY_TICKS, ResourceMapManager
from src.infra.telemetry import StoreTelemetry
from src.learning.regret import RegretCalculator
from src.learning.slow_loop import SlowLoop
from src.orchestrator import fsm_states
from src.orchestrator.debug_logging import DebugLogger
from src.orchestrator.fsm_states import TickContext, TickRunner
from src.prediction import tchebycheff as tchebycheff_module
from src.validation.cusum import Cusum
from src.validation.icp import ICP
from src.validation.quadrants import QuadrantValidator
from src.validation.validator import Validator
from tandemn_system_data.clients import (  # type: ignore[import-untyped]
    GpuMetricStore,
    PostgresClient,
)

log = logging.getLogger("koi.runner")

DEFAULT_TYPICAL_RANGES = {
    "p99_ttft_ms": 1000.0,
    "p99_tpot_ms": 50.0,
    "throughput_token_per_sec": 1000.0,
    "cost_per_token": 1e-5,
    "slo_margin": 1000.0,
}

PARTIAL_ONLINE_ADMISSION_MODES = ("off", "advisory")
FOUNDRY_MODEL_ROUTES = {
    "gpt-5.6-sol": (
        "KOI_FOUNDRY_OPENAI_BASE_URL",
        "KOI_FOUNDRY_OPENAI_API_KEY",
        20_000,
    ),
    "deepseek-v4-pro": (
        "KOI_FOUNDRY_DEEPSEEK_BASE_URL",
        "KOI_FOUNDRY_DEEPSEEK_API_KEY",
        20_000,
    ),
    "cohere-command-a": (
        "KOI_FOUNDRY_COHERE_BASE_URL",
        "KOI_FOUNDRY_COHERE_API_KEY",
        8_000,
    ),
}
FOUNDRY_NO_TEMPERATURE_MODELS = {"gpt-5.6-sol", "deepseek-v4-pro"}


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default=os.getenv("TANDEMN_USER_ID"))
    parser.add_argument(
        "--start-tick",
        "--tick",
        dest="start_tick",
        type=int,
        default=None,
        help="First tick id; default is evidence+1",
    )
    parser.add_argument("--ticks", type=int, default=1, help="Tick count; 0 runs forever")
    parser.add_argument("--tick-interval-sec", type=int, default=300)
    parser.add_argument("--telemetry-window-sec", type=int, default=300)
    parser.add_argument("--openai-base-url")
    parser.add_argument(
        "--openai-model",
        "--model",
        dest="openai_model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.5"),
    )
    parser.add_argument("--api-key")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout-sec", type=float, default=200.0)
    parser.add_argument("--k-p", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--wall-clock-sec", type=float, default=240.0)
    parser.add_argument("--stdout-limit", type=int, default=10000)
    parser.add_argument("--error-limit", type=int, default=30)
    parser.add_argument("--live-agent", action="store_true")
    parser.add_argument("--print-llm", action="store_true")
    parser.add_argument("--log-string-limit", type=int, default=1200)
    parser.add_argument("--log-level", default=os.getenv("KOI_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-dir", default=os.getenv("KOI_LOG_DIR", "logs/koi"))
    parser.add_argument("--run-id", default=os.getenv("KOI_RUN_ID"))
    parser.add_argument("--trace", choices=("no-llm", "errors", "all"), default="no-llm")
    parser.add_argument(
        "--surrogate-lower-quantile",
        type=float,
        default=os.getenv("KOI_SURROGATE_LOWER_QUANTILE", "0.05"),
        help="Conservative fusion residual quantile (KOI_SURROGATE_LOWER_QUANTILE; default: 0.05)",
    )
    parser.add_argument(
        "--surrogate-peer-mode",
        choices=("off", "shadow", "enabled"),
        default=os.getenv("KOI_SURROGATE_PEER_MODE", "shadow"),
        help="External predictor mode; requires the optional tandemn-predictors package",
    )
    parser.add_argument(
        "--surrogate-call-budget",
        type=_positive_int,
        default=os.getenv("KOI_SURROGATE_CALL_BUDGET", "100"),
        help="Maximum distinct surrogate search calls per tick (default: 100)",
    )
    parser.add_argument(
        "--partial-online-admission",
        choices=PARTIAL_ONLINE_ADMISSION_MODES,
        default=os.getenv("KOI_PARTIAL_ONLINE_ADMISSION", "advisory"),
        help="Partial online admission mode (KOI_PARTIAL_ONLINE_ADMISSION; default: advisory)",
    )
    parser.add_argument(
        "--mechanism-mode",
        choices=ablation.MECHANISM_MODES,
        default=os.getenv("KOI_MECHANISM_MODE", "full"),
        help="Causal DAG ablation: 'inert' zeroes EIG and replaces mechanism selection "
        "with a pass-through id (KOI_MECHANISM_MODE; default: full)",
    )
    parser.add_argument(
        "--learning-mode",
        choices=ablation.LEARNING_MODES,
        default=os.getenv("KOI_LEARNING_MODE", "online"),
        help="Online-learning ablation: 'frozen' skips S3 learning, surrogate "
        "calibration, and dead-shape memory (KOI_LEARNING_MODE; default: online)",
    )
    parser.add_argument("--rust-log", default=os.getenv("RUST_LOG", "warn"))
    args = parser.parse_args(argv)
    if args.partial_online_admission not in PARTIAL_ONLINE_ADMISSION_MODES:
        parser.error(
            "--partial-online-admission must be one of: "
            + ", ".join(PARTIAL_ONLINE_ADMISSION_MODES)
        )
    # choices only guards CLI values; an env-var default bypasses it.
    if args.mechanism_mode not in ablation.MECHANISM_MODES:
        parser.error("--mechanism-mode must be one of: " + ", ".join(ablation.MECHANISM_MODES))
    if args.learning_mode not in ablation.LEARNING_MODES:
        parser.error("--learning-mode must be one of: " + ", ".join(ablation.LEARNING_MODES))
    return args


def _is_foundry_endpoint(base_url: str) -> bool:
    """Return whether a base URL targets an Azure Foundry OpenAI v1 endpoint."""
    host = (urlparse(base_url).hostname or "").lower()
    return host.endswith((".openai.azure.com", ".services.ai.azure.com"))


def resolve_llm_settings(
    args: argparse.Namespace,
) -> tuple[str, str | None, float | None, int | None, str | None]:
    """Resolve model-selected Foundry settings without exposing credentials in CLI args."""
    model = str(getattr(args, "openai_model", "gpt-5.5"))
    route = FOUNDRY_MODEL_ROUTES.get(model.lower())
    routed_base_url = os.getenv(route[0]) if route else None
    configured_base_url = (
        getattr(args, "openai_base_url", None)
        or routed_base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("KOI_FOUNDRY_BASE_URL")
    )
    if route and not configured_base_url:
        raise SystemExit(f"{model} requires {route[0]} or --openai-base-url")
    base_url = configured_base_url or "https://api.openai.com/v1"
    foundry = _is_foundry_endpoint(base_url)
    api_key = getattr(args, "api_key", None)
    if not api_key and route and foundry:
        api_key = os.getenv(route[1])
    if not api_key and foundry:
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("KOI_FOUNDRY_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    requested_temperature = getattr(args, "temperature", None)
    if (
        foundry
        and model.lower() in FOUNDRY_NO_TEMPERATURE_MODELS
        and requested_temperature is not None
    ):
        raise SystemExit(f"{model} does not support --temperature on Azure Foundry")
    temperature = (
        requested_temperature if requested_temperature is not None else (None if foundry else 1.0)
    )

    requested_max_tokens = getattr(args, "max_tokens", None)
    max_tokens = requested_max_tokens
    if max_tokens is None and foundry:
        max_tokens = route[2] if route else None
    if max_tokens is None and not foundry:
        max_tokens = 20_000
    return (
        str(base_url),
        api_key,
        temperature,
        max_tokens,
        "max_completion_tokens" if foundry else None,
    )


def configure_logging(level: str, log_file=None) -> None:
    """Configure process logging for the runner."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


def build_runner(args: argparse.Namespace):
    """Build one Store-backed TickRunner and its key debug handles."""
    if not args.user_id:
        raise SystemExit("TANDEMN_USER_ID or --user-id is required")
    base_url, api_key, temperature, max_tokens, token_limit_param = resolve_llm_settings(args)
    if not api_key:
        raise SystemExit("OPENAI_API_KEY, an Azure Foundry key, or --api-key is required")
    args.openai_base_url = base_url
    args.api_key = api_key
    args.temperature = temperature
    args.max_tokens = max_tokens

    agent_tools.configure_surrogate_call_budget(args.surrogate_call_budget)
    agent_tools.configure_partial_online_admission(args.partial_online_admission)
    ablation.configure_mechanism_mode(args.mechanism_mode)
    ablation.configure_learning_mode(args.learning_mode)
    os.environ["RUST_LOG"] = str(args.rust_log)

    client = PostgresClient()
    candidate_graph, mechanism_registry, confidence_service = init_causal_graph(
        args.user_id, postgres_client=client
    )
    if ablation.mechanism_inert():
        ablation.set_passthrough_mechanism_id(
            ensure_passthrough_mechanism(candidate_graph, mechanism_registry, confidence_service)
        )
        log.info(
            "mechanism-mode inert: pass-through mechanism %s",
            ablation.passthrough_mechanism_id(),
        )
    evidence_store = EvidenceService(user_id=args.user_id, postgres_client=client)
    dro = DRO()
    regret = RegretCalculator()
    cusum = Cusum()
    icp = ICP()
    quadrant_validator = QuadrantValidator()
    slow_loop = SlowLoop(
        evidence_store=evidence_store,
        dro=dro,
        regret_calculator=regret,
        objectives=candidate_graph.y,
        typical_ranges={obj: DEFAULT_TYPICAL_RANGES.get(obj, 1.0) for obj in candidate_graph.y},
        cusum=cusum,
        tracked_v_variables=candidate_graph.v,
    )
    # The failure lookback must not collapse to zero when ticks are driven
    # externally with --tick-interval-sec 0 (the simulator does this): a zero
    # window meant no rank failure ever reached a planning tick.
    failure_window_sec = RANK_FAILURE_HISTORY_TICKS * max(
        int(args.tick_interval_sec), int(args.telemetry_window_sec), 60
    )
    resource_map = ResourceMapManager(
        user_id=args.user_id,
        postgres_client=client,
        rank_failure_history_seconds=failure_window_sec,
    )
    telemetry = StoreTelemetry(
        user_id=args.user_id,
        gpu_metric_store=GpuMetricStore(client),
        candidate_graph=candidate_graph,
        tick_interval_sec=args.telemetry_window_sec,
    )
    validator = Validator(
        candidate_graph=candidate_graph,
        mechanism_registry=mechanism_registry,
        resource_map=resource_map,
        partial_online_admission_mode=args.partial_online_admission,
    )
    # Frozen learning severs the surrogate's evidence input: calibration and
    # throughput fusion are recomputed from evidence per call, so an unbound
    # store is what makes the surrogate truly static.
    surrogate = init_surrogate_stack(
        evidence_store=None if ablation.learning_frozen() else evidence_store,
        peer_mode=args.surrogate_peer_mode,
        lower_quantile=args.surrogate_lower_quantile,
    )
    llm = RecordingLLMClient(
        OpenAICompatClient(
            base_url=args.openai_base_url,
            model=args.openai_model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            token_limit_param=token_limit_param,
        ),
        live=args.live_agent,
        print_messages=args.print_llm,
        log_string_limit=args.log_string_limit,
    )
    agent = KoiAgentHarness(
        llm_client=llm,
        specialist_llm_client=llm,
        resource_map=resource_map,
        plan_validator=validator,
        tool_dependencies={
            "slow_loop": slow_loop,
            "dro": dro,
            "evidence_store": evidence_store,
            "mechanism_registry": mechanism_registry,
            "confidence_service": confidence_service,
            "candidate_graph": candidate_graph,
            "eig_module": eig_module,
            "tchebycheff_module": tchebycheff_module,
            "switchcost_module": switchcost_module,
            "surrogate": surrogate,
            "telemetry": telemetry,
            "cusum": cusum,
            "icp": icp,
            "quadrant_validator": quadrant_validator,
            "regret_calculator": regret,
        },
        config={
            "k_p": args.k_p,
            "k_max": args.k_max,
            "wall_clock_sec": args.wall_clock_sec,
            "stdout_limit": args.stdout_limit,
            "max_history_messages": 0,
            "consecutive_error_limit": args.error_limit,
        },
    )
    runner = TickRunner(
        evidence_store=evidence_store,
        telemetry=telemetry,
        cusum=cusum,
        icp=icp,
        quadrant_validator=quadrant_validator,
        confidence_service=confidence_service,
        slow_loop=slow_loop,
        dro=dro,
        mechanism_registry=mechanism_registry,
        resource_map=resource_map,
        agent=agent,
        plan_validator=validator,
        executor=StorePlanExecutor(args.user_id, postgres_client=client),
        candidate_graph=candidate_graph,
        tchebycheff=tchebycheff_module,
        tick_interval_sec=args.tick_interval_sec,
        on_tick_start=agent_tools.reset_tick_caches,
    )
    return runner, evidence_store, agent, llm


def next_tick(evidence_store: Any, requested_tick: int | None) -> int:
    """Return the requested tick or the next persisted evidence tick."""
    if requested_tick is not None:
        return int(requested_tick)
    return int(evidence_store.current_tick()) + 1


def log_tick_summary(ctx: TickContext) -> None:
    """Log the compact outcome of one FSM tick."""
    actions = getattr(ctx.validated_plan, "actions", []) or []
    log.info(
        "tick=%d states=%s evidence_rows=%d actions=%d deploy_acks=%s error=%s",
        ctx.tick,
        [state.value for state in ctx.state_history],
        len(ctx.evidence_rows),
        len(actions),
        ctx.deploy_acks,
        repr(ctx.error) if ctx.error else None,
    )


def clear_tick_buffers(agent: Any, llm: Any) -> None:
    """Drop per-tick debug buffers so continuous runs do not grow forever."""
    events = getattr(getattr(agent, "trace", None), "events", None)
    if isinstance(events, list):
        events.clear()
    calls = getattr(llm, "calls", None)
    if isinstance(calls, list):
        calls.clear()


def _git_output(*args: str) -> str | None:
    """Return bounded Git output, or None when metadata is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _source_revision() -> str | None:
    """Return the current Git revision when repository metadata is available."""
    return _git_output("rev-parse", "HEAD") or None


def _source_dirty() -> bool | None:
    """Return whether Git sees source changes, or None when Git is unavailable."""
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    return None if status is None else bool(status)


def emit_run_manifest(debug_logger: DebugLogger, args: argparse.Namespace) -> None:
    """Persist effective non-secret runner configuration once per run."""
    admission_status_fn = getattr(agent_tools, "get_partial_online_admission_status", None)
    admission_status = (
        admission_status_fn()
        if callable(admission_status_fn)
        else {"mode": args.partial_online_admission, "status": "unavailable"}
    )
    payload: dict[str, Any] = {
        "config": {
            "model": args.openai_model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout_sec": args.timeout_sec,
            "k_p": args.k_p,
            "k_max": args.k_max,
            "wall_clock_sec": args.wall_clock_sec,
            "tick_interval_sec": args.tick_interval_sec,
            "telemetry_window_sec": args.telemetry_window_sec,
            "trace": args.trace,
            "surrogate_peer_mode": args.surrogate_peer_mode,
            "surrogate_lower_quantile": args.surrogate_lower_quantile,
            "surrogate_call_budget": args.surrogate_call_budget,
            "partial_online_admission": args.partial_online_admission,
            "mechanism_mode": args.mechanism_mode,
            "learning_mode": args.learning_mode,
        },
        "surrogate_budget": agent_tools.get_surrogate_budget_status(),
        "partial_online_admission": admission_status,
        "ablation": ablation.ablation_status(),
    }
    revision = _source_revision()
    if revision is not None:
        payload["source_revision"] = revision
    source_dirty = _source_dirty()
    if source_dirty is not None:
        payload["source_dirty"] = source_dirty
    debug_logger.write_event("run_manifest", payload)


def run_ticks(
    *,
    evidence_store: Any,
    agent: Any,
    llm: Any,
    requested_tick: int | None,
    ticks: int,
    run_tick_fn,
    debug_logger: DebugLogger | None = None,
) -> int:
    """Run ticks until count is exhausted or shutdown is requested."""
    if ticks < 0:
        raise SystemExit("--ticks must be >= 0")

    tick = next_tick(evidence_store, requested_tick)
    completed = 0
    exit_code = 0
    while ticks == 0 or completed < ticks:
        log.info("starting koi tick %d", tick)
        ctx = run_tick_fn(tick)
        log_tick_summary(ctx)
        if ctx.error:
            exit_code = 1
        if debug_logger is not None:
            try:
                debug_logger.persist_runner_tick(ctx, agent, llm)
            except Exception:
                log.exception("debug logging failed at tick %d", tick)
        clear_tick_buffers(agent, llm)
        tick += 1
        completed += 1
    return exit_code


def main(
    argv: list[str] | None = None,
    *,
    build_runner_fn=build_runner,
    run_tick_fn=fsm_states.run_tick,
) -> int:
    """Run Koi ticks through the configured loop."""
    args = parse_args(argv)
    debug_logger = DebugLogger(args.log_dir, trace=args.trace, run_id=args.run_id)
    configure_logging(args.log_level, debug_logger.runner_log_path)
    log.info("debug logs: %s", debug_logger.run_dir)
    runner, evidence_store, agent, llm = build_runner_fn(args)
    emit_run_manifest(debug_logger, args)
    if hasattr(runner, "trace"):
        runner.trace = debug_logger
    agent_tools.bind_tools(trace_logger=debug_logger)
    fsm_states.bind_runner(runner)
    try:
        return run_ticks(
            evidence_store=evidence_store,
            agent=agent,
            llm=llm,
            requested_tick=args.start_tick,
            ticks=args.ticks,
            run_tick_fn=run_tick_fn,
            debug_logger=debug_logger,
        )
    except KeyboardInterrupt:
        log.info("shutdown requested")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
