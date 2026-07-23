"""Koi runner wired against Tandemn Store."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from src.agent.agent import KoiAgentHarness
from src.agent.llm_clients import OpenAICompatClient, RecordingLLMClient
from src.agent.tools import agent_tools
from src.bootstrap.initialization import init_causal_graph
from src.core.evidence_service import EvidenceService
from src.cost import switch_cost as switchcost_module
from src.cost.dro import DRO
from src.executor.executor import StorePlanExecutor
from src.exploration import eig as eig_module
from src.infra.resource_map import ResourceMapManager
from src.infra.telemetry import StoreTelemetry
from src.learning.regret import RegretCalculator
from src.learning.slow_loop import SlowLoop
from src.orchestrator import fsm_states
from src.orchestrator.debug_logging import DebugLogger
from src.orchestrator.fsm_states import TickContext, TickRunner
from src.prediction import tchebycheff as tchebycheff_module
from src.prediction.surrogate import SurrogatePrediction
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
    parser.add_argument(
        "--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--timeout-sec", type=float, default=200.0)
    parser.add_argument("--k-p", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--wall-clock-sec", type=float, default=520.0)
    parser.add_argument("--stdout-limit", type=int, default=10000)
    parser.add_argument("--error-limit", type=int, default=30)
    parser.add_argument("--live-agent", action="store_true")
    parser.add_argument("--print-llm", action="store_true")
    parser.add_argument("--log-string-limit", type=int, default=1200)
    parser.add_argument("--log-level", default=os.getenv("KOI_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-dir", default=os.getenv("KOI_LOG_DIR", "logs/koi"))
    parser.add_argument("--run-id", default=os.getenv("KOI_RUN_ID"))
    parser.add_argument("--trace", choices=("summary", "full"), default="summary")
    parser.add_argument("--rust-log", default=os.getenv("RUST_LOG", "warn"))
    return parser.parse_args(argv)


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
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY or --api-key is required")

    os.environ["RUST_LOG"] = str(args.rust_log)

    client = PostgresClient()
    candidate_graph, mechanism_registry, confidence_service = init_causal_graph(
        args.user_id, postgres_client=client
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
    resource_map = ResourceMapManager(user_id=args.user_id, postgres_client=client)
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
    )
    surrogate = SurrogatePrediction(objective="online")
    llm = RecordingLLMClient(
        OpenAICompatClient(
            base_url=args.openai_base_url,
            model=args.openai_model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
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
    if hasattr(runner, "trace"):
        runner.trace = debug_logger
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
