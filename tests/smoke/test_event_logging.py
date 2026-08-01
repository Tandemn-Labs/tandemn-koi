from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from src.core.models import ActionType, Plan, PlanAction
from src.observability.events import ChronologicalEventLogger
from src.orchestrator.fsm_states import FSMState, TickRunner
from src.validation.validator import ValidationResult


class _Kind(Enum):
    SAMPLE = "sample"


@dataclass
class _Payload:
    values: np.ndarray
    kind: _Kind


def test_chronological_logger_writes_ordered_jsonl_and_converts_values(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sink = ChronologicalEventLogger(path, run_id="run-test")
    sink.emit(
        {
            "event": "test.first",
            "component": "test",
            "payload": _Payload(np.asarray([1.0, np.nan]), _Kind.SAMPLE),
            "path": tmp_path,
            "items": {"b", "a"},
            "error": ValueError("bad value"),
            "schema_name": "cannot-overwrite",
            "sequence": 999,
        }
    )
    sink.emit({"event": "test.second", "component": "test", "scalar": np.int64(4)})
    sink.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["sequence"] for record in records] == [1, 2]
    assert all(record["schema_name"] == "koi-event" for record in records)
    assert records[0]["payload"] == {"values": [1.0, "NaN"], "kind": "sample"}
    assert records[0]["items"] == ["a", "b"]
    assert records[0]["error"]["type"] == "ValueError"
    assert records[1]["scalar"] == 4
    assert records[0]["timestamp_epoch_s"] <= records[1]["timestamp_epoch_s"]


def test_chronological_logger_serializes_concurrent_emits(tmp_path: Path) -> None:
    sink = ChronologicalEventLogger(tmp_path / "concurrent.jsonl")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda index: sink.emit(
                    {"event": "test.concurrent", "component": "test", "index": index}
                ),
                range(40),
            )
        )
    sink.close()

    records = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [record["sequence"] for record in records] == list(range(1, 41))
    assert {record["index"] for record in records} == set(range(40))


class _Snapshot:
    state_version = 7
    active_jobs: ClassVar[list[dict]] = []
    pending_jobs: ClassVar[list[dict[str, str]]] = [{"job_id": "job-1"}]

    def active_jobs_summary(self):
        return list(self.active_jobs)

    def pending_jobs_summary(self):
        return list(self.pending_jobs)

    def resources_summary(self):
        return {"env": {"free": 1, "total": 1}}


class _ResourceMap:
    def snapshot_cluster_state(self, tick):
        del tick
        return _Snapshot()

    @staticmethod
    def build_keep_all_plan(snapshot):
        del snapshot
        return {"job-1": {"type": "defer"}}


class _Telemetry:
    def collect_telemetry(self, **kwargs):
        return {"request": kwargs, "rows": []}

    @staticmethod
    def iter_per_rank(bundle):
        del bundle
        return iter(())


class _SlowLoop:
    typical_ranges: ClassVar[dict[str, float]] = {}

    @staticmethod
    def get_sss_wt():
        return {}

    @staticmethod
    def get_sss_z_star_t():
        return {}

    @staticmethod
    def get_sss_cusum_params_v():
        return {}

    @staticmethod
    def get_sss_cusum_params_y():
        return {}

    @staticmethod
    def anneal_targets(tick):
        del tick
        return {}

    @staticmethod
    def slow_update_all(**kwargs):
        return {"input": kwargs, "B_t": 0}


class _Agent:
    @staticmethod
    def run_agent_loop(**kwargs):
        return Plan(kwargs["tick"], [PlanAction("job-1", ActionType.DEFER)])

    @staticmethod
    def receive_validator_feedback(violations):
        return list(violations)


class _Validator:
    @staticmethod
    def val_plan(**kwargs):
        del kwargs
        return ValidationResult(True)


class _Executor:
    @staticmethod
    def send_to_executor(plan):
        return [{"status": "accepted", "job_id": plan.actions[0].job_id}]


def test_tick_runner_emits_states_validation_executor_and_completion(tmp_path: Path) -> None:
    sink = ChronologicalEventLogger(tmp_path / "tick.jsonl", run_id="tick-test")
    graph = SimpleNamespace(x=["instance_type"], edge_table={})
    runner = TickRunner(
        evidence_store=SimpleNamespace(),
        telemetry=_Telemetry(),
        cusum=SimpleNamespace(),
        icp=SimpleNamespace(),
        quadrant_validator=SimpleNamespace(),
        confidence_service=SimpleNamespace(candidate_graph=graph),
        slow_loop=_SlowLoop(),
        dro=SimpleNamespace(target=0.9),
        mechanism_registry=SimpleNamespace(),
        resource_map=_ResourceMap(),
        agent=_Agent(),
        plan_validator=_Validator(),
        executor=_Executor(),
        candidate_graph=graph,
        tick_interval_sec=0,
        event_sink=sink,
    )

    context = runner.run_tick(3)
    sink.close()
    events = [json.loads(line) for line in sink.path.read_text().splitlines()]
    names = [event["event"] for event in events]

    assert context.error is None
    assert names[0] == "tick.started"
    for state in FSMState:
        if state.value.startswith("S") and state not in {FSMState.S7_EXIT_TICK}:
            started = next(
                index
                for index, event in enumerate(events)
                if event["event"] == "tick.state.started" and event["state"] == state.value
            )
            completed = next(
                index
                for index, event in enumerate(events)
                if event["event"] == "tick.state.completed" and event["state"] == state.value
            )
            assert started < completed
    assert "plan.validation.completed" in names
    assert "executor.call.completed" in names
    assert names[-1] == "tick.completed"
    assert events[-1]["state_durations_ms"]["S6_DEPLOY"] >= 0
