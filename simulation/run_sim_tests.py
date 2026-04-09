#!/usr/bin/env python3
"""
simulation/run_sim_tests.py — Autonomous simulation test suite.

Tests all adaptive-replacement + spot-fallback + Beta-prior fixes.
Structured in three tiers:

  Tier 1 (direct)  — pure Python, no server needed            (always runs)
  Tier 2 (server)  — koi HTTP endpoints, no LLM               (always runs)
  Tier 3 (llm)     — full agent loop via mock_orca + koi       (requires ANTHROPIC_API_KEY)

Usage:
    cd /home/orange/Desktop/tandemn/koi
    python simulation/run_sim_tests.py
    ANTHROPIC_API_KEY=sk-ant-... python simulation/run_sim_tests.py
"""

import os
import sys
import json
import time
import signal
import tempfile
import textwrap
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

import requests

# ── make sure koi is importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KOI_PORT  = 18090   # offset from default to avoid clashing with a live instance
ORCA_PORT = 18336
KOI_URL   = f"http://localhost:{KOI_PORT}"
ORCA_URL  = f"http://localhost:{ORCA_PORT}"

# ── result tracking ───────────────────────────────────────────────────────────
_results: list[tuple[str, bool, Optional[str]]] = []
_current_section = ""

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def section(name: str):
    global _current_section
    _current_section = name
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{name}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


def check(name: str, fn):
    """Run one assertion function; record pass/fail."""
    try:
        fn()
        _results.append((_current_section, name, True, None))
        print(f"  {GREEN}✓{RESET} {name}")
    except AssertionError as e:
        msg = str(e) or "assertion failed"
        _results.append((_current_section, name, False, msg))
        print(f"  {RED}✗{RESET} {name}  →  {msg}")
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        _results.append((_current_section, name, False, tb))
        print(f"  {RED}✗{RESET} {name}  →  {tb}")


def skip(name: str, reason: str):
    _results.append((_current_section, name, None, reason))
    print(f"  {YELLOW}–{RESET} {name}  [skipped: {reason}]")


def wait_for(url: str, timeout: int = 15, label: str = "") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    print(f"    {RED}timeout waiting for {label or url}{RESET}")
    return False


@contextmanager
def koi_server(api_key: str = "dummy", orca_url: str = ""):
    """Start koi server as a subprocess; yield; kill on exit."""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": api_key,
        "ORCA_URL": orca_url,
        "KOI_PORT": str(KOI_PORT),
        "KOI_MEMORY_PATH": db.name,
        "KOI_PERFDB_PATH": "./perfdb/perfdb_all.csv",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "koi.server"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    if not wait_for(f"{KOI_URL}/health", timeout=12, label="koi"):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise RuntimeError("Koi server failed to start")
    try:
        yield db.name
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        try:
            os.unlink(db.name)
        except OSError:
            pass


@contextmanager
def mock_orca_server():
    """Start mock_orca as a subprocess with --no-decide; yield; kill on exit.
    Koi server must already be running (mock_orca sends /job/started on init)."""
    # Use a dummy koi-url so init_scenario's /decide + /job/started fail instantly
    # (we register replicas with Koi manually in the test)
    env = {**os.environ}
    proc = subprocess.Popen(
        [sys.executable, "simulation/mock_orca.py",
         "--port", str(ORCA_PORT),
         "--koi-url", "http://localhost:1",
         "--replicas", "2", "--tps", "1200",
         "--model", "Qwen/Qwen3-32B",
         "--no-decide"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    if not wait_for(f"{ORCA_URL}/resources", timeout=20, label="mock_orca"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.kill()
        raise RuntimeError("Mock Orca failed to start")
    # Wait for replicas to appear (init_scenario needs ~3s)
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            state = requests.get(f"{ORCA_URL}/sim/state", timeout=2).json()
            has_replicas = any(info.get("replicas_alive", 0) > 0 for info in state.values())
            if has_replicas:
                break
        except Exception:
            pass
        time.sleep(1)
    try:
        yield proc
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def post(url, data):
    return requests.post(url, json=data, timeout=10).json()


def fresh_memory():
    from koi.tools.memory import AgenticMemory
    return AgenticMemory(":memory:")


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Direct Python (no server)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier1():
    from koi.server import _classify_failure
    from koi.tools.memory import AgenticMemory
    from koi.schemas import (
        MonitoringStatus, MonitoringTrigger, PlacementConfig, EngineConfig,
        JobRequest, TaskType, Objective, ResourceMap, GPUResource,
    )
    from koi.agent import KoiAgent
    from koi.tools.perfdb import PerfDB

    # ── T1.1: failure classifier ──────────────────────────────────────────────
    section("T1.1  Failure reason classification")

    check("SpotInstanceInterruption → spot_preemption", lambda:
        assert_eq(_classify_failure("SpotInstanceInterruption"), "spot_preemption"))
    check("spot preempted → spot_preemption", lambda:
        assert_eq(_classify_failure("instance was spot preempted"), "spot_preemption"))
    check("InsufficientCapacity → no_capacity", lambda:
        assert_eq(_classify_failure("InsufficientCapacity in us-east-1"), "no_capacity"))
    check("no capacity → no_capacity", lambda:
        assert_eq(_classify_failure("no capacity available"), "no_capacity"))
    check("CUDA OOM → oom", lambda:
        assert_eq(_classify_failure("CUDA OOM: tried to allocate 40GB"), "oom"))
    check("OutOfMemoryError → oom", lambda:
        assert_eq(_classify_failure("OutOfMemoryError"), "oom"))
    check("QuotaExceeded → quota", lambda:
        assert_eq(_classify_failure("QuotaExceeded for p5.48xlarge"), "quota"))
    check("unknown → unknown", lambda:
        assert_eq(_classify_failure("something totally different"), "unknown"))
    check("empty string → unknown", lambda:
        assert_eq(_classify_failure(""), "unknown"))

    # ── T1.2: Beta-prior math ─────────────────────────────────────────────────
    section("T1.2  Beta-prior: update_availability + get_failure_summary")

    def t_uninformative():
        m = fresh_memory()
        s = m.get_failure_summary("H100")
        assert abs(s["availability_pct"] - 50.0) < 0.1, f"expected 50%, got {s['availability_pct']}"
        assert s["effective_observations"] == 0

    def t_single_success():
        m = fresh_memory()
        m.update_availability("L40S", "us-east-1", "spot", launched=True)
        s = m.get_failure_summary("L40S", region="us-east-1", market="spot")
        assert s["availability_pct"] > 50.0, f"expected >50% after success, got {s['availability_pct']}"
        assert s["effective_observations"] == 1

    def t_single_failure():
        m = fresh_memory()
        m.update_availability("L40S", "us-east-1", "spot", launched=False)
        s = m.get_failure_summary("L40S", region="us-east-1", market="spot")
        assert s["availability_pct"] < 50.0, f"expected <50% after failure, got {s['availability_pct']}"

    def t_two_preemptions():
        m = fresh_memory()
        m.update_availability("L40S", "us-east-1", "spot", launched=False)
        m.update_availability("L40S", "us-east-1", "spot", launched=False)
        s = m.get_failure_summary("L40S", region="us-east-1", market="spot")
        # Beta(1,3) → mean = 1/4 = 25%
        assert s["availability_pct"] < 35.0, f"expected <35% after 2 failures, got {s['availability_pct']}"
        assert s["effective_observations"] == 2

    def t_uncertainty_shrinks():
        m = fresh_memory()
        s_before = m.get_failure_summary("L40S")
        for _ in range(10):
            m.update_availability("L40S", "us-east-1", "spot", launched=True)
        s_after = m.get_failure_summary("L40S")
        assert s_after["uncertainty_pct"] < s_before["uncertainty_pct"], \
            "uncertainty should shrink with more observations"

    def t_separate_market_priors():
        m = fresh_memory()
        for _ in range(5):
            m.update_availability("L40S", "us-east-1", "spot", launched=False)
        m.update_availability("L40S", "us-east-1", "on_demand", launched=True)
        s_spot = m.get_failure_summary("L40S", region="us-east-1", market="spot")
        s_od   = m.get_failure_summary("L40S", region="us-east-1", market="on_demand")
        assert s_spot["availability_pct"] < 30.0, "spot should be low"
        assert s_od["availability_pct"] > 50.0,   "on_demand should be high"
        assert s_spot["availability_pct"] < s_od["availability_pct"], "spot < on_demand"

    check("uninformative prior = 50%", t_uninformative)
    check("single success raises availability above 50%", t_single_success)
    check("single failure drops availability below 50%", t_single_failure)
    check("two failures → <35% availability", t_two_preemptions)
    check("uncertainty shrinks with more observations", t_uncertainty_shrinks)
    check("spot and on_demand have separate priors", t_separate_market_priors)

    # ── T1.3: spot_preemptions_6h counter ────────────────────────────────────
    section("T1.3  Recent failure counts in summary")

    def t_spot_preemption_count():
        m = fresh_memory()
        dec_id = m.record_decision(
            job_id="job-p1", model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            tp=4, pp=1, dp=1, num_gpus=4, predicted_tps=1000,
            predicted_cost_per_hour=13.35, slo_deadline_hours=8.0,
            objective="cheapest", avg_input_tokens=512, avg_output_tokens=256,
        )
        m.record_outcome(
            decision_id=dec_id, job_id="job-p1",
            status="replica_failed",
            failure_category="spot_preemption",
            diagnosis="SpotInstanceInterruption",
        )
        s = m.get_failure_summary("L40S")
        assert s["spot_preemptions_6h"] == 1, f"expected 1, got {s['spot_preemptions_6h']}"

    def t_no_capacity_count():
        m = fresh_memory()
        m.record_launch_attempt(
            decision_id="dec-1", job_id="job-1",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            region="us-east-1", market="spot", count=1,
            launched=False,
            failure_reason="InsufficientCapacity",
            failure_category="no_capacity",
        )
        s = m.get_failure_summary("L40S")
        assert s["no_capacity_6h"] == 1, f"expected 1, got {s['no_capacity_6h']}"

    def t_last_failure_at_set():
        m = fresh_memory()
        dec_id = m.record_decision(
            job_id="job-x", model_name="test", instance_type="g6e",
            gpu_type="L40S", tp=4, pp=1, dp=1, num_gpus=4, predicted_tps=1000,
            predicted_cost_per_hour=13.35, slo_deadline_hours=8.0,
            objective="cheapest", avg_input_tokens=512, avg_output_tokens=256,
        )
        m.record_outcome(decision_id=dec_id, job_id="job-x",
                         status="replica_failed", failure_category="oom")
        s = m.get_failure_summary("L40S")
        assert s["last_failure_at"] is not None, "last_failure_at should be set"

    check("spot_preemptions_6h increments from outcomes", t_spot_preemption_count)
    check("no_capacity_6h increments from launch_attempts", t_no_capacity_count)
    check("last_failure_at is populated", t_last_failure_at_set)

    # ── T1.4: cost table availability column ─────────────────────────────────
    section("T1.4  Cost table shows availability column")

    def t_cost_table_has_avail_column():
        try:
            perfdb = PerfDB("./perfdb/perfdb_all.csv")
        except Exception:
            perfdb = None

        m = fresh_memory()
        # seed two failures so L40S shows a degraded availability
        m.update_availability("L40S", "us-east-1", "spot", launched=False)
        m.update_availability("L40S", "us-east-1", "spot", launched=False)

        agent = KoiAgent(perfdb=perfdb, memory=m, orca=None, api_key="dummy")

        rm = ResourceMap(vpc_id="test", region="us-east-1", resources=[
            GPUResource(gpu_type="L40S", instance_type="g6e.12xlarge",
                        gpus_per_instance=4, total_gpus=16,
                        gpu_memory_gb=48, cost_per_instance_hour_usd=13.35,
                        region="us-east-1", interconnect="PCIe"),
        ])
        req = JobRequest(
            model_name="Qwen/Qwen3-32B", task_type=TaskType.BATCH,
            avg_input_tokens=512, avg_output_tokens=256,
            num_requests=5000, slo_deadline_hours=2.0,
            objective=Objective.CHEAPEST,
        )
        table = agent._build_cost_table(req, rm)
        assert "Avail" in table, f"'Avail' column missing from cost table:\n{table}"

    def t_cost_table_avail_degraded_after_failures():
        try:
            perfdb = PerfDB("./perfdb/perfdb_all.csv")
        except Exception:
            perfdb = None

        m = fresh_memory()
        for _ in range(4):
            m.update_availability("L40S", "us-east-1", "spot", launched=False)

        agent = KoiAgent(perfdb=perfdb, memory=m, orca=None, api_key="dummy")
        rm = ResourceMap(vpc_id="test", region="us-east-1", resources=[
            GPUResource(gpu_type="L40S", instance_type="g6e.12xlarge",
                        gpus_per_instance=4, total_gpus=16,
                        gpu_memory_gb=48, cost_per_instance_hour_usd=13.35,
                        region="us-east-1", interconnect="PCIe"),
        ])
        req = JobRequest(
            model_name="Qwen/Qwen3-32B", task_type=TaskType.BATCH,
            avg_input_tokens=512, avg_output_tokens=256,
            num_requests=5000, slo_deadline_hours=2.0,
            objective=Objective.CHEAPEST,
        )
        table = agent._build_cost_table(req, rm)
        # Find the avail % for L40S — should be below 30% after 4 failures
        # The format is "XX%±YY%" in the last column
        import re
        avail_matches = re.findall(r"(\d+)%±\d+%", table)
        assert avail_matches, f"No avail% found in table:\n{table}"
        avail_pct = float(avail_matches[0])
        assert avail_pct < 30.0, f"Expected <30% availability after 4 failures, got {avail_pct}%"

    check("cost table contains 'Avail' column header", t_cost_table_has_avail_column)
    check("cost table shows degraded availability after failures", t_cost_table_avail_degraded_after_failures)

    # ── T1.5: FAILED trigger prompt structure ─────────────────────────────────
    section("T1.5  FAILED trigger prompt has FAILURE CONTEXT block")

    def make_failed_trigger(gpu_type="L40S"):
        config = PlacementConfig(
            gpu_type=gpu_type, instance_type="g6e.12xlarge",
            num_gpus=4, num_instances=1, tp=4, pp=1, dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=1),
        )
        return MonitoringTrigger(
            trigger_type=MonitoringStatus.FAILED,
            job_id="chain-r0",
            job_tracker={
                "job_id": "chain-r0", "group_id": "job-abc",
                "config": config.model_dump(),
                "slo_deadline_hours": 2.0, "total_tokens": 6_000_000,
                "predicted_tps": 1200, "smoothed_tps": 0,
                "slo_headroom_pct": 45.0, "elapsed_hours": 0.3,
                "tokens_remaining": 5_000_000,
                "gpu_cache_usage": 0.0, "gpu_sm_util": 0.0, "gpu_mem_bw_util": 0.0,
                "status": "failed",
            },
            diagnosis_hint="Replica died: SpotInstanceInterruption",
        )

    def t_failed_prompt_has_context_block():
        m = fresh_memory()
        m.update_availability("L40S", "us-east-1", "spot", launched=False)
        m.update_availability("L40S", "us-east-1", "spot", launched=False)
        agent = KoiAgent(perfdb=None, memory=m, orca=None, api_key="dummy")
        prompt = agent._build_trigger_prompt(make_failed_trigger())
        assert "FAILURE CONTEXT" in prompt, "FAILURE CONTEXT block missing from FAILED prompt"

    def t_failed_prompt_shows_preemption_count():
        m = fresh_memory()
        dec_id = m.record_decision(
            job_id="j1", model_name="test", instance_type="g6e",
            gpu_type="L40S", tp=4, pp=1, dp=1, num_gpus=4, predicted_tps=1000,
            predicted_cost_per_hour=13.35, slo_deadline_hours=8.0,
            objective="cheapest", avg_input_tokens=512, avg_output_tokens=256,
        )
        m.record_outcome(decision_id=dec_id, job_id="j1",
                         status="replica_failed", failure_category="spot_preemption")
        m.record_outcome(decision_id=dec_id, job_id="j1",
                         status="replica_failed", failure_category="spot_preemption")
        agent = KoiAgent(perfdb=None, memory=m, orca=None, api_key="dummy")
        prompt = agent._build_trigger_prompt(make_failed_trigger())
        assert "Spot preemptions" in prompt, "Spot preemption count missing from prompt"
        assert "2" in prompt, "Preemption count '2' missing from prompt"

    def t_failed_prompt_instructs_failure_summary_tool():
        m = fresh_memory()
        agent = KoiAgent(perfdb=None, memory=m, orca=None, api_key="dummy")
        prompt = agent._build_trigger_prompt(make_failed_trigger())
        assert "get_failure_summary_tool" in prompt, \
            "Prompt doesn't mention get_failure_summary_tool"

    def t_failed_prompt_instructs_on_demand():
        m = fresh_memory()
        agent = KoiAgent(perfdb=None, memory=m, orca=None, api_key="dummy")
        prompt = agent._build_trigger_prompt(make_failed_trigger())
        assert "on_demand=True" in prompt, "Prompt doesn't mention on_demand=True fallback"

    check("FAILED prompt has FAILURE CONTEXT block", t_failed_prompt_has_context_block)
    check("FAILED prompt shows spot preemption count from memory", t_failed_prompt_shows_preemption_count)
    check("FAILED prompt mentions get_failure_summary_tool", t_failed_prompt_instructs_failure_summary_tool)
    check("FAILED prompt mentions on_demand=True for preemptions", t_failed_prompt_instructs_on_demand)

    # ── T1.6: Tool wiring ─────────────────────────────────────────────────────
    section("T1.6  Agent tool wiring")

    def t_failure_summary_tool_present():
        m = fresh_memory()
        agent = KoiAgent(perfdb=None, memory=m, orca=None, api_key="dummy")
        tools = agent._build_tools()
        names = [t.function.name if hasattr(t, "function") else getattr(t, "name", str(t))
                 for t in tools]
        # Also try .__name__ for beta_async_tool wrappers
        names2 = []
        for t in tools:
            try:
                names2.append(t.__name__)
            except AttributeError:
                try:
                    names2.append(t.function.name)
                except AttributeError:
                    names2.append(repr(t))
        all_names = names + names2
        assert any("failure_summary" in n for n in all_names), \
            f"get_failure_summary_tool not in tools: {all_names}"

    def t_scale_chain_tool_has_on_demand_param():
        """scale_chain_tool must accept on_demand parameter."""
        import inspect
        from unittest.mock import MagicMock
        m = fresh_memory()
        mock_orca = MagicMock()
        agent = KoiAgent(perfdb=None, memory=m, orca=mock_orca, api_key="dummy")
        # The tool is defined as a local function inside _build_tools
        # We can check the agent source for the on_demand param
        import ast, textwrap
        src = inspect.getsource(agent._build_tools)
        assert "on_demand" in src, "on_demand param missing from scale_chain_tool source"
        assert "on_demand is not None" in src, "on_demand override logic missing"

    check("get_failure_summary_tool is in agent tool list", t_failure_summary_tool_present)
    check("scale_chain_tool source has on_demand parameter + override logic", t_scale_chain_tool_has_on_demand_param)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Koi HTTP server (no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier2():
    section("T2.1  /job/config-attempted endpoint")

    try:
        with koi_server() as db_path:
            from koi.tools.memory import AgenticMemory

            def t_config_attempted_failure():
                r = post(f"{KOI_URL}/job/config-attempted", {
                    "job_id": "job-ca-1",
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "L40S",
                    "region": "us-east-1",
                    "market": "spot",
                    "launched": False,
                    "failure_reason": "SpotInstanceInterruption",
                })
                assert r.get("launched") is False or r.get("status") == "recorded", \
                    f"unexpected response: {r}"
                # check memory
                m = AgenticMemory(db_path)
                s = m.get_failure_summary("L40S", region="us-east-1", market="spot")
                assert s["availability_pct"] < 50.0, \
                    f"availability should drop after failure, got {s['availability_pct']}"

            def t_config_attempted_success():
                r = post(f"{KOI_URL}/job/config-attempted", {
                    "job_id": "job-ca-2",
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "A100-80GB",
                    "region": "us-west-2",
                    "market": "on_demand",
                    "launched": True,
                    "time_to_launch": 34.2,
                })
                assert r.get("status") == "recorded", f"unexpected response: {r}"
                m = AgenticMemory(db_path)
                s = m.get_failure_summary("A100-80GB", region="us-west-2", market="on_demand")
                assert s["availability_pct"] > 50.0, \
                    f"availability should rise after success, got {s['availability_pct']}"

            def t_spot_and_od_tracked_separately():
                # fail spot, succeed on_demand — should be separate priors
                post(f"{KOI_URL}/job/config-attempted", {
                    "job_id": "job-ca-3", "instance_type": "g6e.12xlarge",
                    "gpu_type": "H100", "region": "us-east-1",
                    "market": "spot", "launched": False,
                    "failure_reason": "InsufficientCapacity",
                })
                post(f"{KOI_URL}/job/config-attempted", {
                    "job_id": "job-ca-3", "instance_type": "p5.48xlarge",
                    "gpu_type": "H100", "region": "us-east-1",
                    "market": "on_demand", "launched": True,
                })
                m = AgenticMemory(db_path)
                s_spot = m.get_failure_summary("H100", region="us-east-1", market="spot")
                s_od   = m.get_failure_summary("H100", region="us-east-1", market="on_demand")
                assert s_spot["availability_pct"] < s_od["availability_pct"], \
                    f"spot({s_spot['availability_pct']}) should be < on_demand({s_od['availability_pct']})"

            check("POST /job/config-attempted failure lowers availability", t_config_attempted_failure)
            check("POST /job/config-attempted success raises availability", t_config_attempted_success)
            check("spot and on_demand tracked separately via config-attempted", t_spot_and_od_tracked_separately)

    except RuntimeError as e:
        for name in ["failure lowers availability", "success raises availability",
                     "spot and on_demand tracked separately via config-attempted"]:
            skip(name, str(e))

    section("T2.2  /job/launch-failed stores failure_category")

    try:
        with koi_server() as db_path:
            from koi.tools.memory import AgenticMemory

            def t_launch_failed_stores_category():
                r = post(f"{KOI_URL}/job/launch-failed", {
                    "job_id": "job-lf-1",
                    "configs_tried": [
                        {"instance_type": "g6e.12xlarge", "gpu_type": "L40S",
                         "region": "us-east-1", "market": "spot"},
                        {"instance_type": "p5.48xlarge", "gpu_type": "H100",
                         "region": "us-east-1", "market": "on_demand"},
                    ],
                    "failure_reasons": [
                        "SpotInstanceInterruption",
                        "InsufficientCapacity",
                    ],
                    "total_time_seconds": 120.0,
                })
                assert r.get("attempts_recorded") == 2, f"expected 2, got {r}"
                m = AgenticMemory(db_path)
                conn = m._conn()
                rows = conn.execute(
                    "SELECT gpu_type, failure_reason, failure_category FROM launch_attempts"
                ).fetchall()
                cats = {r["gpu_type"]: r["failure_category"] for r in rows}
                assert cats.get("L40S") == "spot_preemption", \
                    f"L40S should be spot_preemption, got {cats.get('L40S')}"
                assert cats.get("H100") == "no_capacity", \
                    f"H100 should be no_capacity, got {cats.get('H100')}"

            def t_launch_failed_updates_beta_priors():
                m = AgenticMemory(db_path)
                s_l40s = m.get_failure_summary("L40S", market="spot")
                s_h100 = m.get_failure_summary("H100", market="on_demand")
                assert s_l40s["availability_pct"] < 50.0, "L40S spot should be < 50%"
                assert s_h100["availability_pct"] < 50.0, "H100 on_demand should be < 50%"

            check("/job/launch-failed stores correct failure_category per config", t_launch_failed_stores_category)
            check("/job/launch-failed updates Beta priors for all configs", t_launch_failed_updates_beta_priors)

    except RuntimeError as e:
        for name in ["stores correct failure_category", "updates Beta priors"]:
            skip(name, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Full agent loop (requires ANTHROPIC_API_KEY)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier3(api_key: str):
    section("T3.1  Agent calls get_failure_summary_tool on FAILED trigger")

    try:
        # Koi must start first so mock_orca can register replicas via /job/started
        with koi_server(api_key=api_key, orca_url=ORCA_URL) as db_path:
          with mock_orca_server():
            from koi.tools.memory import AgenticMemory

            # Seed 2 spot preemptions in memory before the trigger
            m = AgenticMemory(db_path)
            m.update_availability("L40S", "unknown", "spot", launched=False)
            m.update_availability("L40S", "unknown", "spot", launched=False)

            # Get running replica IDs from mock_orca
            state = requests.get(f"{ORCA_URL}/sim/state", timeout=5).json()
            replica_ids = []
            group_id = None
            for job_id, job_info in state.items():
                group_id = job_id
                for rid, rinfo in job_info.get("replicas", {}).items():
                    if rinfo.get("phase") == "running":
                        replica_ids.append(rid)

            if not replica_ids:
                skip("agent responded to FAILED trigger (scale_up decision in memory)", "no replicas running in mock_orca")
                skip("scale_up decision recorded in memory", "no replicas running")
                skip("agent uses market=on_demand after 2 spot preemptions", "no replicas running")
                return

            print(f"    [{len(replica_ids)} replicas running, group={group_id}]")

            # Register replicas with Koi (mock_orca used dummy koi-url)
            for rid in replica_ids:
                post(f"{KOI_URL}/job/started", {
                    "job_id": rid, "group_id": group_id,
                    "gpu_type": "L40S", "instance_type": "g6e.12xlarge",
                    "tp": 4, "pp": 1, "dp": 1,
                    "slo_deadline_hours": 2.0, "total_tokens": 6_000_000,
                    "predicted_tps": 1200.0,
                })
            time.sleep(2)  # let monitor pick them up

            # Fire preemption: kill in mock_orca AND send webhook directly to Koi
            target = replica_ids[0]
            print(f"    preempting {target}...")
            post(f"{ORCA_URL}/sim/kill-replica/{target}", {"reason": "SpotInstanceInterruption"})
            post(f"{KOI_URL}/job/replica-failed", {
                "job_id": target, "group_id": group_id,
                "status": "failed", "reason": "SpotInstanceInterruption",
            })

            # Give agent up to 120s to respond
            deadline = time.time() + 120
            trigger_responded = False
            while time.time() < deadline:
                time.sleep(5)
                m2 = AgenticMemory(db_path)
                decs = m2.query_decisions(limit=10)
                scale_decs = [d for d in decs if d.get("triggered_by") == "scale_up"]
                if scale_decs:
                    trigger_responded = True
                    break
                elapsed = int(time.time() - (deadline - 120))
                print(f"    waiting for agent response... ({elapsed}s)")

            def t_agent_responded_to_failed_trigger():
                assert trigger_responded, "Agent did not produce a scale_up decision within 120s"

            def t_agent_scale_decision_exists():
                m2 = AgenticMemory(db_path)
                decs = m2.query_decisions(limit=10)
                scale_decs = [d for d in decs if d.get("triggered_by") == "scale_up"]
                assert scale_decs, "No scale_up decision found in memory"

            check("agent responded to FAILED trigger (scale_up decision in memory)", t_agent_responded_to_failed_trigger)
            check("scale_up decision recorded in memory", t_agent_scale_decision_exists)

            # Check if the agent used on_demand (it should — we seeded 2 preemptions)
            section("T3.2  Agent uses on_demand=True after 2 spot preemptions")

            m3 = AgenticMemory(db_path)
            decs = m3.query_decisions(limit=10)
            scale_decs = [d for d in decs if d.get("triggered_by") == "scale_up"]
            on_demand_used = any(d.get("market") == "on_demand" for d in scale_decs)

            def t_on_demand_used():
                assert on_demand_used, \
                    f"Agent did not use market='on_demand'. Scale decisions: {[{k: d.get(k) for k in ('triggered_by','market','gpu_type')} for d in scale_decs]}"

            check("agent uses market=on_demand after 2 spot preemptions", t_on_demand_used)

    except RuntimeError as e:
        for name in ["agent responded to FAILED trigger (scale_up decision in memory)",
                     "scale_up decision recorded in memory",
                     "agent uses market=on_demand after 2 spot preemptions"]:
            skip(name, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def assert_eq(a, b):
    assert a == b, f"{a!r} != {b!r}"


def print_report():
    passed  = [r for r in _results if r[2] is True]
    failed  = [r for r in _results if r[2] is False]
    skipped = [r for r in _results if r[2] is None]

    print(f"\n{'═'*60}")
    print(f"{BOLD}RESULTS{RESET}")
    print(f"{'═'*60}")

    if failed:
        print(f"\n{RED}{BOLD}FAILED ({len(failed)}){RESET}")
        for sec, name, _, msg in failed:
            print(f"  {RED}✗{RESET} [{sec}] {name}")
            if msg:
                print(f"      {msg}")

    if skipped:
        print(f"\n{YELLOW}SKIPPED ({len(skipped)}){RESET}")
        for sec, name, _, msg in skipped:
            print(f"  {YELLOW}–{RESET} [{sec}] {name}  ({msg})")

    print(f"\n{GREEN}PASSED:  {len(passed)}{RESET}")
    print(f"{RED}FAILED:  {len(failed)}{RESET}")
    print(f"{YELLOW}SKIPPED: {len(skipped)}{RESET}")
    print(f"TOTAL:   {len(_results)}")
    print(f"{'═'*60}\n")

    return len(failed) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    has_llm = bool(api_key and not api_key.startswith("dummy"))

    print(f"{BOLD}Koi Simulation Test Suite{RESET}")
    print(f"LLM tests: {'enabled' if has_llm else 'disabled (set ANTHROPIC_API_KEY to enable)'}")

    run_tier1()
    run_tier2()

    if has_llm:
        run_tier3(api_key)
    else:
        section("T3  LLM agent tests  [skipped]")
        for name in [
            "agent calls get_failure_summary_tool on FAILED trigger",
            "scale_up decision recorded in memory",
            "agent uses market=on_demand after 2 spot preemptions",
        ]:
            skip(name, "no ANTHROPIC_API_KEY")

    ok = print_report()
    sys.exit(0 if ok else 1)
