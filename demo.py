"""
demo.py — End-to-end Koi placement demo using results.json as the perf DB.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo.py

Demonstrates two scenarios:
  1. Batch inference: Qwen-72B on a 100k-row dataset, 8-hour deadline, cheapest
  2. Online serving: Qwen-72B serving endpoint, 50 concurrent users, 35ms TPOT SLO
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from koi import KoiPlacement
from koi.schemas import GPUResource, JobRequest, ResourceMap, TaskType

# ---------------------------------------------------------------------------
# Fixture: VPC resource map (L40S cluster, as per our results.json setup)
# ---------------------------------------------------------------------------

DEMO_RESOURCE_MAP = ResourceMap(
    vpc_id="vpc-demo-tandem-01",
    region="us-east-1",
    resources=[
        GPUResource(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            gpus_per_instance=4,
            total_gpus=16,
            allocated_gpus=0,
            cost_per_instance_hour_usd=4.68,
            gpu_memory_gb=45.5,
            region="us-east-1",
            interconnect="PCIe",
        ),
        GPUResource(
            gpu_type="A10G",
            instance_type="g5.12xlarge",
            gpus_per_instance=4,
            total_gpus=8,
            allocated_gpus=0,
            cost_per_instance_hour_usd=5.67,
            gpu_memory_gb=23.0,
            region="us-east-1",
            interconnect="PCIe",
        ),
    ],
)


def run_batch_demo(koi: KoiPlacement) -> None:
    print("\n" + "="*70)
    print("  DEMO 1: Batch Inference — Qwen-72B on 100k rows, 8hr deadline")
    print("="*70)

    request = JobRequest(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        task_type=TaskType.BATCH,
        avg_input_tokens=512,
        avg_output_tokens=256,
        num_requests=100_000,
        slo_deadline_hours=8.0,
        objective="cheapest",
    )

    print(f"\nJob: {request.model_name}")
    print(f"  {request.num_requests:,} requests × ({request.avg_input_tokens}in + {request.avg_output_tokens}out)")
    print(f"  Total tokens: {request.total_tokens:,}")
    print(f"  Deadline: {request.slo_deadline_hours}h | Objective: {request.objective.value}")

    decision = koi.decide(request, DEMO_RESOURCE_MAP)
    print(decision.display_summary())

    print("\nThinker breakdown:")
    for p in decision.thinker_proposals:
        print(f"  {p.thinker_id} (directive: {p.directive})")
        print(f"    → {p.proposed_config.gpu_type} TP={p.proposed_config.tp} PP={p.proposed_config.pp} DP={p.proposed_config.dp} | conf={p.confidence:.0%}")
        print(f"    Hypothesis: {p.hypothesis[:120]}...")
        print(f"    Reasoning:  {p.reasoning[:120]}...")


def run_online_demo(koi: KoiPlacement) -> None:
    print("\n" + "="*70)
    print("  DEMO 2: Online Serving — Qwen-72B, 50 users, 35ms TPOT SLO")
    print("="*70)

    request = JobRequest(
        model_name="Qwen/Qwen2.5-72B-Instruct",
        task_type=TaskType.ONLINE,
        avg_input_tokens=1024,
        avg_output_tokens=512,
        expected_concurrency=50,
        slo_tpot_ms=35.0,
        slo_ttft_ms=500.0,
        objective="balanced",
    )

    print(f"\nJob: {request.model_name}")
    print(f"  {request.expected_concurrency} concurrent users | {request.avg_input_tokens}in / {request.avg_output_tokens}out")
    print(f"  TPOT SLO: {request.slo_tpot_ms}ms | TTFT SLO: {request.slo_ttft_ms}ms")
    print(f"  Objective: {request.objective.value}")

    decision = koi.decide(request, DEMO_RESOURCE_MAP)
    print(decision.display_summary())


def run_deepseek_batch_demo(koi: KoiPlacement) -> None:
    print("\n" + "="*70)
    print("  DEMO 3: Batch — DeepSeek-R1-Distill-70B, 500k rows, 24hr deadline")
    print("="*70)

    request = JobRequest(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        task_type=TaskType.BATCH,
        avg_input_tokens=800,
        avg_output_tokens=2000,
        num_requests=500_000,
        slo_deadline_hours=24.0,
        objective="cheapest",
    )

    print(f"\nJob: {request.model_name}")
    print(f"  {request.num_requests:,} requests × ({request.avg_input_tokens}in + {request.avg_output_tokens}out)")
    print(f"  Total tokens: {request.total_tokens:,}")
    print(f"  Deadline: {request.slo_deadline_hours}h")

    decision = koi.decide(request, DEMO_RESOURCE_MAP)
    print(decision.display_summary())


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Initialize Koi — will load perfdb from ./perfdb and fall back to ./results.json
    koi = KoiPlacement(
        api_key=api_key,
        perfdb_path="./perfdb",
        llm_model="claude-opus-4-6",
    )

    # Run demos
    run_batch_demo(koi)

    # Uncomment to run more:
    # run_online_demo(koi)
    # run_deepseek_batch_demo(koi)
