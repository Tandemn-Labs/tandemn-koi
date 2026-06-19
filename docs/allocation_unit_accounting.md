# Allocation-Unit Accounting

Koi now separates engine GPU demand from reserved capacity. This matters because
cloud providers allocate and bill whole instances, while future on-prem pools may
allocate discrete GPUs.

## Policy

- `gpu_count` / `tp * pp` remains engine GPU demand.
- Cloud pools reserve and charge one full instance per rank replica.
- On-prem or explicitly discrete pools can reserve per GPU.
- Koi does not pack multiple rank replicas into one cloud instance yet.

Example: a `g6e.12xlarge` rank with `gpu_count=1` still reserves all 4 L40S GPUs
and pays one `g6e.12xlarge` hourly rate per `n_replicas`.

## ResourceMapManager Changes

`src/infra/resource_map.py` now exposes allocation helpers:

- `resolve_allocation_unit(env, config)` finds the pool used by a rank.
- `rank_allocation_summary(rank)` returns engine demand, reserved capacity, and price metadata.
- `rank_capacity_per_replica(rank)` returns the reserved GPU footprint of one replica.
- `rank_capacity_footprint(rank)` returns `n_replicas * capacity_per_replica`.
- `switch_pricing_map()` builds an env/instance pricing table for switch cost.

When an env contains multiple pools for the same GPU type, Koi requires
`config["instance_type"]`; otherwise capacity and pricing are ambiguous.

## Switch Cost Changes

`src/cost/switch_cost.py` now prices `c_parallel` from the full `ChainEntry`.
That preserves both `env` and `config["instance_type"]`, so switch cost can use
resource-map instance prices instead of falling back to `$1/hour`.

Pricing resolution is:

1. explicit `config["hourly_rate"]`
2. `pricing_map[env]["by_instance_type"][instance_type]`
3. `pricing_map[env]["default"]`
4. flat `pricing_map[gpu_type]`
5. `$1/hour` fallback

## Agent Tool Changes

`src/agent/tools/agent_tools.py` now asks the resource map for
`rank_allocation_summary()` when sizing ladders. `size_ladder()` still uses
per-chain throughput from the surrogate, but capacity caps are based on reserved
capacity:

- cloud instance pool: `free_gpus // gpus_per_instance`
- discrete GPU pool: `free_gpus // engine_gpus`

Per-rank diagnostics now include allocation kind, instance type, capacity GPUs
per replica, and price per allocation unit. `compute_sigma()` and
`compute_switching_cost()` pass a resource-map-derived pricing map into switch
cost so canaries are charged with instance prices.

## Validator Changes

`src/validation/validator.py` now validates resource capacity against reserved
allocation footprint. If a rank selects a cloud instance pool, C5 counts
`n_replicas * gpus_per_instance`, not `n_replicas * gpu_count`.

C6 also verifies that a rank's engine GPU demand fits inside the selected
allocation unit. When an env has multiple pools, the rank must include an
`instance_type` so Koi can select the correct footprint and price.

## Prompt Changes

`src/agent/agent.py` now tells the root planner that cloud ranks should carry
`config["instance_type"]`, and that `gpu_count` is engine demand rather than
reserved capacity. This prevents the LLM from treating one GPU on a cloud
instance as if it only reserved one GPU.

## Current Boundary

This change is Koi-local. It does not modify Orca or Tandemn Store schemas.
Running-chain usage still reads `shape_json["count"]` from Tandemn Store, so the
launcher should write that value as the reserved GPU footprint for cloud chains
if it wants Koi's future free-capacity view to stay instance-atomic.

## Example

For a rank:

```python
env = ["reserved", "aws", "us-east-2", "use2-az3", "L40S"]
config = {"instance_type": "g6e.12xlarge", "gpu_type": "L40S", "gpu_count": 1}
n_replicas = 1
```

Koi now interprets this as:

- engine demand: 1 GPU
- reserved capacity: 4 L40S GPUs
- switch-cost rate: one `g6e.12xlarge` hourly price

No packing is attempted; unused GPUs inside the instance are deliberate waste
that the planner can see through capacity pressure and switch cost.
