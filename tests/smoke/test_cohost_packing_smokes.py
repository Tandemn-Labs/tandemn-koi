"""KOI_COHOST_PACKING: several replicas may share one cloud instance.

Flag off must be byte-identical to instance-atomic accounting; flag on packs
replica footprints onto instances with first-fit-decreasing, prices a replica
by the GPUs it occupies, and keeps a full-width replica on an EMPTY instance.
The plan / ladder schema handed to Orca does not change.
"""

import os
import unittest
from unittest.mock import patch

from src.agent.tools import agent_tools
from src.core.models import RankSpec
from src.cost.switch_cost import hourly_rate
from src.infra.packing import (
    COHOST_PACKING_ENV,
    apply_footprints_to_pool,
    effective_allocation_kind,
    pack_footprints,
    packable_replicas,
    packing_shortfall,
    partial_free_slots,
)
from src.infra.resource_map import ResourceMapManager
from src.validation.validator import Validator

ENV = "reserved|aws|us-east-2|use2-az3|A100"
POOL = "p4d.24xlarge"


def _flag(on: bool):
    return patch.dict(os.environ, {COHOST_PACKING_ENV: "1" if on else ""})


class FakeResourceMap:
    market = ("reserved",)

    def __init__(self, total_instances=2, price=32.0):
        self.total_instances = total_instances
        self.price = price

    def scheduling_summary(self):
        return {
            ENV: {
                "market": "reserved",
                "cloud": "aws",
                "region": "us-east-2",
                "zone": "use2-az3",
                "gpu_type": "A100",
                "total": 8 * self.total_instances,
                "pools": [
                    {
                        "instance_type": POOL,
                        "gpu_type": "A100",
                        "gpus_per_instance": 8,
                        "total_instances": self.total_instances,
                        "price_per_instance_hour": self.price,
                    }
                ],
            }
        }


class Manager(ResourceMapManager):
    def __init__(self, chain_counts, total_instances=2):
        super().__init__(user_id="cohost_packing_smoke")
        self.chain_counts = list(chain_counts)
        self.total_instances = total_instances

    def get_resource_map(self, user_id=None):
        return FakeResourceMap(total_instances=self.total_instances)

    def get_running_chains(self, user_id=None):
        return [
            {
                "chain_id": f"chain_{index}",
                "target_node": ENV,
                "shape_json": {"count": count, "gpu_count": count, "instance_type": POOL},
            }
            for index, count in enumerate(self.chain_counts)
        ]

    def get_running_jobs(self, user_id=None):
        return []

    def get_waiting_jobs(self, user_id=None):
        return []


def _plan(gpu_count: int, n_replicas: int, job_id: str = "job_new") -> dict:
    return {
        "actions": [
            {
                "job_id": job_id,
                "type": "place",
                "ladder": [
                    {
                        "role": "aggregate",
                        "env": ENV.split("|"),
                        "config": {
                            "instance_type": POOL,
                            "gpu_count": gpu_count,
                            "tp": gpu_count,
                            "pp": 1,
                        },
                        "n_replicas": n_replicas,
                    }
                ],
            }
        ]
    }


class PackingHelperTests(unittest.TestCase):
    def test_first_fit_decreasing(self):
        self.assertEqual(pack_footprints([1, 1, 8, 4, 4, 2], 8), [8, 8, 4])
        self.assertEqual(pack_footprints([], 8), [])
        self.assertEqual(pack_footprints([3], 8, bins=[6, 2]), [6, 5])
        self.assertEqual(partial_free_slots([8, 6, 2], 8), [6, 2])

    def test_shortfall_needs_an_empty_box_for_full_width(self):
        # Two half-used boxes hold 8 free slots in total, but no tp=8 replica.
        self.assertEqual(packing_shortfall([8], 8, partial_slots=[4, 4], empty_instances=0), 1)
        self.assertEqual(packing_shortfall([8], 8, partial_slots=[4, 4], empty_instances=1), 0)
        self.assertEqual(packing_shortfall([4, 4], 8, partial_slots=[4, 4], empty_instances=0), 0)

    def test_packable_replicas(self):
        self.assertEqual(packable_replicas(2, 8, partial_slots=[6, 3], empty_instances=1), 8)
        self.assertEqual(packable_replicas(8, 8, partial_slots=[6, 3], empty_instances=1), 1)
        self.assertEqual(packable_replicas(16, 8, partial_slots=[], empty_instances=3), 0)
        self.assertEqual(apply_footprints_to_pool([8, 2], 8, [4], 2), ([2], 1))

    def test_flag_controls_effective_kind(self):
        with _flag(False):
            self.assertEqual(effective_allocation_kind(None), "instance")
            self.assertEqual(effective_allocation_kind("gpu"), "gpu")
        with _flag(True):
            self.assertEqual(effective_allocation_kind(None), "packed")
            self.assertEqual(effective_allocation_kind("instance"), "packed")
            self.assertEqual(effective_allocation_kind("gpu"), "gpu")


class ResourceMapPackingTests(unittest.TestCase):
    def test_flag_off_keeps_instance_atomic_accounting(self):
        with _flag(False):
            manager = Manager(chain_counts=[1, 1])
            resources = manager.resources_summary()
            pool = resources[ENV]["pools"][0]
            self.assertNotIn("allocation_kind", pool)
            self.assertNotIn("partial_free_slots", pool)
            self.assertEqual(pool["free_instances"], 0)
            self.assertEqual(resources[ENV]["free"], 0)
            ok, violations = manager.check_resource_feasibility(_plan(1, 1))
            self.assertFalse(ok)
            self.assertIn("requested 1 instances, only 0 free", violations[0])
            summary = manager.rank_allocation_summary(
                RankSpec.from_dict(_plan(1, 1)["actions"][0]["ladder"][0]), resources
            )
            self.assertEqual(summary["allocation_kind"], "instance")
            self.assertEqual(summary["capacity_per_replica"], 8)
            self.assertEqual(summary["price_per_unit_hour"], 32.0)
            self.assertNotIn("packable_replicas", summary)

    def test_flag_on_packs_two_single_gpu_chains_onto_one_box(self):
        with _flag(True):
            manager = Manager(chain_counts=[1, 1])
            resources = manager.resources_summary()
            pool = resources[ENV]["pools"][0]
            self.assertEqual(pool["allocation_kind"], "packed")
            self.assertEqual(pool["free_instances"], 1)
            self.assertEqual(pool["partial_free_slots"], [6])
            self.assertEqual(pool["free"], 14)
            self.assertEqual(resources[ENV]["free"], 14)

            capacity = manager.pool_capacity(resources)[(ENV, POOL)]
            self.assertEqual(capacity["allocation_kind"], "packed")
            self.assertEqual(capacity["available_units"], 14)
            self.assertEqual(capacity["empty_instances"], 1)
            self.assertEqual(capacity["partial_free_slots"], [6])

    def test_flag_on_prices_and_caps_a_replica_by_its_engine_gpus(self):
        with _flag(True):
            manager = Manager(chain_counts=[1, 1])
            resources = manager.resources_summary()
            rank = RankSpec.from_dict(_plan(2, 1)["actions"][0]["ladder"][0])
            summary = manager.rank_allocation_summary(rank, resources)
            self.assertEqual(summary["allocation_kind"], "packed")
            self.assertEqual(summary["capacity_per_replica"], 2)
            self.assertEqual(summary["gpus_per_unit"], 8)
            self.assertEqual(summary["instance_price_per_hour"], 32.0)
            self.assertEqual(summary["price_per_unit_hour"], 8.0)
            # 3 on the partial box (6 free slots) + 4 on the empty box.
            self.assertEqual(summary["packable_replicas"], 7)

            full = RankSpec.from_dict(_plan(8, 1)["actions"][0]["ladder"][0])
            self.assertEqual(
                manager.rank_allocation_summary(full, resources)["packable_replicas"], 1
            )

    def test_flag_on_feasibility_respects_fragmentation(self):
        with _flag(True):
            manager = Manager(chain_counts=[1, 1])
            ok, violations = manager.check_resource_feasibility(_plan(8, 1))
            self.assertTrue(ok, violations)

            ok, violations = manager.check_resource_feasibility(_plan(4, 3))
            self.assertTrue(ok, violations)

            ok, violations = manager.check_resource_feasibility(_plan(8, 2))
            self.assertFalse(ok)
            self.assertEqual(len(violations), 1)
            self.assertIn("requested 16 GPU slots, only 14 free", violations[0])

            # Koi models first-fit packing, so two tp=4 chains share ONE box
            # (the modeled view; Orca does not report node names yet).
            self.assertEqual(
                Manager(chain_counts=[4, 4]).resources_summary()[ENV]["pools"][0]["free_instances"],
                1,
            )

            # Four 6-GPU chains cannot share boxes: 8 slots free in total
            # across four boxes, but no room for one tp=8 replica.
            fragmented = Manager(chain_counts=[6, 6, 6, 6], total_instances=4)
            resources = fragmented.resources_summary()
            self.assertEqual(resources[ENV]["pools"][0]["partial_free_slots"], [2, 2, 2, 2])
            self.assertEqual(resources[ENV]["pools"][0]["free_instances"], 0)
            self.assertEqual(resources[ENV]["free"], 8)
            ok, violations = fragmented.check_resource_feasibility(_plan(8, 1))
            self.assertFalse(ok)
            self.assertEqual(len(violations), 1)
            self.assertIn("do not pack into 0 empty instances", violations[0])
            self.assertIn("short 1 instances", violations[0])
            future = fragmented.simulate_future_resources(_plan(8, 1))[ENV]["pools"][0]
            self.assertEqual(future["free_units_after"], 0)
            self.assertEqual(future["packing_shortfall"], 1)

            requested = fragmented.requested_capacity(_plan(2, 2), resources)[1][(ENV, POOL)]
            self.assertEqual(requested, {"units": 4, "gpus": 4, "footprints": [2, 2]})
            ok, violations = fragmented.check_resource_feasibility(_plan(2, 2))
            self.assertTrue(ok, violations)

    def test_flag_on_validator_c5_reports_packing_shortfall(self):
        with _flag(True):
            manager = Manager(chain_counts=[6, 6, 6, 6], total_instances=4)
            resources = manager.resources_summary()

            class Snapshot:
                @staticmethod
                def resources_summary():
                    return resources

            result = Validator(resource_map=manager).val_plan(_plan(8, 1), Snapshot())
            self.assertFalse(result.feasible)
            self.assertEqual(len(result.violations), 1)
            self.assertTrue(result.violations[0].startswith("C5 capacity: env "))
            self.assertIn("do not pack into 0 empty instances", result.violations[0])

            result = Validator(resource_map=manager).val_plan(_plan(2, 2), Snapshot())
            self.assertTrue(result.feasible, result.violations)

    def test_flag_on_switch_pricing_prorates_hourly_rate(self):
        with _flag(True):
            manager = Manager(chain_counts=[])
            pricing = manager.switch_pricing_map()
            self.assertEqual(pricing[ENV]["packed_gpus_per_instance"], {POOL: 8})
            chain = {"env": ENV, "instance_type": POOL, "gpu_count": 2}
            self.assertEqual(hourly_rate(chain, pricing), 8.0)
            self.assertEqual(hourly_rate({**chain, "gpu_count": 8}, pricing), 32.0)
        with _flag(False):
            pricing = Manager(chain_counts=[]).switch_pricing_map()
            self.assertNotIn("packed_gpus_per_instance", pricing[ENV])
            self.assertEqual(
                hourly_rate({"env": ENV, "instance_type": POOL, "gpu_count": 2}, pricing), 32.0
            )


class AgentToolsPackingTests(unittest.TestCase):
    def _resources(self, manager):
        return manager.resources_summary()

    def test_instance_catalog_and_ladder_costs(self):
        with _flag(True):
            manager = Manager(chain_counts=[1, 1])
            with patch.object(
                agent_tools, "get_resource_map", return_value=self._resources(manager)
            ):
                specs = agent_tools.instance_catalog()
            spec = specs[ENV][POOL]
            self.assertEqual(spec["allocation_kind"], "packed")
            self.assertEqual(spec["gpus_per_instance"], 8)
            self.assertEqual(spec["free_instances"], 14)
            self.assertEqual(spec["available_units"], 14)
            self.assertEqual(spec["empty_instances"], 1)
            self.assertEqual(spec["partial_free_slots"], [6])

            ladder = _plan(2, 3)["actions"][0]["ladder"]
            self.assertEqual(
                agent_tools._ladder_capacity_cost(ladder, specs),
                {("gpu", ENV): 6, ("pool", ENV, POOL): 6},
            )
            self.assertEqual(
                agent_tools._ladder_pool_footprints(ladder, specs), {(ENV, POOL): [2, 2, 2]}
            )
            states = agent_tools._packed_pool_states(specs)
            self.assertEqual(states[(ENV, POOL)], (8, [6], 1))
            self.assertTrue(agent_tools._packed_footprints_fit(states, {(ENV, POOL): [8]}))
            self.assertFalse(agent_tools._packed_footprints_fit(states, {(ENV, POOL): [8, 8]}))
            self.assertTrue(agent_tools._packed_footprints_fit(states, {(ENV, POOL): [8, 4, 2]}))
            self.assertFalse(agent_tools._packed_footprints_fit(states, {(ENV, POOL): [8, 4, 4]}))

            agent_tools._apply_pending_footprints(specs, {(ENV, POOL): [8]})
            self.assertEqual(specs[ENV][POOL]["empty_instances"], 0)
            self.assertEqual(specs[ENV][POOL]["partial_free_slots"], [6])

        with _flag(False):
            manager = Manager(chain_counts=[1, 1])
            with patch.object(
                agent_tools, "get_resource_map", return_value=self._resources(manager)
            ):
                specs = agent_tools.instance_catalog()
            spec = specs[ENV][POOL]
            self.assertEqual(spec["allocation_kind"], "instance")
            self.assertEqual(spec["free_instances"], 0)
            self.assertNotIn("empty_instances", spec)
            self.assertEqual(
                agent_tools._ladder_capacity_cost(_plan(2, 3)["actions"][0]["ladder"], specs),
                {("gpu", ENV): 24, ("pool", ENV, POOL): 3},
            )
            self.assertEqual(agent_tools._packed_pool_states(specs), {})

    def test_generated_tp_ladder_opens_up_for_packed_pools(self):
        kwargs = {"heads": 64, "gpu_cap": 8, "gpu_type": "H100", "model_id": "unknown/model"}
        self.assertEqual(
            agent_tools._generated_tp_options(allocation_kind="instance", **kwargs), [8]
        )
        self.assertEqual(
            agent_tools._generated_tp_options(allocation_kind="packed", **kwargs), [1, 2, 4, 8]
        )
        self.assertEqual(
            agent_tools._generated_tp_options(allocation_kind="gpu", **kwargs), [1, 2, 4, 8]
        )

    def test_joint_selection_rejects_fragmented_full_width_set(self):
        """The solver's per-step packing check: a set that fits the additive
        slot budget is still rejected when its full-width replica has no
        empty box left."""
        with _flag(True):
            manager = Manager(chain_counts=[5, 5], total_instances=3)
            resources = self._resources(manager)
            # State: two boxes with 3 free slots each + one empty box = 14 slots.
            with patch.object(agent_tools, "get_resource_map", return_value=resources):
                specs = agent_tools.instance_catalog()
            states = agent_tools._packed_pool_states(specs)
            self.assertEqual(states[(ENV, POOL)], (8, [3, 3], 1))
            fit = agent_tools._packed_footprints_fit
            # Two tp=3 on the partial boxes and one tp=8 on the empty box.
            self.assertTrue(fit(states, {(ENV, POOL): [3, 3, 8]}))
            # 12 <= 14 slots, but tp=4 opens the empty box and tp=8 then has none.
            self.assertFalse(fit(states, {(ENV, POOL): [4, 8]}))
            self.assertFalse(fit(states, {(ENV, POOL): [8, 8]}))
            # Pending deployments are folded in before selection.
            agent_tools._apply_pending_footprints(specs, {(ENV, POOL): [8]})
            states = agent_tools._packed_pool_states(specs)
            self.assertEqual(states[(ENV, POOL)], (8, [3, 3], 0))
            self.assertFalse(fit(states, {(ENV, POOL): [8]}))
            self.assertTrue(fit(states, {(ENV, POOL): [3, 3]}))


if __name__ == "__main__":
    unittest.main()
