"""Tests for koi/tools/resources.py"""

import pytest
from koi.tools.resources import parse_orca_resources, get_resources


ORCA_SHAPE_C = {
    "instances": [
        {
            "instance_type": "g6e.12xlarge", "gpu_type": "L40S",
            "gpus_per_instance": 4, "vcpus": 48, "quota_family": "G",
            "gpu_memory_gb": 48.0, "interconnect": "PCIe",
            "cost_per_instance_hour_usd": 10.49,
        },
        {
            "instance_type": "p4d.24xlarge", "gpu_type": "A100",
            "gpus_per_instance": 8, "vcpus": 96, "quota_family": "P4_P3_P2",
            "gpu_memory_gb": 40.0, "interconnect": "NVLink",
            "cost_per_instance_hour_usd": 32.77,
        },
        {
            "instance_type": "p4de.24xlarge", "gpu_type": "A100",
            "gpus_per_instance": 8, "vcpus": 96, "quota_family": "P4_P3_P2",
            "gpu_memory_gb": 80.0, "interconnect": "NVLink",
            "cost_per_instance_hour_usd": 40.96,
        },
    ],
    "quotas": [
        {"family": "G", "region": "us-east-1", "market": "on_demand",
         "baseline_vcpus": 192, "used_vcpus": 0},
        {"family": "P4_P3_P2", "region": "us-west-2", "market": "on_demand",
         "baseline_vcpus": 384, "used_vcpus": 0},
    ],
}


class TestShapeC:
    def test_basic_parse(self):
        rm = parse_orca_resources(ORCA_SHAPE_C)
        assert len(rm.resources) == 3
        assert rm.total_available_gpus() > 0

    def test_a100_normalization(self):
        """A100 with 40GB VRAM → A100-40GB, 80GB → A100-80GB."""
        rm = parse_orca_resources(ORCA_SHAPE_C)
        gpu_types = {r.gpu_type for r in rm.resources}
        assert "A100-40GB" in gpu_types
        assert "A100-80GB" in gpu_types
        assert "A100" not in gpu_types  # generic A100 should be normalized

    def test_l40s_preserved(self):
        rm = parse_orca_resources(ORCA_SHAPE_C)
        l40s = [r for r in rm.resources if r.gpu_type == "L40S"]
        assert len(l40s) == 1
        assert l40s[0].gpu_memory_gb == 48.0

    def test_gpu_counts(self):
        rm = parse_orca_resources(ORCA_SHAPE_C)
        l40s = [r for r in rm.resources if r.gpu_type == "L40S"][0]
        # 192 / 48 = 4 instances × 4 GPUs = 16
        assert l40s.total_gpus == 16

    def test_no_cap_on_instances(self):
        """Large quota should produce all available GPUs, no arbitrary cap."""
        data = {
            "instances": [{"instance_type": "g6e.12xlarge", "gpu_type": "L40S",
                           "gpus_per_instance": 4, "vcpus": 48, "quota_family": "G",
                           "gpu_memory_gb": 48.0, "cost_per_instance_hour_usd": 10.49}],
            "quotas": [{"family": "G", "region": "us-east-1", "market": "on_demand",
                        "baseline_vcpus": 960, "used_vcpus": 0}],
        }
        rm = parse_orca_resources(data)
        # 960 / 48 = 20 instances × 4 GPUs = 80
        assert rm.resources[0].total_gpus == 80

    def test_zero_quota_excluded(self):
        data = {
            "instances": [{"instance_type": "g6e.12xlarge", "gpu_type": "L40S",
                           "gpus_per_instance": 4, "vcpus": 48, "quota_family": "G",
                           "gpu_memory_gb": 48.0, "cost_per_instance_hour_usd": 10.49}],
            "quotas": [{"family": "G", "region": "us-east-1", "market": "on_demand",
                        "baseline_vcpus": 0, "used_vcpus": 0}],
        }
        with pytest.raises(ValueError):
            parse_orca_resources(data)


class TestShapeA:
    def test_plain_list(self):
        data = [
            {"gpu_type": "H100", "instance_type": "p5.48xlarge",
             "gpus_per_instance": 8, "total_gpus": 16,
             "cost_per_instance_hour_usd": 98.32, "gpu_memory_gb": 80.0,
             "region": "us-east-1", "interconnect": "NVLink"},
        ]
        rm = parse_orca_resources(data)
        assert len(rm.resources) == 1
        assert rm.resources[0].gpu_type == "H100"


class TestShapeB:
    def test_wrapper(self):
        data = {
            "vpc_id": "vpc-test",
            "region": "us-west-2",
            "resources": [
                {"gpu_type": "A100", "instance_type": "p4de.24xlarge",
                 "gpus_per_instance": 8, "total_gpus": 32,
                 "cost_per_instance_hour_usd": 40.96, "gpu_memory_gb": 80.0,
                 "region": "us-west-2", "interconnect": "NVLink"},
            ],
        }
        rm = parse_orca_resources(data)
        assert rm.vpc_id == "vpc-test"
        assert rm.resources[0].total_gpus == 32


class TestGetResources:
    def test_formatted_output(self):
        rm = parse_orca_resources(ORCA_SHAPE_C)
        text = get_resources(rm)
        assert "AVAILABLE RESOURCES" in text
        assert "L40S" in text
        assert "A100-80GB" in text
