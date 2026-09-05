"""Load model placement constraints from the cluster configuration policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

SUPPORTED_EP = 1

_POLICY_TEXT = """\
A100-40GB:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct -> fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 2, 4, 8
- microsoft/phi-4 -> fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-8B -> bf16 TP: 1, 2
- Qwen/Qwen3-Next-80B-A3B-Thinking -> bf16 TP: 8
- zai-org/GLM-4.5-Air -> bf16 TP: 4, 8
A100-80GB:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct -> fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 2, 4, 8
- microsoft/phi-4 -> fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking -> bf16 TP: 4, 8
- zai-org/GLM-4.5-Air -> bf16 TP: 4, 8
A10G:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct -> fp16 TP: 1, 2, 4
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 4, 8
- microsoft/phi-4 -> fp16 TP: 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> fp16 TP: 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking -> bf16 TP: 8
- zai-org/GLM-4.5-Air -> bf16 TP: 8
H100:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- deepseek-ai/DeepSeek-V3 -> bf16-fp8 TP: 1, 8
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.1-8B -> bf16 TP: 1, 2, 4
- meta-llama/Llama-3.2-1B-Instruct -> fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 2, 4, 8
- microsoft/phi-4 -> fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> bf16 TP: 1, 2, 4, 8
- mistralai/Mixtral-8x7B-v0.1 -> bf16 TP: 1, 2
- moonshotai/Kimi-K2-Instruct -> bf16-fp8 TP: 1, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-235B-A22B -> bf16 TP: 1, 8
- Qwen/Qwen3-30B-A3B-Instruct-2507 -> bf16 TP: 1, 2, 4
- Qwen/Qwen3-Next-80B-A3B-Thinking -> bf16 TP: 4, 8
- zai-org/GLM-4.5-Air -> bf16 TP: 1, 4, 8
H200:
- mistralai/Mixtral-8x22B-v0.1 -> bf16 TP: 1
- mistralai/Mixtral-8x7B-v0.1 -> bf16 TP: 1
- Qwen/Qwen3-235B-A22B -> bf16 TP: 1
- Qwen/Qwen3-30B-A3B-Instruct-2507 -> bf16 TP: 1, 2
L4:
- meta-llama/Llama-3.2-1B-Instruct -> bf16 TP: 1
L40S:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct -> fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 2, 4, 8
- microsoft/phi-4 -> fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> fp16 TP: 1, 2, 4, 8
- mistralai/Mixtral-8x7B-v0.1 -> bf16 TP: 1, 2, 4
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking -> bf16 TP: 4, 8
- zai-org/GLM-4.5-Air -> bf16 TP: 4, 8
MI300X:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- deepseek-ai/DeepSeek-V3 -> bf16-fp8 TP: 1, 2, 4, 8
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct -> fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 1, 2, 4, 8
- microsoft/phi-4 -> fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> fp16 TP: 1, 2, 4, 8
- moonshotai/Kimi-K2-Instruct -> bf16-fp8 TP: 1, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking -> bf16 TP: 1, 2, 4, 8
- zai-org/GLM-4.5-Air -> bf16 TP: 1, 2, 4, 8
RTX3090:
- Qwen/Qwen3-4B -> bf16 TP: 1, 2, 4, 8
- Qwen/Qwen3-8B -> bf16 TP: 1, 2, 4, 8
RTX4090:
- meta-llama/Llama-3.1-8B -> bf16 TP: 1
- Qwen/Qwen3-32B -> bf16 TP: 1
RTXPRO6000:
- meta-llama/Llama-3.1-8B -> bf16 TP: 1, 2
- Qwen/Qwen3-30B-A3B-Instruct-2507 -> bf16 TP: 1, 2
- Qwen/Qwen3-32B -> bf16 TP: 1, 2
RTXPRO6000-RUNPOD:
- BAAI/bge-reranker-v2-m3 -> fp16 TP: 1
- intfloat/multilingual-e5-large -> fp16 TP: 1
- meta-llama/Llama-3.3-70B-Instruct -> bf16 TP: 1, 2, 4, 8
- microsoft/phi-4 -> fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 -> fp16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct -> fp16 TP: 1, 2, 4, 8
- zai-org/GLM-4.5-Air -> bf16 TP: 1, 2, 4, 8
"""

_POLICY_ENTRY = re.compile(r"^- (?P<model>.+) -> (?P<precision>\S+) TP: (?P<tp>\d+(?:, \d+)*)$")
VALID_TP_DEGREES = frozenset({1, 2, 4, 6, 8})
MODEL_MAX_TP = {"microsoft/phi-4": 2}
_HARDWARE_ALIASES = {
    "a100": "a100-80gb",
    "a100-pcie": "a100-40gb",
    "a100-pcie-40gb": "a100-40gb",
    "a100-sxm": "a100-80gb",
    "a100-sxm-80gb": "a100-80gb",
    "a100-80gb": "a100-80gb",
    "a10g-24gb": "a10g",
    "h100-sxm": "h100",
    "h100-sxm-80gb": "h100",
    "h200-sxm": "h200",
    "h200-sxm-141gb": "h200",
    "l4-24gb": "l4",
    "l40s-48gb": "l40s",
    "mi300": "mi300x",
    "mi300x-192gb": "mi300x",
}


@dataclass(frozen=True)
class ConfigPolicyRule:
    """Required precision and permitted tensor-parallel sizes."""

    precision: str
    allowed_tp: tuple[int, ...]


class ConfigPolicy:
    """Resolve model placement constraints for canonical hardware names."""

    def __init__(self, rules: dict[tuple[str, str], ConfigPolicyRule]) -> None:
        self._rules = dict(rules)

    def rule_for(self, hardware: str, model_id: str) -> ConfigPolicyRule | None:
        return self._rules.get((_canonical_hardware(hardware), model_id))

    def rules_for_model(self, model_id: str, hardware: list[str]) -> dict[str, ConfigPolicyRule]:
        return {gpu: rule for gpu in hardware if (rule := self.rule_for(gpu, model_id)) is not None}

    def precision_matches(
        self,
        rule: ConfigPolicyRule,
        weight_dtype: object | None,
        quantization_method: object | None,
    ) -> bool | None:
        """Return None when launch precision is unavailable, else policy compliance."""
        if weight_dtype is None and quantization_method is None:
            return None
        dtype = _canonical_precision(str(weight_dtype or ""))
        expected_dtype, _, expected_quantization = rule.precision.partition("-")
        if dtype and dtype != expected_dtype:
            return False
        if expected_quantization:
            return expected_quantization in str(quantization_method or "").lower()
        quantization = str(quantization_method or "").strip().lower()
        return quantization in {"", "none", "unquantized"}


@cache
def load_config_policy() -> ConfigPolicy:
    """Parse and cache the embedded configuration policy."""
    rules: dict[tuple[str, str], ConfigPolicyRule] = {}
    hardware: str | None = None
    for line_number, raw_line in enumerate(_POLICY_TEXT.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith("- "):
            hardware = _canonical_hardware(line[:-1])
            continue
        if hardware is None:
            raise ValueError(f"policy entry before hardware at line {line_number}")
        match = _POLICY_ENTRY.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid config policy at line {line_number}: {line!r}")
        allowed_tp = tuple(int(value) for value in match.group("tp").split(", "))
        if any(tp not in VALID_TP_DEGREES for tp in allowed_tp):
            raise ValueError(f"invalid TP policy at line {line_number}")
        model_id = match.group("model")
        max_tp = MODEL_MAX_TP.get(model_id)
        if max_tp is not None and any(tp > max_tp for tp in allowed_tp):
            raise ValueError(f"{model_id} exceeds TP policy at line {line_number}")
        key = (hardware, model_id)
        if key in rules:
            raise ValueError(f"duplicate config policy entry at line {line_number}")
        rules[key] = ConfigPolicyRule(match.group("precision"), allowed_tp)
    if not rules:
        raise ValueError("embedded config policy is empty")
    return ConfigPolicy(rules)


def _canonical_hardware(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    return _HARDWARE_ALIASES.get(normalized, normalized)


def _canonical_precision(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"float16", "half", "fp16"}:
        return "fp16"
    if normalized in {"bfloat16", "bf16"}:
        return "bf16"
    return normalized
