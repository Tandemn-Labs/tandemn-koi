### Surrogate Stack Installation

```sudo apt update
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake curl git
```

These are build/runtime dependencies for Dynamo’s Rust/Python bindings.

### 2. Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Verify:

```
uv --version
```

### 3. Create Python Env For koi

From the repo root:

```
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install pip maturin
```

### 4. Install Rust

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 5. Install Dynamo Stable in the same uv venv

```
source .venv/bin/activate
uv pip install 'ai-dynamo[mocker]==1.2.1'
```

`[mocker]` installs AIC support via `aiconfigurator`.

Only use a local Dynamo checkout for Dynamo development. If you do, rebuild and
reinstall all Dynamo pieces from the same checkout/revision so `components/src`,
`lib/bindings/python/src`, and the compiled `dynamo._core` extension match:

```
cd ../dynamo
source ../koi/.venv/bin/activate
cd lib/bindings/python
maturin develop --uv
cd "$(git rev-parse --show-toplevel)"
uv pip install -e '.[mocker]'
```

Mixed Dynamo revisions usually show up as import/signature errors such as
`dynamo.mocker` missing `MockEngineArgs`, missing AIC helpers in
`dynamo._internal.aic`, or `_core` passing more AIC args than Python accepts.


### 6. Configure Hugging Face Access With .env

For koi, model configs are resolved from Hugging Face. Store the token in a repo-local .env file.

go to koi root
```
cd ../../../../
cat > .env <<'EOF'
HF_TOKEN=hf_your_token_here
EOF
echo ".env" >> .gitignore
```
### 7. Verify AIC-Backed DynoSim

```
python -m dynamo.replay \
--input-tokens 1024 \
--output-tokens 128 \
--request-count 10 \
--num-workers 1 \
--replay-mode offline \
--replay-concurrency 2 \
--extra-engine-args '{
    "engine_type": "vllm",
    "block_size": 64,
    "aic_backend": "vllm",
    "aic_backend_version": "0.19.0",
    "aic_system": "h200_sxm",
    "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8",
    "aic_tp_size": 1
}' \
--report-json /tmp/dynosim-aic-smoke.json
```

### 8. Install other deps

```
uv pip install -r requirements.txt
```

### 9. Koi-native composer

`SurrogatePrediction` remains the authoritative AIC backend for DP, scenarios, memory preflight,
PP, and structured errors. `SurrogateComposer` enriches that result without replacing the primary:

```python
from src.prediction.composer import SurrogateComposer
from src.prediction.surrogate import SurrogatePrediction

surrogate = SurrogateComposer(
    primary=SurrogatePrediction(),
    peer_mode="shadow",  # off | shadow | enabled
    evidence_store=evidence_store,
)
```

PerfDB correction defaults to `enabled`. `init_surrogate_stack` reads its CSV path from
`KOI_PERFDB_PATH` when no explicit `perfdb_path` is passed. Without a configured path, the
component remains `unconfigured` and does not alter AIC output.

The composer preserves `compose_prediction(..., scenario=...)` and
`compose_prediction_with_trace(...)`. Its order is:

1. Direct AIC primary and structured safety failures.
2. Omission-safe analytic memory/KV and TP/PP mediator V.
3. Optional PerfDB correction, default off.
4. Versioned Solver/BLIS peers, default shadow.
5. Evidence-gated throughput fusion.
6. One residual-calibration pass for eligible V/Y nodes.
7. Re-derived cost and SLO outcomes.

Analytic parallelism never applies a second PP correction to AIC Y. Learned fusion throughput is
not residual-calibrated again. Cost and SLO margin are derived after any eligible correction.
Schema-v3 lineage records normalized inputs, component statuses/versions/coverage, raw/final
values, fusion/calibration decisions, timings, and the failure stage. Structured primary failures
still propagate unchanged and prevent all later stages from running.

The local EvidenceService preserves new lineage fields in a process-local sidecar when the
installed Tandemn Store wire model lacks them. Restart-durable learned fusion therefore requires
the Store schema to add `deployment_id`, `evidence_available_timestamp_utc`, and
`prediction_lineage`.
