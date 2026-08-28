### Surrogate Stack Installation

Production predictions through `SurrogateComposer` and `AICBackend` are pinned to
`AIC_Direct`. The DynoSim setup below supports isolated low-level legacy tests only.

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

### 9. Optional Tandemn predictors

The solver adapters are maintained in `Tandemn-Labs/LLM_placement_solver`, not
in this repository. Install a sibling checkout into Koi's environment:

```bash
uv pip install --python .venv/bin/python -e ../LLM_placement_solver
.venv/bin/python -c "import predictor_compare"
```

This package requires Gurobi and a working license. Configure peer mode with:

```bash
export KOI_SURROGATE_PEER_MODE=enabled
# or: .venv/bin/python -m src.orchestrator.runner --surrogate-peer-mode enabled ...
```

Modes are `off` (disabled), `shadow` (record only), and `enabled` (permit learned
fusion). The default is `shadow`; the CLI flag overrides the environment variable.
Missing peer dependencies produce one warning and Koi continues with AIC,
analytic, and PerfDB estimates.

### AIC Profile Matching

Koi builds a cached support index from AIC's public aggregate silicon support
matrix. For candidates with complete hardware and model catalog facts, it ranks
the five closest supported GPU/model/dtype profiles using operation signatures
and workload-range penalties. The requested workload and VRAM are unchanged.

The selected proxy's prefill and decode normalization is passed to DynoSim via
`speedup_ratio` and `decode_speedup_ratio`, so queueing and percentile latency
are simulated at the normalized service rate. A typed AIC coverage miss retries
the next ranked profile; failed profile/workload slices are cached. Prediction
lineage records the selected profile, distance, confidence, and phase ratios.
