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
