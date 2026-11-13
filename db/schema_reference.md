# Koi schema reference

Documentation snapshot of the frozen initial application schema, preserving the
original SQL and comments in table order. The authoritative source is
[`20261109000000_initial_schema.sql`](migrations/20261109000000_initial_schema.sql).
[`schema.sql`](schema.sql) is the generated database dump. This document is not
executed by dbmate; approved future changes belong in new migrations.

Dbmate's `schema_migrations` bookkeeping table is not part of the application
schema below. Migration directives and the blocked rollback are omitted.

## Shared types

```sql
-- Initial Koi PostgreSQL schema.
-- Python APIs own input validation and application-level defaults.

CREATE TYPE job_status AS ENUM (
    'running',
    'queued',
    'finished'
);

CREATE TYPE job_kind AS ENUM ('online', 'batch');

CREATE TYPE token_distribution AS ENUM (
    'Constant',
    'Uniform',
    'Normal',
    'LogNormal',
    'Exponential'
);

CREATE TYPE arrival_pattern AS ENUM (
    'Poisson',
    'Constant',
    'Bursty',
    'Diurnal'
);
```

## Append-only enforcement

```sql
CREATE FUNCTION reject_append_only_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
END;
$$;
```

## model_architecture

```sql
CREATE TABLE model_architecture (
    model_id TEXT PRIMARY KEY,
    model_params_b DOUBLE PRECISION,
    model_size_gb DOUBLE PRECISION,
    num_hidden_layers INTEGER,
    hidden_size INTEGER,
    num_attn_heads INTEGER,
    num_kv_heads INTEGER,
    attn_heads_per_kv_head INTEGER,
    intermediate_size INTEGER,
    max_pos_embeddings INTEGER,
    vocab_size INTEGER,
    is_moe BOOLEAN,
    num_routed_experts INTEGER,
    num_active_experts INTEGER,
    flops_per_param DOUBLE PRECISION,
    weight_dtype TEXT,
    weight_quantization_method TEXT,
    weight_quantization_bits INTEGER,
    sliding_window_size INTEGER,
    min_chain_warmup_time DOUBLE PRECISION
);
```

## job_engine_defaults

```sql
CREATE TABLE job_engine_defaults (
    model_id TEXT PRIMARY KEY REFERENCES model_architecture (model_id),
    max_model_len INTEGER,
    gpu_mem_util DOUBLE PRECISION,
    max_num_seq JSONB,
    max_num_batched_tokens JSONB,
    block_size JSONB,
    kv_cache_dtype JSONB,
    attn_backend TEXT,
    activation_quantization_method TEXT,
    activation_dtype TEXT,
    prefix_cache_enabled BOOLEAN,
    chunked_prefill_enabled BOOLEAN,
    lmcache_enabled BOOLEAN,
    spec_decoding_enabled BOOLEAN,
    draft_model_id TEXT,
    spec_decoding_method TEXT,
    num_speculative_tokens INTEGER,
    spec_acceptance_threshold DOUBLE PRECISION,
    cuda_graph_enabled BOOLEAN,
    torch_compile_enabled BOOLEAN,
    preemption_policy TEXT,
    pd_enabled BOOLEAN,
    kv_transfer_method TEXT,
    chunk_size INTEGER,
    max_chunked_steps_per_request INTEGER,
    extra_args TEXT[],
    misc_kubernetes_labels TEXT[]
);
```

## hardware_catalog

```sql
CREATE TABLE hardware_catalog (
    cloud_name TEXT PRIMARY KEY,
    catalog JSONB
);
```

## jobs

```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    kind job_kind NOT NULL,
    status job_status NOT NULL,
    model_id TEXT NOT NULL REFERENCES model_architecture (model_id)
);
```

## job_intent

```sql
CREATE TABLE job_intent (
    job_id TEXT PRIMARY KEY REFERENCES jobs (job_id),
    target_p99_ttft_ms DOUBLE PRECISION CHECK (target_p99_ttft_ms > 0),
    target_p99_tpot_ms DOUBLE PRECISION CHECK (target_p99_tpot_ms > 0),
    deadline_hrs DOUBLE PRECISION CHECK (deadline_hrs >= 0),
    isl_token_avg DOUBLE PRECISION,
    isl_token_min DOUBLE PRECISION,
    isl_token_max DOUBLE PRECISION,
    isl_distribution_type token_distribution,
    osl_token_avg DOUBLE PRECISION,
    osl_token_min DOUBLE PRECISION,
    osl_token_max DOUBLE PRECISION,
    osl_distribution_type token_distribution,
    request_arrival_rate DOUBLE PRECISION CHECK (request_arrival_rate > 0),
    max_concurrent_streaming DOUBLE PRECISION CHECK (max_concurrent_streaming > 0),
    request_arrival_pattern arrival_pattern,
    peak_to_mean_ratio DOUBLE PRECISION CHECK (peak_to_mean_ratio >= 1),
    workload_prefix_concentration DOUBLE PRECISION
        CHECK (workload_prefix_concentration BETWEEN 0 AND 1),
    multi_turn_ratio DOUBLE PRECISION CHECK (multi_turn_ratio BETWEEN 0 AND 1),
    shared_prefix_length_avg DOUBLE PRECISION CHECK (shared_prefix_length_avg >= 0),
    is_session_affinity BOOLEAN,
    total_token_budget BIGINT CHECK (total_token_budget > 0),
    priority_class TEXT,
    scheduling_policy TEXT,
    router_policy TEXT,
    CHECK (
        isl_token_min >= 0
        AND isl_token_min <= isl_token_avg
        AND isl_token_avg <= isl_token_max
    ),
    CHECK (
        osl_token_min >= 0
        AND osl_token_min <= osl_token_avg
        AND osl_token_avg <= osl_token_max
    )
);
```

## job_engine_arguments

```sql
CREATE TABLE job_engine_arguments (
    job_id TEXT REFERENCES jobs (job_id),
    rank_id TEXT,
    max_model_len INTEGER,
    gpu_mem_util DOUBLE PRECISION,
    max_num_seq INTEGER,
    max_num_batched_tokens INTEGER,
    block_size INTEGER,
    kv_cache_dtype TEXT,
    attn_backend TEXT,
    activation_quantization_method TEXT,
    activation_dtype TEXT,
    prefix_cache_enabled BOOLEAN,
    chunked_prefill_enabled BOOLEAN,
    lmcache_enabled BOOLEAN,
    spec_decoding_enabled BOOLEAN,
    draft_model_id TEXT,
    spec_decoding_method TEXT,
    num_speculative_tokens INTEGER,
    spec_acceptance_threshold DOUBLE PRECISION,
    cuda_graph_enabled BOOLEAN,
    torch_compile_enabled BOOLEAN,
    preemption_policy TEXT,
    pd_enabled BOOLEAN,
    kv_transfer_method TEXT,
    chunk_size INTEGER,
    max_chunked_steps_per_request INTEGER,
    extra_args TEXT[],
    misc_kubernetes_labels TEXT[],
    PRIMARY KEY (job_id, rank_id)
);
```

## master_table

```sql
CREATE TABLE master_table (
    tick INTEGER NOT NULL CHECK (tick >= 0),
    job_id TEXT NOT NULL REFERENCES jobs (job_id),
    rank_id TEXT NOT NULL,
    mechanisms TEXT[] NOT NULL DEFAULT '{}',
    -- debt, probably need removal soon - not needed for now
    cusum_per_mechanism JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(cusum_per_mechanism) = 'object'),
    q_label_per_mechanism JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(q_label_per_mechanism) = 'object'),
    icp_result_per_edge JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(icp_result_per_edge) = 'object'),
    -- shortcut: snapshot resolved X per evidence row; replace with joins over immutable tick-scoped source revisions when available.
    x_snapshot JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(x_snapshot) = 'object'),
    reasoning TEXT NOT NULL,
    job_status job_status NOT NULL,
    rank_beta DOUBLE PRECISION,
    rank_dro DOUBLE PRECISION,
    v_predicted JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(v_predicted) = 'object'),
    y_predicted JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(y_predicted) = 'object'),
    PRIMARY KEY (tick, job_id, rank_id)
);

CREATE INDEX master_table_rank_history_idx
    ON master_table (job_id, rank_id, tick DESC);

CREATE TRIGGER master_table_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON master_table
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## placement

The foreign key to `plan_actions` is added after that table is defined below,
matching the original execution order.

```sql
CREATE TABLE placement (
    plan_id TEXT,
    job_id TEXT REFERENCES jobs (job_id),
    rank_id TEXT,
    cluster_id TEXT,
    role TEXT,
    max_replicas INTEGER,
    min_replicas INTEGER,
    engine_name TEXT,
    engine_version TEXT,
    market TEXT,
    cloud TEXT,
    region TEXT,
    zone TEXT,
    accelerator_type TEXT,
    tensor_parallelism INTEGER,
    pipeline_parallelism INTEGER,
    moe_expert_parallelism INTEGER,
    moe_tensor_parallelism INTEGER,
    PRIMARY KEY (plan_id, job_id, rank_id)
);

CREATE TRIGGER placement_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON placement
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## slow_state_job

```sql
CREATE TABLE slow_state_job (
    tick INTEGER,
    job_id TEXT REFERENCES jobs (job_id),
    w_t JSONB,
    z_star JSONB,
    PRIMARY KEY (tick, job_id)
);

CREATE TRIGGER slow_state_job_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON slow_state_job
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## slow_state_cluster

```sql
CREATE TABLE slow_state_cluster (
    tick INTEGER,
    cluster_id TEXT,
    lambda_t DOUBLE PRECISION,
    beta_t DOUBLE PRECISION,
    b_t INTEGER,
    epsilon_dro DOUBLE PRECISION,
    PRIMARY KEY (tick, cluster_id)
);

CREATE TRIGGER slow_state_cluster_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON slow_state_cluster
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## causal_nodes

```sql
CREATE TABLE causal_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT,
    description TEXT,
    unit TEXT
);
```

## causal_edges

```sql
CREATE TABLE causal_edges (
    edge_id TEXT PRIMARY KEY,
    src TEXT REFERENCES causal_nodes (node_id),
    dst TEXT REFERENCES causal_nodes (node_id),
    src_type TEXT,
    dst_type TEXT,
    status TEXT,
    alpha DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    visit_count INTEGER,
    last_touched_tick INTEGER,
    q_histogram JSONB,
    envs_seen JSONB,
    q3_frequency DOUBLE PRECISION
);
```

## causal_mechanisms

```sql
CREATE TABLE causal_mechanisms (
    mechanism_id TEXT PRIMARY KEY,
    name TEXT,
    edge_ids TEXT[],
    scope JSONB,
    narrative TEXT,
    status TEXT,
    alpha DOUBLE PRECISION,
    beta DOUBLE PRECISION,
    visit_count INTEGER,
    last_touched_tick INTEGER,
    q_histogram JSONB,
    envs_last_seen JSONB,
    inspection_count INTEGER
);
```

## plans

```sql
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY,
    tick INTEGER,
    tick_rationale TEXT,
    created_at TIMESTAMPTZ,
    status TEXT
);
```

## plan_actions

```sql
CREATE TABLE plan_actions (
    plan_id TEXT REFERENCES plans (plan_id),
    job_id TEXT REFERENCES jobs (job_id),
    action TEXT,
    rationale TEXT,
    PRIMARY KEY (plan_id, job_id)
);

CREATE TRIGGER plan_actions_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON plan_actions
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();

ALTER TABLE placement
    ADD CONSTRAINT placement_action_fk
    FOREIGN KEY (plan_id, job_id)
    REFERENCES plan_actions (plan_id, job_id);
```

## plan_rank_predictions

```sql
CREATE TABLE plan_rank_predictions (
    plan_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    rank_id TEXT NOT NULL,
    v_predicted JSONB NOT NULL
        CHECK (jsonb_typeof(v_predicted) = 'object'),
    y_predicted JSONB NOT NULL
        CHECK (jsonb_typeof(y_predicted) = 'object'),
    mechanism_id TEXT REFERENCES causal_mechanisms (mechanism_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (plan_id, job_id, rank_id),
    FOREIGN KEY (plan_id, job_id, rank_id)
        REFERENCES placement (plan_id, job_id, rank_id)
);

CREATE TRIGGER plan_rank_predictions_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON plan_rank_predictions
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## resource_map_snapshots

```sql
CREATE TABLE resource_map_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    stored_at TIMESTAMPTZ,
    schema_version TEXT,
    cluster JSONB,
    collected_at TIMESTAMPTZ,
    source_logical_counter NUMERIC(20, 0),
    capacity_pools JSONB,
    observed_jobs JSONB,
    topology JSONB,
    installed_components JSONB,
    api_resources JSONB
);

CREATE INDEX resource_map_snapshots_cluster_time_idx
    ON resource_map_snapshots (
        (cluster ->> 'cluster_id'),
        collected_at DESC NULLS LAST
    );

CREATE TRIGGER resource_map_snapshots_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON resource_map_snapshots
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## tick_resource_map

```sql
CREATE TABLE tick_resource_map (
    tick INTEGER,
    cluster_id TEXT,
    snapshot_id TEXT REFERENCES resource_map_snapshots (snapshot_id),
    PRIMARY KEY (tick, cluster_id)
);

CREATE TRIGGER tick_resource_map_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON tick_resource_map
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```

## rank_metrics

```sql
CREATE TABLE rank_metrics (
    tick INTEGER,
    job_id TEXT REFERENCES jobs (job_id),
    rank_id TEXT,
    cluster_id TEXT,
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    collected_at TIMESTAMPTZ,
    v_observed JSONB,
    y_observed JSONB,
    PRIMARY KEY (tick, job_id, rank_id, cluster_id)
);

CREATE INDEX rank_metrics_history_idx
    ON rank_metrics (job_id, rank_id, cluster_id, tick DESC);

CREATE TRIGGER rank_metrics_append_only
BEFORE UPDATE OR DELETE OR TRUNCATE ON rank_metrics
FOR EACH STATEMENT
EXECUTE FUNCTION reject_append_only_mutation();
```
