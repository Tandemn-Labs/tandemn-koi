-- migrate:up
-- Initial Koi PostgreSQL schema.
-- Python APIs own input validation and application-level defaults.

CREATE TYPE job_status AS ENUM (
    'running',
    'queued',
    'finished',
    'pending'
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

CREATE TABLE job_engine_defaults (
    model_id TEXT REFERENCES model_architecture (model_id),
    engine_name TEXT,
    engine_version TEXT,
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
    misc_kubernetes_labels TEXT[],
    PRIMARY KEY (model_id, engine_name)
);

CREATE TABLE hardware_catalog (
    cloud_name TEXT PRIMARY KEY,
    catalog JSONB
);

CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    kind job_kind NOT NULL,
    status job_status NOT NULL,
    model_id TEXT NOT NULL REFERENCES model_architecture (model_id),
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
    job_status job_status NOT NULL DEFAULT 'pending',
    rank_beta DOUBLE PRECISION,
    rank_dro DOUBLE PRECISION,
    v_predicted JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(v_predicted) = 'object'),
    y_predicted JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(y_predicted) = 'object'),
    v_observed JSONB CHECK (jsonb_typeof(v_observed) = 'object'),
    y_observed JSONB CHECK (jsonb_typeof(y_observed) = 'object'),
    -- TODO: clarify why prediction_lineage needs persistence and which fields are required.
    prediction_lineage JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(prediction_lineage) = 'object'),
    mechanism_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tick, job_id, rank_id)
);

CREATE INDEX master_table_rank_history_idx
    ON master_table (job_id, rank_id, tick DESC);

CREATE TABLE placement (
    job_id TEXT REFERENCES jobs (job_id),
    rank_id TEXT,
    cluster_id TEXT,
    pool_id TEXT,
    node_type_id TEXT,
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

CREATE TABLE slow_state_job (
    tick INTEGER,
    job_id TEXT REFERENCES jobs (job_id),
    w_t JSONB,
    z_star JSONB,
    PRIMARY KEY (tick, job_id)
);

CREATE TABLE slow_state_cluster (
    tick INTEGER,
    cluster_id TEXT,
    lambda_t DOUBLE PRECISION,
    beta_t DOUBLE PRECISION,
    b_t INTEGER,
    epsilon_dro DOUBLE PRECISION,
    PRIMARY KEY (tick, cluster_id)
);

CREATE TABLE causal_nodes (
    node_id TEXT PRIMARY KEY,
    node_type TEXT,
    description TEXT,
    unit TEXT
);

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

ALTER TABLE master_table
    ADD CONSTRAINT master_table_mechanism_id_fkey
    FOREIGN KEY (mechanism_id) REFERENCES causal_mechanisms (mechanism_id);

CREATE TABLE plans (
    tick INTEGER PRIMARY KEY CHECK (tick >= 0),
    tick_rationale TEXT,
    created_at TIMESTAMPTZ,
    status TEXT
);

CREATE TABLE plan_actions (
    tick INTEGER REFERENCES plans (tick),
    job_id TEXT REFERENCES jobs (job_id),
    action TEXT,
    rationale TEXT,
    PRIMARY KEY (tick, job_id)
);

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
    PRIMARY KEY (tick, job_id, rank_id)
);

CREATE INDEX rank_metrics_history_idx
    ON rank_metrics (job_id, rank_id, tick DESC);

-- migrate:down
DO $$
BEGIN
    RAISE EXCEPTION 'Initial schema rollback is disabled';
END;
$$;
