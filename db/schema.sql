\restrict dbmate

-- Dumped from database version 16.14
-- Dumped by pg_dump version 18.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: arrival_pattern; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.arrival_pattern AS ENUM (
    'Poisson',
    'Constant',
    'Bursty',
    'Diurnal'
);


--
-- Name: job_kind; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.job_kind AS ENUM (
    'online',
    'batch'
);


--
-- Name: job_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.job_status AS ENUM (
    'running',
    'queued',
    'finished'
);


--
-- Name: token_distribution; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.token_distribution AS ENUM (
    'Constant',
    'Uniform',
    'Normal',
    'LogNormal',
    'Exponential'
);


--
-- Name: reject_append_only_mutation(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.reject_append_only_mutation() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: causal_edges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.causal_edges (
    edge_id text NOT NULL,
    src text,
    dst text,
    src_type text,
    dst_type text,
    status text,
    alpha double precision,
    beta double precision,
    visit_count integer,
    last_touched_tick integer,
    q_histogram jsonb,
    envs_seen jsonb,
    q3_frequency double precision
);


--
-- Name: causal_mechanisms; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.causal_mechanisms (
    mechanism_id text NOT NULL,
    name text,
    edge_ids text[],
    scope jsonb,
    narrative text,
    status text,
    alpha double precision,
    beta double precision,
    visit_count integer,
    last_touched_tick integer,
    q_histogram jsonb,
    envs_last_seen jsonb,
    inspection_count integer
);


--
-- Name: causal_nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.causal_nodes (
    node_id text NOT NULL,
    node_type text,
    description text,
    unit text
);


--
-- Name: hardware_catalog; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hardware_catalog (
    cloud_name text NOT NULL,
    catalog jsonb
);


--
-- Name: job_engine_arguments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_engine_arguments (
    job_id text NOT NULL,
    rank_id text NOT NULL,
    max_model_len integer,
    gpu_mem_util double precision,
    max_num_seq integer,
    max_num_batched_tokens integer,
    block_size integer,
    kvcache_dtype text,
    attn_backend text,
    activation_quantization_method text,
    activation_dtype text,
    prefix_cache_enabled boolean,
    chunked_prefill_enabled boolean,
    lmcache_enabled boolean,
    spec_decoding_enabled boolean,
    draft_model_id text,
    spec_decoding_method text,
    num_speculative_tokens integer,
    spec_acceptance_threshold double precision,
    cuda_graph_enabled boolean,
    torch_compile_enabled boolean,
    preemption_policy text,
    pd_enabled boolean,
    kv_transfer_method text,
    chunk_size integer,
    max_chunked_steps_per_request integer,
    extra_args text[],
    misc_kubernetes_labels text[]
);


--
-- Name: job_engine_defaults; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_engine_defaults (
    model_id text NOT NULL,
    max_model_len integer,
    gpu_mem_util double precision,
    max_num_seq jsonb,
    max_num_batched_tokens jsonb,
    block_size jsonb,
    kvcache_dtype jsonb,
    attn_backend text,
    activation_quantization_method text,
    activation_dtype text,
    prefix_cache_enabled boolean,
    chunked_prefill_enabled boolean,
    lmcache_enabled boolean,
    spec_decoding_enabled boolean,
    draft_model_id text,
    spec_decoding_method text,
    num_speculative_tokens integer,
    spec_acceptance_threshold double precision,
    cuda_graph_enabled boolean,
    torch_compile_enabled boolean,
    preemption_policy text,
    pd_enabled boolean,
    kv_transfer_method text,
    chunk_size integer,
    max_chunked_steps_per_request integer,
    extra_args text[],
    misc_kubernetes_labels text[]
);


--
-- Name: job_intent; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_intent (
    job_id text NOT NULL,
    target_p99_ttft_ms double precision,
    target_p99_tpot_ms double precision,
    deadline_hrs double precision,
    isl_token_avg double precision,
    isl_token_min double precision,
    isl_token_max double precision,
    isl_distribution_type public.token_distribution,
    osl_token_avg double precision,
    osl_token_min double precision,
    osl_token_max double precision,
    osl_distribution_type public.token_distribution,
    request_arrival_rate double precision,
    max_concurrent_streaming double precision,
    request_arrival_pattern public.arrival_pattern,
    peak_to_mean_ratio double precision,
    workload_prefix_concentration double precision,
    multi_turn_ratio double precision,
    shared_prefix_length_avg double precision,
    is_session_affinity boolean,
    total_token_budget bigint,
    priority_class text,
    scheduling_policy text,
    router_policy text,
    CONSTRAINT job_intent_check CHECK (((isl_token_min >= (0)::double precision) AND (isl_token_min <= isl_token_avg) AND (isl_token_avg <= isl_token_max))),
    CONSTRAINT job_intent_check1 CHECK (((osl_token_min >= (0)::double precision) AND (osl_token_min <= osl_token_avg) AND (osl_token_avg <= osl_token_max))),
    CONSTRAINT job_intent_deadline_hrs_check CHECK ((deadline_hrs >= (0)::double precision)),
    CONSTRAINT job_intent_max_concurrent_streaming_check CHECK ((max_concurrent_streaming > (0)::double precision)),
    CONSTRAINT job_intent_multi_turn_ratio_check CHECK (((multi_turn_ratio >= (0)::double precision) AND (multi_turn_ratio <= (1)::double precision))),
    CONSTRAINT job_intent_peak_to_mean_ratio_check CHECK ((peak_to_mean_ratio >= (1)::double precision)),
    CONSTRAINT job_intent_request_arrival_rate_check CHECK ((request_arrival_rate > (0)::double precision)),
    CONSTRAINT job_intent_shared_prefix_length_avg_check CHECK ((shared_prefix_length_avg >= (0)::double precision)),
    CONSTRAINT job_intent_target_p99_tpot_ms_check CHECK ((target_p99_tpot_ms > (0)::double precision)),
    CONSTRAINT job_intent_target_p99_ttft_ms_check CHECK ((target_p99_ttft_ms > (0)::double precision)),
    CONSTRAINT job_intent_total_token_budget_check CHECK ((total_token_budget > 0)),
    CONSTRAINT job_intent_workload_prefix_concentration_check CHECK (((workload_prefix_concentration >= (0)::double precision) AND (workload_prefix_concentration <= (1)::double precision)))
);


--
-- Name: jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.jobs (
    job_id text NOT NULL,
    kind public.job_kind NOT NULL,
    status public.job_status NOT NULL,
    model_id text NOT NULL
);


--
-- Name: master_table; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.master_table (
    tick integer NOT NULL,
    job_id text NOT NULL,
    rank_id text NOT NULL,
    mechanisms text[] DEFAULT '{}'::text[] NOT NULL,
    cusum_per_mechanism jsonb DEFAULT '{}'::jsonb NOT NULL,
    q_label_per_mechanism jsonb DEFAULT '{}'::jsonb NOT NULL,
    icp_result_per_edge jsonb DEFAULT '{}'::jsonb NOT NULL,
    x_snapshot jsonb DEFAULT '{}'::jsonb NOT NULL,
    reasoning text NOT NULL,
    job_status public.job_status NOT NULL,
    rank_beta double precision,
    rank_dro double precision,
    v_predicted jsonb DEFAULT '{}'::jsonb NOT NULL,
    y_predicted jsonb DEFAULT '{}'::jsonb NOT NULL,
    CONSTRAINT master_table_cusum_per_mechanism_check CHECK ((jsonb_typeof(cusum_per_mechanism) = 'object'::text)),
    CONSTRAINT master_table_icp_result_per_edge_check CHECK ((jsonb_typeof(icp_result_per_edge) = 'object'::text)),
    CONSTRAINT master_table_q_label_per_mechanism_check CHECK ((jsonb_typeof(q_label_per_mechanism) = 'object'::text)),
    CONSTRAINT master_table_tick_check CHECK ((tick >= 0)),
    CONSTRAINT master_table_v_predicted_check CHECK ((jsonb_typeof(v_predicted) = 'object'::text)),
    CONSTRAINT master_table_x_snapshot_check CHECK ((jsonb_typeof(x_snapshot) = 'object'::text)),
    CONSTRAINT master_table_y_predicted_check CHECK ((jsonb_typeof(y_predicted) = 'object'::text))
);


--
-- Name: model_architecture; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_architecture (
    model_id text NOT NULL,
    model_params_b double precision,
    model_size_gb double precision,
    num_hidden_layers integer,
    hidden_size integer,
    num_attn_heads integer,
    num_kv_heads integer,
    attn_heads_per_kv_head integer,
    intermediate_size integer,
    max_pos_embeddings integer,
    vocab_size integer,
    is_moe boolean,
    num_routed_experts integer,
    num_active_experts integer,
    flops_per_param double precision,
    weight_dtype text,
    weight_quantization_method text,
    weight_quantization_bits integer,
    sliding_window_size integer,
    min_chain_warmup_time double precision
);


--
-- Name: placement; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.placement (
    plan_id text NOT NULL,
    job_id text NOT NULL,
    rank_id text NOT NULL,
    cluster_id text,
    role text,
    max_replicas integer,
    min_replicas integer,
    engine_name text,
    engine_version text,
    market text,
    cloud text,
    region text,
    zone text,
    accelerator_type text,
    tensor_parallelism integer,
    pipeline_parallelism integer,
    moe_expert_parallelism integer,
    moe_tensor_parallelism integer
);


--
-- Name: plan_actions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plan_actions (
    plan_id text NOT NULL,
    job_id text NOT NULL,
    action text,
    rationale text
);


--
-- Name: plan_rank_predictions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plan_rank_predictions (
    plan_id text NOT NULL,
    job_id text NOT NULL,
    rank_id text NOT NULL,
    v_predicted jsonb NOT NULL,
    y_predicted jsonb NOT NULL,
    mechanism_id text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT plan_rank_predictions_v_predicted_check CHECK ((jsonb_typeof(v_predicted) = 'object'::text)),
    CONSTRAINT plan_rank_predictions_y_predicted_check CHECK ((jsonb_typeof(y_predicted) = 'object'::text))
);


--
-- Name: plans; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plans (
    plan_id text NOT NULL,
    tick integer,
    tick_rationale text,
    created_at timestamp with time zone,
    status text
);


--
-- Name: rank_metrics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rank_metrics (
    tick integer NOT NULL,
    job_id text NOT NULL,
    rank_id text NOT NULL,
    cluster_id text NOT NULL,
    window_start timestamp with time zone,
    window_end timestamp with time zone,
    collected_at timestamp with time zone,
    v_observed jsonb,
    y_observed jsonb
);


--
-- Name: resource_map_snapshots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.resource_map_snapshots (
    snapshot_id text NOT NULL,
    stored_at timestamp with time zone,
    schema_version text,
    cluster jsonb,
    collected_at timestamp with time zone,
    source_logical_counter numeric(20,0),
    capacity_pools jsonb,
    observed_jobs jsonb,
    topology jsonb,
    installed_components jsonb,
    api_resources jsonb
);


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: slow_state_cluster; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.slow_state_cluster (
    tick integer NOT NULL,
    cluster_id text NOT NULL,
    lambda_t double precision,
    beta_t double precision,
    b_t integer,
    epsilon_dro double precision
);


--
-- Name: slow_state_job; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.slow_state_job (
    tick integer NOT NULL,
    job_id text NOT NULL,
    w_t jsonb,
    z_star jsonb
);


--
-- Name: tick_resource_map; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tick_resource_map (
    tick integer NOT NULL,
    cluster_id text NOT NULL,
    snapshot_id text
);


--
-- Name: causal_edges causal_edges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_edges
    ADD CONSTRAINT causal_edges_pkey PRIMARY KEY (edge_id);


--
-- Name: causal_mechanisms causal_mechanisms_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_mechanisms
    ADD CONSTRAINT causal_mechanisms_pkey PRIMARY KEY (mechanism_id);


--
-- Name: causal_nodes causal_nodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_nodes
    ADD CONSTRAINT causal_nodes_pkey PRIMARY KEY (node_id);


--
-- Name: hardware_catalog hardware_catalog_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hardware_catalog
    ADD CONSTRAINT hardware_catalog_pkey PRIMARY KEY (cloud_name);


--
-- Name: job_engine_arguments job_engine_arguments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_engine_arguments
    ADD CONSTRAINT job_engine_arguments_pkey PRIMARY KEY (job_id, rank_id);


--
-- Name: job_engine_defaults job_engine_defaults_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_engine_defaults
    ADD CONSTRAINT job_engine_defaults_pkey PRIMARY KEY (model_id);


--
-- Name: job_intent job_intent_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_intent
    ADD CONSTRAINT job_intent_pkey PRIMARY KEY (job_id);


--
-- Name: jobs jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_pkey PRIMARY KEY (job_id);


--
-- Name: master_table master_table_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.master_table
    ADD CONSTRAINT master_table_pkey PRIMARY KEY (tick, job_id, rank_id);


--
-- Name: model_architecture model_architecture_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_architecture
    ADD CONSTRAINT model_architecture_pkey PRIMARY KEY (model_id);


--
-- Name: placement placement_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.placement
    ADD CONSTRAINT placement_pkey PRIMARY KEY (plan_id, job_id, rank_id);


--
-- Name: plan_actions plan_actions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_actions
    ADD CONSTRAINT plan_actions_pkey PRIMARY KEY (plan_id, job_id);


--
-- Name: plan_rank_predictions plan_rank_predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_rank_predictions
    ADD CONSTRAINT plan_rank_predictions_pkey PRIMARY KEY (plan_id, job_id, rank_id);


--
-- Name: plans plans_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plans
    ADD CONSTRAINT plans_pkey PRIMARY KEY (plan_id);


--
-- Name: rank_metrics rank_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rank_metrics
    ADD CONSTRAINT rank_metrics_pkey PRIMARY KEY (tick, job_id, rank_id, cluster_id);


--
-- Name: resource_map_snapshots resource_map_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.resource_map_snapshots
    ADD CONSTRAINT resource_map_snapshots_pkey PRIMARY KEY (snapshot_id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: slow_state_cluster slow_state_cluster_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.slow_state_cluster
    ADD CONSTRAINT slow_state_cluster_pkey PRIMARY KEY (tick, cluster_id);


--
-- Name: slow_state_job slow_state_job_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.slow_state_job
    ADD CONSTRAINT slow_state_job_pkey PRIMARY KEY (tick, job_id);


--
-- Name: tick_resource_map tick_resource_map_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tick_resource_map
    ADD CONSTRAINT tick_resource_map_pkey PRIMARY KEY (tick, cluster_id);


--
-- Name: master_table_rank_history_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX master_table_rank_history_idx ON public.master_table USING btree (job_id, rank_id, tick DESC);


--
-- Name: rank_metrics_history_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX rank_metrics_history_idx ON public.rank_metrics USING btree (job_id, rank_id, cluster_id, tick DESC);


--
-- Name: resource_map_snapshots_cluster_time_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX resource_map_snapshots_cluster_time_idx ON public.resource_map_snapshots USING btree (((cluster ->> 'cluster_id'::text)), collected_at DESC NULLS LAST);


--
-- Name: master_table master_table_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER master_table_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.master_table FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: placement placement_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER placement_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.placement FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: plan_actions plan_actions_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER plan_actions_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.plan_actions FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: plan_rank_predictions plan_rank_predictions_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER plan_rank_predictions_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.plan_rank_predictions FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: rank_metrics rank_metrics_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER rank_metrics_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.rank_metrics FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: resource_map_snapshots resource_map_snapshots_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER resource_map_snapshots_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.resource_map_snapshots FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: slow_state_cluster slow_state_cluster_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER slow_state_cluster_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.slow_state_cluster FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: slow_state_job slow_state_job_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER slow_state_job_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.slow_state_job FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: tick_resource_map tick_resource_map_append_only; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER tick_resource_map_append_only BEFORE DELETE OR UPDATE OR TRUNCATE ON public.tick_resource_map FOR EACH STATEMENT EXECUTE FUNCTION public.reject_append_only_mutation();


--
-- Name: causal_edges causal_edges_dst_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_edges
    ADD CONSTRAINT causal_edges_dst_fkey FOREIGN KEY (dst) REFERENCES public.causal_nodes(node_id);


--
-- Name: causal_edges causal_edges_src_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_edges
    ADD CONSTRAINT causal_edges_src_fkey FOREIGN KEY (src) REFERENCES public.causal_nodes(node_id);


--
-- Name: job_engine_arguments job_engine_arguments_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_engine_arguments
    ADD CONSTRAINT job_engine_arguments_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: job_engine_defaults job_engine_defaults_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_engine_defaults
    ADD CONSTRAINT job_engine_defaults_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.model_architecture(model_id);


--
-- Name: job_intent job_intent_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_intent
    ADD CONSTRAINT job_intent_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: jobs jobs_model_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_model_id_fkey FOREIGN KEY (model_id) REFERENCES public.model_architecture(model_id);


--
-- Name: master_table master_table_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.master_table
    ADD CONSTRAINT master_table_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: placement placement_action_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.placement
    ADD CONSTRAINT placement_action_fk FOREIGN KEY (plan_id, job_id) REFERENCES public.plan_actions(plan_id, job_id);


--
-- Name: placement placement_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.placement
    ADD CONSTRAINT placement_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: plan_actions plan_actions_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_actions
    ADD CONSTRAINT plan_actions_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: plan_actions plan_actions_plan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_actions
    ADD CONSTRAINT plan_actions_plan_id_fkey FOREIGN KEY (plan_id) REFERENCES public.plans(plan_id);


--
-- Name: plan_rank_predictions plan_rank_predictions_mechanism_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_rank_predictions
    ADD CONSTRAINT plan_rank_predictions_mechanism_id_fkey FOREIGN KEY (mechanism_id) REFERENCES public.causal_mechanisms(mechanism_id);


--
-- Name: plan_rank_predictions plan_rank_predictions_plan_id_job_id_rank_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_rank_predictions
    ADD CONSTRAINT plan_rank_predictions_plan_id_job_id_rank_id_fkey FOREIGN KEY (plan_id, job_id, rank_id) REFERENCES public.placement(plan_id, job_id, rank_id);


--
-- Name: rank_metrics rank_metrics_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rank_metrics
    ADD CONSTRAINT rank_metrics_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: slow_state_job slow_state_job_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.slow_state_job
    ADD CONSTRAINT slow_state_job_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(job_id);


--
-- Name: tick_resource_map tick_resource_map_snapshot_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tick_resource_map
    ADD CONSTRAINT tick_resource_map_snapshot_id_fkey FOREIGN KEY (snapshot_id) REFERENCES public.resource_map_snapshots(snapshot_id);


--
-- PostgreSQL database dump complete
--

\unrestrict dbmate


--
-- Dbmate schema migrations
--

INSERT INTO public.schema_migrations (version) VALUES
    ('20261109000000');
