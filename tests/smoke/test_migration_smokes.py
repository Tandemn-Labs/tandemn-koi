"""Verify dbmate against disposable databases on local PostgreSQL."""

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL") or not shutil.which("dbmate"),
    reason="Requires dbmate and TEST_DATABASE_URL pointing to local PostgreSQL",
)


@pytest.fixture
def migration_database():
    url = os.environ["TEST_DATABASE_URL"]
    parsed = urlsplit(url)
    assert parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    name = f"koi_migration_test_{uuid4().hex}"
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            yield urlunsplit(parsed._replace(path=f"/{name}"))
        finally:
            conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def test_concurrent_and_repeated_migrations(migration_database):
    migrations = Path(__file__).resolve().parents[2] / "db" / "migrations"
    env = {**os.environ, "DATABASE_URL": migration_database}
    command = [
        "dbmate",
        "--wait",
        "--migration-lock",
        "--no-dump-schema",
        "--migrations-dir",
        str(migrations),
        "migrate",
        "--strict",
    ]
    with (
        subprocess.Popen(command, env=env) as first,
        subprocess.Popen(command, env=env) as second,
    ):
        assert first.wait(timeout=60) == 0
        assert second.wait(timeout=60) == 0
    subprocess.run(command, env=env, check=True, timeout=60)

    with psycopg.connect(migration_database) as conn:
        assert conn.execute("SELECT version FROM schema_migrations").fetchall() == [
            ("20261109000000",)
        ]
        assert conn.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name <> 'schema_migrations'
            """
        ).fetchone() == (13,)
        for table in (
            "job_intent",
            "job_engine_arguments",
            "plan_rank_predictions",
            "rank_metrics",
            "resource_map_snapshots",
            "tick_resource_map",
        ):
            assert conn.execute("SELECT to_regclass(%s)", (f"public.{table}",)).fetchone() == (
                None,
            )
        for table, key in (
            ("master_table", "PRIMARY KEY (tick, job_id, rank_id)"),
            ("placement", "PRIMARY KEY (job_id, rank_id)"),
            ("plans", "PRIMARY KEY (tick)"),
            ("plan_actions", "PRIMARY KEY (tick, job_id)"),
            ("job_engine_defaults", "PRIMARY KEY (model_id, engine_name)"),
        ):
            assert conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = %s::regclass AND contype = 'p'",
                (f"public.{table}",),
            ).fetchone() == (key,)
        assert (
            conn.execute(
                """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND column_name = 'plan_id'
            """
            ).fetchall()
            == []
        )
        assert conn.execute(
            """
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'placement'
              AND column_name IN ('cluster_id', 'pool_id', 'node_type_id')
            ORDER BY ordinal_position
            """
        ).fetchall() == [("cluster_id", "text"), ("pool_id", "text"), ("node_type_id", "text")]
        assert conn.execute(
            """
            SELECT data_type, is_nullable, column_default FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'master_table'
              AND column_name = 'prediction_lineage'
            """
        ).fetchone() == ("jsonb", "NO", "'{}'::jsonb")
        assert conn.execute(
            """
            SELECT pg_get_constraintdef(oid) FROM pg_constraint
            WHERE conrelid = 'public.master_table'::regclass
              AND conname = 'master_table_prediction_lineage_check'
            """
        ).fetchone() == ("CHECK ((jsonb_typeof(prediction_lineage) = 'object'::text))",)
        assert conn.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'job_engine_defaults'
              AND column_name IN ('engine_name', 'engine_version')
            ORDER BY column_name
            """
        ).fetchall() == [("engine_name",), ("engine_version",)]
        assert conn.execute(
            """
            SELECT table_name, column_name, data_type FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN ('job_engine_defaults', 'placement')
              AND column_name IN ('kv_cache_dtype', 'chunked_prefill_enabled')
            ORDER BY table_name, column_name
            """
        ).fetchall() == [
            ("job_engine_defaults", "chunked_prefill_enabled", "boolean"),
            ("job_engine_defaults", "kv_cache_dtype", "jsonb"),
            ("placement", "chunked_prefill_enabled", "boolean"),
            ("placement", "kv_cache_dtype", "text"),
        ]
        assert conn.execute(
            """
            SELECT table_name, column_name, data_type FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN ('jobs', 'job_engine_defaults')
              AND column_name IN ('extra_args', 'misc_kubernetes_labels')
            ORDER BY table_name, column_name
            """
        ).fetchall() == [
            ("jobs", "extra_args", "ARRAY"),
            ("jobs", "misc_kubernetes_labels", "ARRAY"),
        ]
        assert (
            conn.execute(
                """
            SELECT tgname FROM pg_trigger AS t
            JOIN pg_class AS c ON c.oid = t.tgrelid
            JOIN pg_namespace AS n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND NOT t.tgisinternal
            """
            ).fetchall()
            == []
        )
        assert conn.execute(
            "SELECT to_regprocedure('public.reject_append_only_mutation()')"
        ).fetchone() == (None,)

        conn.execute("INSERT INTO model_architecture (model_id) VALUES ('model-1')")
        conn.execute(
            "INSERT INTO job_engine_defaults (model_id, engine_name, max_num_seq) "
            "VALUES ('model-1', 'vllm', '64'), ('model-1', 'sglang', '128')"
        )
        with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
            conn.execute(
                "INSERT INTO job_engine_defaults (model_id, engine_name) VALUES ('model-1', 'vllm')"
            )
        with pytest.raises(psycopg.errors.NotNullViolation), conn.transaction():
            conn.execute("INSERT INTO job_engine_defaults (model_id) VALUES ('model-1')")
        conn.execute(
            "INSERT INTO jobs "
            "(job_id, kind, status, model_id, target_p99_ttft_ms, extra_args, "
            "misc_kubernetes_labels) "
            "VALUES ('job-1', 'online', 'queued', 'model-1', 100, %s, %s)",
            (["--enable-prefix-caching"], ["team=serving"]),
        )
        assert conn.execute(
            "SELECT extra_args, misc_kubernetes_labels FROM jobs WHERE job_id = 'job-1'"
        ).fetchone() == (["--enable-prefix-caching"], ["team=serving"])
        conn.execute("INSERT INTO plans (tick) VALUES (1), (2)")
        with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
            conn.execute("INSERT INTO plans (tick) VALUES (1)")
        conn.execute(
            "INSERT INTO plan_actions (tick, job_id, action) "
            "VALUES (1, 'job-1', 'place'), (2, 'job-1', 'keep')"
        )
        with pytest.raises(psycopg.errors.ForeignKeyViolation), conn.transaction():
            conn.execute("INSERT INTO plan_actions (tick, job_id) VALUES (3, 'job-1')")
        with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
            conn.execute("INSERT INTO plan_actions (tick, job_id) VALUES (1, 'job-1')")
        with pytest.raises(psycopg.errors.CheckViolation), conn.transaction():
            conn.execute("UPDATE jobs SET target_p99_ttft_ms = -1 WHERE job_id = 'job-1'")
        conn.execute("UPDATE jobs SET target_p99_ttft_ms = 200 WHERE job_id = 'job-1'")
        conn.execute(
            "INSERT INTO placement (job_id, rank_id, tensor_parallelism, max_num_seq) "
            "VALUES ('job-1', 'rank-1', 2, 64)"
        )
        with pytest.raises(psycopg.errors.UniqueViolation), conn.transaction():
            conn.execute("INSERT INTO placement (job_id, rank_id) VALUES ('job-1', 'rank-1')")
        assert conn.execute(
            "UPDATE placement SET max_num_seq = 128 "
            "WHERE job_id = 'job-1' AND rank_id = 'rank-1' RETURNING max_num_seq"
        ).fetchone() == (128,)
        conn.execute("INSERT INTO causal_mechanisms (mechanism_id) VALUES ('mechanism-1')")
        assert conn.execute(
            "INSERT INTO master_table "
            "(tick, job_id, rank_id, reasoning, v_predicted, y_predicted, mechanism_id) "
            "VALUES (1, 'job-1', 'rank-1', 'initial prediction', "
            "'{\"utilization\": 0.5}', '{\"ttft_ms\": 100}', 'mechanism-1') "
            "RETURNING job_status, v_observed, y_observed"
        ).fetchone() == ("pending", None, None)
        conn.execute(
            "INSERT INTO master_table (tick, job_id, rank_id, reasoning) "
            "VALUES (2, 'job-1', 'rank-1', 'next tick')"
        )
        assert conn.execute(
            "UPDATE master_table SET v_observed = '{\"utilization\": [0.6]}', "
            "y_observed = '{\"ttft_ms\": [110]}', job_status = 'running' "
            "WHERE tick = 1 AND job_id = 'job-1' AND rank_id = 'rank-1' "
            "RETURNING v_predicted, y_predicted, v_observed, y_observed"
        ).fetchone() == (
            {"utilization": 0.5},
            {"ttft_ms": 100},
            {"utilization": [0.6]},
            {"ttft_ms": [110]},
        )
        assert conn.execute(
            "SELECT job_status, v_observed, y_observed FROM master_table "
            "WHERE tick = 2 AND job_id = 'job-1' AND rank_id = 'rank-1'"
        ).fetchone() == ("pending", None, None)
        assert (
            conn.execute(
                "DELETE FROM placement WHERE job_id = 'job-1' AND rank_id = 'rank-1'"
            ).rowcount
            == 1
        )
        assert conn.execute("SELECT count(*) FROM master_table").fetchone() == (2,)


def test_failed_migration_rolls_back(migration_database, tmp_path):
    (tmp_path / "20261109000001_failing.sql").write_text(
        "-- migrate:up\nCREATE TABLE migration_check (id INTEGER);\n"
        "SELECT 1 / 0;\n-- migrate:down\nDROP TABLE migration_check;\n"
    )
    result = subprocess.run(
        [
            "dbmate",
            "--migration-lock",
            "--no-dump-schema",
            "--migrations-dir",
            str(tmp_path),
            "migrate",
            "--strict",
        ],
        env={**os.environ, "DATABASE_URL": migration_database},
        check=False,
        timeout=60,
    )
    assert result.returncode != 0
    with psycopg.connect(migration_database) as conn:
        assert conn.execute("SELECT to_regclass('public.migration_check')").fetchone() == (None,)
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone() == (0,)
