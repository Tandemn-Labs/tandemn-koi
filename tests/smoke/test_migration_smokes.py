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
        ).fetchone() == (19,)
        assert conn.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'plan_rank_predictions'
            ORDER BY ordinal_position
            """
        ).fetchall() == [
            ("plan_id",),
            ("job_id",),
            ("rank_id",),
            ("v_predicted",),
            ("y_predicted",),
            ("mechanism_id",),
            ("created_at",),
        ]
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
            SELECT tgname FROM pg_trigger
            WHERE tgrelid = 'public.plan_rank_predictions'::regclass
              AND NOT tgisinternal
            """
        ).fetchall() == [("plan_rank_predictions_append_only",)]


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
