import os

import pytest
from db.postgres import postgres_pool
from psycopg.pq import TransactionStatus

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="Requires TEST_DATABASE_URL pointing to local PostgreSQL",
)


def test_postgres_transactions_and_pool_cleanup():
    with postgres_pool(os.environ["TEST_DATABASE_URL"]) as pool:
        with pool.connection() as conn:
            conn.execute("CREATE TEMP TABLE pool_check (value INTEGER)")
            with conn.transaction():
                conn.execute("INSERT INTO pool_check VALUES (%s)", (1,))

            with pytest.raises(ValueError, match="abort transaction"), conn.transaction():
                conn.execute("INSERT INTO pool_check VALUES (%s)", (2,))
                raise ValueError("abort transaction")

            assert conn.execute("SELECT value FROM pool_check").fetchall() == [{"value": 1}]
            assert conn.info.transaction_status == TransactionStatus.IDLE
            conn.execute("DROP TABLE pool_check")

        with pool.connection() as conn:
            assert conn.execute("SELECT 1 AS value").fetchone() == {"value": 1}

    assert pool.closed


def test_postgres_pool_closes_on_failure():
    with (
        pytest.raises(ValueError, match="startup failed"),
        postgres_pool(os.environ["TEST_DATABASE_URL"]) as pool,
    ):
        raise ValueError("startup failed")

    assert pool.closed
