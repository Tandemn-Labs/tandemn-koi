"""Process-owned PostgreSQL connections; domain services own their SQL."""

from collections.abc import Iterator
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


@contextmanager
def postgres_pool(database_url: str) -> Iterator[ConnectionPool]:
    """Open one pool at startup and close it on exit, including failures.

    Pass this pool to services in the same process. Group related writes using
    ``with pool.connection() as conn, conn.transaction():``; autocommit keeps
    standalone reads from leaving idle transactions open.
    """
    with ConnectionPool(
        database_url,
        min_size=1,
        max_size=4,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    ) as pool:
        pool.wait()
        yield pool
