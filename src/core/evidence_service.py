"""In-memory EvidenceRow ledger with secondary indexes.

The service is the v0 append-only evidence store used by S2/S3, ICP, EIG,
slow-loop recalibration, and agent diagnostics. Tandemn Store persistence is
the eventual backend, but callers should depend on this query contract rather
than the current dictionary implementation.
"""

from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import numpy as np
from src.core.models import EvidenceRow
from src.validation.cusum import CusumResult
from src.validation.icp import ICPResult
from src.validation.quadrants import Quadrant
from tandemn_system_data.clients import (  # type: ignore[import-untyped]
    EvidenceStore as StoreEvidenceStore,
)
from tandemn_system_data.clients import (
    PostgresClient,
)
from tandemn_system_data.models import (  # type: ignore[import-untyped]
    EvidenceRow as StoreEvidenceRow,
)

DEFAULT_ROW_READ_LIMIT = 200


def _coerce_enum(value: Any, enum_cls):
    """Best-effort enum round-trip for values serialized through JSON."""
    if value is None or isinstance(value, enum_cls):
        return value
    text = str(value.value if hasattr(value, "value") else value).rsplit(".", 1)[-1]
    for member in enum_cls:
        if text in (member.name, member.value, member.name.lower(), str(member.value).lower()):
            return member
    return value


def _normalize_quadrants(raw: dict[str, Any]) -> dict[str, object | None]:
    return {mid: _coerce_enum(q, Quadrant) for mid, q in (raw or {}).items()}


def _normalize_icp_results(raw: dict[str, Any]) -> dict[str, object]:
    return {edge_id: _coerce_enum(result, ICPResult) for edge_id, result in (raw or {}).items()}


def _normalize_cusum_pairs(raw: dict[str, Any]) -> dict[str, tuple[object, object]]:
    out: dict[str, tuple[object, object]] = {}
    for mid, value in (raw or {}).items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            out[mid] = (
                _coerce_enum(value[0], CusumResult),
                _coerce_enum(value[1], CusumResult),
            )
    return out


def _normalize_array_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON list values back to arrays while leaving scalars intact."""
    out: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = np.asarray(value, dtype=float)
        else:
            out[key] = value
    return out


def _to_store_row(row: EvidenceRow) -> StoreEvidenceRow:
    """Convert Koi's EvidenceRow dataclass to Tandemn Store's wire row."""
    return StoreEvidenceRow(**asdict(row))


def _from_store_row(row: StoreEvidenceRow) -> EvidenceRow:
    """Convert Tandemn Store's EvidenceRow wire row to Koi's dataclass."""
    data = asdict(row)
    for field in ("residuals_per_v", "residuals_per_y"):
        data[field] = _normalize_array_dict(data.get(field, {}))
    data["cusum_per_mechanism"] = _normalize_cusum_pairs(row.cusum_per_mechanism)
    data["q_label_per_mechanism"] = _normalize_quadrants(row.q_label_per_mechanism)
    data["icp_result_per_edge"] = _normalize_icp_results(row.icp_result_per_edge)
    return EvidenceRow(**data)


class EvidenceService:
    """Koi-compatible EvidenceRow API backed by Tandemn Store."""

    def __init__(
        self,
        user_id: str,
        postgres_client=None,
    ):
        self.user_id = user_id
        self._postgres_client = postgres_client or PostgresClient()
        self._store = StoreEvidenceStore(self._postgres_client)

    def append_row(self, row: EvidenceRow) -> str:
        """Persist one evidence row.

        Args:
            row: EvidenceRow produced by S2 for one deployed rank.

        Returns:
            The stored row_id.

        Raises:
            ValueError: If row_id already exists. Evidence rows are replayable
                facts, so duplicate ids indicate non-idempotent ingestion.
        """
        if self._store.get(row.row_id) is not None:
            raise ValueError(f"Row with ID {row.row_id} already exists")
        self._store.put(self.user_id, _to_store_row(row))
        return row.row_id

    def get_row(self, job_id: str, rank_id: str) -> list[EvidenceRow]:
        """Return rows for one (job_id, rank_id) pair."""
        return self._convert_many(self._store.rows_for_rank(self.user_id, job_id, rank_id))

    def get_rows_in_window(self, window: tuple[int, int]) -> list[EvidenceRow]:
        """Return rows with tick in the inclusive (start_tick, end_tick) window."""
        start_tick, end_tick = window
        return self._convert_many(self._store.rows_in_window(self.user_id, start_tick, end_tick))

    def get_all_rows(self, limit: int | None = DEFAULT_ROW_READ_LIMIT) -> list[EvidenceRow]:
        """Return rows in tick order, optionally capped to the latest N rows."""
        rows = self.get_rows_in_window((0, self.current_tick()))
        if limit is not None:
            return rows[-int(limit) :] if int(limit) > 0 else []
        return rows

    def retrieve_similar_rows(
        self,
        job_features: dict[str, Any],
        top_k: int = 200,
    ) -> list[EvidenceRow]:
        """Return recent rows with the same workload type when available."""
        return self._convert_many(
            self._store.retrieve_similar_rows(self.user_id, job_features, top_k=top_k)
        )

    def get_rows_for_edge(self, edge_id: str, limit: int | None = None) -> list[EvidenceRow]:
        """Return rows whose ICP result touched edge_id."""
        return self._convert_many(self._store.rows_for_edge(self.user_id, edge_id, limit))

    def get_rows_for_mechanism(
        self,
        mechanism_id: str,
        limit: int | None = None,
    ) -> list[EvidenceRow]:
        """Return rows where the mechanism was applicable to the rank."""
        return self._convert_many(self._store.rows_for_mechanism(self.user_id, mechanism_id, limit))

    def get_rows_for_job(self, job_id: str) -> list[EvidenceRow]:
        """Return all rows for one job id."""
        return self._convert_many(self._store.rows_for_job(self.user_id, job_id))

    def get_rows_for_environment(self, envs: Any) -> list[EvidenceRow]:
        """Return rows observed in one ICP environment label."""
        return self._convert_many(
            self._store.rows_for_environment(self.user_id, self._env_tuple(envs))
        )

    def get_recently_decided(self, window: int) -> list[EvidenceRow]:
        """Return rows in the last window ticks, regardless of Q decision state."""
        return self._convert_many(self._store.recently_decided(self.user_id, window))

    def count_visits_per_edge(self, edge_id: str) -> int:
        """Return row count indexed to one edge."""
        return len(self.get_rows_for_edge(edge_id))

    def envs_for_edge(self, edge_id: str) -> set:
        """Return environments where an edge has evidence rows.

        EIG's validator-support gate needs the env set itself, not just the
        count, because a new candidate may add one more environment.
        """
        return {row.env_label for row in self.get_rows_for_edge(edge_id)}

    def count_envs_per_edge(self, edge_id: str) -> int:
        """Return distinct environment count for one edge."""
        return len(self.envs_for_edge(edge_id))

    def current_tick(self) -> int:
        """Return the latest tick observed by the store."""
        return self._store.current_tick(self.user_id)

    def get_residual_history_per_v(self, v_name: str, window: int) -> np.ndarray:
        """Return concatenated V residuals for CUSUM recalibration."""
        return self._residual_history(v_name, window, "residuals_per_v")

    def get_residual_history_per_y(self, y_name: str, window: int) -> np.ndarray:
        """Return concatenated Y residuals for CUSUM recalibration."""
        return self._residual_history(y_name, window, "residuals_per_y")

    def _residual_history(self, name: str, window: int, field: str) -> np.ndarray:
        """Concatenate residual arrays from recent rows for one variable."""
        cutoff = max(0, self.current_tick() - int(window))
        chunks = []
        for row in self.get_rows_in_window((cutoff, self.current_tick())):
            arr = getattr(row, field, {}).get(name)
            if arr is not None and len(arr) > 0:
                chunks.append(np.asarray(arr, dtype=float))
        if not chunks:
            return np.array([], dtype=float)
        return np.concatenate(chunks)

    def last_touched_per_edge(self, edge_id: str) -> int | None:
        """Return latest tick for rows indexed to one edge, if any."""
        rows = self.get_rows_for_edge(edge_id)
        if not rows:
            return None
        return max(row.tick for row in rows)

    def q3_rate_window(self, edge_id: str, window: tuple[int, int]) -> float:
        """Return fraction of decided edge rows with any Q3 mechanism label."""
        start_tick, end_tick = window
        q3_count = 0
        decided_count = 0

        for row in self.get_rows_for_edge(edge_id):
            if row.tick < start_tick or row.tick > end_tick:
                continue

            q_labels = [q for q in row.q_label_per_mechanism.values() if q is not None]
            if not q_labels:
                continue

            decided_count += 1
            if any((q.value if hasattr(q, "value") else q) == "Q3" for q in q_labels):
                q3_count += 1

        if decided_count == 0:
            return 0.0
        return q3_count / decided_count

    # def write_theory_narrative(self, evidence_row_id, narrative):
    #     # Placeholder: write a narrative to an EvidenceRow, return Bool
    # TODO

    # def aggregate(self):
    #     # Placeholder: aggregate evidence (dashboard use)
    # TODO
    #     pass

    def iter_decided_per_mechanism(
        self, window: int, tick: int | None = None
    ) -> Iterator[tuple[EvidenceRow, str, object]]:
        """Yield decided (row, mechanism_id, q_label) triples in a window.

        A single EvidenceRow can contribute multiple triples because S2 fans
        one rank's telemetry out to every applicable mechanism.
        """
        upper = self.current_tick() if tick is None else int(tick)
        cutoff = max(0, upper - window)
        for row in self.get_rows_in_window((cutoff, upper)):
            for mid, q in row.q_label_per_mechanism.items():
                if q is not None:
                    yield row, mid, q

    @staticmethod
    def _convert_many(rows) -> list[EvidenceRow]:
        return [_from_store_row(row) for row in rows]

    @staticmethod
    def _env_tuple(envs: Any) -> tuple[str, ...]:
        if isinstance(envs, str):
            return tuple(envs.split("|"))
        if isinstance(envs, (list, tuple)):
            return tuple(str(part) for part in envs)
        return (str(envs),)
