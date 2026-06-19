"""In-memory EvidenceRow ledger with secondary indexes.

The service is the v0 append-only evidence store used by S2/S3, ICP, EIG,
slow-loop recalibration, and agent diagnostics. Tandemn Store persistence is
the eventual backend, but callers should depend on this query contract rather
than the current dictionary implementation.
"""

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import numpy as np
from src.core.models import EvidenceRow

# TODO - Will have to modify this to work with TandemnStore
DEFAULT_ROW_READ_LIMIT = 200


class EvidenceService:
    """Append-only EvidenceRow store plus lookup indexes."""

    def __init__(self):
        self._row_by_id: dict[str, EvidenceRow] = {}
        self._by_tick: defaultdict[int, list[str]] = defaultdict(list)
        self._by_job: defaultdict[str, list[str]] = defaultdict(list)
        self._by_rank: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        self._by_mechanism: defaultdict[str, list[str]] = defaultdict(list)
        self._by_edge: defaultdict[str, list[str]] = defaultdict(list)
        self._by_env: defaultdict[Any, list[str]] = defaultdict(list)
        self._by_workload_type: defaultdict[str, list[str]] = defaultdict(list)
        self._current_tick = 0

    def append_row(self, row: EvidenceRow) -> str:
        """Append one evidence row and update all lookup indexes.

        Args:
            row: EvidenceRow produced by S2 for one deployed rank.

        Returns:
            The stored row_id.

        Raises:
            ValueError: If row_id already exists. Evidence rows are replayable
                facts, so duplicate ids indicate non-idempotent ingestion.
        """
        # the idea is that, it appends the row to the evidence store,
        # keep it a placeholder for now
        # given that we will save this in TandemnStore instead of InMemoryDictionary
        if row.row_id in self._row_by_id:
            raise ValueError(f"Row with ID {row.row_id} already exists")
        self._row_by_id[row.row_id] = row
        self._by_tick[row.tick].append(row.row_id)
        self._by_job[row.job_id].append(row.row_id)
        self._by_rank[(row.job_id, row.rank_id)].append(row.row_id)
        self._by_env[row.env_label].append(row.row_id)
        workload_type = self._workload_type(row.W_observed)
        if workload_type is not None:
            self._by_workload_type[workload_type].append(row.row_id)
        for mechanism_id in row.mechanism_ids:
            self._by_mechanism[mechanism_id].append(row.row_id)
        for edge_id in row.icp_result_per_edge:
            self._by_edge[edge_id].append(row.row_id)
        self._current_tick = max(self._current_tick, row.tick)
        return row.row_id

    def get_row(self, job_id: str, rank_id: str) -> list[EvidenceRow]:
        """Return rows for one (job_id, rank_id) pair."""
        row_ids = self._by_rank.get((job_id, rank_id), [])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_in_window(self, window: tuple[int, int]) -> list[EvidenceRow]:
        """Return rows with tick in the inclusive (start_tick, end_tick) window."""
        start_tick, end_tick = window
        rows = []
        for tick in range(start_tick, end_tick + 1):
            row_ids = self._by_tick.get(tick, [])
            for row_id in row_ids:
                rows.append(self._row_by_id[row_id])
        return rows

    def get_all_rows(self, limit: int | None = DEFAULT_ROW_READ_LIMIT) -> list[EvidenceRow]:
        """Return rows in tick order, optionally capped to the latest N rows."""
        if limit is not None:
            row_ids = self._latest_row_ids(int(limit))
            return [self._row_by_id[row_id] for row_id in row_ids]

        row_ids = []
        for tick in sorted(self._by_tick):
            row_ids.extend(self._by_tick[tick])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def retrieve_similar_rows(
        self,
        job_features: dict[str, Any],
        top_k: int = 200,
    ) -> list[EvidenceRow]:
        """Return recent rows with the same workload type when available.

        This is the temporary in-memory stand-in for a future Tandemn Store
        retrieval/KNN path. It preserves the same top_k contract used by
        z_star and agent diagnostics.
        """
        # TODO: Replace this type-only in-memory filter with KNN over TandemnStore
        # components (EvidenceRow, profiling DB, etc.) once TandemnStore is integrated.
        top_k = int(top_k)
        if top_k <= 0:
            return []

        workload_type = self._workload_type(job_features)
        if workload_type is None:
            row_ids = self._latest_row_ids(top_k)
            return [self._row_by_id[row_id] for row_id in row_ids]

        row_ids = self._by_workload_type.get(workload_type, [])[-top_k:]
        if not row_ids:
            row_ids = self._latest_row_ids(top_k)
        return [self._row_by_id[row_id] for row_id in row_ids]

    def _latest_row_ids(self, limit: int) -> list[str]:
        """Return latest row ids, preserving chronological order in output."""
        limit = int(limit)
        if limit <= 0:
            return []

        row_ids = []
        for tick in sorted(self._by_tick, reverse=True):
            for row_id in reversed(self._by_tick[tick]):
                row_ids.append(row_id)
                if len(row_ids) == limit:
                    return list(reversed(row_ids))
        return list(reversed(row_ids))

    @staticmethod
    def _workload_type(features: dict[str, Any] | None) -> str | None:
        """Extract the normalized workload type key used by the v0 retriever."""
        if not features:
            return None
        value = features.get("type") or features.get("workload_type")
        return str(value).lower() if value is not None else None

    def get_rows_for_edge(self, edge_id: str, limit: int | None = None) -> list[EvidenceRow]:
        """Return rows whose ICP result touched edge_id."""
        row_ids = self._by_edge.get(edge_id, [])
        if limit is not None:
            row_ids = row_ids[-limit:]
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_for_mechanism(
        self,
        mechanism_id: str,
        limit: int | None = None,
    ) -> list[EvidenceRow]:
        """Return rows where the mechanism was applicable to the rank."""
        row_ids = self._by_mechanism.get(mechanism_id, [])
        if limit is not None:
            row_ids = row_ids[-limit:]
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_for_job(self, job_id: str) -> list[EvidenceRow]:
        """Return all rows for one job id."""
        row_ids = self._by_job.get(job_id, [])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_for_environment(self, envs: Any) -> list[EvidenceRow]:
        """Return rows observed in one ICP environment label."""
        row_ids = self._by_env.get(envs, [])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_recently_decided(self, window: int) -> list[EvidenceRow]:
        """Return rows in the last window ticks, regardless of Q decision state."""
        rows = []
        cutoff = max(0, self._current_tick - window)
        for tick in range(cutoff, self._current_tick + 1):
            for row_id in self._by_tick.get(tick, []):
                row = self._row_by_id[row_id]
                rows.append(row)
        return rows

    def count_visits_per_edge(self, edge_id: str) -> int:
        """Return row count indexed to one edge."""
        return len(self._by_edge.get(edge_id, []))

    def envs_for_edge(self, edge_id: str) -> set:
        """Return environments where an edge has evidence rows.

        EIG's validator-support gate needs the env set itself, not just the
        count, because a new candidate may add one more environment.
        """
        return {self._row_by_id[row_id].env_label for row_id in self._by_edge.get(edge_id, [])}

    def count_envs_per_edge(self, edge_id: str) -> int:
        """Return distinct environment count for one edge."""
        return len(self.envs_for_edge(edge_id))

    def current_tick(self) -> int:
        """Return the latest tick observed by the store."""
        return self._current_tick

    def get_residual_history_per_v(self, v_name: str, window: int) -> np.ndarray:
        """Return concatenated V residuals for CUSUM recalibration."""
        return self._residual_history(v_name, window, "residuals_per_v")

    def get_residual_history_per_y(self, y_name: str, window: int) -> np.ndarray:
        """Return concatenated Y residuals for CUSUM recalibration."""
        return self._residual_history(y_name, window, "residuals_per_y")

    def _residual_history(self, name: str, window: int, field: str) -> np.ndarray:
        """Concatenate residual arrays from recent rows for one variable."""
        cutoff = max(0, self._current_tick - int(window))
        chunks = []
        for tick in range(cutoff, self._current_tick + 1):
            for row_id in self._by_tick.get(tick, ()):
                arr = getattr(self._row_by_id[row_id], field, {}).get(name)
                if arr is not None and len(arr) > 0:
                    chunks.append(np.asarray(arr, dtype=float))
        if not chunks:
            return np.array([], dtype=float)
        return np.concatenate(chunks)

    def last_touched_per_edge(self, edge_id: str) -> int | None:
        """Return latest tick for rows indexed to one edge, if any."""
        row_ids = self._by_edge.get(edge_id, [])
        if not row_ids:
            return None
        return max(self._row_by_id[row_id].tick for row_id in row_ids)

    def q3_rate_window(self, edge_id: str, window: tuple[int, int]) -> float:
        """Return fraction of decided edge rows with any Q3 mechanism label."""
        start_tick, end_tick = window
        q3_count = 0
        decided_count = 0

        for row_id in self._by_edge.get(edge_id, []):
            row = self._row_by_id[row_id]
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
        upper = self._current_tick if tick is None else int(tick)
        cutoff = max(0, upper - window)
        for t in range(cutoff, upper + 1):
            for row_id in self._by_tick.get(t, ()):
                row = self._row_by_id[row_id]
                for mid, q in row.q_label_per_mechanism.items():
                    if q is not None:
                        yield row, mid, q
