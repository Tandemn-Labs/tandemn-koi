from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from src.core.models import EvidenceRow

# TODO - Will have to modify this to work with TandemnStore


class EvidenceService:
    def __init__(self):
        self._row_by_id = {}
        self._by_tick = defaultdict(list)
        self._by_job = defaultdict(list)
        self._by_rank = defaultdict(list)
        self._by_mechanism = defaultdict(list)
        self._by_edge = defaultdict(list)
        self._by_env = defaultdict(list)
        self._current_tick = 0

    def append_row(self, row: EvidenceRow):
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
        for mechanism_id in row.mechanism_ids:
            self._by_mechanism[mechanism_id].append(row.row_id)
        for edge_id in row.icp_result_per_edge:
            self._by_edge[edge_id].append(row.row_id)
        self._current_tick = max(self._current_tick, row.tick)
        return row.row_id

    def get_row(self, job_id, rank_id) -> list[EvidenceRow]:
        # Placeholder: return EvidenceRow for (job_id, rank_id)
        row_ids = self._by_rank.get((job_id, rank_id), [])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_in_window(self, window) -> list[EvidenceRow]:
        # Placeholder: return all EvidenceRows within the given time window
        # Window is an object that is (start_tick, end_tick)
        start_tick, end_tick = window
        rows = []
        for tick in range(start_tick, end_tick + 1):
            row_ids = self._by_tick.get(tick, [])
            for row_id in row_ids:
                rows.append(self._row_by_id[row_id])
        return rows

    def get_rows_for_edge(self, edge_id, limit=None) -> list[EvidenceRow]:
        # TODO - Data type for Limit is Int
        row_ids = self._by_edge.get(edge_id, [])
        if limit is not None:
            row_ids = row_ids[-limit:]
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_for_mechanism(self, mechanism_id, limit=None) -> list[EvidenceRow]:
        # Placeholder: return EvidenceRows for all ranks with this mechanism attached
        row_ids = self._by_mechanism.get(mechanism_id, [])
        if limit is not None:
            row_ids = row_ids[-limit:]
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_for_job(self, job_id) -> list[EvidenceRow]:
        row_ids = self._by_job.get(job_id, [])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_rows_for_environment(self, envs) -> list[EvidenceRow]:
        row_ids = self._by_env.get(envs, [])
        return [self._row_by_id[row_id] for row_id in row_ids]

    def get_recently_decided(self, window) -> list[EvidenceRow]:
        rows = []
        cutoff = max(0, self._current_tick - window)
        for tick in range(cutoff, self._current_tick + 1):
            for row_id in self._by_tick.get(tick, []):
                row = self._row_by_id[row_id]
                rows.append(row)
        return rows

    def count_visits_per_edge(self, edge_id) -> int:
        return len(self._by_edge.get(edge_id, []))

    def envs_for_edge(self, edge_id) -> set:
        # eig.py gate_validator_support needs the env set itself, not just the count
        return {self._row_by_id[row_id].env_label for row_id in self._by_edge.get(edge_id, [])}

    def count_envs_per_edge(self, edge_id) -> int:
        return len(self.envs_for_edge(edge_id))

    def current_tick(self) -> int:
        # regret.py and agent tools read the store's notion of "now"
        return self._current_tick

    def get_residual_history_per_v(self, v_name, window) -> np.ndarray:
        # concatenated residuals for one V over the last `window` ticks;
        # feeds slow_loop.recalibrate_cusum_params (delta, h) calibration
        return self._residual_history(v_name, window, "residuals_per_v")

    def get_residual_history_per_y(self, y_name, window) -> np.ndarray:
        return self._residual_history(y_name, window, "residuals_per_y")

    def _residual_history(self, name, window, field) -> np.ndarray:
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

    def last_touched_per_edge(self, edge_id) -> int | None:
        row_ids = self._by_edge.get(edge_id, [])
        if not row_ids:
            return None
        return max(self._row_by_id[row_id].tick for row_id in row_ids)

    def q3_rate_window(self, edge_id, window) -> float:
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
        # tick=None means "as of now" - the store substitutes its own current tick
        upper = self._current_tick if tick is None else int(tick)
        cutoff = max(0, upper - window)
        for t in range(cutoff, upper + 1):
            for row_id in self._by_tick.get(t, ()):
                row = self._row_by_id[row_id]
                for mid, q in row.q_label_per_mechanism.items():
                    if q is not None:
                        yield row, mid, q


if __name__ == "__main__":
    import numpy as np
    from src.validation.icp import ICPResult
    from src.validation.quadrants import Quadrant

    def make_row(
        row_id,
        tick,
        job_id,
        rank_id,
        env_label,
        mechanism_ids,
        icp_result_per_edge,
        q_label_per_mechanism,
    ):
        return EvidenceRow(
            row_id=row_id,
            tick=tick,
            deploy_timestamp_utc=float(tick),
            job_id=job_id,
            rank_id=rank_id,
            env_label=env_label,
            X={"batch_size": 8},
            W_observed={"request_rate": 10.0},
            V_observed_trajectory={"kv_cache_pressure": np.array([0.2, 0.3])},
            V_predicted_trajectory={"kv_cache_pressure": np.array([0.2, 0.2])},
            y_observed_trajectory={"ttft_ms": np.array([100.0, 110.0])},
            y_predicted={"ttft_ms": 100.0},
            y_observed_mean={"ttft_ms": 105.0},
            residuals_per_v={"kv_cache_pressure": np.array([0.0, 0.1])},
            residuals_per_y={"ttft_ms": np.array([0.0, 10.0])},
            mechanism_ids=mechanism_ids,
            cusum_per_mechanism={},
            q_label_per_mechanism=q_label_per_mechanism,
            icp_result_per_edge=icp_result_per_edge,
            w_t_snapshot={"ttft_ms": 1.0},
            z_star_snapshot={"ttft_ms": 100.0},
            J_realized=-5.0,
            sigma_realized=1.0,
        )

    store = EvidenceService()
    env_a = ("aws", "us-east-1", "on_demand", "H100")
    env_b = ("aws", "us-west-2", "spot", "H100")

    row_1 = make_row(
        row_id="row_1",
        tick=1,
        job_id="job_1",
        rank_id="rank_1",
        env_label=env_a,
        mechanism_ids=["M1", "M2"],
        icp_result_per_edge={"e1": ICPResult.ACCEPT, "e2": ICPResult.UNDECIDED},
        q_label_per_mechanism={"M1": Quadrant.Q1, "M2": None},
    )
    row_2 = make_row(
        row_id="row_2",
        tick=2,
        job_id="job_1",
        rank_id="rank_1",
        env_label=env_b,
        mechanism_ids=["M1"],
        icp_result_per_edge={"e1": ICPResult.REJECT},
        q_label_per_mechanism={"M1": Quadrant.Q3},
    )
    row_3 = make_row(
        row_id="row_3",
        tick=3,
        job_id="job_2",
        rank_id="rank_2",
        env_label=env_a,
        mechanism_ids=["M2"],
        icp_result_per_edge={"e2": ICPResult.REJECT},
        q_label_per_mechanism={"M2": Quadrant.Q4},
    )

    print("append_row:", store.append_row(row_1), store.append_row(row_2), store.append_row(row_3))
    print("get_row(job_1, rank_1):", [row.row_id for row in store.get_row("job_1", "rank_1")])
    print("get_rows_in_window(1, 2):", [row.row_id for row in store.get_rows_in_window((1, 2))])
    print("get_rows_for_edge(e1):", [row.row_id for row in store.get_rows_for_edge("e1")])
    print(
        "get_rows_for_edge(e1, limit=1):",
        [row.row_id for row in store.get_rows_for_edge("e1", limit=1)],
    )
    print("get_rows_for_mechanism(M1):", [row.row_id for row in store.get_rows_for_mechanism("M1")])
    print("get_rows_for_job(job_1):", [row.row_id for row in store.get_rows_for_job("job_1")])
    print(
        "get_rows_for_environment(env_a):",
        [row.row_id for row in store.get_rows_for_environment(env_a)],
    )
    print("get_recently_decided(1):", [row.row_id for row in store.get_recently_decided(1)])
    print("count_visits_per_edge(e1):", store.count_visits_per_edge("e1"))
    print("count_envs_per_edge(e1):", store.count_envs_per_edge("e1"))
    print("last_touched_per_edge(e1):", store.last_touched_per_edge("e1"))
    print("q3_rate_window(e1, (1, 3)):", store.q3_rate_window("e1", (1, 3)))
    print(
        "iter_decided_per_mechanism(3, 3):",
        [(row.row_id, mid, q) for row, mid, q in store.iter_decided_per_mechanism(3, 3)],
    )

    assert [row.row_id for row in store.get_row("job_1", "rank_1")] == ["row_1", "row_2"]
    assert [row.row_id for row in store.get_rows_in_window((1, 2))] == ["row_1", "row_2"]
    assert [row.row_id for row in store.get_rows_for_edge("e1")] == ["row_1", "row_2"]
    assert [row.row_id for row in store.get_rows_for_edge("e1", limit=1)] == ["row_2"]
    assert [row.row_id for row in store.get_rows_for_mechanism("M1")] == ["row_1", "row_2"]
    assert [row.row_id for row in store.get_rows_for_job("job_1")] == ["row_1", "row_2"]
    assert [row.row_id for row in store.get_rows_for_environment(env_a)] == ["row_1", "row_3"]
    assert [row.row_id for row in store.get_recently_decided(1)] == ["row_2", "row_3"]
    assert store.count_visits_per_edge("e1") == 2
    assert store.count_envs_per_edge("e1") == 2
    assert store.last_touched_per_edge("e1") == 2
    assert store.q3_rate_window("e1", (1, 3)) == 0.5
    assert [(row.row_id, mid, q) for row, mid, q in store.iter_decided_per_mechanism(3, 3)] == [
        (row_1.row_id, "M1", Quadrant.Q1),
        (row_2.row_id, "M1", Quadrant.Q3),
        (row_3.row_id, "M2", Quadrant.Q4),
    ]
    print("All EvidenceService smoke tests passed.")
