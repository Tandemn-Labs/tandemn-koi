"""Check the shared observation types without changing the legacy collector."""

import unittest
from typing import get_type_hints

import numpy as np
from src.core.causal_graph_store_adapter import (
    DEFAULT_EDGE_TABLE_PATH,
    parse_seed_candidate_graph,
)
from src.core.telemetry_models import ObservedV, ObservedY
from src.infra.telemetry import RankTelemetry


class TelemetryModelsSmokeTests(unittest.TestCase):
    def test_all_graph_observations_are_optional_array_fields(self):
        graph = parse_seed_candidate_graph(DEFAULT_EDGE_TABLE_PATH)

        for schema, names in ((ObservedV, graph.v), (ObservedY, graph.y)):
            with self.subTest(schema=schema.__name__):
                hints = get_type_hints(schema)
                self.assertEqual(set(hints), set(names))
                self.assertEqual(schema.__required_keys__, frozenset())
                self.assertEqual(schema.__optional_keys__, frozenset(names))
                for name, value_type in hints.items():
                    with self.subTest(field=name):
                        self.assertIs(value_type, np.ndarray)

    def test_empty_observations_are_ordinary_dictionaries(self):
        for observations in (ObservedV(), ObservedY()):
            self.assertIsInstance(observations, dict)
            self.assertEqual(observations, {})

    def test_partial_observations_fit_existing_rank_output_without_copying_arrays(self):
        queue = np.array([0.0, 2.0])
        throughput = np.array([0.0, 20.0])
        v: ObservedV = {"depth_req_q": queue}
        y: ObservedY = {"throughput_token_per_sec": throughput}

        rank = RankTelemetry(
            job_id="job_1",
            rank_id="rank_1",
            v_observed=dict(v),
            v_predicted={},
            y_observed=dict(y),
            y_predicted={},
        )

        self.assertIs(rank.v_observed["depth_req_q"], queue)
        self.assertIs(rank.y_observed["throughput_token_per_sec"], throughput)
        np.testing.assert_array_equal(rank.v_observed["depth_req_q"], [0.0, 2.0])
        np.testing.assert_array_equal(rank.y_observed["throughput_token_per_sec"], [0.0, 20.0])
        self.assertEqual(set(rank.v_observed), {"depth_req_q"})
        self.assertEqual(set(rank.y_observed), {"throughput_token_per_sec"})
        self.assertEqual(rank.v_predicted, {})
        self.assertEqual(rank.y_predicted, {})


if __name__ == "__main__":
    unittest.main()
