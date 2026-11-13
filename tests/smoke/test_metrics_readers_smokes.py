"""Test the Prometheus reader with local response objects, not a live service."""

import io
import json
import unittest
import urllib.error
import urllib.parse
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
from src.infra.metrics.backends.prometheus import PrometheusReader
from src.infra.metrics.base import MetricQueryError

START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(minutes=1)


def _payload(values=None):
    if values is None:
        values = [[START.timestamp(), "0"], [END.timestamp(), "20"]]
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [{"metric": {"job_id": "job_1", "rank_id": "rank_1"}, "values": values}],
        },
    }


def _reader(payload):
    opener = Mock(return_value=io.BytesIO(json.dumps(payload).encode()))
    return PrometheusReader("https://metrics.example", opener=opener), opener


class PrometheusReaderSmokeTests(unittest.TestCase):
    def test_posts_query_with_authentication_and_does_not_connect_at_startup(self):
        opener = Mock(return_value=io.BytesIO(json.dumps(_payload()).encode()))
        reader = PrometheusReader(
            "https://metrics.example/prefix/",
            timeout_seconds=3.0,
            bearer_token="test-token",
            opener=opener,
        )
        opener.assert_not_called()
        query = 'sum by (job_id, rank_id) (vllm:num_requests_waiting{job_id="job_1"})'

        result = reader.query_range(query, start=START, end=END, step_seconds=10)

        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://metrics.example/prefix/api/v1/query_range")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(request.get_header("Content-type"), "application/x-www-form-urlencoded")
        self.assertEqual(opener.call_args.kwargs, {"timeout": 3.0})
        self.assertEqual(
            urllib.parse.parse_qs(request.data.decode()),
            {
                "query": [query],
                "start": [str(START.timestamp())],
                "end": [str(END.timestamp())],
                "step": ["10"],
            },
        )
        self.assertEqual(len(result.series), 1)

    def test_returns_all_rank_series_without_chain_identifiers(self):
        payload = _payload()
        payload["data"]["result"].append(
            {
                "metric": {"job_id": "job_1", "rank_id": "rank_2"},
                "values": [[START.timestamp(), "10"], [END.timestamp(), "30"]],
            }
        )
        reader, _ = _reader(payload)

        result = reader.query_range("rank_query", start=START, end=END)

        self.assertEqual(
            [series.labels["rank_id"] for series in result.series], ["rank_1", "rank_2"]
        )
        self.assertEqual(result.series[0].values.dtype, np.dtype(np.float64))
        self.assertEqual(result.series[0].timestamps.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(result.series[0].values, [0.0, 20.0])
        np.testing.assert_array_equal(
            result.series[0].timestamps, [START.timestamp(), END.timestamp()]
        )
        self.assertIs(result.series[0].timestamps.base, result.series[0].values.base)
        self.assertEqual(result.warnings, ())

    def test_preserves_warnings_and_information(self):
        payload = _payload()
        payload.update(warnings=["partial response: cluster unavailable"], infos=["query note"])
        reader, _ = _reader(payload)

        result = reader.query_range("rank_query", start=START, end=END)

        self.assertEqual(result.warnings, ("partial response: cluster unavailable",))
        self.assertEqual(result.infos, ("query note",))

    def test_non_finite_samples_are_omitted_with_a_warning_not_replaced_by_zero(self):
        reader, _ = _reader(
            _payload(
                [
                    [START.timestamp(), "0"],
                    [START.timestamp() + 10, "NaN"],
                    [START.timestamp() + 20, "+Inf"],
                    [END.timestamp(), "20"],
                ]
            )
        )

        result = reader.query_range("rank_query", start=START, end=END)

        np.testing.assert_array_equal(result.series[0].values, [0.0, 20.0])
        np.testing.assert_array_equal(
            result.series[0].timestamps, [START.timestamp(), END.timestamp()]
        )
        self.assertEqual(result.warnings, ("Dropped 2 non-finite samples from a series",))

    def test_no_data_and_all_invalid_data_remain_distinct(self):
        payload = _payload()
        payload["data"]["result"] = []
        reader, _ = _reader(payload)
        empty = reader.query_range("rank_query", start=START, end=END)
        reader, _ = _reader(_payload([[START.timestamp(), "NaN"]]))
        invalid = reader.query_range("rank_query", start=START, end=END)

        self.assertEqual(empty.series, ())
        self.assertEqual(empty.warnings, ())
        self.assertEqual(invalid.series, ())
        self.assertTrue(invalid.warnings)

    def test_query_errors_and_malformed_responses_are_not_empty_results(self):
        for payload in (
            {"status": "error", "error": "bad query"},
            [],
            {"status": "success", "data": {"resultType": "vector", "result": []}},
            {"status": "success", "data": {"resultType": "matrix", "result": {}}},
        ):
            with self.subTest(payload=payload):
                reader, _ = _reader(payload)
                with self.assertRaises(MetricQueryError):
                    reader.query_range("rank_query", start=START, end=END)

    def test_transport_and_json_errors_are_reported(self):
        for error in (urllib.error.URLError("unreachable"), TimeoutError("timeout")):
            with self.subTest(error=error):
                reader = PrometheusReader("https://metrics.example", opener=Mock(side_effect=error))
                with self.assertRaisesRegex(MetricQueryError, "request failed"):
                    reader.query_range("rank_query", start=START, end=END)
        reader = PrometheusReader(
            "https://metrics.example", opener=Mock(return_value=io.BytesIO(b"not JSON"))
        )
        with self.assertRaisesRegex(MetricQueryError, "request failed"):
            reader.query_range("rank_query", start=START, end=END)

    def test_invalid_query_arguments_do_not_send_requests(self):
        for changes in (
            {"query": " "},
            {"start": START.replace(tzinfo=None)},
            {"end": START - timedelta(seconds=1)},
            {"step_seconds": 0},
            {"step_seconds": -1},
            {"step_seconds": True},
            {"step_seconds": 1.5},
        ):
            with self.subTest(changes=changes):
                reader, opener = _reader(_payload())
                arguments = {"query": "rank_query", "start": START, "end": END}
                arguments.update(changes)
                with self.assertRaises(ValueError):
                    reader.query_range(**arguments)
                opener.assert_not_called()

    def test_configuration_rejects_invalid_urls_and_timeouts(self):
        for url in (
            "",
            "metrics.example",
            "file:///tmp/metrics",
            "https://user:secret@metrics.example",
            "https://metrics.example?setting=value",
            "https://metrics.example#fragment",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                PrometheusReader(url)
        for timeout in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                PrometheusReader("https://metrics.example", timeout_seconds=timeout)

    def test_invalid_authentication_does_not_expose_the_token(self):
        for token in ("", "private\ntoken", "private\rtoken", True):
            with self.subTest(token=token):
                opener = Mock()
                with self.assertRaises(ValueError) as error:
                    PrometheusReader("https://metrics.example", bearer_token=token, opener=opener)
                self.assertNotIn("private", str(error.exception))
                opener.assert_not_called()

    def test_default_reader_does_not_forward_credentials_through_redirects(self):
        with patch("urllib.request.build_opener") as build_opener:
            build_opener.return_value.open.return_value = io.BytesIO(
                json.dumps(_payload()).encode()
            )
            reader = PrometheusReader("https://metrics.example", bearer_token="test-token")
            reader.query_range("rank_query", start=START, end=END)

        redirect_handler = build_opener.call_args.args[0]
        request = build_opener.return_value.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertIsNone(
            redirect_handler.redirect_request(
                request, None, 302, "Found", {}, "https://other.example"
            )
        )

    def test_aware_times_are_converted_to_epoch_seconds(self):
        reader, opener = _reader(_payload())
        local_zone = timezone(timedelta(hours=5))

        reader.query_range(
            "rank_query", start=START.astimezone(local_zone), end=END.astimezone(local_zone)
        )

        body = urllib.parse.parse_qs(opener.call_args.args[0].data.decode())
        self.assertEqual(body["start"], [str(START.timestamp())])
        self.assertEqual(body["end"], [str(END.timestamp())])

    def test_invalid_samples_and_times_are_rejected(self):
        for values in (
            [[START.timestamp()]],
            [[START.timestamp(), "not-a-number"]],
            [[START.timestamp(), True]],
            [[START.timestamp() - 1, "10"]],
            [[START.timestamp(), "10"], [START.timestamp(), "20"]],
            [[END.timestamp(), "10"], [START.timestamp(), "20"]],
        ):
            with self.subTest(values=values):
                reader, _ = _reader(_payload(values))
                with self.assertRaises(MetricQueryError):
                    reader.query_range("rank_query", start=START, end=END)

    def test_invalid_labels_messages_and_raw_histograms_are_rejected(self):
        for changes in (
            {"metric": {"job_id": 123}},
            {"values": "not-a-list"},
            {"histograms": [[START.timestamp(), {"count": "1"}]]},
        ):
            with self.subTest(changes=changes):
                payload = _payload()
                payload["data"]["result"][0].update(changes)
                reader, _ = _reader(payload)
                with self.assertRaises(MetricQueryError):
                    reader.query_range("rank_query", start=START, end=END)
        for name in ("warnings", "infos"):
            reader, _ = _reader({**_payload(), name: "not-a-list"})
            with self.subTest(name=name), self.assertRaises(MetricQueryError):
                reader.query_range("rank_query", start=START, end=END)


if __name__ == "__main__":
    unittest.main()
