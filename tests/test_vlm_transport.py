"""Network failures must be bounded, measured and secret-safe."""

import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/policies"))
from vlm_agent.vlm_client import VLMClient, VLMError


class TransportTest(unittest.TestCase):
    def test_explicit_medium_overrides_disabled_profile_thinking(self):
        client = self.client()
        client.extra_body = {"reasoning_effort": "none", "enable_thinking": False}
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
        with patch("urllib.request.urlopen", return_value=response) as request:
            client.complete([], reasoning_effort="medium")
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertTrue(payload["enable_thinking"])

    def client(self, **options):
        return VLMClient({"model": "test", "base_url": "https://example.invalid/v1", "api_key": "private-key"}, **options)

    def test_permanent_http_error_does_not_retry_or_echo_remote_body(self):
        failure = urllib.error.HTTPError("https://example.invalid", 401, "unauthorized", {}, io.BytesIO(b"private-key"))
        with patch("urllib.request.urlopen", side_effect=failure) as request, patch("time.sleep") as sleep:
            with self.assertRaises(VLMError) as caught:
                self.client(max_attempts=3).complete([])
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertIsNotNone(caught.exception.latency_s)
        self.assertNotIn("private-key", str(caught.exception))

    def test_bad_request_is_retried_and_its_body_is_kept_for_diagnosis(self):
        bad = urllib.error.HTTPError("https://example.invalid", 400, "bad request", {},
                                     io.BytesIO(b'{"error": "image too large private-key"}'))
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode())
        with patch("urllib.request.urlopen", side_effect=[bad, response]) as request, patch("time.sleep"):
            result = self.client(max_attempts=3).complete([])
        self.assertEqual(result["text"], "ok")
        self.assertEqual(request.call_count, 2)
        self.assertIn("image too large", result["retry_history"][0]["error"])
        self.assertNotIn("private-key", str(result))
        bad = [urllib.error.HTTPError("https://example.invalid", 400, "bad request", {}, io.BytesIO(b"nope"))
               for _ in range(2)]
        with patch("urllib.request.urlopen", side_effect=bad) as request, patch("time.sleep"):
            with self.assertRaises(VLMError) as caught:
                self.client(max_attempts=2).complete([])
        self.assertEqual(request.call_count, 2)
        self.assertTrue(caught.exception.retryable)
        self.assertIn("HTTP 400: nope", str(caught.exception))

    def test_error_body_redacts_keys_crossing_display_and_read_limits(self):
        for body, expected in (
            (b"x" * 295 + b"private-key", "HTTP 400: " + "x" * 295 + "<reda"),
            (b" " * 595 + b"private-key", "HTTP 400: <redacted>"),
        ):
            with self.subTest(body_length=len(body)):
                failure = urllib.error.HTTPError("https://example.invalid", 400, "bad request", {},
                                                 io.BytesIO(body))
                with patch("urllib.request.urlopen", side_effect=failure):
                    with self.assertRaises(VLMError) as caught:
                        self.client(max_attempts=1).complete([])
                self.assertEqual(caught.exception.retry_history[0]["error"], expected)
                self.assertTrue(str(caught.exception).endswith(expected))

    def test_default_timeout_retries_five_times_and_retains_failures(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("socket timed out")) as request, patch("time.sleep") as sleep:
            with self.assertRaises(VLMError) as caught:
                self.client().complete([])
        self.assertEqual(request.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual(caught.exception.attempts, 6)
        self.assertEqual(len(caught.exception.retry_history), 6)
        self.assertTrue(caught.exception.retryable)
        self.assertGreaterEqual(caught.exception.latency_s, 0)

    def test_rate_limit_then_network_failure_recovers_with_same_request(self):
        rate_limit = urllib.error.HTTPError("https://example.invalid", 429, "limited",
                                           {"Retry-After": "7"}, io.BytesIO(b"private-key"))
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"},
                                                       "finish_reason": "stop"}]}).encode())
        with patch("urllib.request.urlopen", side_effect=[rate_limit, ConnectionResetError(), response]) as request, \
                patch("time.sleep") as sleep:
            result = self.client().complete([])
        self.assertEqual(result["text"], "ok")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["retry_history"][0]["status_code"], 429)
        self.assertGreaterEqual(sleep.call_args_list[0].args[0], 7)
        self.assertEqual(len({call.args[0].data for call in request.call_args_list}), 1)
        self.assertNotIn("private-key", str(result))

    def test_expired_deadline_reports_zero_actual_requests(self):
        with patch("urllib.request.urlopen") as request:
            with self.assertRaises(VLMError) as caught:
                self.client(max_attempts=3).complete([], deadline=time.monotonic() - 1)
        request.assert_not_called()
        self.assertEqual(caught.exception.attempts, 0)

    def test_retry_backoff_cannot_exceed_remaining_budget(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()) as request, patch("time.sleep") as sleep:
            with self.assertRaises(VLMError):
                self.client(max_attempts=3, retry_delay_s=10).complete([], deadline=time.monotonic() + 1)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
