"""Offline unit tests for api_client (no internet needed).  Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api_client as api  # noqa: E402


def resp(status=200, json_body=None, text="", headers=None, bad_json=False):
    r = mock.Mock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    r.reason = "reason"
    r.text = text
    if bad_json:
        r.json.side_effect = ValueError("Expecting value")
    else:
        r.json.return_value = json_body
    return r


def session_with(*outcomes):
    s = mock.Mock()
    s.get.side_effect = list(outcomes)
    return s


class TestSettings(unittest.TestCase):
    def test_timeouts_and_attempts(self):
        self.assertEqual(api.TIMEOUT, (20, 90))
        self.assertEqual(api.MAX_ATTEMPTS, 3)

    def test_backoff_is_exponential(self):
        self.assertEqual([api._backoff_delay(i) for i in (1, 2, 3)], [2, 4, 8])


class TestRetry(unittest.TestCase):
    def setUp(self):
        api._force_ipv4 = False

    def test_success_first_try(self):
        s = session_with(resp(json_body={"ok": 1}))
        sleeps = []
        self.assertEqual(api.get_json("https://x.test/a", session=s, sleep=sleeps.append), {"ok": 1})
        self.assertEqual(sleeps, [])
        self.assertEqual(s.get.call_args.kwargs["timeout"], (20, 90))

    def test_recovers_after_connect_timeouts(self):
        s = session_with(requests.exceptions.ConnectTimeout("t"), requests.exceptions.ConnectTimeout("t"),
                         resp(json_body={"ok": 1}))
        sleeps = []
        self.assertEqual(api.get_json("https://x.test/a", session=s, sleep=sleeps.append), {"ok": 1})
        self.assertEqual(sleeps, [2, 4])           # exponential backoff
        self.assertEqual(s.get.call_count, 3)

    def test_gives_up_after_three_attempts_connectivity(self):
        s = session_with(*[requests.exceptions.ConnectTimeout("t")] * 5)
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None, api_name="Weather API")
        self.assertEqual(s.get.call_count, 3)
        self.assertEqual(cm.exception.kind, "connectivity")
        self.assertEqual(cm.exception.attempts, 3)

    def test_connection_error_is_connectivity(self):
        s = session_with(*[requests.exceptions.ConnectionError("dns")] * 3)
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "connectivity")

    def test_read_timeout_is_availability(self):
        s = session_with(*[requests.exceptions.ReadTimeout("slow")] * 3)
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "availability")

    def test_http_500_retried_then_availability(self):
        s = session_with(*[resp(503, text="down")] * 3)
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "availability")
        self.assertEqual(s.get.call_count, 3)

    def test_http_429_uses_retry_after(self):
        s = session_with(resp(429, headers={"Retry-After": "7"}), resp(json_body={"ok": 1}))
        sleeps = []
        api.get_json("https://x.test/a", session=s, sleep=sleeps.append)
        self.assertEqual(sleeps, [7.0])

    def test_http_400_not_retried_and_reason_shown(self):
        s = session_with(resp(400, json_body={"error": True, "reason": "Latitude must be in range"}))
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "request")
        self.assertIn("Latitude", cm.exception.message)
        self.assertEqual(s.get.call_count, 1)

    def test_error_flag_in_200_body(self):
        s = session_with(resp(200, json_body={"error": True, "reason": "bad"}))
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "request")

    def test_invalid_json_retried_then_data_error(self):
        s = session_with(*[resp(200, bad_json=True)] * 3)
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "data")
        self.assertEqual(s.get.call_count, 3)

    def test_invalid_json_then_recovery(self):
        s = session_with(resp(200, bad_json=True), resp(json_body={"ok": 1}))
        self.assertEqual(api.get_json("https://x.test/a", session=s, sleep=lambda _: None), {"ok": 1})

    def test_validator_failure_is_data_error_no_retry(self):
        s = session_with(resp(json_body={"nope": 1}))
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None, validator=api._validate_weather)
        self.assertEqual(cm.exception.kind, "data")
        self.assertEqual(s.get.call_count, 1)

    def test_ssl_error_is_connectivity(self):
        s = session_with(*[requests.exceptions.SSLError("cert")] * 3)
        with self.assertRaises(api.APIError) as cm:
            api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertEqual(cm.exception.kind, "connectivity")
        self.assertIn("SSL", cm.exception.message)

    def test_ipv4_fallback_enabled_after_connect_failure(self):
        api._force_ipv4 = False
        s = session_with(requests.exceptions.ConnectTimeout("t"), resp(json_body={"ok": 1}))
        api.get_json("https://x.test/a", session=s, sleep=lambda _: None)
        self.assertTrue(api._force_ipv4)
        api._force_ipv4 = False


class TestEndpoints(unittest.TestCase):
    def test_geocode_not_found(self):
        with mock.patch.object(api, "get_json", return_value={}):
            with self.assertRaises(ValueError):
                api.geocode("Nowhereville")

    def test_geocode_ok(self):
        with mock.patch.object(api, "get_json", return_value={"results": [{"name": "Ludhiana"}]}):
            self.assertEqual(api.geocode("Ludhiana")["name"], "Ludhiana")

    def test_each_endpoint_uses_correct_url_and_validator(self):
        with mock.patch.object(api, "get_json", return_value={}) as g:
            api.fetch_weather(30.9, 75.85)
            api.fetch_air(30.9, 75.85)
            api.fetch_history(30.9, 75.85, "2026-01-31")
        urls = [c.args[0] for c in g.call_args_list]
        self.assertEqual(urls, [api.FORECAST_URL, api.AIR_URL, api.ARCHIVE_URL])
        hist_params = g.call_args_list[2].args[1]
        self.assertEqual(hist_params["end_date"], "2026-01-31")
        self.assertEqual(hist_params["start_date"], "2025-11-03")  # 89 days earlier -> 90 records


if __name__ == "__main__":
    unittest.main()
