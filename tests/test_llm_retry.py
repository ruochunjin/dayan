"""Retry/backoff behavior of the AnthropicClient (offline, mocked transport).

GSM8K-30 batch hit the endpoint's 403 rate limit after the default 3
retries while a sibling benchmark ran concurrently — the same failure the
jiujiubiao client was hardened against (retries 3 -> 6).  Pinned here.
"""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest import mock

from dayan.llm import AnthropicClient


def _client() -> AnthropicClient:
    return AnthropicClient(base_url="https://example.test", api_key="k", model="m")


def _ok_response(text: str = "pong"):
    payload = json.dumps({"content": [{"type": "text", "text": text}]}).encode()
    response = mock.MagicMock()
    response.read.return_value = payload
    response.__enter__ = lambda s: s
    response.__exit__ = mock.Mock(return_value=False)
    return response


def _http_403() -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, None)


class RetryTests(unittest.TestCase):
    def test_default_retries_is_six(self) -> None:
        self.assertEqual(_client().retries, 6)

    def test_recovers_from_repeated_403(self) -> None:
        client = _client()
        with mock.patch("urllib.request.urlopen") as urlopen, mock.patch(
            "time.sleep"
        ) as sleep:
            urlopen.side_effect = [_http_403(), _http_403(), _http_403(), _ok_response()]
            self.assertEqual(client.chat("hi"), "pong")
            # backoff waits happened for each failure
            self.assertEqual(sleep.call_count, 3)

    def test_exhaustion_raises_runtime_error(self) -> None:
        client = AnthropicClient(
            base_url="https://example.test", api_key="k", model="m", retries=2
        )
        with mock.patch("urllib.request.urlopen") as urlopen, mock.patch("time.sleep"):
            urlopen.side_effect = [_http_403(), _http_403()]
            with self.assertRaises(RuntimeError):
                client.chat("hi")


if __name__ == "__main__":
    unittest.main()
