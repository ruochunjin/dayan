"""LLM interface for DaYan: protocol, stdlib OpenAI-compatible client, FakeLLM.

Same conventions as the sibling tools: environment-driven configuration
(``DAYAN_API_KEY`` / ``DAYAN_BASE_URL`` / ``DAYAN_MODEL``, falling back to
``OPENAI_*``), plain HTTP (no SDK), nothing created at import time.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable, Protocol

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class LLM(Protocol):
    def chat(self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        """Single-turn completion of one prompt."""
        ...


class ChatClient:
    """Stdlib-only OpenAI-compatible chat client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        url = (
            base_url
            or os.environ.get("DAYAN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        self.url = url
        self.api_key = api_key or os.environ.get("DAYAN_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model or os.environ.get("DAYAN_MODEL") or os.environ.get("OPENAI_MODEL") or ""
        if not self.model:
            raise ValueError("Set DAYAN_MODEL (or pass model=) to the endpoint's model name")
        self.timeout = timeout or float(os.environ.get("DAYAN_TIMEOUT", "180"))
        self.retries = retries or int(os.environ.get("DAYAN_RETRIES", "3"))

    def chat(self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                return payload["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                last = error
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries} retries: {last}")


class FakeLLM:
    """Scripted LLM for offline tests: regex rules tried in order."""

    def __init__(
        self,
        rules: list[tuple[str, str | Callable[[str], str]]] | None = None,
        default: str = "{}",
    ) -> None:
        self.rules = [(re.compile(p, re.S | re.I), r) for p, r in (rules or [])]
        self.default = default
        self.calls: list[str] = []

    def chat(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        for pattern, response in self.rules:
            if pattern.search(prompt):
                return response(prompt) if callable(response) else response
        return self.default


__all__ = ["ChatClient", "FakeLLM", "LLM"]
