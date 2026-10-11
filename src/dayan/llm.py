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


def _anthropic_credentials() -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model) for the Anthropic-messages backend.

    Environment first (``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_MODEL``); falls back to the Claude Code settings file's
    ``env`` block so a K3-style endpoint configured for Claude Code works
    in-process without exporting anything.  Never logged or printed.
    """

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL")
    if base_url and api_key and model:
        return base_url, api_key, model
    settings = os.path.join(os.path.expanduser("~"), ".claude", "settings.json")
    try:
        with open(settings, encoding="utf-8-sig") as handle:
            env = json.load(handle).get("env", {})
        base_url = base_url or env.get("ANTHROPIC_BASE_URL")
        api_key = api_key or env.get("ANTHROPIC_API_KEY")
        model = model or env.get("ANTHROPIC_MODEL")
    except (OSError, json.JSONDecodeError):
        pass
    if not (base_url and api_key and model):
        raise RuntimeError(
            "Anthropic backend needs ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY/ANTHROPIC_MODEL"
        )
    return base_url, api_key, model


class AnthropicClient:
    """Stdlib-only Anthropic messages client (e.g. a K3 coding endpoint).

    Extended thinking is explicitly disabled — the endpoint defaults to it,
    which eats the whole token budget on small requests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        if base_url and api_key and model:
            self.base_url, self.api_key, self.model = base_url, api_key, model
        else:
            self.base_url, self.api_key, self.model = _anthropic_credentials()
        self.url = self.base_url.rstrip("/") + "/v1/messages"
        self.timeout = timeout or float(os.environ.get("DAYAN_TIMEOUT", "180"))
        self.retries = retries or int(os.environ.get("DAYAN_RETRIES", "3"))

    def chat(self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or 4096,
            "thinking": {"type": "disabled"},
            "temperature": temperature,
        }
        data = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        import random

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                return "".join(
                    block.get("text", "")
                    for block in payload.get("content", [])
                    if block.get("type") == "text"
                )
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                last = error
                status = getattr(error, "code", None)
                if status in (403, 429, 500, 502, 503, 529):
                    time.sleep(min(90.0, 5.0 * (2 ** attempt)) + random.random())
                else:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries} retries: {last}")


def default_client(backend: str | None = None):
    """Build the environment-configured client (``DAYAN_BACKEND``).

    ``anthropic`` for a K3-style messages endpoint; ``openai`` (default) for
    any OpenAI-compatible endpoint.
    """

    name = (backend or os.environ.get("DAYAN_BACKEND") or "openai").lower()
    if name == "anthropic":
        return AnthropicClient()
    return ChatClient()


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


__all__ = ["AnthropicClient", "ChatClient", "FakeLLM", "LLM", "default_client"]
