"""Direct LLM workflow for the Logical DA pipeline (CrewAI-free).

Each stage renders the same prompts as the CrewAI version and parses one
JSON object from the response, with a single reformat repair retry — the
orchestration the pipeline needs, without the framework dependency.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dayan.augment.data import Schema
from dayan.augment.filtering import Attribute
from dayan.augment.prompts import (
    extraction_prompt,
    filtering_prompt,
    generation_prompt,
    verification_prompt,
)
from dayan.llm import LLM


class AgentResponseError(ValueError):
    """A task completed without a usable JSON object."""


def parse_json_payload(value: Any) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""

    if isinstance(value, dict):
        return value
    raw = value if isinstance(value, str) else str(value)
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for fenced in re.findall(r"^```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL | re.MULTILINE):
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AgentResponseError("LLM response did not contain a JSON object")


class SimpleWorkflow:
    """Workflow protocol implementation over any injected LLM."""

    def __init__(self, llm: LLM, *, solver: str = "auto") -> None:
        self.llm = llm
        self.solver = solver

    def _run(self, prompt: str) -> dict[str, Any]:
        raw = self.llm.chat(prompt, temperature=0.0)
        try:
            return parse_json_payload(raw)
        except AgentResponseError:
            repair = (
                "Reformat the following response into exactly one valid JSON "
                "object. Preserve its content and identifiers. Return JSON only.\n\n"
                + str(raw)[:30000]
            )
            return parse_json_payload(self.llm.chat(repair, temperature=0.0))

    def extract(self, source: dict[str, Any], schema: Schema) -> dict[str, Any]:
        return self._run(extraction_prompt(source, schema))

    def filter_combinations(
        self,
        source: dict[str, Any],
        schema: Schema,
        attributes: list[Attribute],
        combinations: list[dict[str, str]],
        limit: int,
    ) -> dict[str, Any]:
        return self._run(filtering_prompt(source, schema, attributes, combinations, limit))

    def generate(
        self,
        source: dict[str, Any],
        schema: Schema,
        requests: list[dict[str, Any]],
        stage: str,
    ) -> dict[str, Any]:
        return self._run(generation_prompt(source, schema, requests, stage))

    def verify(
        self,
        candidates: list[dict[str, Any]],
        schema: Schema,
        labels: tuple[str, ...],
    ) -> dict[str, Any]:
        return self._run(verification_prompt(candidates, schema, labels, solver=self.solver))


__all__ = ["AgentResponseError", "SimpleWorkflow", "parse_json_payload"]
