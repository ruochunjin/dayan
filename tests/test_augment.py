"""Offline tests for the augment subpackage (Logical DA)."""

from __future__ import annotations

import unittest
from pathlib import Path

from dayan.augment.data import Dataset, infer_schema
from dayan.augment.workflow import SimpleWorkflow, parse_json_payload
from dayan.llm import FakeLLM


class ParseJsonTests(unittest.TestCase):
    def test_plain_and_fenced_json(self) -> None:
        self.assertEqual(parse_json_payload('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_payload('```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(parse_json_payload('prose {"a": 3} epilogue'), {"a": 3})

    def test_reject_non_json(self) -> None:
        with self.assertRaises(Exception):
            parse_json_payload("no json here")


class WorkflowTests(unittest.TestCase):
    def test_extract_with_fake_llm(self) -> None:
        llm = FakeLLM(
            rules=[(r".*", '{"attributes": [{"name": "color", "values": ["red", "blue"]}]}')],
        )
        workflow = SimpleWorkflow(llm)
        schema = infer_schema(Dataset(Path("toy.json"), [{"text": "All cats are animals.", "label": "True"}], "json"))
        out = workflow.extract({"text": "All cats are animals.", "label": "True"}, schema)
        self.assertIn("attributes", out)

    def test_repair_retry_on_bad_json(self) -> None:
        llm = FakeLLM(
            rules=[
                (r"Reformat the following", '{"fixed": true}'),
                (r".*", "not json"),
            ]
        )
        workflow = SimpleWorkflow(llm)
        out = workflow.extract({"text": "x", "label": "True"}, _schema())
        self.assertEqual(out, {"fixed": True})


def _schema():
    return infer_schema(Dataset(Path("toy.json"), [{"text": "x", "label": "True"}], "json"))


class DatasetTests(unittest.TestCase):
    def test_infer_schema_nli(self) -> None:
        schema = infer_schema(
            Dataset(Path("toy.json"), [{"premise": "p", "hypothesis": "h", "label": "True"}], "json")
        )
        self.assertEqual(schema.kind, "nli")


if __name__ == "__main__":
    unittest.main(verbosity=2)
