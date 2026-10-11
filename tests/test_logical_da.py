"""Logical DA end-to-end: verified augmentation on syllogism-style samples
with the K3 workflow and the Z3 symbolic backend.

Gated by DAYAN_RUN_E2E=1.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

_ENABLED = os.environ.get("DAYAN_RUN_E2E") == "1"
REPORT = Path(__file__).parent.parent / "benchmarks" / "logical_da_rte.json"

_ROWS = [
    {"id": "r1", "premise": "All cats are animals.", "hypothesis": "Tom is an animal, given that Tom is a cat.", "label": "True"},
    {"id": "r2", "premise": "No dog can fly.", "hypothesis": "Rex can fly, and Rex is a dog.", "label": "False"},
    {"id": "r3", "premise": "Every mammal breathes air.", "hypothesis": "Whales breathe air because whales are mammals.", "label": "True"},
    {"id": "r4", "premise": "Some birds sing.", "hypothesis": "All birds sing.", "label": "False"},
]


@unittest.skipUnless(_ENABLED, "requires DAYAN_RUN_E2E=1")
class LogicalDaE2ETests(unittest.TestCase):
    def test_verified_augmentation_with_z3(self) -> None:
        from dayan.augment import PipelineConfig, SimpleWorkflow, run_dataset
        from dayan.augment.data import Dataset
        from dayan.llm import AnthropicClient

        path = Path("benchmarks/data/rte_toy.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_ROWS, ensure_ascii=False), encoding="utf-8")
        dataset = Dataset(path, list(_ROWS), "json")

        workflow = SimpleWorkflow(AnthropicClient(), solver="z3")
        result = run_dataset(
            dataset,
            workflow,
            config=PipelineConfig(symbolic_solver="z3", initial_candidates=3, seed_count=2),
        )
        report = result.report
        print("\n[logical-da] report:", json.dumps(report, ensure_ascii=False)[:600])
        self.assertGreaterEqual(report["processed_records"], 1)
        self.assertGreaterEqual(report["generated_records"], 1)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            json.dumps(
                {
                    "report": report,
                    "audit": result.audit[:40],
                    "output_rows": result.rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # Z3 must have participated somewhere in the audit (symbolic path).
        audit_text = json.dumps(result.audit, ensure_ascii=False)
        self.assertIn("z3", audit_text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
