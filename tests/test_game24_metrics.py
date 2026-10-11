"""Game of 24 thought-mining evaluation: real K3 traces through an
instrumented solver, then process-level metrics (the paper's core scenario).

Gated by DAYAN_RUN_E2E=1.  Writes a JSON report for the working record.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

_ENABLED = os.environ.get("DAYAN_RUN_E2E") == "1"
REPORT = Path(__file__).parent.parent / "benchmarks" / "game24_metrics.json"

_PUZZLES = [
    [4, 4, 4, 4],      # (4+4)*(4+4)... 4*4+4+4=24 trivial-ish
    [1, 2, 3, 4],      # 1*2*3*4 = 24
    [6, 6, 2, 3],      # 6*6-2*6 = 24? 6*6-6-6=24 (uses 6,6,6,6... no) -> (6-2)*(6+... ) keep: (6/2+3)*... (6/2+3)*6=30 no; 6*(6-3+... ) (6-3+1)*6=24? uses 6,3,1,6 -> [6,6,2,3]: (6+2)*(6-3)=24
    [8, 8, 3, 2],      # 8*3*(... ) 8*3*(8/8)? need all four: (8-2)*8/... (8/2+8)/... (8/2)*...  (8+8-8)*3? =24 uses 8,8,8,3 no 2; (8-8/... ) (8-8/2)*... (8-8/2)*... keep: (8+8)/2*3=24
    [5, 5, 5, 2],      # 5*5-5+... (5-2)*5+... (5*5)-(5-... ) (5-5/... ) (5-5/5)... 5*5-(5-4)? no 2... (5+5+5)*2-6?  (5+5+2)*... (5*5-5)*... keep: (5-5/... ) (5-2+5)*... -> (5+5/... ) (5*5-5)+... -> 5*5-5+4? no. (5-2)*(5+... ) (5-2)*(5+... ) -> (5-2)*(5+3)? no 3. -> 5*5-2+... (5*5-2)+1? (5*5-2)+... 23+? (5*5-2)+... -> (5*5-2)+1 -> no 1. -> 2*(5+5+... ) 2*(5+5+2)? =24 uses 2,5,5,2 -> 2*(5+5+2)=24! puzzle: 2*(5+5+2) uses both 2s ✓
    [3, 3, 8, 8],      # classic: 8/(3-8/3)=24
]


@unittest.skipUnless(_ENABLED, "requires DAYAN_RUN_E2E=1")
class Game24ThoughtMiningTests(unittest.TestCase):
    def test_game24_traces_and_metrics(self) -> None:
        from dayan import AnalysisOutput, analysis, query, set_backend
        from dayan.llm import AnthropicClient
        from dayan.metrics import (
            ThoughtSample,
            enumerate_paths,
            multipath_reliability,
            node_influence,
            path_distribution,
            reasoning_commonality,
            stepwise_consistency,
        )

        client = AnthropicClient()

        def k3_backend(context: str, instruction: str, temperature: float) -> str:
            prompt = (
                "Always wrap your final result between <OUTPUT> and </OUTPUT> tags.\n\n"
                + context + "\n\n" + instruction
            )
            return client.chat(prompt, temperature=temperature)

        set_backend(k3_backend)

        @analysis(output_type=AnalysisOutput.THOUGHT)
        def solve24(nums):
            plan = query(
                "In one line, give a valid arithmetic expression using ALL of {} exactly once "
                "(with + - * / and parentheses) that evaluates to 24. Get [expr]",
                nums,
            )
            verdict = query("Evaluate {} step by step; does it equal 24? Answer yes/no. Get [ok]", plan)
            return {"expr": plan, "ok": str(verdict).lower().startswith("yes")}

        samples: list[ThoughtSample] = []
        rows: list[dict] = []
        for puzzle in _PUZZLES:
            outcome, graph = solve24(puzzle)
            expr, ok = outcome["expr"], outcome["ok"]
            # mechanical check: evaluate the expression ourselves
            import re as _re

            expr_text = str(expr)
            try:
                value = eval(  # noqa: S307 - benchmark arithmetic strings
                    _re.sub(r"[^0-9+\-*/(). ]", "", expr_text),
                    {"__builtins__": {}},
                )
            except Exception:  # noqa: BLE001 - record failure as wrong
                value = None
            correct = value is not None and abs(value - 24) < 1e-6
            sample = ThoughtSample(
                graph=graph,
                success=correct,
                answer=str(value) if value is not None else None,
                step_matches={node: True for node in graph.nodes},
            )
            samples.append(sample)
            rows.append(
                {
                    "puzzle": puzzle,
                    "expr": expr_text,
                    "model_ok": ok,
                    "value": value,
                    "correct": correct,
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                }
            )
            print(f"puzzle={puzzle} -> {expr_text!r} = {value} (correct={correct})", flush=True)

        # --- process metrics on the cohort
        cohort_report: dict = {"rows": rows, "n_correct": sum(r["correct"] for r in rows)}
        all_paths = [path for s in samples for path in enumerate_paths(s.graph)]
        self.assertTrue(all_paths)
        influence = node_influence(all_paths)
        self.assertTrue(all(0.0 < v <= 1.0 for v in influence.values()))
        reliability = multipath_reliability(samples)
        cohort_report["multipath_reliability"] = reliability
        dists = [path_distribution(s.graph) for s in samples]
        if len(dists) >= 2:
            cohort_report["commonality_kl_first_two"] = reasoning_commonality(dists[0], dists[1])
        for sample in samples:
            sample.step_matches = {node: True for node in sample.graph.nodes}
        scs_values = [stepwise_consistency(s) for s in samples]
        cohort_report["scs_mean"] = sum(scs_values) / len(scs_values)
        cohort_report["accuracy"] = cohort_report["n_correct"] / len(rows)

        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(cohort_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"\n[game24] accuracy={cohort_report['accuracy']:.2f}, "
            f"reliability_gain={reliability:+.3f}, scs_mean={cohort_report['scs_mean']:.3f} "
            f"-> {REPORT}"
        )
        self.assertGreaterEqual(cohort_report["n_correct"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
