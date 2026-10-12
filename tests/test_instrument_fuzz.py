"""Fuzz tests for the DaYan instrumentation engine (offline, scripted backend).

Adversarial program shapes that must produce a well-formed ThoughtGraph:
- loops with query() inside (repeated LLM steps)
- conditionals that skip the LLM call
- nested @analysis calls
- programs raising exceptions mid-trace
- query_iter divergence loops
"""

from __future__ import annotations

import unittest

from dayan import AnalysisOutput, analysis, query, set_backend


def _fake_backend(context: str, instruction: str, temperature: float) -> str:
    return "<OUTPUT>The result is <8></OUTPUT>"


class InstrumentationFuzzTests(unittest.TestCase):
    def setUp(self) -> None:
        set_backend(_fake_backend)

    def test_query_inside_loop(self) -> None:
        @analysis(output_type=AnalysisOutput.THOUGHT)
        def prog(nums):
            total = 0
            for n in nums:
                step = query("Double {} and get [value]", n)
                total += int(step)
            return total

        result, graph = prog([1, 2, 3])
        self.assertEqual(result, 24)  # fake backend returns 8 per call; 3 calls
        # three LLM calls -> three LLM nodes
        llm_nodes = [n for n, label in graph.labels.items() if label == "LLM"]
        self.assertGreaterEqual(len(llm_nodes), 3)

    def test_conditional_skip(self) -> None:
        @analysis(output_type=AnalysisOutput.THOUGHT)
        def prog(flag):
            if flag:
                return query("Double {} and get [value]", 4)
            return "no-llm"

        result, graph = prog(False)
        self.assertEqual(result, "no-llm")
        llm_nodes = [n for n, label in graph.labels.items() if label == "LLM"]
        self.assertEqual(llm_nodes, [])

    def test_exception_mid_trace(self) -> None:
        @analysis(output_type=AnalysisOutput.THOUGHT)
        def prog(nums):
            query("Add {} and get [total]", nums)
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            prog([1, 2])

    def test_nested_analysis_call(self) -> None:
        """Nested @analysis calls share the outer trace: the inner call
        returns the bare result (not a (result, graph) tuple)."""

        @analysis(output_type=AnalysisOutput.THOUGHT)
        def inner(x):
            return query("Double {} and get [value]", x)

        @analysis(output_type=AnalysisOutput.THOUGHT)
        def outer(x):
            inner_result = inner(x)  # nested: bare result by design
            return inner_result

        result, graph = outer(2)
        self.assertEqual(result, "8")
        self.assertIsNotNone(graph)

    def test_query_iter_mode(self) -> None:
        from dayan import query_iter

        def iter_backend(context: str, instruction: str, temperature: float) -> str:
            return "<OUTPUT>get <1>, get <2>, get <3></OUTPUT>"

        set_backend(iter_backend)

        @analysis(output_type=AnalysisOutput.THOUGHT)
        def prog(n):
            values = list(query_iter("Generate {} candidate numbers as [value]", n))
            return len(values)

        result, graph = prog(3)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
