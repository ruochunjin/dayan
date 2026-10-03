"""Offline tests for the thought-graph core (instrumentation → DFG → ThoughtGraph)."""

from __future__ import annotations

import unittest

import dayan
from dayan import AnalysisOutput, analysis, query, set_backend
from dayan.graph.graph import ThoughtGraph


def _fake_backend(context: str, instruction: str, temperature: float) -> str:
    """Deterministic LLM stand-in answering the template's single slot."""
    return "<OUTPUT>The result is <8></OUTPUT>"


@analysis(output_type=AnalysisOutput.THOUGHT)
def tiny_program(nums):
    step = query("Add the numbers in {} and get [total]", nums)
    final = int(step) * 2
    return final


class ThoughtGraphBuildTests(unittest.TestCase):
    def test_instrumented_program_builds_thought_graph(self) -> None:
        set_backend(_fake_backend)
        result, graph = tiny_program([4, 4])
        self.assertEqual(result, 16)
        self.assertIsInstance(graph, ThoughtGraph)
        labels = sorted(graph.labels.values())
        self.assertIn("ROOT", labels)  # the program parameter
        self.assertIn("LLM", labels)  # the query() output
        # The LLM node depends on the ROOT node (nums was used in the prompt).
        self.assertTrue(graph.edges)

    def test_trace_and_dfg_outputs(self) -> None:
        set_backend(_fake_backend)

        @analysis(output_type=AnalysisOutput.TRACE)
        def plain(x):
            y = query("Double {} and get [value]", x)
            return y

        result, trace = plain(4)
        self.assertEqual(result, "8")
        self.assertTrue(trace.events)

        @analysis(output_type=AnalysisOutput.DATA_FLOW)
        def to_dfg(x):
            y = query("Double {} and get [value]", x)
            z = int(y) + 1
            return z

        _, dfg = to_dfg(4)
        self.assertTrue(dfg.all_nodes)

    def test_manual_graph_paths(self) -> None:
        graph = ThoughtGraph()
        graph.add_node("root", "ROOT", {"value": None})
        graph.add_node("a", "LLM", {"value": "A"})
        graph.add_node("b", "LLM", {"value": "B"})
        graph.add_edge("root", "a")
        graph.add_edge("a", "b")
        paths = dayan.enumerate_paths(graph)
        self.assertEqual(paths, [["root", "a", "b"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
