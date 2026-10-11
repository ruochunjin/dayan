"""Metric discrimination tests: the reasoning-efficiency metric must
distinguish branched (backtracking-heavy) reasoning from linear chains —
the core behavioral claim of the Thought Mining paper's κ/λ measures.
"""

from __future__ import annotations

import unittest

from dayan.graph.graph import ThoughtGraph
from dayan.metrics import reasoning_efficiency
from dayan.mining import mine_frequent_patterns


def _linear_graph(depth: int = 6) -> ThoughtGraph:
    graph = ThoughtGraph()
    previous = None
    for layer in range(depth):
        node = f"n{layer}"
        graph.add_node(node, "LLM", {"value": layer})
        if previous is not None:
            graph.add_edge(previous, node)
        previous = node
    return graph


def _branched_graph(depth: int = 4, fanout: int = 3) -> ThoughtGraph:
    graph = ThoughtGraph()
    frontier = ["root"]
    graph.add_node("root", "ROOT", {"value": None})
    serial = 0
    for _layer in range(depth):
        next_frontier: list[str] = []
        for parent in frontier:
            for branch in range(fanout):
                serial += 1
                child = f"b{serial}"
                graph.add_node(child, "LLM", {"value": serial})
                graph.add_edge(parent, child)
                next_frontier.append(child)
        frontier = next_frontier
    return graph


class EfficiencyDiscriminationTests(unittest.TestCase):
    def test_branched_reasoning_has_higher_branching(self) -> None:
        """κ (branching) must be larger for the backtracking-style graph."""

        linear = _linear_graph()
        branched = _branched_graph()
        linear_patterns = mine_frequent_patterns([linear, linear], length=3, min_support=2)
        branched_patterns = mine_frequent_patterns([branched, branched], length=3, min_support=2)

        linear_eff = reasoning_efficiency(linear_patterns)
        branched_eff = reasoning_efficiency(branched_patterns)
        print(
            f"\n[discrimination] linear κ={linear_eff.avg_branching:.2f} "
            f"vs branched κ={branched_eff.avg_branching:.2f}"
        )
        self.assertGreater(branched_eff.avg_branching, linear_eff.avg_branching)
        self.assertEqual(linear_eff.avg_branching, 1.0)

    def test_frequent_patterns_rank_central_structure(self) -> None:
        """The hub pattern must win support over leaf patterns."""

        graph = _branched_graph(depth=3, fanout=3)
        patterns = mine_frequent_patterns([graph, graph], length=2, min_support=2)
        self.assertTrue(patterns)
        top = patterns[0]
        self.assertGreaterEqual(top.support, 2)
        # Hub-adjacent patterns carry the largest branching factor.
        self.assertGreater(top.branching, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
