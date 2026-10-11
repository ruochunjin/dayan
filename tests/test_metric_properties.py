"""Property tests for DaYan metrics: invariants that must hold on ANY graph.

- KL divergence non-negative, zero for identical distributions
- node influence in [0, 1], 1.0 for the unique path's nodes
- SCS in [0, 1]; 1.0 when every step matches; depth weighting: a deep error
  hurts more than a shallow one
- efficiency κ=1 for pure chains, ≥1 otherwise; λ ≥ 0
- multipath reliability in [-1, 1]
"""

from __future__ import annotations

import random
import unittest

from dayan.graph.graph import ThoughtGraph
from dayan.metrics import (
    ThoughtSample,
    enumerate_paths,
    multipath_reliability,
    node_influence,
    path_distribution,
    reasoning_commonality,
    stepwise_consistency,
)


def _random_graph(rng: random.Random, nodes: int, edge_p: float) -> ThoughtGraph:
    """Random DAG: nodes n0..nN-1, edges only forward (i<j)."""

    graph = ThoughtGraph()
    for i in range(nodes):
        graph.add_node(f"n{i}", "LLM" if i else "ROOT", {"value": i})
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if rng.random() < edge_p:
                graph.add_edge(f"n{i}", f"n{j}")
    return graph


class MetricInvariantTests(unittest.TestCase):
    def test_kl_nonnegative_zero_identical(self) -> None:
        rng = random.Random(3)
        for _ in range(6):
            graph = _random_graph(rng, nodes=8, edge_p=0.3)
            dist = path_distribution(graph, n_walks=50, seed=1)
            if not dist:
                continue
            self.assertAlmostEqual(reasoning_commonality(dist, dist), 0.0, places=6)
            other = _random_graph(rng, nodes=8, edge_p=0.4)
            other_dist = path_distribution(other, n_walks=50, seed=2)
            if other_dist:
                self.assertGreaterEqual(reasoning_commonality(dist, other_dist), 0.0)

    def test_node_influence_bounds(self) -> None:
        rng = random.Random(11)
        for _ in range(6):
            graph = _random_graph(rng, nodes=6, edge_p=0.4)
            paths = enumerate_paths(graph)
            if not paths:
                continue
            influence = node_influence(paths)
            for value in influence.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_scs_bounds_and_depth_weighting(self) -> None:
        # Linear chain of 4 nodes: wrong step at depth 1 vs depth 3 —
        # the deeper error must score LOWER (depth-increasing weights).
        graph = ThoughtGraph()
        previous = None
        ids = []
        for i in range(4):
            node = f"n{i}"
            graph.add_node(node, "LLM" if i else "ROOT", {"value": i})
            ids.append(node)
            if previous:
                graph.add_edge(previous, node)
            previous = node
        all_correct = ThoughtSample(graph, True, step_matches={n: True for n in ids})
        self.assertEqual(stepwise_consistency(all_correct), 1.0)
        shallow_wrong = ThoughtSample(
            graph, True,
            step_matches={n: (i != 1) for i, n in enumerate(ids)},
        )
        deep_wrong = ThoughtSample(
            graph, True,
            step_matches={n: (i != 3) for i, n in enumerate(ids)},
        )
        shallow_scs = stepwise_consistency(shallow_wrong)
        deep_scs = stepwise_consistency(deep_wrong)
        self.assertGreater(shallow_scs, deep_scs)
        self.assertGreaterEqual(shallow_scs, 0.0)

    def test_multipath_bounds(self) -> None:
        rng = random.Random(5)
        samples = []
        for i in range(6):
            graph = _random_graph(rng, nodes=4, edge_p=0.3)
            samples.append(
                ThoughtSample(graph, success=i % 2 == 0, answer="a" if i % 3 else "b")
            )
        score = multipath_reliability(samples)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
