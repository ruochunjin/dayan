"""Offline tests for metrics, mining, and optimization."""

from __future__ import annotations

import unittest

from dayan import (
    Pattern,
    PatternProcessReward,
    ThoughtSample,
    contrastive_patterns,
    crucial_node_judgment,
    inject_rules_into_prompt,
    mine_frequent_patterns,
    multipath_reliability,
    node_influence,
    path_distribution,
    patterns_to_prompt_rules,
    reasoning_commonality,
    stepwise_consistency,
)
from dayan.graph.graph import ThoughtGraph


def make_graph(branches: int = 2, depth: int = 2, prefix: str = "") -> ThoughtGraph:
    """root -> LLM_i -> LLM_ij chains."""
    graph = ThoughtGraph()
    graph.add_node(f"{prefix}root", "ROOT", {"value": None})
    for i in range(branches):
        mid = f"{prefix}a{i}"
        graph.add_node(mid, "LLM", {"value": f"step{i}"})
        graph.add_edge(f"{prefix}root", mid)
        for j in range(depth):
            leaf = f"{prefix}a{i}b{j}"
            graph.add_node(leaf, "LLM", {"value": f"leaf{j}"})
            graph.add_edge(mid, leaf)
    return graph


class MetricTests(unittest.TestCase):
    def test_node_influence(self) -> None:
        graph = make_graph()
        influence = node_influence(__import__("dayan").enumerate_paths(graph))
        self.assertAlmostEqual(influence["root"], 1.0)
        self.assertAlmostEqual(influence["a0"], 0.5)

    def test_crucial_node_judgment_correlation(self) -> None:
        graph = make_graph(branches=2, depth=1)
        # LLM scores aligned with structural influence.
        sample = ThoughtSample(
            graph,
            success=True,
            node_scores={"root": 1.0, "a0": 0.6, "a1": 0.6, "a0b0": 0.1, "a1b0": 0.1},
        )
        score = crucial_node_judgment(sample)
        self.assertGreater(score, 0.5)
        # Anti-correlated scores score lower.
        flipped = ThoughtSample(
            graph, success=True,
            node_scores={"root": 0.1, "a0": 0.4, "a1": 0.4, "a0b0": 0.9, "a1b0": 0.9},
        )
        self.assertLess(crucial_node_judgment(flipped), score)

    def test_commonality_identical_distributions_zero_kl(self) -> None:
        graph = make_graph()
        dist_a = path_distribution(graph, n_walks=20, seed=1)
        self.assertAlmostEqual(reasoning_commonality(dist_a, dist_a), 0.0, places=6)
        other = path_distribution(make_graph(prefix="x"), n_walks=20, seed=1)
        self.assertAlmostEqual(reasoning_commonality(dist_a, other), 0.0, places=6)

    def test_multipath_reliability_aggregation_gain(self) -> None:
        samples = [
            ThoughtSample(make_graph(), success=False, answer="wrong"),
            ThoughtSample(make_graph(), success=False, answer="wrong"),
            ThoughtSample(make_graph(), success=True, answer="right"),
            ThoughtSample(make_graph(), success=True, answer="right"),
        ]
        gain = multipath_reliability(samples)
        self.assertGreaterEqual(gain, 0.0)

    def test_stepwise_consistency(self) -> None:
        graph = make_graph(branches=1, depth=1)  # root -> a0 -> a0b0
        good = ThoughtSample(
            graph, success=True,
            step_matches={"root": True, "a0": True, "a0b0": True},
        )
        self.assertAlmostEqual(stepwise_consistency(good), 1.0, places=2)
        bad = ThoughtSample(
            graph, success=False,
            step_matches={"root": True, "a0": True, "a0b0": False},
        )
        self.assertLess(stepwise_consistency(bad), 1.0)


class MiningAndOptimizeTests(unittest.TestCase):
    def test_mine_frequent_patterns(self) -> None:
        graphs = [make_graph(prefix="g0"), make_graph(prefix="g1"), make_graph(prefix="g2")]
        patterns = mine_frequent_patterns(graphs, length=2, min_support=2)
        self.assertTrue(patterns)
        root_llm = next(p for p in patterns if p.sequence == ("ROOT", "LLM"))
        self.assertEqual(root_llm.graph_support, 3)
        self.assertEqual(root_llm.first_layer, 0)
        self.assertGreaterEqual(root_llm.branching, 1)

    def test_contrastive_and_prompt_rules(self) -> None:
        good = [make_graph(prefix="s0"), make_graph(prefix="s1")]
        bad_graphs = []
        for k in range(2):
            g = ThoughtGraph()
            g.add_node(f"f{k}root", "ROOT", {"value": None})
            g.add_node(f"f{k}x", "LLM", {"value": "x"})
            g.add_node(f"f{k}y", "LLM", {"value": "y"})
            g.add_node(f"f{k}z", "LLM", {"value": "z"})
            g.add_edge(f"f{k}root", f"f{k}x")
            g.add_edge(f"f{k}x", f"f{k}y")
            g.add_edge(f"f{k}y", f"f{k}z")
            bad_graphs.append(g)
        success_p, failure_p = contrastive_patterns(good, bad_graphs, length=4, min_support=2)
        rules = patterns_to_prompt_rules(success_p, failure_p)
        self.assertTrue(any(r.startswith("AVOID") for r in rules))
        prompt = inject_rules_into_prompt("Solve the task.", rules)
        self.assertIn("Reasoning guidelines", prompt)

    def test_pattern_process_reward(self) -> None:
        reward_fn = PatternProcessReward(
            success_patterns=[Pattern(("ROOT", "LLM", "LLM"), support=3, graph_support=2, branching=1.0, first_layer=0)],
            failure_patterns=[Pattern(("LLM", "LLM", "LLM"), support=2, graph_support=2, branching=3.0, first_layer=0)],
        )
        steps = reward_fn.score_trajectory(["ROOT", "LLM", "LLM", "LLM"])
        rewards = [s.reward for s in steps]
        self.assertIn(1.0, rewards)   # success pattern
        self.assertIn(-1.0, rewards)  # failure pattern
        self.assertEqual(reward_fn(["ROOT", "LLM", "LLM", "LLM"]), sum(rewards))
        self.assertEqual(reward_fn(["ROOT"]), 0.0)


class StarMiningTests(unittest.TestCase):
    """Star-subgraph mining: center label + sorted multiset of child labels."""

    def test_frequent_star_found_with_graph_support(self) -> None:
        from dayan import mine_frequent_stars

        graphs = [make_graph(branches=2, depth=2, prefix=f"g{k}") for k in range(3)]
        stars = mine_frequent_stars(graphs, min_arity=2, min_support=2)
        # ROOT fans out to two LLM children in every graph.
        root_star = next(s for s in stars if s.center == "ROOT")
        self.assertEqual(root_star.children, ("LLM", "LLM"))
        self.assertEqual(root_star.graph_support, 3)
        self.assertEqual(root_star.support, 3)
        self.assertEqual(root_star.first_layer, 0)
        self.assertEqual(root_star.arity, 2)

    def test_edge_insertion_order_does_not_matter(self) -> None:
        from dayan import mine_frequent_stars

        def graph_with_order(prefix: str, order: list[str]) -> ThoughtGraph:
            g = ThoughtGraph()
            g.add_node(f"{prefix}c", "PLAN", {"value": None})
            for i, lab in enumerate(order):
                g.add_node(f"{prefix}{i}", lab, {"value": None})
                g.add_edge(f"{prefix}c", f"{prefix}{i}")
            return g

        a = graph_with_order("a", ["LLM", "TOOL", "LLM"])
        b = graph_with_order("b", ["TOOL", "LLM", "LLM"])
        stars = mine_frequent_stars([a, b], min_arity=3, min_support=2)
        self.assertEqual(len(stars), 1)
        self.assertEqual(stars[0].children, ("LLM", "LLM", "TOOL"))
        self.assertEqual(stars[0].graph_support, 2)

    def test_min_arity_and_min_support_filter(self) -> None:
        from dayan import mine_frequent_stars

        chains = []
        for k in range(4):
            g = ThoughtGraph()
            g.add_node(f"c{k}", "ROOT", {"value": None})
            g.add_node(f"c{k}x", "LLM", {"value": None})
            g.add_edge(f"c{k}", f"c{k}x")
            chains.append(g)
        # out-degree-1 nodes are not stars at min_arity=2.
        self.assertEqual(mine_frequent_stars(chains, min_arity=2, min_support=2), [])
        graphs = [make_graph(prefix=f"h{k}") for k in range(2)]
        self.assertEqual(mine_frequent_stars(graphs, min_arity=2, min_support=5), [])

    def test_render_is_readable(self) -> None:
        from dayan import mine_frequent_stars

        stars = mine_frequent_stars([make_graph(prefix="r0"), make_graph(prefix="r1")],
                                    min_arity=2, min_support=2)
        rendered = next(s for s in stars if s.center == "ROOT").render()
        self.assertIn("ROOT", rendered)
        self.assertIn("LLM", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
