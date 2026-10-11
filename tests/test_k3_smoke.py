"""K3 end-to-end smoke: instrumented reasoning program → ThoughtGraph → metrics.

Gated by HUFU_RUN_E2E-style flag: DAYAN_RUN_E2E=1 (and requires the
Anthropic/K3 credentials resolvable by dayan.llm).
"""

from __future__ import annotations

import os
import unittest

_ENABLED = os.environ.get("DAYAN_RUN_E2E") == "1"


@unittest.skipUnless(_ENABLED, "requires DAYAN_RUN_E2E=1")
class K3ThoughtGraphSmokeTests(unittest.TestCase):
    def test_instrumented_program_with_k3(self) -> None:
        from dayan import AnalysisOutput, analysis, query, set_backend
        from dayan.llm import AnthropicClient

        client = AnthropicClient()

        def k3_backend(context: str, instruction: str, temperature: float) -> str:
            prompt = (
                "Always wrap your final result between <OUTPUT> and </OUTPUT> tags.\n\n"
                + context + "\n\n" + instruction
            )
            return client.chat(prompt, temperature=temperature)

        set_backend(k3_backend)

        @analysis(output_type=AnalysisOutput.THOUGHT)
        def two_step(nums):
            total = query("Add the numbers in {} and get [total]", nums)
            result = query("Compute {} times 3 and get [product]", total)
            return result

        result, graph = two_step([4, 4])
        self.assertTrue(str(result).strip())
        labels = sorted(graph.labels.values())
        self.assertIn("ROOT", labels)
        self.assertIn("LLM", labels)
        self.assertTrue(graph.edges)
        print(f"\n[k3-smoke] result={result!r}; nodes={len(graph.nodes)}, edges={len(graph.edges)}")

        # Metrics must compute on a real trace without error.
        from dayan.metrics import (
            ThoughtSample,
            enumerate_paths,
            node_influence,
            path_distribution,
            stepwise_consistency,
        )

        paths = enumerate_paths(graph)
        self.assertTrue(paths)
        influence = node_influence(paths)
        self.assertTrue(influence)
        dist = path_distribution(graph)
        self.assertTrue(dist)
        sample = ThoughtSample(
            graph=graph,
            success=True,
            answer=result,
            step_matches={node: True for node in graph.nodes},
        )
        scs = stepwise_consistency(sample)
        self.assertEqual(scs, 1.0)  # every step marked correct
        print(f"[k3-smoke] paths={len(paths)}, scs={scs:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
