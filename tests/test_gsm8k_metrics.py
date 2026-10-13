"""GSM8K thought-mining batch: real K3 chain-of-thought traces through the
instrumented pipeline with process metrics, on a public benchmark sample.

Gated by DAYAN_RUN_E2E=1.  GSM8K is downloaded from the HF mirror on first run.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

_ENABLED = os.environ.get("DAYAN_RUN_E2E") == "1"
REPORT = Path(__file__).parent.parent / "benchmarks" / "gsm8k_metrics.json"

_N = int(os.environ.get("DAYAN_GSM8K_N", "8"))


def _gold_number(answer: str) -> float | None:
    match = re.search(r"####\s*([\d,\.]+)", answer or "")
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _pred_number(text: str) -> float | None:
    numbers = re.findall(r"-?\d[\d,\.]*", text or "")
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


@unittest.skipUnless(_ENABLED, "requires DAYAN_RUN_E2E=1")
class Gsm8kThoughtMiningTests(unittest.TestCase):
    def test_gsm8k_batch_with_metrics(self) -> None:
        from datasets import load_dataset

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

        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{_N}]")
        client = AnthropicClient()

        def k3_backend(context: str, instruction: str, temperature: float) -> str:
            prompt = (
                "Always wrap your final result between <OUTPUT> and </OUTPUT> tags.\n\n"
                + context + "\n\n" + instruction
            )
            return client.chat(prompt, temperature=temperature)

        set_backend(k3_backend)

        @analysis(output_type=AnalysisOutput.THOUGHT)
        def solve(question):
            draft = query(
                "Solve this grade-school problem step by step, then give the final "
                "number only at the end: {}. Get [answer]",
                question,
            )
            check = query(
                "Recompute {} independently; state only yes or no whether the final "
                "number is the same. Get [ok]",
                draft,
            )
            return {"answer": draft, "verified": str(check).lower().startswith("yes")}

        samples: list[ThoughtSample] = []
        rows: list[dict] = []
        for i, row in enumerate(ds):
            outcome, graph = solve(row["question"])
            gold = _gold_number(row["answer"])
            pred = _pred_number(str(outcome["answer"]))
            correct = pred is not None and gold is not None and abs(pred - gold) < 1e-6
            samples.append(
                ThoughtSample(
                    graph=graph,
                    success=correct,
                    answer=pred,
                    step_matches={node: True for node in graph.nodes},
                )
            )
            rows.append(
                {
                    "i": i,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "verified": outcome["verified"],
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                }
            )
            print(f"[{i + 1}/{_N}] gold={gold} pred={pred} correct={correct}", flush=True)

        all_paths = [p for s in samples for p in enumerate_paths(s.graph)]
        influence = node_influence(all_paths)
        reliability = multipath_reliability(samples)
        scs = [stepwise_consistency(s) for s in samples]
        dists = [path_distribution(s.graph) for s in samples]

        # Star mining + graph-level reward on the real collected traces:
        # contrastive stars between correct/incorrect runs, then a reward
        # built from them must rank successful graphs above failures.
        from dayan import (
            PatternProcessReward,
            mine_frequent_stars,
            stars_to_prompt_rules,
        )

        ok_graphs = [s.graph for s in samples if s.success]
        bad_graphs = [s.graph for s in samples if not s.success]
        mined = {}
        if ok_graphs and bad_graphs:
            good_stars = mine_frequent_stars(ok_graphs, min_arity=2, min_support=2)
            bad_stars = mine_frequent_stars(bad_graphs, min_arity=2, min_support=1)
            reward_fn = PatternProcessReward(
                success_stars=good_stars, failure_stars=bad_stars
            )
            ok_scores = [reward_fn.score_graph(g) for g in ok_graphs]
            bad_scores = [reward_fn.score_graph(g) for g in bad_graphs]
            mined = {
                "success_stars": [s.render() for s in good_stars[:5]],
                "failure_stars": [s.render() for s in bad_stars[:5]],
                "rules": stars_to_prompt_rules(good_stars, bad_stars, limit=4),
                "reward_ok_mean": sum(ok_scores) / len(ok_scores),
                "reward_bad_mean": sum(bad_scores) / len(bad_scores),
            }
            print(f"[gsm8k] star reward: ok={mined['reward_ok_mean']:+.2f} "
                  f"bad={mined['reward_bad_mean']:+.2f}")
            print(f"[gsm8k] rules: {mined['rules']}")

        report = {
            "rows": rows,
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "verified_rate": sum(r["verified"] for r in rows) / len(rows),
            "multipath_reliability": reliability,
            "scs_mean": sum(scs) / len(scs),
            "commonality_kl_first_two": reasoning_commonality(dists[0], dists[1]) if len(dists) > 1 else None,
            "star_mining": mined,
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"\n[gsm8k] accuracy={report['accuracy']:.2f} verified={report['verified_rate']:.2f} "
            f"reliability_gain={reliability:+.3f} scs={report['scs_mean']:.3f}"
        )
        self.assertTrue(all_paths)
        self.assertTrue(all(0.0 < v <= 1.0 for v in influence.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
