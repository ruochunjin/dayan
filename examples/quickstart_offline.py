"""Offline quickstart: instrument a reasoning program, build its Thought
Graph, evaluate and mine it — no LLM endpoint needed (scripted backend).

Run from the repository root:  python examples/quickstart_offline.py
"""

import dayan
from dayan import AnalysisOutput, analysis, query, set_backend


def scripted_backend(context: str, instruction: str, temperature: float) -> str:
    # A deterministic stand-in: always fills the single [slot] with 8.
    return "<OUTPUT>The result is <8></OUTPUT>"


@analysis(output_type=AnalysisOutput.THOUGHT)
def solve(nums):
    total = query("Add the numbers in {} and get [total]", nums)  # LLM step
    doubled = int(total) * 2  # plain Python step (transparent in the graph)
    return doubled


def main() -> None:
    set_backend(scripted_backend)
    result, graph = solve([4, 4])
    print("program result:", result)
    graph.print_summary()

    # Process evaluation
    paths = dayan.enumerate_paths(graph)
    print("\npaths:", paths)
    print("node influence:", dayan.node_influence(paths))

    # Pattern mining over a small corpus (same program, several runs)
    graphs = [solve([1, 2])[1] for _ in range(3)] + [graph]
    patterns = dayan.mine_frequent_patterns(graphs, length=2, min_support=2)
    for p in patterns:
        print(f"pattern {p.render()}  support={p.support} κ={p.branching} λ={p.first_layer}")
    print("efficiency:", dayan.reasoning_efficiency(patterns))

    # Patterns -> prompt rules / RL process reward
    rules = dayan.patterns_to_prompt_rules(patterns)
    print("\nprompt rule:", rules[0] if rules else "-")
    reward_fn = dayan.PatternProcessReward(patterns)
    print("trajectory reward:", reward_fn(["ROOT", "LLM"]))


if __name__ == "__main__":
    main()
