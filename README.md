# DaYan 大衍

> *大衍之数五十，其用四十有九。* — The I Ching's Great Expansion generates
> structured hexagram states through a formal derivation procedure.
> **DaYan** does the same for LLM reasoning: it turns the reasoning *process*
> into structured data — a Thought Graph — and then analyzes it.

DaYan is a toolkit for **modeling, evaluating, and enhancing LLM reasoning as
data**. Where standard evaluation only checks final answers, DaYan answers
*how* the model reasoned — and feeds the findings back as improvements.

```text
reasoning program (your code calling an LLM)
  → @analysis instrumentation (sys.settrace, no source changes)
  → program trace → data-flow graph → Thought Graph
        (nodes = LLM outputs "thoughts", edges = data dependencies)
  → process evaluation: 5 metrics beyond accuracy
  → pattern mining: frequent thinking patterns (successful vs. failed)
  → enhancement: prompt rules / verifiable RL process rewards
```

An auxiliary subpackage, `dayan.augment`, provides **symbolically verified
data augmentation** (Logical DA): LLM-generated samples whose labels are
checked by Z3 / Prover9 / rule / CSP solvers before they enter your training
set.

## Quickstart

```python
from dayan import analysis, query, set_backend, AnalysisOutput

# Inject any LLM: (context, instruction, temperature) -> str
set_backend(my_llm_fn)

@analysis(output_type=AnalysisOutput.THOUGHT)
def solve(nums):
    step = query("Add the numbers in {} and get [total]", nums)  # LLM step
    return int(step) * 2

result, thought_graph = solve([4, 4])
print(result)                       # program output
print(thought_graph.print_summary())  # nodes/edges of the reasoning process
```

Evaluate a cohort of reasoning runs:

```python
from dayan import ThoughtSample, mine_frequent_patterns, reasoning_efficiency, stepwise_consistency

patterns = mine_frequent_patterns([g1, g2, g3], length=3, min_support=2)
print(reasoning_efficiency(patterns))          # κ (branching) / λ (first layer)
print(stepwise_consistency(ThoughtSample(g1, success=True, step_matches={...})))
```

Mine patterns from successful vs. failed runs and turn them into prompt
rules or RL process rewards:

```python
from dayan import contrastive_patterns, patterns_to_prompt_rules, PatternProcessReward

good, bad = contrastive_patterns(success_graphs, failure_graphs)
rules = patterns_to_prompt_rules(good, bad)          # inject into prompts
reward_fn = PatternProcessReward(good, bad)           # per-step RL reward
reward = reward_fn(label_sequence_of_a_trajectory)
```

## Verified data augmentation

```bash
dayan augment --dataset records.json --output augmented.json --solver z3
```

The pipeline (attribute extraction → diversity filtering → label-flipped
generation → symbolic verification → reflection) runs on any OpenAI-compatible
endpoint; `DAYAN_API_KEY` / `DAYAN_BASE_URL` / `DAYAN_MODEL` configure it
(falling back to `OPENAI_*`).  Solvers: `z3`, `prover9`, `rules`, `csp`
(auto-detected), with an `llm` fallback.

## Installation

```bash
pip install -e .              # core is dependency-free
pip install -e ".[llm]"       # default OpenAI-compatible backend
pip install -e ".[augment]"   # z3-solver for label verification
pip install -e ".[dev]"       # tests + lint
```

## Research lineage

- **Thought Mining for Large Language Model Reasoning** — the Thought Graph
  data model, the five process metrics, and both enhancement strategies.
- **Logical DA** (ACL 2025 Findings) — multi-agent data augmentation with
  symbolic label verification.

If you use DaYan, please cite these works (see CITATION.cff).

## License

Apache-2.0 (see LICENSE).
