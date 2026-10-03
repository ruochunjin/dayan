"""DaYan (大衍): thought-graph modeling, evaluation, and enhancement.

Named after the I Ching's 大衍之数 — the formal generative procedure that
produces structured hexagram states — DaYan models an LLM's reasoning
process as data: instrument a reasoning program, get its Thought Graph,
evaluate the *process* (five metrics), mine thinking patterns, and feed them
back as prompt rules or verifiable RL rewards.  An auxiliary subpackage
(``dayan.augment``) provides symbolically verified data augmentation.

    from dayan import analysis, AnalysisOutput, query, set_backend

    set_backend(my_llm_fn)          # inject any LLM
    result, thought_graph = solve()  # @analysis(output_type=THOUGHT)
"""

from dayan.graph import (
    AnalysisOutput,
    DataFlowGraph,
    ProgramTrace,
    ThoughtGraph,
    VariableNode,
    analysis,
    query,
    query_iter,
    set_backend,
)
from dayan.llm import LLM, ChatClient, FakeLLM
from dayan.metrics import (
    ThoughtSample,
    crucial_node_judgment,
    enumerate_paths,
    multipath_reliability,
    node_influence,
    path_distribution,
    reasoning_commonality,
    reasoning_efficiency,
    stepwise_consistency,
)
from dayan.mining import Pattern, contrastive_patterns, mine_frequent_patterns
from dayan.optimize import (
    PatternProcessReward,
    inject_rules_into_prompt,
    patterns_to_prompt_rules,
)

__version__ = "0.1.0"

__all__ = [
    "AnalysisOutput",
    "ChatClient",
    "DataFlowGraph",
    "FakeLLM",
    "LLM",
    "Pattern",
    "PatternProcessReward",
    "ProgramTrace",
    "ThoughtGraph",
    "ThoughtSample",
    "VariableNode",
    "__version__",
    "analysis",
    "contrastive_patterns",
    "crucial_node_judgment",
    "enumerate_paths",
    "inject_rules_into_prompt",
    "mine_frequent_patterns",
    "multipath_reliability",
    "node_influence",
    "path_distribution",
    "patterns_to_prompt_rules",
    "query",
    "query_iter",
    "reasoning_commonality",
    "reasoning_efficiency",
    "set_backend",
    "stepwise_consistency",
]
