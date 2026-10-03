"""Process-level evaluation metrics for LLM reasoning (Thought Mining paper).

Five metrics over thought graphs — reasoning is evaluated as a *process*,
not just by final-answer accuracy:

1. **Reasoning Efficiency** — branching factor κ and first-occurrence layer λ
   of frequent patterns; large κ / early λ indicate backtracking-heavy,
   inefficient reasoning.
2. **Crucial Node Judgment (NIS)** — correlation between LLM-assigned node
   scores and true node influence (share of successful paths through the
   node), plus local stability across node subsets.
3. **Reasoning Commonality** — KL divergence between models' reasoning-path
   distributions (random-walk sampled).
4. **Multipath Reasoning Reliability** — path aggregation: does combining
   multiple reasoning paths yield the correct answer more often than the
   average single path?
5. **Stepwise Consistency (SCS)** — depth-weighted per-step correctness
   along reasoning paths.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence

from dayan.graph.graph import ThoughtGraph

# ----------------------------------------------------------------------
# path utilities
# ----------------------------------------------------------------------


def _children(graph: ThoughtGraph) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for src, dst in graph.edges:
        adjacency[src].append(dst)
    return {node: sorted(targets) for node, targets in adjacency.items()}


def _roots(graph: ThoughtGraph) -> list[str]:
    has_incoming = {dst for _, dst in graph.edges}
    return sorted(node for node in graph.nodes if node not in has_incoming)


def enumerate_paths(graph: ThoughtGraph, max_depth: int = 64) -> list[list[str]]:
    """All simple root→terminal paths (capped by ``max_depth``)."""

    adjacency = _children(graph)
    paths: list[list[str]] = []

    def walk(node: str, prefix: list[str]) -> None:
        if len(prefix) >= max_depth or node not in adjacency:
            paths.append(prefix)
            return
        for child in adjacency[node]:
            if child not in prefix:  # simple paths only
                walk(child, prefix + [child])

    for root in _roots(graph):
        walk(root, [root])
    return paths or [[r] for r in _roots(graph)]


def random_walk_paths(
    graph: ThoughtGraph, n_walks: int = 100, max_depth: int = 32, seed: int = 42
) -> list[list[str]]:
    """Random-walk path sampling from a root, as used for Commonality."""

    adjacency = _children(graph)
    rng = random.Random(seed)
    paths = []
    for _ in range(n_walks):
        for root in _roots(graph) or sorted(graph.nodes):
            node, path = root, [root]
            while node in adjacency and len(path) < max_depth:
                node = rng.choice(adjacency[node])
                path.append(node)
            paths.append(path)
    return paths


def node_layers(graph: ThoughtGraph) -> dict[str, int]:
    """BFS layer of each node (roots = layer 0)."""

    adjacency = _children(graph)
    layers: dict[str, int] = {}
    frontier = [(root, 0) for root in _roots(graph)]
    for node, depth in frontier:
        if node in layers and layers[node] <= depth:
            continue
        layers[node] = depth
        frontier.extend((child, depth + 1) for child in adjacency.get(node, []))
    return layers


# ----------------------------------------------------------------------
# sample container
# ----------------------------------------------------------------------


@dataclass(slots=True)
class ThoughtSample:
    """One reasoning run: its thought graph plus outcome metadata."""

    graph: ThoughtGraph
    success: bool
    answer: Hashable | None = None
    node_scores: Mapping[str, float] | None = None  # LLM-assigned s_v
    step_matches: Mapping[str, bool] | None = None  # ground-truth step match m(v)


# ----------------------------------------------------------------------
# 1. Reasoning Efficiency
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EfficiencyReport:
    avg_branching: float  # mean κ over frequent patterns (lower is better)
    avg_first_layer: float  # mean λ (later onset is better)
    pattern_count: int


def reasoning_efficiency(
    patterns: Sequence[Any], graph: ThoughtGraph | None = None
) -> EfficiencyReport:
    """Aggregate κ/λ over mined patterns (see ``dayan.mining``).

    Each pattern must expose ``support``, ``branching`` (κ) and
    ``first_layer`` (λ) attributes — as produced by
    ``dayan.mining.mine_frequent_patterns``.
    """

    if not patterns:
        return EfficiencyReport(0.0, 0.0, 0)
    total_support = sum(max(p.support, 1) for p in patterns)
    avg_branching = sum(p.branching * max(p.support, 1) for p in patterns) / total_support
    avg_layer = sum(p.first_layer * max(p.support, 1) for p in patterns) / total_support
    return EfficiencyReport(avg_branching, avg_layer, len(patterns))


# ----------------------------------------------------------------------
# 2. Crucial Node Judgment (NIS)
# ----------------------------------------------------------------------


def node_influence(paths: Sequence[Sequence[str]]) -> dict[str, float]:
    """ψ(v): share of (successful) paths passing through each node."""

    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(set(path))
    total = max(len(paths), 1)
    return {node: count / total for node, count in counts.items()}


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def crucial_node_judgment(
    sample: ThoughtSample,
    *,
    n_subsets: int = 5,
    subset_size: int | None = None,
    weight_global: float = 0.5,
    seed: int = 42,
) -> float:
    """CNJ = ω₁·ρ + ω₂·(1/(1+Var ρ_subsets)), in [-1, 1] → higher is better.

    Requires ``sample.node_scores`` (LLM-assigned).  Node influence is
    computed over the sample's successful-path set — for a single graph, the
    path set; pass samples with ``success`` filtered upstream for cohorts.
    """

    if not sample.node_scores:
        raise ValueError("node_scores are required for crucial node judgment")
    paths = enumerate_paths(sample.graph)
    influence = node_influence(paths)
    nodes = sorted(set(sample.node_scores) & set(influence))
    if len(nodes) < 2:
        return 0.0
    scores = [sample.node_scores[v] for v in nodes]
    influences = [influence[v] for v in nodes]
    rho = _pearson(scores, influences)

    rng = random.Random(seed)
    size = subset_size or max(2, len(nodes) // 2)
    subset_rhos = []
    for _ in range(n_subsets):
        if size >= len(nodes):
            subset = nodes
        else:
            subset = rng.sample(nodes, size)
        idx = [nodes.index(v) for v in subset]
        subset_rhos.append(_pearson([scores[i] for i in idx], [influences[i] for i in idx]))
    variance = sum((r - rho) ** 2 for r in subset_rhos) / max(len(subset_rhos), 1)
    stability = 1.0 / (1.0 + variance)
    return weight_global * rho + (1 - weight_global) * stability


# ----------------------------------------------------------------------
# 3. Reasoning Commonality
# ----------------------------------------------------------------------


def path_distribution(
    graph: ThoughtGraph, n_walks: int = 100, seed: int = 42
) -> dict[tuple[str, ...], float]:
    """Normalized distribution over label-sequences of random-walk paths."""

    paths = random_walk_paths(graph, n_walks=n_walks, seed=seed)
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(graph.labels.get(node, "?") for node in path) for path in paths
    )
    total = sum(counts.values()) or 1
    return {path: count / total for path, count in counts.items()}


def reasoning_commonality(
    dist_a: Mapping[tuple[str, ...], float],
    dist_b: Mapping[tuple[str, ...], float],
    *,
    epsilon: float = 1e-9,
) -> float:
    """KL divergence KL(A‖B) between two models' path distributions.

    Lower = more similar reasoning behavior.  ``epsilon`` regularizes zeros.
    """

    keys = set(dist_a) | set(dist_b)
    return sum(
        (pa := dist_a.get(key, epsilon)) * math.log(pa / dist_b.get(key, epsilon))
        for key in keys
    )


# ----------------------------------------------------------------------
# 4. Multipath Reasoning Reliability
# ----------------------------------------------------------------------


def multipath_reliability(
    samples: Sequence[ThoughtSample],
    *,
    aggregate: Callable[[list[Hashable]], Hashable] | None = None,
    gold: Callable[[ThoughtSample], Hashable] | None = None,
) -> float:
    """Path aggregation gain: accuracy of the majority-aggregated answer
    minus mean single-path accuracy, over a cohort of samples.

    ``aggregate`` defaults to majority vote; ``gold`` defaults to comparing
    against the majority answer of successful samples.  Returns a value in
    [-1, 1]; positive means aggregation helps.
    """

    if aggregate is None:
        aggregate = lambda answers: Counter(answers).most_common(1)[0][0]  # noqa: E731
    singles = [1.0 if s.success else 0.0 for s in samples]
    answers = [s.answer for s in samples if s.answer is not None]
    if not answers:
        return 0.0
    aggregated = aggregate(answers)
    if gold is not None:
        correct = sum(1 for s in samples if gold(s) == aggregated)
        aggregated_acc = correct / len(samples)
    else:
        aggregated_acc = 1.0 if any(s.success for s in samples) else 0.0
    return aggregated_acc - (sum(singles) / len(singles))


# ----------------------------------------------------------------------
# 5. Stepwise Consistency (SCS)
# ----------------------------------------------------------------------


def stepwise_consistency(sample: ThoughtSample) -> float:
    """Depth-weighted per-step correctness along the sample's paths.

    ``sample.step_matches`` maps node id → ground-truth match.  Per path,
    SCS = Σᵢ wᵢ·m(vᵢ) / Σᵢ wᵢ with wᵢ = (i+1)/n (depth-increasing weights);
    averaged over paths.  In [0, 1]; 1.0 means every step of every path is
    correct, and deep steps count more than shallow ones.
    """

    if not sample.step_matches:
        raise ValueError("step_matches are required for stepwise consistency")
    paths = enumerate_paths(sample.graph)
    if not paths:
        return 0.0
    total = 0.0
    for path in paths:
        n = len(path)
        weights = [(i + 1) / n for i in range(n)]
        earned = sum(
            w * (1.0 if sample.step_matches.get(node) else 0.0)
            for w, node in zip(weights, path)
        )
        total += earned / sum(weights)
    return total / len(paths)


__all__ = [
    "EfficiencyReport",
    "ThoughtSample",
    "crucial_node_judgment",
    "enumerate_paths",
    "node_influence",
    "node_layers",
    "path_distribution",
    "random_walk_paths",
    "reasoning_commonality",
    "reasoning_efficiency",
    "multipath_reliability",
    "stepwise_consistency",
]
