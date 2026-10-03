"""Lightweight frequent pattern mining over thought graphs.

A *pattern* is a short label-sequence path (length ``k``) that recurs across
a corpus of thought graphs — the dependency-free replacement for external
frequent-subgraph miners (GraMi).  Each pattern carries the two structural
properties the Reasoning Efficiency metric needs:

- ``branching`` (κ): maximum out-degree among instances of the pattern —
  high κ means the model fans out / backtracks a lot at this point;
- ``first_layer`` (λ): earliest BFS layer where the pattern occurs —
  early-onset patterns indicate inefficiency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from dayan.graph.graph import ThoughtGraph
from dayan.metrics import enumerate_paths, node_layers


@dataclass(frozen=True, slots=True)
class Pattern:
    """A recurring label-sequence pattern with structural statistics."""

    sequence: tuple[str, ...]
    support: int  # number of path occurrences across the corpus
    graph_support: int  # number of distinct graphs containing it
    branching: float  # κ: max out-degree over member nodes
    first_layer: int  # λ: earliest layer observed

    def render(self) -> str:
        """Human-readable rule form, used in prompt optimization."""

        return " → ".join(self.sequence)


def _paths_with_nodes(graph: ThoughtGraph, max_depth: int) -> list[list[str]]:
    return [p for p in enumerate_paths(graph, max_depth=max_depth) if len(p) >= 2]


def mine_frequent_patterns(
    graphs: Sequence[ThoughtGraph],
    *,
    length: int = 3,
    min_support: int = 2,
    max_depth: int = 64,
) -> list[Pattern]:
    """Mine frequent label-sequences of ``length`` from a graph corpus.

    Support counts path occurrences; a pattern is returned when its support
    reaches ``min_support``.  Sorted by support (desc), then sequence.
    """

    if length < 2:
        raise ValueError("pattern length must be >= 2")
    occurrences: Counter[tuple[str, ...]] = Counter()
    graph_membership: dict[tuple[str, ...], set[int]] = {}
    branching: dict[tuple[str, ...], float] = {}
    layers: dict[tuple[str, ...], int] = {}

    for graph_id, graph in enumerate(graphs):
        graph_layers = node_layers(graph)
        children: dict[str, set[str]] = {}
        for src, dst in graph.edges:
            children.setdefault(src, set()).add(dst)
        for path in _paths_with_nodes(graph, max_depth):
            label_path = tuple(graph.labels.get(node, "?") for node in path)
            for start in range(len(label_path) - length + 1):
                seq = label_path[start : start + length]
                occurrences[seq] += 1
                graph_membership.setdefault(seq, set()).add(graph_id)
                member_nodes = path[start : start + length]
                kappa = max(len(children.get(node, ())) for node in member_nodes)
                branching[seq] = max(branching.get(seq, 0.0), float(kappa))
                first = min(graph_layers.get(node, 0) for node in member_nodes)
                layers[seq] = min(layers.get(seq, first), first)

    patterns = [
        Pattern(
            sequence=seq,
            support=count,
            graph_support=len(graph_membership[seq]),
            branching=branching[seq],
            first_layer=layers[seq],
        )
        for seq, count in occurrences.items()
        if count >= min_support
    ]
    patterns.sort(key=lambda p: (-p.support, p.sequence))
    return patterns


def contrastive_patterns(
    success_graphs: Sequence[ThoughtGraph],
    failure_graphs: Sequence[ThoughtGraph],
    *,
    length: int = 3,
    min_support: int = 2,
) -> tuple[list[Pattern], list[Pattern]]:
    """Patterns frequent in successful runs but rare in failed ones (and vice versa).

    Returns ``(success_patterns, failure_patterns)`` — the raw material for
    prompt rules and process rewards in ``dayan.optimize``.
    """

    good = mine_frequent_patterns(success_graphs, length=length, min_support=min_support)
    bad = mine_frequent_patterns(failure_graphs, length=length, min_support=min_support)
    bad_seqs = {p.sequence for p in bad}
    good_seqs = {p.sequence for p in good}
    return (
        [p for p in good if p.sequence not in bad_seqs],
        [p for p in bad if p.sequence not in good_seqs],
    )


__all__ = ["Pattern", "contrastive_patterns", "mine_frequent_patterns"]
