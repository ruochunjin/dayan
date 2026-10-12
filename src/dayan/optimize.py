"""Thought-driven optimization: mined patterns → prompt rules / RL rewards.

Two enhancement strategies from the Thought Mining paper, both derived from
contrastive pattern mining (successful vs. failed reasoning runs):

1. **Prompt optimization** — render success patterns as explicit rules to
   inject into prompts (interpretable, no training);
2. **Process reward** — a verifiable per-step reward for RL (GRPO-style):
   a reasoning step earns +1 when its label sequence matches a success
   pattern and −1 when it matches a failure pattern.  Being pattern-derived
   rather than model-derived, the reward is explainable and resistant to
   reward hacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dayan.mining import Pattern, StarPattern


def patterns_to_prompt_rules(
    success_patterns: Sequence[Pattern],
    failure_patterns: Sequence[Pattern] = (),
    *,
    limit: int = 10,
) -> list[str]:
    """Render mined patterns as natural-language reasoning rules."""

    rules = []
    for pattern in success_patterns[:limit]:
        rules.append(
            f"DO: follow the reasoning shape [{pattern.render()}] "
            f"(seen in {pattern.graph_support} successful runs)"
        )
    for pattern in failure_patterns[:limit]:
        rules.append(
            f"AVOID: the reasoning shape [{pattern.render()}] "
            f"(seen in {pattern.graph_support} failed runs)"
        )
    return rules


def stars_to_prompt_rules(
    success_stars: Sequence[StarPattern],
    failure_stars: Sequence[StarPattern] = (),
    *,
    limit: int = 10,
) -> list[str]:
    """Render mined star (branching) motifs as reasoning rules.

    Stars carry the branching information path patterns cannot: a failure
    star with high arity is an inefficiency signature (fan-out/backtrack),
    a success star pins the productive branching shape.
    """

    rules = []
    for star in success_stars[:limit]:
        rules.append(
            f"DO: at [{star.center}] steps, branch into "
            f"{{{', '.join(star.children)}}} "
            f"(seen in {star.graph_support} successful runs)"
        )
    for star in failure_stars[:limit]:
        rules.append(
            f"AVOID: at [{star.center}] steps, fanning out {star.arity} ways "
            f"into {{{', '.join(star.children)}}} "
            f"(seen in {star.graph_support} failed runs)"
        )
    return rules


def inject_rules_into_prompt(prompt: str, rules: Sequence[str]) -> str:
    """Prepend mined rules to a prompt in a standard format."""

    if not rules:
        return prompt
    block = "Reasoning guidelines derived from past runs:\n" + "\n".join(
        f"{i + 1}. {rule}" for i, rule in enumerate(rules)
    )
    return f"{block}\n\n{prompt}"


@dataclass(frozen=True, slots=True)
class StepReward:
    """One verifiable reward decision for a reasoning step."""

    step_index: int
    label_sequence: tuple[str, ...]
    reward: float
    matched_pattern: str | None


class PatternProcessReward:
    """Pattern-based per-step reward (GRPO-compatible callable).

    Score a reasoning trajectory (sequence of step labels) by sliding the
    mined pattern windows over it: success-pattern matches give +``gain``,
    failure-pattern matches −``penalty``.
    """

    def __init__(
        self,
        success_patterns: Sequence[Pattern] = (),
        failure_patterns: Sequence[Pattern] = (),
        *,
        success_stars: Sequence[StarPattern] = (),
        failure_stars: Sequence[StarPattern] = (),
        gain: float = 1.0,
        penalty: float = 1.0,
    ) -> None:
        self.patterns: dict[tuple[str, ...], float] = {}
        for pattern in success_patterns:
            self.patterns[pattern.sequence] = gain
        for pattern in failure_patterns:
            self.patterns[pattern.sequence] = -penalty
        self.star_patterns: dict[tuple[str, tuple[str, ...]], float] = {}
        for star in success_stars:
            self.star_patterns[(star.center, star.children)] = gain
        for star in failure_stars:
            self.star_patterns[(star.center, star.children)] = -penalty
        self.gain, self.penalty = gain, penalty

    def score_trajectory(self, labels: Sequence[str]) -> list[StepReward]:
        """Score every window-aligned step of one trajectory."""

        rewards: list[StepReward] = []
        if not self.patterns:
            return rewards
        length = max(len(seq) for seq in self.patterns)
        for start in range(len(labels)):
            for size in range(2, length + 1):
                window = tuple(labels[start : start + size])
                if window in self.patterns:
                    rewards.append(
                        StepReward(
                            step_index=start,
                            label_sequence=window,
                            reward=self.patterns[window],
                            matched_pattern=" → ".join(window),
                        )
                    )
        return rewards

    def __call__(self, labels: Sequence[str]) -> float:
        """Total trajectory reward (the RL-facing interface)."""

        return sum(step.reward for step in self.score_trajectory(labels))

    def score_graph(self, graph) -> float:
        """Star-motif reward over one ThoughtGraph's branching structure.

        Every node whose (label, sorted child-label multiset) matches a mined
        star contributes the star's reward — the graph-level complement of
        the sequence-based ``score_trajectory``.  Deterministic and immune to
        edge insertion order.
        """

        if not self.star_patterns:
            return 0.0
        children: dict[str, list[str]] = {}
        for src, dst in graph.edges:
            children.setdefault(src, []).append(dst)
        total = 0.0
        for center_node, child_nodes in children.items():
            if len(child_nodes) < 2:
                continue
            key = (
                graph.labels.get(center_node, "?"),
                tuple(sorted(graph.labels.get(child, "?") for child in child_nodes)),
            )
            total += self.star_patterns.get(key, 0.0)
        return total


__all__ = [
    "PatternProcessReward",
    "StepReward",
    "inject_rules_into_prompt",
    "patterns_to_prompt_rules",
    "stars_to_prompt_rules",
]
