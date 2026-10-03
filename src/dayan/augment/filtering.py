"""Bounded attribute combinations and diversity filtering."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .data import Schema, sample_text


@dataclass(frozen=True)
class Attribute:
    name: str
    description: str
    current_value: str
    alternatives: tuple[str, ...]


def parse_attributes(raw: Any, limit: int = 6) -> list[Attribute]:
    values = raw.get("attributes") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        return []

    attributes: list[Attribute] = []
    names: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name", "")).strip()
        current = str(value.get("current_value", "")).strip()
        choices = value.get("alternatives", value.get("possible_values", []))
        if not name or not current or not isinstance(choices, list):
            continue
        if name.casefold() in names:
            continue
        alternatives: list[str] = []
        for choice in choices:
            text = str(choice).strip()
            if (
                text
                and text.casefold() != current.casefold()
                and text.casefold() not in {item.casefold() for item in alternatives}
            ):
                alternatives.append(text)
        if not alternatives:
            continue
        names.add(name.casefold())
        attributes.append(
            Attribute(
                name=name,
                description=str(value.get("description", "")).strip(),
                current_value=current,
                alternatives=tuple(alternatives[:4]),
            )
        )
        if len(attributes) >= limit:
            break
    return attributes


def build_combinations(
    attributes: list[Attribute],
    max_combinations: int = 12,
    seed: int = 42,
) -> list[dict[str, str]]:
    if max_combinations < 1:
        raise ValueError("max_combinations must be positive")
    if not attributes:
        return [{}]

    base = {attribute.name: attribute.current_value for attribute in attributes}
    singles: list[dict[str, str]] = []
    for attribute in attributes:
        for alternative in attribute.alternatives:
            variant = dict(base)
            variant[attribute.name] = alternative
            singles.append(variant)

    pairs: list[dict[str, str]] = []
    for first, second in combinations(attributes, 2):
        for first_value in first.alternatives[:2]:
            for second_value in second.alternatives[:2]:
                variant = dict(base)
                variant[first.name] = first_value
                variant[second.name] = second_value
                pairs.append(variant)

    rng = random.Random(seed)
    rng.shuffle(singles)
    rng.shuffle(pairs)
    result: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    # Reserve part of the budget for combinations with two simultaneous changes.
    single_budget = max(1, (max_combinations + 1) // 2) if pairs else max_combinations
    for group, budget in ((singles, single_budget), (pairs, max_combinations)):
        for variant in group:
            key = tuple(sorted(variant.items()))
            if key in seen:
                continue
            seen.add(key)
            result.append(variant)
            if len(result) >= max_combinations or (
                group is singles and len(result) >= budget
            ):
                break
        if len(result) >= max_combinations:
            break

    if len(result) < max_combinations:
        for variant in singles:
            key = tuple(sorted(variant.items()))
            if key not in seen:
                seen.add(key)
                result.append(variant)
            if len(result) >= max_combinations:
                break
    return result or [{}]


def selected_indices(raw: Any, size: int, limit: int) -> list[int]:
    choices = raw.get("selected_indices") if isinstance(raw, dict) else None
    if not isinstance(choices, list):
        return list(range(min(size, limit)))
    selected: list[int] = []
    for value in choices:
        if isinstance(value, int) and 0 <= value < size and value not in selected:
            selected.append(value)
        if len(selected) >= limit:
            break
    return selected


def _lexical_vector(text: str) -> Counter[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)
    characters = re.sub(r"\s+", " ", text.casefold())
    features = [f"word:{token}" for token in tokens]
    features.extend(
        f"pair:{tokens[index]}_{tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )
    features.extend(
        f"char:{characters[index:index + 3]}"
        for index in range(max(0, len(characters) - 2))
    )
    return Counter(features)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    common = left.keys() & right.keys()
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class DiversityFilter:
    """Use Sentence-BERT when available, with a local lexical fallback."""

    def __init__(self, backend: str = "auto") -> None:
        if backend not in {"auto", "sbert", "lexical"}:
            raise ValueError(f"Unknown diversity backend: {backend}")
        self.requested_backend = backend
        self.backend = "lexical"
        self.error: str | None = None
        self._model: Any = None

    def _get_model(self) -> Any | None:
        if self.requested_backend == "lexical":
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.backend = "sbert"
            return self._model
        except Exception as exc:
            self.error = str(exc)
            if self.requested_backend == "sbert":
                raise RuntimeError(
                    "Sentence-BERT could not be loaded. Install sentence-transformers "
                    "and make the all-MiniLM-L6-v2 model available."
                ) from exc
            self.backend = "lexical"
            return None

    def select(
        self,
        samples: list[dict[str, Any]],
        schema: Schema,
        count: int,
        reference: dict[str, Any],
    ) -> list[int]:
        if count <= 0 or not samples:
            return []
        if len(samples) <= count:
            return list(range(len(samples)))

        texts = [sample_text(reference, schema)] + [
            sample_text(sample, schema) for sample in samples
        ]
        model = self._get_model()
        if model is not None:
            embeddings = model.encode(texts, normalize_embeddings=True)
            similarity = [
                [
                    sum(float(a) * float(b) for a, b in zip(left, right))
                    for right in embeddings
                ]
                for left in embeddings
            ]
        else:
            vectors = [_lexical_vector(text) for text in texts]
            similarity = [
                [_cosine(left, right) for right in vectors]
                for left in vectors
            ]

        # Paper Eq. 2 maximizes total pairwise cosine distance among seeds.
        candidate_indices = list(range(1, len(texts)))
        if count == 1:
            best = min(candidate_indices, key=lambda index: (similarity[index][0], index))
            return [best - 1]

        if math.comb(len(samples), count) <= 50000:
            best_group: tuple[int, ...] | None = None
            best_score = float("-inf")
            best_reference_distance = float("-inf")
            for group in combinations(candidate_indices, count):
                score = sum(
                    1.0 - similarity[left][right]
                    for left, right in combinations(group, 2)
                )
                reference_distance = sum(1.0 - similarity[index][0] for index in group)
                if (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and reference_distance > best_reference_distance + 1e-12
                    )
                ):
                    best_group = group
                    best_score = score
                    best_reference_distance = reference_distance
            assert best_group is not None
            return [index - 1 for index in best_group]

        # Large user-specified candidate pools use a bounded greedy approximation.
        selected = [
            min(candidate_indices, key=lambda index: (similarity[index][0], index))
        ]
        remaining = set(candidate_indices) - set(selected)
        while remaining and len(selected) < count:
            next_index = max(
                remaining,
                key=lambda index: (
                    sum(1.0 - similarity[index][other] for other in selected),
                    1.0 - similarity[index][0],
                    -index,
                ),
            )
            selected.append(next_index)
            remaining.remove(next_index)
        return [index - 1 for index in selected]
