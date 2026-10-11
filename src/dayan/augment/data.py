"""Dataset loading, schema inference, and format-preserving sample validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_OPTION_RE = re.compile(
    r"^\s*(?:\(([A-Za-z0-9]{1,3})\)|([A-Za-z0-9]{1,3})[).:])\s*(.*?)\s*$"
)
_SPACE_RE = re.compile(r"\s+")
_ALLOWED_SUFFIXES = {".json", ".jsonl"}


class DatasetError(ValueError):
    """The input cannot be used as a labeled reasoning dataset."""


@dataclass(frozen=True)
class Dataset:
    path: Path
    rows: list[dict[str, Any]]
    source_format: str  # "json" or "jsonl"


@dataclass(frozen=True)
class Schema:
    kind: str  # "nli", "qa", or "text"
    text_fields: tuple[str, ...]
    label_field: str
    observed_labels: tuple[str, ...]
    dataset_hint: str

    def label_space(self, row: dict[str, Any]) -> tuple[str, ...]:
        if self.kind == "qa" and isinstance(row.get("options"), list):
            options = option_map(row["options"])
            if options:
                return tuple(options)

        if self.kind == "nli":
            hint = self.dataset_hint.upper()
            observed = {canonical_relation(value) for value in self.observed_labels}
            if hint == "CB" or {"contradiction", "neutral"} & observed:
                return ("entailment", "contradiction", "neutral")
            if hint == "RTE" or "not_entailment" in observed:
                return ("entailment", "not_entailment")

        if self.kind == "qa":
            labels = tuple(str(value) for value in self.observed_labels)
            if len(labels) > 8 or any(len(label) > 40 for label in labels):
                return (str(row[self.label_field]),)
            return labels

        return self.observed_labels

    def resolve_label(self, prediction: Any, row: dict[str, Any]) -> str | None:
        if prediction is None:
            return None
        value = str(prediction).strip()
        if not value:
            return None
        labels = self.label_space(row)
        if self.kind == "qa" and isinstance(row.get("options"), list):
            options = option_map(row["options"])
            upper = value.upper()
            if upper in options:
                return upper
            match = _OPTION_RE.match(value)
            if match:
                key = (match.group(1) or match.group(2)).upper()
                if key in options:
                    return key
            for key, text in options.items():
                if value.casefold() == text.casefold():
                    return key
            return None

        if self.kind == "nli":
            relation = canonical_relation(value)
            allowed = {canonical_relation(label): label for label in labels}
            # Index the entailment-form of every label too: datasets labeled
            # plain "True"/"False" must resolve solver relations
            # "entailment"/"contradiction" back to them (Logical DA z3 fix).
            for label in labels:
                canon = canonical_relation(label)
                mapped = {
                    "true": "entailment",
                    "entailed": "entailment",
                    "false": "contradiction",
                    "contradicted": "contradiction",
                }.get(canon)
                if mapped:
                    allowed.setdefault(mapped, label)
            if relation in allowed:
                return allowed[relation]
            relation = {
                "true": "entailment",
                "entailed": "entailment",
                "false": "contradiction",
                "contradicted": "contradiction",
            }.get(relation, relation)
            if relation in allowed:
                return allowed[relation]
            if relation in {"contradiction", "neutral"} and "not_entailment" in allowed:
                return allowed["not_entailment"]
            return None

        for label in labels:
            if value.casefold() == str(label).casefold():
                return label
        return None

    def truth_label(self, truth: str, row: dict[str, Any]) -> str | None:
        if truth not in {"true", "false", "unknown"}:
            return None
        if self.kind == "nli":
            desired = {
                "true": "entailment",
                "false": "contradiction",
                "unknown": "neutral",
            }[truth]
            return self.resolve_label(desired, row)
        if self.kind == "qa" and isinstance(row.get("options"), list):
            aliases = {
                "true": {"true", "yes", "entailed"},
                "false": {"false", "no", "contradicted"},
                "unknown": {"unknown", "uncertain", "undetermined", "not enough information", "cannot be determined"},
            }[truth]
            for key, text in option_map(row["options"]).items():
                normalized = _SPACE_RE.sub(" ", text.strip().lower())
                if normalized in aliases:
                    return key
        return None


def canonical_relation(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return {
        "no_entailment": "not_entailment",
        "non_entailment": "not_entailment",
        "not_entailed": "not_entailment",
        "contradictory": "contradiction",
        "unknown": "neutral",
        "uncertain": "neutral",
    }.get(key, key)


def option_map(options: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for option in options:
        if not isinstance(option, str):
            return {}
        match = _OPTION_RE.match(option)
        if not match:
            return {}
        key = (match.group(1) or match.group(2)).upper()
        if key in result or not match.group(3).strip():
            return {}
        result[key] = match.group(3).strip()
    return result


def discover_inputs(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise DatasetError(f"Unsupported input extension: {path}")
        return [path]
    if not path.is_dir():
        raise DatasetError(f"Input does not exist: {path}")

    candidates = sorted(
        candidate for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in _ALLOWED_SUFFIXES
        and not any(part.startswith(".") for part in candidate.relative_to(path).parts)
    )
    training = [
        candidate for candidate in candidates
        if candidate.stem.lower() == "train"
        or candidate.stem.lower().startswith("train_")
    ]
    if training:
        # Some legacy folders contain the same JSON array twice, as train.json
        # and train.jsonl. Prefer the clearly named JSON file in directory mode.
        chosen: dict[tuple[Path, str], Path] = {}
        for candidate in training:
            key = (candidate.parent, candidate.stem.lower())
            if key not in chosen or candidate.suffix.lower() == ".json":
                chosen[key] = candidate
        return sorted(chosen.values())
    if len(candidates) == 1:
        return candidates
    if not candidates:
        raise DatasetError(f"No JSON or JSONL data files found under {path}")
    raise DatasetError(
        f"{path} contains multiple data files but no train.json/train.jsonl. "
        "Pass one file explicitly to avoid augmenting evaluation data."
    )


def load_dataset(path: Path) -> Dataset:
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        raise DatasetError(f"Input is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    f"{path}:{line_number}: invalid JSONL: {exc.msg}"
                ) from exc
        source_format = "jsonl"
    else:
        if isinstance(parsed, list):
            rows = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            rows = parsed["data"]
        elif isinstance(parsed, dict):
            rows = [parsed]
        else:
            raise DatasetError(f"{path}: top-level JSON must be an object or array")
        source_format = (
            "jsonl"
            if isinstance(parsed, dict)
            and not isinstance(parsed.get("data"), list)
            and path.suffix.lower() == ".jsonl"
            else "json"
        )

    if not rows:
        raise DatasetError(f"Input contains no records: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetError(f"{path}: record {index} is not a JSON object")
    return Dataset(path=path, rows=rows, source_format=source_format)


def _dataset_hint(dataset: Dataset) -> str:
    """Keep automatic solver routing when an original dataset file is moved."""
    parent = dataset.path.parent.name
    known = {
        "folio": "FOLIO",
        "proofwriter": "ProofWriter",
        "prontoqa": "ProntoQA",
        "logicaldeduction": "LogicalDeduction",
        "arlsat": "AR-LSAT",
        "cb": "CB",
        "rte": "RTE",
    }
    normalized = re.sub(r"[^a-z0-9]", "", parent.lower())
    if normalized in known:
        return known[normalized]
    identifier = str(dataset.rows[0].get("id", ""))
    for prefix, hint in (
        (r"^folio[_-]", "FOLIO"),
        (r"^proofwriter[_-]", "ProofWriter"),
        (r"^prontoqa[_-]", "ProntoQA"),
        (r"^logical[_-]?deduction[_-]", "LogicalDeduction"),
        (r"^ar[_-]?lsat[_-]", "AR-LSAT"),
    ):
        if re.match(prefix, identifier, re.IGNORECASE):
            return hint
    return parent


def infer_schema(dataset: Dataset) -> Schema:
    first = dataset.rows[0]
    if {"premise", "hypothesis", "label"} <= first.keys():
        kind, fields, label_field = "nli", ("premise", "hypothesis"), "label"
    elif {"context", "question", "answer"} <= first.keys():
        kind, fields, label_field = "qa", ("context", "question"), "answer"
    elif {"text", "label"} <= first.keys():
        kind, fields, label_field = "text", ("text",), "label"
    else:
        raise DatasetError(
            f"{dataset.path}: records need premise/hypothesis/label, "
            "context/question/answer, or text/label"
        )

    seen: list[str] = []
    for index, row in enumerate(dataset.rows):
        for field in (*fields, label_field):
            if field not in row:
                raise DatasetError(f"{dataset.path}: record {index} lacks {field!r}")
        if not all(isinstance(row[field], str) and row[field].strip() for field in fields):
            raise DatasetError(f"{dataset.path}: record {index} has an empty text field")
        if row[label_field] is None or str(row[label_field]).strip() == "":
            raise DatasetError(f"{dataset.path}: record {index} has an empty label")
        if "options" in row and row["options"] is not None:
            if not isinstance(row["options"], list) or not option_map(row["options"]):
                raise DatasetError(
                    f"{dataset.path}: record {index} has invalid labeled options"
                )
            answer = str(row[label_field]).strip().upper()
            if kind == "qa" and answer not in option_map(row["options"]):
                raise DatasetError(
                    f"{dataset.path}: record {index} answer {answer!r} "
                    "is not an option key"
                )
        label = str(row[label_field])
        if label not in seen:
            seen.append(label)
    return Schema(
        kind=kind,
        text_fields=fields,
        label_field=label_field,
        observed_labels=tuple(seen),
        dataset_hint=_dataset_hint(dataset),
    )


def sample_text(row: dict[str, Any], schema: Schema) -> str:
    parts = [str(row[field]) for field in schema.text_fields]
    if isinstance(row.get("options"), list):
        parts.extend(str(value) for value in row["options"])
    return "\n".join(parts)


def text_signature(row: dict[str, Any], schema: Schema) -> str:
    return _SPACE_RE.sub(" ", sample_text(row, schema).strip()).casefold()


def cast_label_like(label: str, original: Any) -> Any:
    """Keep numeric and Boolean label types from the source dataset."""
    if isinstance(original, bool):
        if label.lower() in {"true", "false"}:
            return label.lower() == "true"
        raise DatasetError(f"Cannot convert label {label!r} to bool")
    if isinstance(original, int):
        try:
            return int(label)
        except ValueError as exc:
            raise DatasetError(f"Cannot convert label {label!r} to int") from exc
    if isinstance(original, float):
        try:
            return float(label)
        except ValueError as exc:
            raise DatasetError(f"Cannot convert label {label!r} to float") from exc
    return label


def build_candidate(
    source: dict[str, Any],
    generated: Any,
    schema: Schema,
    label: str,
) -> dict[str, Any]:
    if isinstance(generated, dict):
        generated = generated.get("sample", generated.get("data", generated))
    if not isinstance(generated, dict):
        raise DatasetError("Agent returned no sample object")

    candidate = dict(source)
    for field in schema.text_fields:
        value = generated.get(field, source[field])
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"Generated {field!r} is empty or not text")
        candidate[field] = value.strip()

    if "options" in source and source["options"] is not None:
        options = generated.get("options", source["options"])
        if not isinstance(options, list) or len(options) != len(source["options"]):
            raise DatasetError("Generated options changed the number of options")
        original_keys = tuple(option_map(source["options"]))
        generated_keys = tuple(option_map(options))
        if not generated_keys or generated_keys != original_keys:
            raise DatasetError("Generated options changed their keys or order")
        candidate["options"] = options
    elif "options" in generated:
        raise DatasetError("Generated options were added to an option-free sample")

    if text_signature(candidate, schema) == text_signature(source, schema):
        raise DatasetError("Generated sample is identical to its source")
    candidate[schema.label_field] = cast_label_like(label, source[schema.label_field])
    return candidate


def assign_identity(
    source: dict[str, Any],
    candidate: dict[str, Any],
    serial: int,
    next_numeric_idx: int | None,
) -> int | None:
    if "id" in source:
        candidate["id"] = f"{source['id']}__logical_da_{serial}"
    if "idx" in source:
        if isinstance(source["idx"], int) and next_numeric_idx is not None:
            candidate["idx"] = next_numeric_idx
            next_numeric_idx += 1
        else:
            candidate["idx"] = f"{source['idx']}__logical_da_{serial}"
    return next_numeric_idx


def next_idx(rows: list[dict[str, Any]]) -> int | None:
    values = [row["idx"] for row in rows if isinstance(row.get("idx"), int)]
    return max(values) + 1 if values else None


def write_dataset(path: Path, rows: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if fmt == "json":
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        elif fmt == "jsonl":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            raise ValueError(f"Unknown output format: {fmt}")
    temporary.replace(path)
