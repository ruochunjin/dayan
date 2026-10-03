"""Two-stage Logical DA pipeline with feedback and label correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .data import (
    Dataset,
    DatasetError,
    Schema,
    assign_identity,
    build_candidate,
    cast_label_like,
    infer_schema,
    next_idx,
    text_signature,
)
from .filtering import (
    DiversityFilter,
    build_combinations,
    parse_attributes,
    selected_indices,
)
from .prompts import sample_view
from .symbolic_backends import SOLVERS, select_solver
from .verification import Verification, evaluate_verification


class Workflow(Protocol):
    def extract(self, source: dict[str, Any], schema: Schema) -> dict[str, Any]: ...

    def filter_combinations(
        self,
        source: dict[str, Any],
        schema: Schema,
        attributes: list[Any],
        combinations: list[dict[str, str]],
        limit: int,
    ) -> dict[str, Any]: ...

    def generate(
        self,
        source: dict[str, Any],
        schema: Schema,
        requests: list[dict[str, Any]],
        stage: str,
    ) -> dict[str, Any]: ...

    def verify(
        self,
        candidates: list[dict[str, Any]],
        schema: Schema,
        labels: tuple[str, ...],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PipelineConfig:
    initial_candidates: int = 5
    seed_count: int = 3
    max_generated_per_record: int | None = None
    reflection_rounds: int = 1
    batch_size: int = 3
    max_combinations: int = 12
    seed: int = 42
    max_records: int | None = None
    require_symbolic: bool = False
    symbolic_solver: str = "auto"

    def __post_init__(self) -> None:
        for name in ("initial_candidates", "seed_count", "batch_size", "max_combinations"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.reflection_rounds < 0:
            raise ValueError("reflection_rounds cannot be negative")
        if self.max_generated_per_record is not None and self.max_generated_per_record < 0:
            raise ValueError("max_generated_per_record cannot be negative")
        if self.max_records is not None and self.max_records < 1:
            raise ValueError("max_records must be positive")
        if self.symbolic_solver not in SOLVERS:
            raise ValueError(f"symbolic_solver must be one of {SOLVERS}")
        if self.require_symbolic and self.symbolic_solver == "llm":
            raise ValueError("--strict-symbolic cannot be combined with --solver llm")


@dataclass
class CandidateState:
    id: str
    seed: dict[str, Any]
    target_label: str
    candidate: dict[str, Any] | None = None
    verification: Verification | None = None
    rounds: int = 0
    error: str | None = None


@dataclass
class RunResult:
    rows: list[dict[str, Any]]
    report: dict[str, Any]
    audit: list[dict[str, Any]]


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _response_items(raw: Any, key: str) -> dict[str, dict[str, Any]]:
    values = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"Agent response lacks a {key!r} array")
    indexed: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        item_id = value.get("id")
        if isinstance(item_id, str) and item_id not in indexed:
            indexed[item_id] = value
    return indexed


def _final_plan(
    seeds: list[dict[str, Any]],
    labels: tuple[str, ...],
    cap: int | None,
) -> list[tuple[int, str]]:
    if not seeds or not labels:
        return []
    plan: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for rotation in range(len(seeds)):
        for label_index, label in enumerate(labels):
            seed_index = (label_index + rotation) % len(seeds)
            pair = (seed_index, label)
            if pair not in seen:
                seen.add(pair)
                plan.append(pair)
    if cap is None:
        limit = max(6, len(labels))
    elif cap == 0:
        limit = len(plan)
    else:
        limit = cap
    return plan[:limit]


def _generate(
    workflow: Workflow,
    source: dict[str, Any],
    schema: Schema,
    requests: list[dict[str, Any]],
    stage: str,
    batch_size: int,
    errors: list[str],
) -> dict[str, Any]:
    generated: dict[str, Any] = {}
    for batch in _chunks(requests, batch_size):
        try:
            indexed = _response_items(
                workflow.generate(source, schema, batch, stage), "candidates"
            )
        except ValueError as exc:
            errors.append(f"{stage} generation response: {exc}")
            continue
        for request in batch:
            item = indexed.get(request["id"])
            if item is None:
                errors.append(f"{stage} generation omitted {request['id']}")
            elif "sample" not in item:
                errors.append(f"{stage} generation omitted sample for {request['id']}")
            else:
                generated[request["id"]] = item["sample"]
    return generated


def _verify_states(
    workflow: Workflow,
    states: list[CandidateState],
    schema: Schema,
    labels: tuple[str, ...],
    batch_size: int,
    require_symbolic: bool,
    solver: str,
    errors: list[str],
) -> None:
    active = [state for state in states if state.candidate is not None]
    for batch in _chunks(active, batch_size):
        payload = [
            {"id": state.id, "sample": state.candidate}
            for state in batch
            if state.candidate is not None
        ]
        try:
            indexed = _response_items(
                workflow.verify(payload, schema, labels), "verifications"
            )
        except ValueError as exc:
            errors.append(f"verification response: {exc}")
            indexed = {}
        for state in batch:
            if state.candidate is None:
                continue
            raw = indexed.get(state.id)
            state.verification = evaluate_verification(
                raw, schema, state.candidate, require_symbolic, solver=solver
            )
            if not state.verification.valid:
                state.error = state.verification.error


def _feedback(state: CandidateState) -> str:
    if state.candidate is None:
        return "The previous generation returned no usable sample. Generate a complete new sample."
    if state.verification is None:
        return state.error or "The previous sample was not verified."
    if state.verification.valid:
        return (
            f"The independent verifier inferred {state.verification.label!r}, "
            f"but the requested target label is {state.target_label!r}. "
            f"Evidence: {state.verification.explanation}. "
            "Revise the logical facts or question so the target label is truly correct."
        )
    return (
        "The previous sample could not be verified. "
        f"Reason: {state.verification.error or 'invalid logical encoding'}. "
        "Generate a clearer, logically consistent sample."
    )


def run_dataset(
    dataset: Dataset,
    workflow: Workflow,
    config: PipelineConfig | None = None,
    diversity: DiversityFilter | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> RunResult:
    config = config or PipelineConfig()
    diversity = diversity or DiversityFilter()
    schema = infer_schema(dataset)
    output_rows = [dict(row) for row in dataset.rows]
    all_signatures = {text_signature(row, schema) for row in dataset.rows}
    numeric_idx = next_idx(dataset.rows)
    serial = 1
    known_ids = {str(row["id"]) for row in dataset.rows if "id" in row}
    known_idxs = {str(row["idx"]) for row in dataset.rows if "idx" in row}
    audit: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "source": str(dataset.path),
        "input_records": len(dataset.rows),
        "processed_records": 0,
        "generated_records": 0,
        "relabeled_records": 0,
        "rejected_candidates": 0,
        "records_without_augmentation": 0,
        "initial_samples": 0,
        "reflection_attempts": 0,
        "verification_modes": {},
        "preferred_solver": select_solver(schema, config.symbolic_solver),
        "symbolic_solvers": {},
        "diversity_backend": diversity.backend,
    }

    selected_rows = (
        dataset.rows[: config.max_records] if config.max_records is not None
        else dataset.rows
    )
    for row_index, source in enumerate(selected_rows):
        labels = schema.label_space(source)
        entry: dict[str, Any] = {
            "source_index": row_index,
            "source_id": source.get("id", source.get("idx", row_index)),
            "label_space": list(labels),
            "initial_samples": 0,
            "selected_seeds": 0,
            "candidates": [],
            "errors": [],
        }
        errors: list[str] = entry["errors"]

        try:
            attributes = parse_attributes(workflow.extract(source, schema))
        except ValueError as exc:
            attributes = []
            errors.append(f"attribute extraction response: {exc}")

        combinations = build_combinations(
            attributes,
            max_combinations=config.max_combinations,
            seed=config.seed + row_index,
        )
        if attributes:
            try:
                filtered = workflow.filter_combinations(
                    source,
                    schema,
                    attributes,
                    combinations,
                    config.initial_candidates,
                )
                indices = selected_indices(
                    filtered, len(combinations), config.initial_candidates
                )
            except ValueError as exc:
                errors.append(f"attribute filtering response: {exc}")
                indices = list(range(min(len(combinations), config.initial_candidates)))
        else:
            indices = []

        initial_requests = [
            {"id": f"i{position}", "combination": combinations[index]}
            for position, index in enumerate(indices)
        ]
        initial_raw = _generate(
            workflow,
            source,
            schema,
            initial_requests,
            "initial",
            config.batch_size,
            errors,
        )
        initial: list[dict[str, Any]] = []
        initial_signatures: set[str] = set()
        for request in initial_requests:
            item = initial_raw.get(request["id"])
            if item is None:
                continue
            try:
                candidate = build_candidate(
                    source, item, schema, str(source[schema.label_field])
                )
            except DatasetError as exc:
                errors.append(f"initial {request['id']}: {exc}")
                continue
            signature = text_signature(candidate, schema)
            if signature in initial_signatures or signature in all_signatures:
                errors.append(f"initial {request['id']}: duplicate text")
                continue
            initial_signatures.add(signature)
            initial.append(candidate)

        entry["initial_samples"] = len(initial)
        report["initial_samples"] += len(initial)
        if initial:
            selected = diversity.select(initial, schema, config.seed_count, source)
            seeds = [initial[index] for index in selected]
        else:
            seeds = [source]
            entry["used_original_as_seed"] = True
        entry["selected_seeds"] = len(seeds)

        plan = _final_plan(seeds, labels, config.max_generated_per_record)
        states: list[CandidateState] = []
        final_requests: list[dict[str, Any]] = []
        for position, (seed_index, target) in enumerate(plan):
            state = CandidateState(
                id=f"f{position}",
                seed=seeds[seed_index],
                target_label=target,
            )
            states.append(state)
            final_requests.append(
                {
                    "id": state.id,
                    "seed": sample_view(state.seed, schema, False),
                    "target_label": target,
                }
            )

        final_raw = _generate(
            workflow,
            source,
            schema,
            final_requests,
            "final",
            config.batch_size,
            errors,
        )
        pending_signatures: set[str] = set()
        for state in states:
            item = final_raw.get(state.id)
            if item is None:
                state.error = "Generator returned no sample"
                continue
            try:
                candidate = build_candidate(
                    state.seed, item, schema, state.target_label
                )
            except DatasetError as exc:
                state.error = str(exc)
                continue
            signature = text_signature(candidate, schema)
            if signature in all_signatures or signature in pending_signatures:
                state.error = "Duplicate text"
                continue
            pending_signatures.add(signature)
            state.candidate = candidate

        _verify_states(
            workflow,
            states,
            schema,
            labels,
            config.batch_size,
            config.require_symbolic,
            config.symbolic_solver,
            errors,
        )

        for _ in range(config.reflection_rounds):
            needs_repair = [
                state for state in states
                if state.verification is None
                or not state.verification.valid
                or state.verification.label != state.target_label
            ]
            if not needs_repair:
                break
            repair_requests = [
                {
                    "id": state.id,
                    "seed": sample_view(state.seed, schema, False),
                    "target_label": state.target_label,
                    "previous_sample": (
                        sample_view(state.candidate, schema, False)
                        if state.candidate is not None else None
                    ),
                    "feedback": _feedback(state),
                }
                for state in needs_repair
            ]
            repaired_raw = _generate(
                workflow,
                source,
                schema,
                repair_requests,
                "final",
                config.batch_size,
                errors,
            )
            revised: list[CandidateState] = []
            occupied = {
                text_signature(state.candidate, schema)
                for state in states if state.candidate is not None
            }
            for state in needs_repair:
                state.rounds += 1
                report["reflection_attempts"] += 1
                item = repaired_raw.get(state.id)
                if item is None:
                    continue
                try:
                    candidate = build_candidate(
                        state.seed, item, schema, state.target_label
                    )
                except DatasetError as exc:
                    errors.append(f"reflection {state.id}: {exc}")
                    continue
                signature = text_signature(candidate, schema)
                if signature in all_signatures or signature in occupied:
                    errors.append(f"reflection {state.id}: duplicate text")
                    continue
                occupied.add(signature)
                state.candidate = candidate
                state.verification = None
                state.error = None
                revised.append(state)
            _verify_states(
                workflow,
                revised,
                schema,
                labels,
                config.batch_size,
                config.require_symbolic,
                config.symbolic_solver,
                errors,
            )

        accepted_for_source = 0
        for state in states:
            record = {
                "id": state.id,
                "requested_label": state.target_label,
                "reflection_rounds": state.rounds,
            }
            if state.candidate is None or state.verification is None:
                record.update(status="rejected", error=state.error or "No candidate")
                report["rejected_candidates"] += 1
                entry["candidates"].append(record)
                continue
            verification = state.verification
            if not verification.valid:
                record.update(
                    status="rejected",
                    error=verification.error or state.error or "Verification failed",
                    verification_method=verification.method,
                )
                if verification.solver:
                    record["symbolic_solver"] = verification.solver
                report["rejected_candidates"] += 1
                entry["candidates"].append(record)
                continue
            signature = text_signature(state.candidate, schema)
            if signature in all_signatures:
                record.update(status="rejected", error="Duplicate text")
                report["rejected_candidates"] += 1
                entry["candidates"].append(record)
                continue

            final_label = verification.label
            assert final_label is not None
            generated = dict(state.candidate)
            generated[schema.label_field] = cast_label_like(
                final_label, source[schema.label_field]
            )
            while (
                ("id" in source and f"{source['id']}__logical_da_{serial}" in known_ids)
                or (
                    "idx" in source
                    and not isinstance(source["idx"], int)
                    and f"{source['idx']}__logical_da_{serial}" in known_idxs
                )
            ):
                serial += 1
            numeric_idx = assign_identity(
                source, generated, serial, numeric_idx
            )
            serial += 1
            if "id" in generated:
                known_ids.add(str(generated["id"]))
            if "idx" in generated:
                known_idxs.add(str(generated["idx"]))
            all_signatures.add(signature)
            output_rows.append(generated)
            accepted_for_source += 1
            report["generated_records"] += 1
            modes = report["verification_modes"]
            modes[verification.method] = modes.get(verification.method, 0) + 1
            if verification.solver:
                solvers = report["symbolic_solvers"]
                solvers[verification.solver] = solvers.get(verification.solver, 0) + 1
                record["symbolic_solver"] = verification.solver
            corrected = final_label != state.target_label
            if corrected:
                report["relabeled_records"] += 1
            record.update(
                status="relabeled" if corrected else "accepted",
                final_label=final_label,
                verification_method=verification.method,
                explanation=verification.explanation,
            )
            if "id" in generated:
                record["output_id"] = generated["id"]
            if "idx" in generated:
                record["output_idx"] = generated["idx"]
            if verification.error:
                record["symbolic_error"] = verification.error
            entry["candidates"].append(record)

        if accepted_for_source == 0:
            report["records_without_augmentation"] += 1
        report["processed_records"] += 1
        audit.append(entry)
        if progress is not None:
            progress(row_index + 1, len(selected_rows), accepted_for_source)

    report["diversity_backend"] = diversity.backend
    if diversity.error:
        report["diversity_fallback_reason"] = diversity.error
    return RunResult(rows=output_rows, report=report, audit=audit)
