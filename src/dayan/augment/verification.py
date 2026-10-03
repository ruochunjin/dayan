"""Safe SMT verification of generated labels, with an independent LLM fallback.

Only SMT-LIB declarations and assertions are parsed by Z3. Model-generated Python
is never executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .data import Schema, option_map
from .symbolic_backends import (
    SymbolicError,
    select_solver,
    solve_csp,
    solve_fol,
    solve_rules,
)

_ALLOWED_COMMANDS = {
    "set-logic",
    "declare-const",
    "declare-fun",
    "declare-sort",
    "declare-datatype",
    "declare-datatypes",
    "define-fun",
    "define-sort",
    "assert",
}
_ALLOWED_RULES = {"must_true", "must_false", "could_true", "could_false"}
_SIMPLE_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Verification:
    label: str | None
    method: str
    explanation: str
    error: str | None = None
    solver: str | None = None

    @property
    def valid(self) -> bool:
        return self.label is not None


def _without_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _top_level_forms(text: str) -> list[str]:
    forms: list[str] = []
    depth = 0
    start = -1
    for index, character in enumerate(text):
        if character == "(":
            if depth == 0:
                start = index
            depth += 1
            if depth > 80:
                raise SymbolicError("SMT expression is too deeply nested")
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise SymbolicError("Unbalanced SMT parentheses")
            if depth == 0:
                forms.append(text[start : index + 1])
                start = -1
        elif depth == 0 and not character.isspace():
            raise SymbolicError("Text outside an SMT command")
    if depth != 0:
        raise SymbolicError("Unbalanced SMT parentheses")
    return forms


def _validate_base(program: str) -> str:
    if not isinstance(program, str) or not program.strip():
        raise SymbolicError("No SMT program")
    if len(program) > 20000:
        raise SymbolicError("SMT program is too long")
    cleaned = _without_comments(program).strip()
    forms = _top_level_forms(cleaned)
    if not forms or len(forms) > 150:
        raise SymbolicError("SMT program has no commands or too many commands")
    for form in forms:
        match = re.match(r"^\(\s*([^\s()]+)", form)
        if not match or match.group(1) not in _ALLOWED_COMMANDS:
            raise SymbolicError("SMT program contains an unsupported command")
    if not any(re.match(r"^\(\s*assert\b", form) for form in forms):
        raise SymbolicError("SMT program contains no assertions")
    return cleaned


def _validate_formula(formula: str) -> str:
    if not isinstance(formula, str) or not formula.strip():
        raise SymbolicError("No SMT formula")
    if len(formula) > 4000:
        raise SymbolicError("SMT formula is too long")
    cleaned = _without_comments(formula).strip()
    if _SIMPLE_SYMBOL.fullmatch(cleaned):
        return cleaned
    forms = _top_level_forms(cleaned)
    if len(forms) != 1:
        raise SymbolicError("Expected one SMT formula")
    command = re.match(r"^\(\s*([^\s()]+)", forms[0])
    if command and command.group(1) in _ALLOWED_COMMANDS:
        raise SymbolicError("Expected an expression, not a command")
    return cleaned


def _check(program: str, extra_assertion: str | None = None, timeout_ms: int = 3000) -> str:
    try:
        import z3
    except ImportError as exc:
        raise SymbolicError(
            "z3-solver is not installed; symbolic verification is unavailable"
        ) from exc

    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    full_program = program
    if extra_assertion is not None:
        full_program += f"\n(assert {extra_assertion})"
    try:
        solver.from_string(full_program)
        result = solver.check()
    except Exception as exc:
        raise SymbolicError(f"Z3 could not parse or execute the program: {exc}") from exc
    if result == z3.sat:
        return "sat"
    if result == z3.unsat:
        return "unsat"
    raise SymbolicError(f"Z3 returned unknown: {solver.reason_unknown()}")


def _require_satisfiable(program: str) -> None:
    if _check(program) != "sat":
        raise SymbolicError("The generated premises are inconsistent")


def solve_truth(program: str, hypothesis: str) -> str:
    """Return true, false, or unknown using entailment and contradiction checks."""
    base = _validate_base(program)
    proposition = _validate_formula(hypothesis)
    _require_satisfiable(base)
    positive = _check(base, proposition)
    negative = _check(base, f"(not {proposition})")
    if negative == "unsat":
        return "true"
    if positive == "unsat":
        return "false"
    return "unknown"


def solve_options(
    program: str,
    option_formulas: dict[str, str],
    selection_rule: str,
) -> str:
    """Return the unique option satisfying the question's logical selection rule."""
    base = _validate_base(program)
    rule = str(selection_rule).strip().lower().replace("-", "_").replace(" ", "_")
    if rule not in _ALLOWED_RULES:
        raise SymbolicError(f"Unsupported option selection rule: {selection_rule!r}")
    if not isinstance(option_formulas, dict) or not option_formulas:
        raise SymbolicError("No symbolic option formulas")
    _require_satisfiable(base)

    matches: list[str] = []
    for key, raw_formula in option_formulas.items():
        formula = _validate_formula(raw_formula)
        positive = _check(base, formula)
        negative = _check(base, f"(not {formula})")
        match = {
            "must_true": negative == "unsat",
            "must_false": positive == "unsat",
            "could_true": positive == "sat",
            "could_false": negative == "sat",
        }[rule]
        if match:
            matches.append(str(key).upper())
    if len(matches) != 1:
        raise SymbolicError(
            f"The solver found {len(matches)} matching options; expected exactly one"
        )
    return matches[0]


def _require_complete_options(
    formulas: dict[str, Any], candidate: dict[str, Any]
) -> None:
    if not isinstance(candidate.get("options"), list):
        return
    expected = set(option_map(candidate["options"]))
    actual = {str(key).upper() for key in formulas}
    if not expected or actual != expected:
        raise SymbolicError("Symbolic formulas must cover every answer option")


def _symbolic_label(
    backend: str, raw: dict[str, Any], schema: Schema, candidate: dict[str, Any]
) -> str:
    if backend == "prover9":
        truth = solve_fol(raw.get("fol_premises"), raw.get("fol_hypothesis"))
        label = schema.truth_label(truth, candidate)
    elif backend == "rules":
        truth = solve_rules(
            raw.get("rule_facts"), raw.get("rule_rules"), raw.get("rule_query")
        )
        label = schema.truth_label(truth, candidate)
    elif backend == "csp":
        formulas = raw.get("csp_options")
        if not isinstance(formulas, dict):
            raise SymbolicError("No CSP option formulas")
        _require_complete_options(formulas, candidate)
        answer = solve_csp(
            raw.get("csp_variables"),
            raw.get("csp_constraints"),
            formulas,
            raw.get("selection_rule", ""),
        )
        label = schema.resolve_label(answer, candidate)
    elif backend == "z3":
        base = raw.get("smt2_base")
        if not isinstance(base, str) or not base.strip():
            raise SymbolicError("No SMT program")
        hypothesis = raw.get("hypothesis_formula")
        formulas = raw.get("option_formulas")
        if isinstance(hypothesis, str) and hypothesis.strip():
            truth = solve_truth(base, hypothesis)
            label = schema.truth_label(truth, candidate)
        elif isinstance(formulas, dict) and formulas:
            _require_complete_options(formulas, candidate)
            answer = solve_options(base, formulas, raw.get("selection_rule", ""))
            label = schema.resolve_label(answer, candidate)
        else:
            raise SymbolicError("No hypothesis or option formulas")
    else:
        raise SymbolicError(f"Unsupported symbolic backend: {backend}")
    if label is None:
        raise SymbolicError("Solver result cannot be mapped to a dataset label")
    return label


def evaluate_verification(
    raw: Any,
    schema: Schema,
    candidate: dict[str, Any],
    require_symbolic: bool = False,
    solver: str = "auto",
) -> Verification:
    """Prefer the dataset's solver, then other provided encodings in auto mode."""
    if not isinstance(raw, dict):
        return Verification(None, "invalid", "", "Verifier did not return an object")

    explanation = str(raw.get("explanation", raw.get("evidence", ""))).strip()
    independent = schema.resolve_label(
        raw.get("predicted_label", raw.get("label")), candidate
    )
    preferred = select_solver(schema, solver)
    available = {
        "prover9": bool(raw.get("fol_premises")),
        "rules": bool(raw.get("rule_facts")) and bool(raw.get("rule_query")),
        "csp": bool(raw.get("csp_variables")) and bool(raw.get("csp_options")),
        "z3": bool(raw.get("smt2_base")),
    }
    order = [preferred]
    if solver == "auto":
        order += [name for name in ("prover9", "rules", "csp", "z3") if name != preferred]
    errors: list[str] = []
    for backend in order:
        if not available.get(backend):
            continue
        try:
            symbolic_label = _symbolic_label(backend, raw, schema, candidate)
        except SymbolicError as exc:
            errors.append(f"{backend}: {exc}")
            continue
        if independent is not None and independent != symbolic_label:
            return Verification(
                None,
                "conflict",
                explanation,
                f"{backend} inferred {symbolic_label}; LLM inferred {independent}",
                backend,
            )
        return Verification(symbolic_label, "symbolic", explanation, solver=backend)

    symbolic_error = "; ".join(errors) or None
    if require_symbolic:
        return Verification(
            None,
            "invalid",
            explanation,
            symbolic_error or "Symbolic verification was required but not provided",
        )
    if independent is not None:
        return Verification(
            independent,
            "llm_fallback",
            explanation,
            symbolic_error,
        )
    return Verification(
        None,
        "invalid",
        explanation,
        symbolic_error or "Verifier did not infer a valid dataset label",
    )
