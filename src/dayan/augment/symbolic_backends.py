"""Task-specific symbolic solvers used by the Logical DA verifier.

The rule and finite-domain CSP backends implement the safe subset used by the
original Pyke and python-constraint programs. They parse data structures; no
model-generated Python is evaluated. Prover9 is invoked as an external binary.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from .data import Schema

SOLVERS = ("auto", "prover9", "rules", "csp", "z3", "llm")
_SELECTION_RULES = {"must_true", "must_false", "could_true", "could_false"}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TERM_NAME = re.compile(r"^[?$]?[A-Za-z_][A-Za-z0-9_]*$")
_ATOM = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*$")


class SymbolicError(ValueError):
    """A symbolic encoding is invalid or a solver cannot establish an answer."""


def select_solver(schema: Schema, requested: str = "auto") -> str:
    """Pick the paper's solver family for known datasets; allow an explicit override."""
    if requested not in SOLVERS:
        raise ValueError(f"Unknown solver {requested!r}; choose one of {SOLVERS}")
    if requested != "auto":
        return requested
    hint = schema.dataset_hint.replace("-", "").replace("_", "").upper()
    by_dataset = {
        "FOLIO": "prover9",
        "PROOFWRITER": "rules",
        "PRONTOQA": "rules",
        "LOGICALDEDUCTION": "csp",
        "ARLSAT": "z3",
    }
    if hint in by_dataset:
        return by_dataset[hint]
    if schema.kind in {"nli", "qa"}:
        return "z3"
    return "llm"


def _fol_formula(raw: str) -> str:
    if not isinstance(raw, str):
        raise SymbolicError("Prover9 formula must be text")
    formula = raw.strip()
    if not formula or len(formula) > 2000 or "\n" in formula or "\r" in formula:
        raise SymbolicError("Prover9 formula is empty, multiline, or too long")
    if ":::" in formula:
        formula = formula.split(":::", 1)[0].strip()
    formula = re.sub(r"∀\s*([A-Za-z_][A-Za-z0-9_]*)", r"all \1 ", formula)
    formula = re.sub(r"∃\s*([A-Za-z_][A-Za-z0-9_]*)", r"exists \1 ", formula)
    for source, target in (
        ("↔", "<->"),
        ("→", "->"),
        ("¬", "-"),
        ("∧", "&"),
        ("∨", "|"),
        ("≠", "!="),
    ):
        formula = formula.replace(source, target)
    if "⊕" in formula:
        raise SymbolicError("Expand exclusive-or into conjunction, disjunction, and negation")
    if not re.fullmatch(r"[A-Za-z0-9_(),&|=<>!+\- \t]+", formula):
        raise SymbolicError("Prover9 formula contains an unsupported character")
    depth = 0
    for character in formula:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if depth < 0 or depth > 80:
            raise SymbolicError("Prover9 formula has invalid parentheses")
    if depth:
        raise SymbolicError("Prover9 formula has invalid parentheses")
    return formula


def unavailable_reason(backend: str) -> str | None:
    """Describe a missing optional executable or package before an LLM run."""
    if backend == "z3" and importlib.util.find_spec("z3") is None:
        return "z3-solver is not installed"
    if backend == "prover9":
        requested = os.getenv("PROVER9_BINARY")
        if not (shutil.which(requested) if requested else shutil.which("prover9")):
            return "Prover9 executable was not found; install it or set PROVER9_BINARY"
    return None


def _prover9_binary() -> str:
    requested = os.getenv("PROVER9_BINARY")
    binary = shutil.which(requested) if requested else shutil.which("prover9")
    if not binary:
        raise SymbolicError(
            "Prover9 executable was not found; install Prover9 or set PROVER9_BINARY"
        )
    return binary


def _prover9_proves(
    binary: str, premises: list[str], goal: str, timeout_seconds: float
) -> bool:
    script = (
        "formulas(assumptions).\n"
        + "\n".join(f"{item}." for item in premises)
        + "\nend_of_list.\nformulas(goals).\n"
        + goal
        + ".\nend_of_list.\n"
    )
    try:
        completed = subprocess.run(
            [binary],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SymbolicError("Prover9 timed out") from exc
    except OSError as exc:
        raise SymbolicError(f"Prover9 could not start: {exc}") from exc
    output = completed.stdout + "\n" + completed.stderr
    if "THEOREM PROVED" in output:
        return True
    if "SEARCH FAILED" in output:
        return False
    excerpt = output[-300:].strip()
    raise SymbolicError(
        f"Prover9 did not return a conclusive proof status "
        f"(exit {completed.returncode}): {excerpt}"
    )


def solve_fol(
    premises: list[str], hypothesis: str, timeout_seconds: float = 4.0
) -> str:
    """Return true, false, or unknown from two Prover9 proof attempts."""
    if not isinstance(premises, list) or not 1 <= len(premises) <= 100:
        raise SymbolicError("Expected 1-100 first-order premises")
    normalized = [_fol_formula(item) for item in premises]
    conclusion = _fol_formula(hypothesis)
    if sum(map(len, normalized)) > 12000:
        raise SymbolicError("First-order premises are too long")
    binary = _prover9_binary()
    positive = _prover9_proves(binary, normalized, conclusion, timeout_seconds)
    negative = _prover9_proves(binary, normalized, f"-({conclusion})", timeout_seconds)
    if positive and negative:
        raise SymbolicError("The first-order premises are inconsistent")
    if positive:
        return "true"
    if negative:
        return "false"
    return "unknown"


@dataclass(frozen=True)
class _RuleAtom:
    predicate: str
    terms: tuple[str, ...]
    truth: bool

    @property
    def signature(self) -> tuple[str, int, bool]:
        return self.predicate, len(self.terms), self.truth

    @property
    def proposition(self) -> tuple[str, tuple[str, ...]]:
        return self.predicate, self.terms


@dataclass(frozen=True)
class _Rule:
    body: tuple[_RuleAtom, ...]
    head: tuple[_RuleAtom, ...]


def _variable(term: str) -> bool:
    return term.startswith(("$", "?"))


def _parse_atom(raw: str, ground: bool = False) -> _RuleAtom:
    if not isinstance(raw, str) or len(raw) > 300:
        raise SymbolicError("Rule atom is missing or too long")
    match = _ATOM.fullmatch(raw)
    if not match:
        raise SymbolicError(f"Invalid rule atom: {raw!r}")
    arguments = [part.strip() for part in match.group(2).split(",")]
    if not 2 <= len(arguments) <= 7:
        raise SymbolicError("Rule atoms need terms and an explicit true/false argument")
    if any(not _TERM_NAME.fullmatch(part) for part in arguments[:-1]):
        raise SymbolicError(f"Rule atom has an invalid term: {raw!r}")
    if arguments[-1].lower() not in {"true", "false"}:
        raise SymbolicError("The final rule atom argument must be True or False")
    terms = tuple(arguments[:-1])
    if ground and any(_variable(term) for term in terms):
        raise SymbolicError("Facts and queries must not contain variables")
    return _RuleAtom(match.group(1), terms, arguments[-1].lower() == "true")


def _parse_rule(raw: str) -> _Rule:
    if not isinstance(raw, str) or len(raw) > 1600:
        raise SymbolicError("Rule is missing or too long")
    parts = raw.split(">>>")
    if len(parts) != 2:
        raise SymbolicError("Rules must use antecedent >>> consequent")
    body = tuple(_parse_atom(part.strip()) for part in parts[0].split("&&"))
    head = tuple(_parse_atom(part.strip()) for part in parts[1].split("&&"))
    bound = {term for atom in body for term in atom.terms if _variable(term)}
    required = {term for atom in head for term in atom.terms if _variable(term)}
    if not required <= bound:
        raise SymbolicError("Every variable in a rule head must occur in its body")
    return _Rule(body, head)


def _match_rule(
    body: tuple[_RuleAtom, ...],
    index: dict[tuple[str, int, bool], list[_RuleAtom]],
    deadline: float,
    position: int = 0,
    bindings: dict[str, str] | None = None,
):
    if time.monotonic() > deadline:
        raise SymbolicError("Rule solver timed out")
    if position == len(body):
        yield bindings or {}
        return
    bindings = bindings or {}
    pattern = body[position]
    for fact in index.get(pattern.signature, ()):
        next_bindings = dict(bindings)
        for expected, actual in zip(pattern.terms, fact.terms):
            if _variable(expected):
                if expected in next_bindings and next_bindings[expected] != actual:
                    break
                next_bindings[expected] = actual
            elif expected != actual:
                break
        else:
            yield from _match_rule(
                body, index, deadline, position + 1, next_bindings
            )


def _instantiate(atom: _RuleAtom, bindings: dict[str, str]) -> _RuleAtom:
    return _RuleAtom(
        atom.predicate,
        tuple(bindings[term] if _variable(term) else term for term in atom.terms),
        atom.truth,
    )


def solve_rules(
    facts: list[str], rules: list[str], query: str, timeout_ms: int = 3000
) -> str:
    """Finite forward chaining with the explicit-negation semantics used for Pyke."""
    if not isinstance(facts, list) or not 1 <= len(facts) <= 300:
        raise SymbolicError("Expected 1-300 rule facts")
    if not isinstance(rules, list) or len(rules) > 100:
        raise SymbolicError("Expected at most 100 rules")
    known = {_parse_atom(item, ground=True) for item in facts}
    parsed_rules = [_parse_rule(item) for item in rules]
    goal = _parse_atom(query, ground=True)
    deadline = time.monotonic() + timeout_ms / 1000
    for _ in range(100):
        if time.monotonic() > deadline:
            raise SymbolicError("Rule solver timed out")
        index: dict[tuple[str, int, bool], list[_RuleAtom]] = {}
        for atom in known:
            index.setdefault(atom.signature, []).append(atom)
        derived: set[_RuleAtom] = set()
        for rule in parsed_rules:
            for bindings in _match_rule(rule.body, index, deadline):
                derived.update(_instantiate(atom, bindings) for atom in rule.head)
                if len(known) + len(derived) > 5000:
                    raise SymbolicError("Rule closure is too large")
        fresh = derived - known
        if not fresh:
            break
        known.update(fresh)
    else:
        raise SymbolicError("Rule closure exceeded 100 rounds")
    by_proposition: dict[tuple[str, tuple[str, ...]], bool] = {}
    for atom in known:
        key = atom.proposition
        if key in by_proposition and by_proposition[key] != atom.truth:
            raise SymbolicError("Rule facts are inconsistent")
        by_proposition[key] = atom.truth
    established = by_proposition.get(goal.proposition)
    if established is None:
        return "unknown"
    return "true" if established == goal.truth else "false"


def _csp_term_valid(term: Any, variables: dict[str, tuple[int, ...]], depth: int) -> None:
    if depth > 20:
        raise SymbolicError("CSP expression is too deeply nested")
    if type(term) is int and abs(term) <= 100000:
        return
    if isinstance(term, str) and term in variables:
        return
    if isinstance(term, dict):
        op = term.get("op")
        if not isinstance(op, str):
            raise SymbolicError("CSP numeric operation must be a string")
        if op in {"add", "sub"}:
            _csp_term_valid(term.get("left"), variables, depth + 1)
            _csp_term_valid(term.get("right"), variables, depth + 1)
            return
        if op == "abs":
            _csp_term_valid(term.get("arg"), variables, depth + 1)
            return
    raise SymbolicError(f"Unsupported CSP numeric term: {term!r}")


def _csp_predicate_valid(
    expression: Any, variables: dict[str, tuple[int, ...]], depth: int = 0
) -> None:
    if depth > 20 or not isinstance(expression, dict):
        raise SymbolicError("CSP predicate is missing or too deeply nested")
    op = expression.get("op")
    if not isinstance(op, str):
        raise SymbolicError("CSP operation must be a string")
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        _csp_term_valid(expression.get("left"), variables, depth + 1)
        _csp_term_valid(expression.get("right"), variables, depth + 1)
    elif op == "all_different":
        names = expression.get("vars")
        if (
            not isinstance(names, list)
            or not 2 <= len(names) <= 12
            or any(not isinstance(name, str) or name not in variables for name in names)
            or len(names) != len(set(names))
        ):
            raise SymbolicError("all_different needs 2-12 distinct variable names")
    elif op in {"and", "or"}:
        children = expression.get("args")
        if not isinstance(children, list) or not 2 <= len(children) <= 30:
            raise SymbolicError(f"{op} needs 2-30 predicates")
        for child in children:
            _csp_predicate_valid(child, variables, depth + 1)
    elif op == "not":
        _csp_predicate_valid(expression.get("arg"), variables, depth + 1)
    elif op == "implies":
        _csp_predicate_valid(expression.get("if"), variables, depth + 1)
        _csp_predicate_valid(expression.get("then"), variables, depth + 1)
    else:
        raise SymbolicError(f"Unsupported CSP operation: {op!r}")


def _csp_term(term: Any, assignment: dict[str, int]) -> int | None:
    if type(term) is int:
        return term
    if isinstance(term, str):
        return assignment.get(term)
    op = term["op"]
    if op == "abs":
        value = _csp_term(term["arg"], assignment)
        return None if value is None else abs(value)
    left = _csp_term(term["left"], assignment)
    right = _csp_term(term["right"], assignment)
    if left is None or right is None:
        return None
    return left + right if op == "add" else left - right


def _csp_predicate(
    expression: dict[str, Any], assignment: dict[str, int]
) -> bool | None:
    op = expression["op"]
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left = _csp_term(expression["left"], assignment)
        right = _csp_term(expression["right"], assignment)
        if left is None or right is None:
            return None
        return {
            "eq": left == right,
            "ne": left != right,
            "lt": left < right,
            "le": left <= right,
            "gt": left > right,
            "ge": left >= right,
        }[op]
    if op == "all_different":
        values = [assignment[name] for name in expression["vars"] if name in assignment]
        if len(values) != len(set(values)):
            return False
        return True if len(values) == len(expression["vars"]) else None
    if op == "not":
        value = _csp_predicate(expression["arg"], assignment)
        return None if value is None else not value
    if op == "implies":
        left = _csp_predicate(expression["if"], assignment)
        right = _csp_predicate(expression["then"], assignment)
        if left is False or right is True:
            return True
        if left is True and right is False:
            return False
        return None
    values = [_csp_predicate(item, assignment) for item in expression["args"]]
    if op == "and":
        if False in values:
            return False
        return True if all(value is True for value in values) else None
    if True in values:
        return True
    return False if all(value is False for value in values) else None


def solve_csp(
    variables: dict[str, list[int]],
    constraints: list[dict[str, Any]],
    options: dict[str, dict[str, Any]],
    selection_rule: str,
    timeout_ms: int = 3000,
) -> str:
    """Solve a finite-domain CSP and select one option over all satisfying models."""
    if not isinstance(variables, dict) or not 1 <= len(variables) <= 12:
        raise SymbolicError("CSP needs 1-12 variables")
    domains: dict[str, tuple[int, ...]] = {}
    for name, values in variables.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise SymbolicError("CSP variable names must be simple identifiers")
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 20
            or any(type(value) is not int or abs(value) > 100000 for value in values)
            or len(values) != len(set(values))
        ):
            raise SymbolicError(f"CSP variable {name!r} has an invalid finite domain")
        domains[name] = tuple(values)
    if not isinstance(constraints, list) or not 1 <= len(constraints) <= 100:
        raise SymbolicError("CSP needs 1-100 constraints")
    if not isinstance(options, dict) or not 2 <= len(options) <= 12:
        raise SymbolicError("CSP needs 2-12 option formulas")
    normalized_options: dict[str, dict[str, Any]] = {}
    for key, expression in options.items():
        if not isinstance(key, str) or not _NAME.fullmatch(key):
            raise SymbolicError("CSP option keys must be simple identifiers")
        upper = key.upper()
        if upper in normalized_options:
            raise SymbolicError("CSP option keys must be unique")
        normalized_options[upper] = expression
    for expression in [*constraints, *normalized_options.values()]:
        _csp_predicate_valid(expression, domains)
    rule = str(selection_rule).strip().lower().replace("-", "_").replace(" ", "_")
    if rule not in _SELECTION_RULES:
        raise SymbolicError(f"Unsupported option selection rule: {selection_rule!r}")

    ordered = sorted(domains, key=lambda name: len(domains[name]))
    assignment: dict[str, int] = {}
    any_true = {key: False for key in normalized_options}
    all_true = {key: True for key in normalized_options}
    deadline = time.monotonic() + timeout_ms / 1000
    nodes = 0
    solutions = 0

    def search(position: int) -> None:
        nonlocal nodes, solutions
        nodes += 1
        if nodes > 2000000 or time.monotonic() > deadline:
            raise SymbolicError("CSP search timed out or exceeded its node limit")
        if position == len(ordered):
            solutions += 1
            if solutions > 200000:
                raise SymbolicError("CSP has too many solutions to check")
            for key, expression in normalized_options.items():
                value = _csp_predicate(expression, assignment)
                if value is None:
                    raise SymbolicError("CSP option was not fully grounded")
                any_true[key] |= value
                all_true[key] &= value
            return
        name = ordered[position]
        for value in domains[name]:
            assignment[name] = value
            if all(_csp_predicate(item, assignment) is not False for item in constraints):
                search(position + 1)
            del assignment[name]

    search(0)
    if solutions == 0:
        raise SymbolicError("The generated CSP constraints are inconsistent")
    matches = [
        key
        for key in normalized_options
        if {
            "must_true": all_true[key],
            "must_false": not any_true[key],
            "could_true": any_true[key],
            "could_false": not all_true[key],
        }[rule]
    ]
    if len(matches) != 1:
        raise SymbolicError(
            f"The CSP found {len(matches)} matching options; expected exactly one"
        )
    return matches[0]
