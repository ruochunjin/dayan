"""Trusted data augmentation subpackage (Logical DA).

Attribute extraction → diversity filtering → label-flipped generation →
symbolic label verification (Z3 / Prover9 / rule / CSP backends) → feedback
reflection.  Orchestration goes through any injected LLM workflow; the
default is ``SimpleWorkflow`` (no CrewAI dependency).
"""

from dayan.augment.pipeline import PipelineConfig, RunResult, run_dataset
from dayan.augment.symbolic_backends import SOLVERS, select_solver
from dayan.augment.workflow import SimpleWorkflow

__all__ = [
    "PipelineConfig",
    "RunResult",
    "SOLVERS",
    "SimpleWorkflow",
    "run_dataset",
    "select_solver",
]
