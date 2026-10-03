"""Thought Graph: model LLM reasoning processes as analyzable graphs.

Instrument a reasoning program with ``@analysis``, let it call ``query`` /
``query_iter`` for LLM steps, and receive the program trace, the data-flow
graph, or the Thought Graph (nodes = LLM outputs, edges = data dependencies).
"""

from dayan.graph.analysis import AnalysisOutput, analysis
from dayan.graph.graph import DataFlowGraph, ThoughtGraph
from dayan.graph.invoke import query, query_iter, set_backend
from dayan.graph.trace import ProgramTrace, VariableNode

__all__ = [
    "AnalysisOutput",
    "DataFlowGraph",
    "ProgramTrace",
    "ThoughtGraph",
    "VariableNode",
    "analysis",
    "query",
    "query_iter",
    "set_backend",
]
