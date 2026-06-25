"""
Insight Agent — an autonomous data analyst.

Upload a CSV, ask a question in plain English, and a LangGraph agent will:
plan -> write its own pandas/matplotlib code -> run it in a safe executor ->
read the result/errors -> self-correct -> return charts + a written answer + the code.
"""

__version__ = "1.0.0"
__author__ = "Ankit Saxena"

from .graph import build_graph, run_agent
from .executor import run_code, check_code_safety, ExecutionResult
from .llm import get_llm, available_providers

__all__ = [
    "build_graph",
    "run_agent",
    "run_code",
    "check_code_safety",
    "ExecutionResult",
    "get_llm",
    "available_providers",
]
