"""Shared state for the Insight Agent's LangGraph state machine."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """The blackboard that flows between graph nodes.

    `total=False` so nodes can return partial updates; LangGraph merges them.
    """

    # --- inputs -----------------------------------------------------------
    question: str           # the user's natural-language question
    df: Any                 # the pandas DataFrame under analysis (not serialised)
    schema: str             # text summary of df (shape, dtypes, head) for prompts

    # --- working memory ---------------------------------------------------
    plan: str               # the planner's analysis plan
    code: str               # the most recent generated code
    observation: str        # text the agent "sees" after executing its code
    last_error: Optional[str]  # traceback/error from the last run, if any
    n_tries: int            # how many times code has been executed
    max_tries: int          # retry budget before giving up gracefully

    # --- outputs ----------------------------------------------------------
    figures: list[str]      # base64 PNGs from the successful run
    answer: str             # final written analysis
    success: bool           # did the last execution succeed?
    steps: list[dict[str, Any]]  # full trace for the UI / debugging
