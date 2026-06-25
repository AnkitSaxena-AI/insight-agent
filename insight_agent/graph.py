"""
The Insight Agent's reasoning loop, as a LangGraph state machine.

    plan -> gen_code -> execute -> (error & retries left?) --yes--> gen_code
                                          |
                                          no (success OR budget spent)
                                          v
                                     synthesize -> END

`build_graph(llm)` closes over an injected `llm`, so the offline test-suite can
pass a stub and exercise the entire graph with zero network calls.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from . import prompts
from .executor import run_code
from .state import AgentState

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull the Python out of a model reply, whether or not it's fenced."""
    if not text:
        return ""
    match = _CODE_RE.search(text)
    return (match.group(1) if match else text).strip()


def _content(message: Any) -> str:
    """Get text from a LangChain message OR a plain string (stub LLM)."""
    if message is None:
        return ""
    return getattr(message, "content", message)


def build_graph(llm, executor: Callable = run_code, timeout: float = 20.0):
    """Compile the agent graph.

    `llm` only needs an `.invoke(messages) -> message` method, where `messages`
    is a list of (role, content) tuples — satisfied by LangChain chat models and
    by the test stub alike.
    """

    def plan_node(state: AgentState) -> dict:
        messages = [
            ("system", prompts.PLANNER_SYSTEM),
            ("human", prompts.PLANNER_USER.format(
                schema=state["schema"], question=state["question"])),
        ]
        plan = _content(llm.invoke(messages)).strip()
        steps = state.get("steps", []) + [{"node": "plan", "content": plan}]
        return {
            "plan": plan,
            "steps": steps,
            "n_tries": 0,
            "max_tries": state.get("max_tries", 5),
            "last_error": None,
        }

    def gen_code_node(state: AgentState) -> dict:
        error_block = ""
        if state.get("last_error"):
            error_block = prompts.CODER_RETRY_BLOCK.format(
                last_code=state.get("code", ""), last_error=state["last_error"])
        messages = [
            ("system", prompts.CODER_SYSTEM),
            ("human", prompts.CODER_USER.format(
                schema=state["schema"],
                question=state["question"],
                plan=state.get("plan", ""),
                error_block=error_block,
            )),
        ]
        code = extract_code(_content(llm.invoke(messages)))
        steps = state.get("steps", []) + [{
            "node": "gen_code", "content": code,
            "attempt": state.get("n_tries", 0) + 1,
        }]
        return {"code": code, "steps": steps}

    def execute_node(state: AgentState) -> dict:
        result = executor(state["code"], state["df"], timeout=timeout)
        steps = state.get("steps", []) + [{
            "node": "execute",
            "ok": result.ok,
            "observation": result.observation,
            "figures": result.figures,
        }]
        update: dict[str, Any] = {
            "observation": result.observation,
            "n_tries": state.get("n_tries", 0) + 1,
            "success": result.ok,
            "steps": steps,
        }
        if result.ok:
            update["last_error"] = None
            update["figures"] = result.figures
        else:
            update["last_error"] = result.error or result.observation
        return update

    def synthesize_node(state: AgentState) -> dict:
        messages = [
            ("system", prompts.SYNTH_SYSTEM),
            ("human", prompts.SYNTH_USER.format(
                question=state["question"],
                code=state.get("code", ""),
                observation=state.get("observation", ""),
            )),
        ]
        answer = _content(llm.invoke(messages)).strip()
        steps = state.get("steps", []) + [{"node": "synthesize", "content": answer}]
        return {"answer": answer, "steps": steps}

    def route_after_execute(state: AgentState) -> str:
        if state.get("success"):
            return "synthesize"
        if state.get("n_tries", 0) >= state.get("max_tries", 5):
            return "synthesize"  # out of retries -> answer from what we have
        return "gen_code"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("gen_code", gen_code_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "gen_code")
    graph.add_edge("gen_code", "execute")
    graph.add_conditional_edges(
        "execute", route_after_execute,
        {"gen_code": "gen_code", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_agent(
    llm,
    df,
    question: str,
    max_tries: int = 5,
    timeout: float = 20.0,
) -> dict:
    """Convenience wrapper: summarise df, build the graph, run it, return final state."""
    app = build_graph(llm, timeout=timeout)
    initial: AgentState = {
        "question": question,
        "df": df,
        "schema": prompts.summarize_dataframe(df),
        "max_tries": max_tries,
        "steps": [],
    }
    return app.invoke(initial, config={"recursion_limit": 50})
