"""
Offline smoke tests for the Insight Agent.

These run with ZERO network access and NO LLM key:
  * the safe executor is tested directly (runs code, captures figures, blocks
    dangerous code, reports errors as text);
  * the full LangGraph agent is driven by a *stub* LLM that deliberately emits
    broken code first, so we prove the graph observes the error and self-corrects.

Run:  pytest -q      (from the project root)
"""
from __future__ import annotations

import pandas as pd

from insight_agent.executor import check_code_safety, run_code
from insight_agent.graph import build_graph, extract_code


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "A"],
            "value": [10, 20, 30, 40, 50],
        }
    )


# Broken on purpose: 'valeu' is a typo -> KeyError when executed.
BAD_CODE = "print(df['valeu'].sum())"

# The corrected version: prints totals, draws a chart, returns a trailing value.
GOOD_CODE = """
totals = df.groupby('category')['value'].sum()
print(totals)
totals.plot(kind='bar', title='Total value by category')
plt.xlabel('category')
plt.ylabel('value')
totals
""".strip()


class _Msg:
    """Minimal stand-in for a LangChain AIMessage."""

    def __init__(self, content: str):
        self.content = content


class StubLLM:
    """Deterministic, offline LLM. Routes by inspecting the system prompt and
    returns broken code on the first coding turn, fixed code on the retry."""

    def __init__(self):
        self.invocations: list[str] = []

    def invoke(self, messages):
        system = messages[0][1]
        human = messages[1][1]
        self.invocations.append(system.split("\n", 1)[0])

        if "planning module" in system:
            return _Msg("1. Group by 'category'. 2. Sum 'value'. 3. Bar chart of totals.")
        if "coding module" in system:
            if "PREVIOUS CODE FAILED" in human:          # this is the retry
                return _Msg(f"```python\n{GOOD_CODE}\n```")
            return _Msg(f"```python\n{BAD_CODE}\n```")    # first attempt: broken
        if "synthesis module" in system:
            return _Msg("Category A totals 90 and B totals 60, so A is the larger segment.")
        return _Msg("")


# --------------------------------------------------------------------------- #
# Executor tests
# --------------------------------------------------------------------------- #
def test_executor_runs_prints_and_captures_figure():
    result = run_code(GOOD_CODE, sample_df())
    assert result.ok, result.error
    assert "category" in result.stdout            # printed the groupby result
    assert len(result.figures) == 1               # exactly one chart captured
    assert result.figures[0]                       # non-empty base64 PNG
    assert "90" in result.result_repr             # trailing expression value


def test_executor_reports_error_as_text():
    result = run_code(BAD_CODE, sample_df())
    assert not result.ok
    assert "KeyError" in (result.error or "")
    assert "ERROR" in result.observation          # surfaced to the agent


def test_safety_blocks_dangerous_code():
    for snippet in [
        "import os",
        "open('/etc/passwd').read()",
        "__import__('os').system('ls')",
        "df.__class__.__bases__",
        "eval('2+2')",
    ]:
        verdict = check_code_safety(snippet)
        assert not verdict.ok, f"should have blocked: {snippet}"

    # run_code must refuse to run it, too
    blocked = run_code("import os; os.listdir('.')", sample_df())
    assert not blocked.ok
    assert "safety check" in (blocked.error or "")


def test_safety_allows_normal_analysis():
    ok_code = "import numpy as np\nprint(np.mean(df['value']))"
    assert check_code_safety(ok_code).ok


def test_extract_code_handles_fenced_and_bare():
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    assert extract_code("print(2)") == "print(2)"


# --------------------------------------------------------------------------- #
# Full-graph test (stub LLM, fully offline) — the headline test
# --------------------------------------------------------------------------- #
def test_graph_self_corrects_and_synthesizes():
    llm = StubLLM()
    app = build_graph(llm)
    final = app.invoke(
        {
            "question": "Which category has the higher total value?",
            "df": sample_df(),
            "schema": "Shape: 5 rows x 2 columns\nColumns: category, value",
            "max_tries": 5,
            "steps": [],
        },
        config={"recursion_limit": 50},
    )

    # It eventually succeeded ...
    assert final["success"] is True
    # ... but only after the first attempt failed and it retried (n_tries == 2).
    assert final["n_tries"] == 2
    # A chart made it through to the final state.
    assert len(final["figures"]) == 1
    # A written answer was synthesised.
    assert "A" in final["answer"]

    # The trace visited every node, with two coding attempts and two executions.
    nodes = [s["node"] for s in final["steps"]]
    assert nodes.count("gen_code") == 2
    assert nodes.count("execute") == 2
    assert nodes[0] == "plan"
    assert nodes[-1] == "synthesize"

    # First execution errored; second succeeded.
    executes = [s for s in final["steps"] if s["node"] == "execute"]
    assert executes[0]["ok"] is False
    assert executes[1]["ok"] is True
