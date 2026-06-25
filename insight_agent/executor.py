"""
Safe(-ish) Python executor for the Insight Agent.

The agent writes pandas / matplotlib code as plain text. This module runs that
code against the user's DataFrame and captures everything the agent needs to
"observe" on the next turn: printed output, the value of a trailing expression,
any matplotlib figures, and a traceback if it blew up.

Defence in depth (NOT a hardened OS sandbox):
  1. Static AST allow-list  -> reject dangerous imports / calls BEFORE running.
  2. Restricted namespace    -> only a curated set of builtins + (df, pd, np, plt).
  3. Headless backend        -> matplotlib "Agg" (thread-safe, no display needed).
  4. Timeout                 -> run in a worker thread; abandon if it overruns.

This is appropriate for a trusted, single-user analyst demo. If you ever expose
it to untrusted users, put real OS-level isolation (container + seccomp) around it.
"""
from __future__ import annotations

import ast
import base64
import builtins as _builtins
import io
import threading
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless + thread-safe; MUST be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

# Modules the generated code is allowed to import (root package is enough).
ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "datetime", "calendar",
    "collections", "itertools", "functools", "json", "re",
    "matplotlib", "seaborn", "scipy",
}

# Bare/attribute names that must never be *called*.
BANNED_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "memoryview", "breakpoint", "help", "exit", "quit",
}

# A conservative, safe subset of builtins exposed to the generated code.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bytes", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hasattr", "int",
    "isinstance", "issubclass", "len", "list", "map", "max", "min", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip",
]
SAFE_BUILTINS = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES}


# --------------------------------------------------------------------------- #
# Static safety check
# --------------------------------------------------------------------------- #

@dataclass
class SafetyVerdict:
    ok: bool
    reason: str = ""


def check_code_safety(code: str) -> SafetyVerdict:
    """Parse `code` and reject obviously dangerous constructs before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return SafetyVerdict(False, f"SyntaxError while parsing code: {exc}")

    for node in ast.walk(tree):
        # --- imports -------------------------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return SafetyVerdict(False, f"Import of '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                return SafetyVerdict(False, f"Import from '{node.module}' is not allowed.")

        # --- dangerous calls ----------------------------------------------
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
                return SafetyVerdict(False, f"Call to '{func.id}(...)' is not allowed.")
            if isinstance(func, ast.Attribute) and func.attr in BANNED_CALLS:
                return SafetyVerdict(False, f"Call to '.{func.attr}(...)' is not allowed.")

        # --- dunder access (classic sandbox-escape vector) ----------------
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return SafetyVerdict(False, f"Access to dunder attribute '{node.attr}' is not allowed.")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                return SafetyVerdict(False, f"Access to '{node.id}' is not allowed.")

    return SafetyVerdict(True)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

@dataclass
class ExecutionResult:
    ok: bool
    stdout: str = ""
    result_repr: str = ""
    error: Optional[str] = None
    figures: list[str] = field(default_factory=list)  # base64-encoded PNGs

    @property
    def observation(self) -> str:
        """Compact, text-only view the agent 'sees' after running its code."""
        parts: list[str] = []
        if self.stdout.strip():
            parts.append("STDOUT:\n" + self.stdout.strip())
        if self.result_repr.strip():
            parts.append("RESULT:\n" + self.result_repr.strip())
        if self.figures:
            parts.append(f"[{len(self.figures)} matplotlib figure(s) produced]")
        if self.error:
            parts.append("ERROR:\n" + self.error.strip())
        if not parts:
            parts.append("(code ran successfully but produced no printed output)")
        return "\n\n".join(parts)


def _capture_figures() -> list[str]:
    """Save every open matplotlib figure to a base64 PNG, then close them."""
    figures: list[str] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            figures.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        except Exception:  # noqa: BLE001 - a broken figure shouldn't kill the run
            pass
    plt.close("all")
    return figures


def run_code(
    code: str,
    df: "pd.DataFrame",
    timeout: float = 20.0,
    max_output_chars: int = 6000,
) -> ExecutionResult:
    """
    Run `code` against a defensive copy of `df` in a restricted namespace.

    Returns an ExecutionResult with captured stdout, the repr of a trailing
    expression (REPL-style), any figures (base64 PNG), and a traceback on error.
    """
    verdict = check_code_safety(code)
    if not verdict.ok:
        return ExecutionResult(ok=False, error=f"Blocked by safety check: {verdict.reason}")

    safe_globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df.copy(),  # agent can't mutate the caller's frame across turns
    }
    container: dict[str, ExecutionResult] = {}

    def worker() -> None:
        out = io.StringIO()
        try:
            plt.close("all")
            tree = ast.parse(code)
            # Pull a trailing bare expression so we can show its value REPL-style.
            trailing_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                trailing_expr = ast.Expression(tree.body.pop().value)
                ast.fix_missing_locations(trailing_expr)
            ast.fix_missing_locations(tree)

            result_repr = ""
            with redirect_stdout(out):
                exec(compile(tree, "<agent_code>", "exec"), safe_globals)  # noqa: S102
                if trailing_expr is not None:
                    value = eval(  # noqa: S307 - sandboxed namespace, our own AST
                        compile(trailing_expr, "<agent_expr>", "eval"), safe_globals
                    )
                    if value is not None:
                        result_repr = repr(value)

            container["result"] = ExecutionResult(
                ok=True,
                stdout=out.getvalue(),
                result_repr=result_repr,
                figures=_capture_figures(),
            )
        except Exception:  # noqa: BLE001 - we want the traceback as text, not a crash
            container["result"] = ExecutionResult(
                ok=False,
                stdout=out.getvalue(),
                error=traceback.format_exc(limit=3),
                figures=_capture_figures(),
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return ExecutionResult(ok=False, error=f"Execution timed out after {timeout:.0f}s.")

    result = container.get("result") or ExecutionResult(ok=False, error="Unknown execution failure.")

    # Keep observations from blowing up the LLM context window.
    if len(result.stdout) > max_output_chars:
        result.stdout = result.stdout[:max_output_chars] + "\n...[output truncated]"
    if len(result.result_repr) > max_output_chars:
        result.result_repr = result.result_repr[:max_output_chars] + "\n...[output truncated]"
    return result
