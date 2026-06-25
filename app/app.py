"""
Insight Agent — Streamlit demo.

Upload a CSV (or pick a bundled sample), ask a question in plain English, and
watch a LangGraph agent plan, write its own pandas/matplotlib code, run it in a
safe executor, self-correct on errors, and return charts + a written answer.

Run locally:
    streamlit run app/app.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable when run as `streamlit run app/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # load a local .env if present (never committed)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001 - dotenv is optional
    pass

from insight_agent.graph import run_agent  # noqa: E402
from insight_agent.llm import get_llm  # noqa: E402

SAMPLES_DIR = ROOT / "data" / "samples"

SAMPLE_QUESTIONS = {
    "walmart_sales.csv": [
        "Which store has the highest average weekly sales?",
        "How do weekly sales compare on holiday vs non-holiday weeks?",
        "Is there a relationship between fuel price and weekly sales?",
    ],
    "telco_churn.csv": [
        "What is the overall churn rate, and how does it differ by contract type?",
        "Do customers with fiber optic internet churn more than DSL customers?",
        "How does monthly charge relate to churn?",
    ],
    "house_sales.csv": [
        "How does price vary with the number of bedrooms?",
        "What is the average price of waterfront vs non-waterfront homes?",
        "Which factors are most correlated with price?",
    ],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLES_DIR / name)


def show_figures(figures: list[str]) -> None:
    for b64 in figures:
        st.image(base64.b64decode(b64), use_container_width=True)


def render_trace(steps: list[dict]) -> None:
    """Render the agent's step-by-step reasoning."""
    attempt = 0
    for step in steps:
        node = step["node"]
        if node == "plan":
            st.markdown("**🧭 Plan**")
            st.markdown(step["content"])
        elif node == "gen_code":
            attempt = step.get("attempt", attempt + 1)
            st.markdown(f"**⌨️ Generated code — attempt {attempt}**")
            st.code(step["content"], language="python")
        elif node == "execute":
            if step["ok"]:
                st.markdown("**✅ Executed — success**")
            else:
                st.markdown("**❌ Executed — error (the agent will self-correct)**")
            if step.get("observation"):
                st.text(step["observation"][:1500])
            if step.get("figures"):
                show_figures(step["figures"])
        elif node == "synthesize":
            st.markdown("**📝 Synthesised answer** (shown above)")
        st.divider()


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Insight Agent — Autonomous Data Analyst",
                   page_icon="🤖", layout="wide")

st.title("🤖 Insight Agent")
st.caption(
    "An autonomous data analyst. Upload a CSV, ask in plain English, and the agent "
    "**plans → writes its own pandas/matplotlib code → runs it → self-corrects → "
    "explains the answer.**"
)

# ---- sidebar: configuration ---------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.radio("LLM provider", ["groq", "gemini"], index=0,
                        help="Groq is free and fast (recommended). Gemini needs "
                             "`pip install langchain-google-genai`.")
    key_label = "Groq API key" if provider == "groq" else "Google API key"
    key_help = ("Free key: https://console.groq.com/keys" if provider == "groq"
                else "Free key: https://aistudio.google.com/app/apikey")
    env_key = os.getenv("GROQ_API_KEY") if provider == "groq" else os.getenv("GOOGLE_API_KEY")
    api_key = st.text_input(key_label, value=env_key or "", type="password", help=key_help)

    with st.expander("Advanced"):
        max_tries = st.slider("Max self-correction attempts", 1, 8, 5)
        timeout = st.slider("Per-execution timeout (s)", 5, 60, 20)

    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. **Plan** the analysis\n"
        "2. **Write** pandas/matplotlib code\n"
        "3. **Execute** it in a safe sandbox\n"
        "4. **Observe** output/errors & retry\n"
        "5. **Synthesise** a written answer"
    )

# ---- data source ---------------------------------------------------------- #
st.subheader("1 · Choose your data")
col_src, col_prev = st.columns([1, 2])

with col_src:
    source = st.radio("Data source", ["Bundled sample", "Upload a CSV"])
    df: pd.DataFrame | None = None
    chosen_name = ""
    if source == "Bundled sample":
        samples = sorted(p.name for p in SAMPLES_DIR.glob("*.csv")) if SAMPLES_DIR.exists() else []
        if samples:
            chosen_name = st.selectbox("Sample dataset", samples)
            df = load_sample(chosen_name)
        else:
            st.warning("No bundled samples found in data/samples/.")
    else:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            chosen_name = uploaded.name

with col_prev:
    if df is not None:
        st.markdown(f"**Preview — `{chosen_name}`**  ·  {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True, height=260)

# ---- question ------------------------------------------------------------- #
st.subheader("2 · Ask a question")

if "question" not in st.session_state:
    st.session_state.question = ""

suggestions = SAMPLE_QUESTIONS.get(chosen_name, [])
if suggestions:
    st.write("Try one of these:")
    cols = st.columns(len(suggestions))
    for i, q in enumerate(suggestions):
        if cols[i].button(q, key=f"sugg_{i}"):
            st.session_state.question = q

question = st.text_input("Your question", key="question",
                         placeholder="e.g. Which category drives the most revenue?")

run = st.button("🚀 Analyze", type="primary", disabled=df is None)

# ---- run ------------------------------------------------------------------ #
if run:
    if df is None:
        st.error("Please choose a dataset first.")
    elif not question.strip():
        st.error("Please enter a question.")
    elif not api_key.strip():
        st.error(f"Please enter your {key_label} in the sidebar (it's free).")
    else:
        try:
            llm = get_llm(provider=provider, api_key=api_key.strip())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not initialise the LLM: {exc}")
            st.stop()

        with st.spinner("The agent is thinking, coding, and self-correcting…"):
            try:
                final = run_agent(llm, df, question.strip(),
                                  max_tries=max_tries, timeout=float(timeout))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Agent run failed: {exc}")
                st.stop()

        # ---- final answer ------------------------------------------------- #
        st.subheader("3 · Answer")
        if final.get("success"):
            st.success(final.get("answer", "(no answer produced)"))
        else:
            st.warning(
                "The agent couldn't fully solve this within the retry budget — "
                "best effort below.\n\n" + final.get("answer", "")
            )

        if final.get("figures"):
            show_figures(final["figures"])

        attempts = final.get("n_tries", 0)
        st.caption(f"Solved in {attempts} execution attempt(s).")

        with st.expander("🔍 Agent trace — how it figured this out", expanded=not final.get("success")):
            render_trace(final.get("steps", []))

        with st.expander("🧩 Final code"):
            st.code(final.get("code", ""), language="python")

st.markdown("---")
st.caption("Built with LangGraph · LangChain · Groq/Gemini · pandas · matplotlib · Streamlit")
