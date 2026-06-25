"""
LLM provider factory.

Groq is the default (free, fast, no credit card). Gemini is an optional
alternative. Imports are intentionally *lazy* so the offline test-suite — which
injects a stub LLM — needs neither provider package installed, and so the
Streamlit app only imports the provider the user actually selects.
"""
from __future__ import annotations

import os

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


def available_providers() -> list[str]:
    return ["groq", "gemini"]


def get_llm(
    provider: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
):
    """Return a LangChain chat model for the chosen provider.

    Resolution order for `provider`: argument -> $LLM_PROVIDER -> "groq".
    Resolution order for the key: `api_key` arg -> provider env var.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            ) from exc
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys"
            )
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=temperature,
            api_key=key,
        )

    if provider in ("gemini", "google"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - install-time guidance
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/app/apikey"
            )
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            temperature=temperature,
            google_api_key=key,
        )

    raise ValueError(f"Unknown provider '{provider}'. Choose one of: {available_providers()}")
