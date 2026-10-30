"""
LLM providers for RAG Jazz.

Providers:
  - groq   — Groq free-tier API (production)
  - ollama — local Ollama (development)
  - none   — retrieval only

Any provider failure (missing key, 429, connection error) returns
(None, error_message) so the caller can fall back to retrieval-only.
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional, Tuple

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a jazz piano and music theory expert.
    Answer the user's question using ONLY the context passages provided below.
    If the context does not contain enough information, say so honestly.
    Cite your sources by referencing the article title and section in parentheses.
""")

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_MODEL = "llama3.2"


def resolve_provider() -> str:
    """Return active provider name: groq | ollama | none."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"groq", "ollama", "none"}:
        return explicit
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "none"


def generate_answer(
    question: str,
    context: str,
    provider: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an LLM answer from retrieved context.

    Returns (answer, None) on success, or (None, error) on failure.
    """
    provider = (provider or resolve_provider()).lower()
    if provider == "none":
        return None, "LLM provider is set to none"
    if provider == "groq":
        return _query_groq(question, context)
    if provider == "ollama":
        return _query_ollama(question, context)
    return None, f"Unknown LLM provider: {provider}"


def _query_groq(question: str, context: str) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None, "GROQ_API_KEY is not set"

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        from groq import Groq
    except ImportError:
        return None, "The 'groq' package is not installed. Run: pip install groq"

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            return None, "Groq returned an empty response"
        return content, None
    except Exception as exc:  # noqa: BLE001 — surface any provider failure for fallback
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        msg = str(exc)
        if status == 429 or "429" in msg or "rate_limit" in msg.lower():
            return None, "Groq rate limit reached"
        return None, f"Groq error: {msg}"


def _query_ollama(question: str, context: str) -> Tuple[Optional[str], Optional[str]]:
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        import ollama as ol
    except ImportError:
        return None, "The 'ollama' Python package is not installed. Run: pip install ollama"

    try:
        response = ol.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"], None
    except ConnectionError:
        return None, (
            "Cannot connect to Ollama. Make sure Ollama is running.\n"
            "  Install: https://ollama.com/download\n"
            f"  Then:    ollama pull {model}"
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "not found" in msg.lower():
            return None, f"Model '{model}' not found. Run: ollama pull {model}"
        return None, f"Ollama error: {msg}"
