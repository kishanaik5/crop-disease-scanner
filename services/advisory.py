"""Optional LLM enrichment of the advisory via the Gemini REST API.

Gated behind ``GEMINI_API_KEY``: when no key is provided the app simply shows the
raw knowledge-base text. We call the REST endpoint directly (no SDK dependency)
and auto-resolve a available 'flash' model so the code keeps working as model
names change. The key is sent only in the request and never stored or logged.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import requests

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = 30
# Preference order; the first available model supporting generateContent is used.
_PREFERRED = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash"]


@lru_cache(maxsize=4)
def _resolve_model(api_key: str) -> str:
    """Return a usable Gemini model name for generateContent (cached per key)."""
    try:
        resp = requests.get(f"{_BASE}/models", params={"key": api_key}, timeout=_TIMEOUT)
        resp.raise_for_status()
        models: List[dict] = resp.json().get("models", [])
        available = {
            m["name"].split("/")[-1]
            for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        }
        for pref in _PREFERRED:
            if pref in available:
                return pref
        flash = sorted(n for n in available if "flash" in n)
        if flash:
            return flash[0]
        if available:
            return sorted(available)[0]
    except requests.exceptions.RequestException:
        pass
    return _PREFERRED[0]  # best-effort default; generateContent will report errors


def rewrite_advisory(api_key: str, advisory_text: str, language: str) -> str:
    """Rewrite the advisory in ``language``, farmer-friendly, preserving structure.

    Args:
        api_key: Gemini API key (request-only; never persisted).
        advisory_text: the plain-text advisory from the knowledge base.
        language: target language name (e.g. "Hindi").

    Returns:
        The rewritten advisory text.

    Raises:
        RuntimeError: on API/network failure, with a user-friendly message.
    """
    model = _resolve_model(api_key)
    prompt = (
        f"You are an agricultural extension officer advising a smallholder farmer. "
        f"Rewrite the crop-disease advisory below in {language}, in simple, "
        f"encouraging, farmer-friendly language. Keep these labelled sections: "
        f"Symptoms, Likely cause, Organic treatment, Chemical treatment, Prevention. "
        f"Be concise and practical. Do not invent facts beyond the advisory.\n\n"
        f"--- ADVISORY ---\n{advisory_text}"
    )
    try:
        resp = requests.post(
            f"{_BASE}/models/{model}:generateContent",
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Gemini returned an unexpected response.") from exc
