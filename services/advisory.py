"""Optional LLM enrichment of the advisory via the Gemini REST API.

Gated behind ``GEMINI_API_KEY``: when no key is provided the app simply shows the
raw knowledge-base text. We call the REST endpoint directly (no SDK dependency)
and auto-resolve a available 'flash' model so the code keeps working as model
names change. The key is sent only in the request and never stored or logged.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import io
from typing import List, Optional

from PIL import Image
import requests

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = 30
_PREFERRED = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]


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
    return _PREFERRED[0]


def rewrite_advisory(
    api_key: str,
    advisory_text: str,
    language: str,
    crop: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> str:
    """Perform expert plant pathology analysis and farmer-friendly advisory generation.

    Args:
        api_key: Gemini API key (request-only; never persisted).
        advisory_text: the plain-text advisory from the knowledge base.
        language: target language name (e.g. "Hindi").
        crop: claimed name of the crop (e.g. "Tomato", "Potato").
        image: optional PIL Image of the leaf for multimodal vision analysis.

    Returns:
        The comprehensive diagnosis and advisory text.

    Raises:
        RuntimeError: on API/network failure, with a user-friendly message.
    """
    primary_model = _resolve_model(api_key)
    models_to_try = [primary_model] + [m for m in _PREFERRED if m != primary_model]

    crop_name = crop.strip() if crop and crop.strip() and crop.lower() != "unknown" else "plant"
    lang_instruction = (
        f"Generate the full response in {language} in simple, practical, encouraging, farmer-friendly language."
        if language and language.lower() != "english"
        else "Generate the full response in clear, practical, farmer-friendly English."
    )

    prompt = f"""Role: You are an expert Plant Pathologist and Agricultural Advisory System.

Task:
Analyze the attached image of a plant leaf to perform a comprehensive diagnosis and advisory generation.

[LANGUAGE INSTRUCTION]: {lang_instruction}

[CONTEXT]:
The user claims this is a "{crop_name}" plant.

Primary Objectives:
1. Identify the Plant accurately.
   - CRITICAL CHECK: Does the image match the user's claim of "{crop_name}"? 
   - If the image is CLEARLY a different plant, flag this in the "scientific_name" field as "MISMATCH: Detected [Actual Plant] vs User Claim [User Plant]".
   - If it is non-plant material, return "INVALID_IMAGE" in the common_name.
2. Identify the Disease. If no disease is visible, return "Healthy".
3. Explain the CAUSE of the disease (how and why it occurs).
4. Explain disease spread (seed, soil, wind, rain, insects).
5. Provide integrated disease management:
   - Organic & biological practices
   - Chemical practices (active ingredients only)

--- REFERENCE DIAGNOSTIC BASELINE ---
{advisory_text}"""

    parts = []
    if image is not None:
        try:
            small = image.convert("RGB")
            small.thumbnail((768, 768))
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=85)
            b64_img = base64.b64encode(buf.getvalue()).decode()
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_img
                }
            })
        except Exception:
            pass

    parts.append({"text": prompt})

    last_err = None
    for model in models_to_try:
        try:
            resp = requests.post(
                f"{_BASE}/models/{model}:generateContent",
                params={"key": api_key},
                json={"contents": [{"parts": parts}]},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.RequestException as exc:
            last_err = exc
            continue
        except (KeyError, IndexError) as exc:
            raise RuntimeError("Gemini returned an unexpected response.") from exc

    raise RuntimeError(f"Gemini request failed: {last_err}")
