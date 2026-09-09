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
_PREFERRED = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-flash-latest"]


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


class AdvisoryResult(str):
    """String subclass that carries parsed structured pathology data."""

    parsed: dict = {}


def format_advisory_markdown(parsed: dict) -> str:
    """Render the structured pathology JSON into clean, farmer-friendly markdown."""
    plant = parsed.get("plant_info", {})
    disease = parsed.get("disease_info", {})
    mgmt = parsed.get("management", {})

    plant_common = plant.get("common_name", "Plant")
    plant_sci = plant.get("scientific_name", "")
    disease_common = disease.get("common_name", "Undetermined")
    disease_sci = disease.get("scientific_name", "")
    pathogen_type = disease.get("pathogen_type", "Pathogen")
    severity = disease.get("severity", "Moderate")
    cause = disease.get("cause", "Etiology under review.")
    symptoms = disease.get("symptoms", "Leaf lesions or discoloration.")
    spread = disease.get("disease_spread", "Airborne, water splash, or seedborne transmission.")
    organic = mgmt.get("organic_practices", [])
    chemical = mgmt.get("chemical_practices", [])

    sci_badge = f" (*{plant_sci}*)" if plant_sci and "MISMATCH" not in plant_sci else f" — **{plant_sci}**" if plant_sci else ""
    disease_sci_badge = f" (*{disease_sci}*)" if disease_sci else ""
    org_md = "\n".join(f"* {p}" for p in organic) if organic else "* Practice proper spacing, clean cultivation, and drip irrigation."
    chem_md = "\n".join(f"* **{c}**" for c in chemical) if chemical else "* Consult local agricultural extension officers for registered active ingredients."

    return (
        f"### **Plant Pathology Diagnosis & Agricultural Advisory**\n\n"
        f"---\n\n"
        f"### **1. Plant & Disease Identification**\n"
        f"* **Host Plant:** {plant_common}{sci_badge}\n"
        f"* **Disease Diagnosis:** **{disease_common}**{disease_sci_badge}\n"
        f"* **Pathogen Type:** {pathogen_type} (Severity: {severity})\n\n"
        f"---\n\n"
        f"### **2. Symptoms & Cause**\n"
        f"* **Symptoms:** {symptoms}\n"
        f"* **Underlying Cause:** {cause}\n\n"
        f"---\n\n"
        f"### **3. How the Disease Spreads**\n"
        f"* {spread}\n\n"
        f"---\n\n"
        f"### **4. Integrated Disease Management (IDM) Advisory**\n\n"
        f"#### **A. Organic & Biological Practices**\n"
        f"{org_md}\n\n"
        f"#### **B. Chemical Control (Active Ingredients Only)**\n"
        f"{chem_md}"
    )


def rewrite_advisory(
    api_key: str,
    advisory_text: str,
    language: str,
    crop: Optional[str] = None,
    image: Optional[Image.Image] = None,
) -> AdvisoryResult:
    """Perform expert plant pathology analysis and farmer-friendly advisory generation.

    Args:
        api_key: Gemini API key (request-only; never persisted).
        advisory_text: the plain-text advisory from the knowledge base.
        language: target language name (e.g. "Hindi").
        crop: claimed name of the crop (e.g. "Tomato", "Potato").
        image: optional PIL Image of the leaf for multimodal vision analysis.

    Returns:
        AdvisoryResult (str) containing formatted markdown and parsed JSON dictionary in ``.parsed``.

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

Output Rules:
- STRICT JSON ONLY
- No markdown
- No extra text

JSON FORMAT:
{{
    "plant_info": {{
        "common_name": "",
        "scientific_name": ""
    }},
    "disease_info": {{
        "common_name": "",
        "scientific_name": "",
        "pathogen_type": "",
        "cause": "",
        "symptoms": "",
        "disease_spread": "",
        "severity": ""
    }},
    "management": {{
        "organic_practices": [],
        "chemical_practices": []
    }}
}}"""

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
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {"response_mime_type": "application/json"},
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            clean = raw_text
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            parsed = {}
            try:
                import json
                parsed = json.loads(clean)
            except Exception:
                import json
                import re
                m = re.search(r"(\{[\s\S]*\})", clean)
                if m:
                    parsed = json.loads(m.group(1))

            formatted_md = format_advisory_markdown(parsed) if parsed else raw_text
            res = AdvisoryResult(formatted_md)
            res.parsed = parsed
            return res

        except requests.exceptions.RequestException as exc:
            last_err = exc
            continue
        except (KeyError, IndexError) as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Gemini request failed: {last_err}")
