"""Configuration & secret loading for Crop Disease Scanner.

The app runs fully on the bundled knowledge base with no secret. An optional
``GEMINI_API_KEY`` (env var or pasted in the UI) enables a multilingual,
farmer-friendly rewrite of the advisory. The key is never written to disk or logs.
"""
from __future__ import annotations

import os
from typing import Optional

GEMINI_API_KEY_ENV: str = "GEMINI_API_KEY"

# Public PlantVillage image-classification model (MobileNetV2, 38 classes).
MODEL_ID: str = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# Languages offered for the optional LLM rewrite.
LANGUAGES = ["English", "Hindi", "Kannada", "Telugu", "Tamil", "Marathi", "Bengali"]

# Below this top-1 probability we advise expert confirmation and surface top-3.
LOW_CONFIDENCE_THRESHOLD: float = 0.55


def get_gemini_api_key(ui_key: Optional[str] = None) -> Optional[str]:
    """Resolve the Gemini API key: UI input wins, else the env var, else None."""
    if ui_key and ui_key.strip():
        return ui_key.strip()
    env_key = os.environ.get(GEMINI_API_KEY_ENV)
    return env_key.strip() if env_key else None
