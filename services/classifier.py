"""Leaf-disease image classifier using a public PlantVillage model.

Wraps ``transformers.pipeline("image-classification")`` with the pre-trained
MobileNetV2 model from the Hugging Face Hub. The pipeline is heavy to construct,
so it is cached as a Streamlit resource and built only once per session.
"""
from __future__ import annotations

from typing import List, Tuple

import streamlit as st
from PIL import Image

from utils.config import MODEL_ID


@st.cache_resource(show_spinner="Loading disease-detection model…")
def get_classifier():
    """Build (once) and return the cached image-classification pipeline."""
    from transformers import pipeline  # imported lazily so the app starts fast

    return pipeline("image-classification", model=MODEL_ID)


def classify(image: Image.Image, top_k: int = 3) -> List[Tuple[str, float]]:
    """Classify a leaf image and return the top-k ``(label, confidence)`` pairs.

    Args:
        image: a PIL image (RGB) of a single leaf.
        top_k: number of ranked predictions to return.

    Returns:
        A list of ``(label, score)`` sorted by descending confidence.

    Raises:
        RuntimeError: if inference fails, with a user-friendly message.
    """
    try:
        clf = get_classifier()
        results = clf(image.convert("RGB"), top_k=top_k)
        return [(r["label"], float(r["score"])) for r in results]
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the UI
        raise RuntimeError(f"Image classification failed: {exc}") from exc
