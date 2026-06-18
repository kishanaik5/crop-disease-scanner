"""Optional Gemini-vision lesion detection with bounding-box overlay.

Gated behind ``GEMINI_API_KEY``. Given a leaf image and the predicted disease, it
asks Gemini to return tight bounding boxes around each affected region (on a
0–1000 coordinate scale) and draws them on the image. Purely additive — the
classifier + knowledge base work without it.
"""
from __future__ import annotations

import base64
import io
import json
from typing import List

import requests
from PIL import Image, ImageDraw

from services.advisory import _BASE

Box = List[int]  # [ymin, xmin, ymax, xmax] on a 0..1000 scale

_VISION_TIMEOUT = 60  # vision calls are slower than text
_MAX_SIDE = 768       # downscale before upload (boxes are 0–1000 scale, so safe)
# Fallback chain — flash models go in and out of 429/503; try several.
_VISION_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash"]


def detect_disease_boxes(api_key: str, image: Image.Image, disease_name: str) -> List[Box]:
    """Return bounding boxes around regions affected by ``disease_name``.

    Args:
        api_key: Gemini API key (request-only; never persisted).
        image: the leaf image (PIL).
        disease_name: the predicted disease, used to focus detection.

    Returns:
        A list of ``[ymin, xmin, ymax, xmax]`` boxes on a 0–1000 scale (possibly
        empty if nothing distinct is found).

    Raises:
        RuntimeError: on API/network failure with a user-friendly message.
    """
    small = image.convert("RGB")
    small.thumbnail((_MAX_SIDE, _MAX_SIDE))  # shrink in place, preserving aspect
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        f"Analyze this plant leaf image. Identify the regions affected by "
        f"'{disease_name}'.\n"
        "1. Draw a TIGHT bounding box around EACH distinct lesion or affected area.\n"
        "2. Ignore healthy tissue and background.\n"
        "3. Use a 0 to 1000 scale for [ymin, xmin, ymax, xmax].\n"
        'Return strict JSON: {"detections": [{"label": "<disease>", '
        '"box_2d": [ymin, xmin, ymax, xmax]}]}'
    )
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    # Try each candidate model; skip ones that are overloaded (503) or rate-limited (429).
    last_status = None
    for model in _VISION_MODELS:
        try:
            resp = requests.post(
                f"{_BASE}/models/{model}:generateContent",
                params={"key": api_key}, json=payload, timeout=_VISION_TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Gemini detection request failed: {exc}") from exc
        if resp.status_code in (429, 503):
            last_status = resp.status_code
            continue
        try:
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            detections = json.loads(text).get("detections", [])
            boxes: List[Box] = []
            for d in detections:
                box = d.get("box_2d")
                if isinstance(box, list) and len(box) == 4:
                    boxes.append([int(v) for v in box])
            return boxes
        except (requests.exceptions.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Gemini returned an unexpected detection response: {exc}") from exc
    raise RuntimeError(
        f"All Gemini vision models are busy right now (last status {last_status}). Try again shortly."
    )


def draw_boxes(image: Image.Image, boxes: List[Box]) -> Image.Image:
    """Draw bounding boxes (0–1000 scale) on a copy of the image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    width = max(2, int(min(w, h) * 0.005))
    for ymin, xmin, ymax, xmax in boxes:
        draw.rectangle(
            [xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h],
            outline="#FF3B30", width=width,
        )
    return img
