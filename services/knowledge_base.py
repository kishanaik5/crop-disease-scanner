"""Knowledge-based agent: maps a disease label to structured treatment advice.

The knowledge base (``data/disease_kb.json``) is authored from general agronomy
knowledge and keyed by the exact labels of the PlantVillage classifier. This
module loads it and applies simple, explainable reasoning rules on top of the
classifier's predictions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.config import LOW_CONFIDENCE_THRESHOLD

_KB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "disease_kb.json")


@dataclass
class Advisory:
    """The assembled advisory for one prediction."""

    label: str
    crop: str
    healthy: bool
    confidence: float
    symptoms: str
    likely_cause: str
    organic_treatment: str
    chemical_treatment: str
    prevention: str
    note: str  # reasoning note (low confidence / healthy / ok)


def load_kb() -> Dict[str, dict]:
    """Load the disease knowledge base from JSON."""
    with open(_KB_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _lookup(kb: Dict[str, dict], label: str) -> Optional[dict]:
    """Case-insensitive lookup of a label in the KB."""
    if label in kb:
        return kb[label]
    lowered = {k.lower(): k for k in kb}
    key = lowered.get(label.lower())
    return kb[key] if key else None


def build_advisory(
    predictions: List[Tuple[str, float]], kb: Optional[Dict[str, dict]] = None
) -> Advisory:
    """Assemble an advisory from ranked ``(label, confidence)`` predictions.

    Reasoning rules:
      * If the top-1 confidence is below the threshold, add a note advising the
        user to compare the top-3 and confirm with an expert.
      * If the top prediction is a 'healthy' class, return monitoring tips rather
        than treatment.

    Raises:
        ValueError: if ``predictions`` is empty.
    """
    if not predictions:
        raise ValueError("No predictions to build an advisory from.")
    kb = kb or load_kb()
    top_label, top_conf = predictions[0]
    entry = _lookup(kb, top_label)

    if entry is None:  # label not in KB — degrade gracefully
        return Advisory(
            label=top_label, crop="Unknown", healthy=False, confidence=top_conf,
            symptoms="Not in the knowledge base.", likely_cause="Unknown.",
            organic_treatment="Consult a local agronomist for identification.",
            chemical_treatment="Consult a local agronomist before applying chemicals.",
            prevention="Send a clear, well-lit leaf photo to an extension service for confirmation.",
            note="This label is not covered by the knowledge base; please verify with an expert.",
        )

    healthy = bool(entry.get("healthy", False))
    if healthy:
        note = "The plant looks healthy — these are monitoring tips, not treatments."
    elif top_conf < LOW_CONFIDENCE_THRESHOLD:
        note = (
            f"Confidence is low ({top_conf:.0%}). Compare the top-3 results below and "
            "confirm with an expert before acting."
        )
    else:
        note = f"Top match with {top_conf:.0%} confidence."

    return Advisory(
        label=top_label, crop=entry.get("crop", "Unknown"), healthy=healthy,
        confidence=top_conf, symptoms=entry.get("symptoms", ""),
        likely_cause=entry.get("likely_cause", ""),
        organic_treatment=entry.get("organic_treatment", ""),
        chemical_treatment=entry.get("chemical_treatment", ""),
        prevention=entry.get("prevention", ""), note=note,
    )


def advisory_to_text(adv: Advisory) -> str:
    """Flatten an advisory into plain text (used as the LLM rewrite input)."""
    return (
        f"Crop: {adv.crop}\n"
        f"Diagnosis: {adv.label} (confidence {adv.confidence:.0%})\n"
        f"Symptoms: {adv.symptoms}\n"
        f"Likely cause: {adv.likely_cause}\n"
        f"Organic treatment: {adv.organic_treatment}\n"
        f"Chemical treatment: {adv.chemical_treatment}\n"
        f"Prevention: {adv.prevention}"
    )
