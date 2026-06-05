"""Crop Disease Scanner — Streamlit UI.

Upload a leaf photo → classify the disease (public PlantVillage model) → a
knowledge-based agent assembles treatment cards → optional multilingual rewrite.
Business logic lives in ``services/``; this file is the UI shell only.
"""
from __future__ import annotations

import glob
import os

import streamlit as st
from PIL import Image

from services.advisory import rewrite_advisory
from services.classifier import classify
from services.knowledge_base import advisory_to_text, build_advisory, load_kb
from utils.config import LANGUAGES, get_gemini_api_key

st.set_page_config(page_title="Crop Disease Scanner", page_icon="🩺", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_IMAGES = sorted(glob.glob(os.path.join(DATA_DIR, "sample_leaf_*")))

st.title("🩺 Crop Disease Scanner")
st.caption(
    "Upload a leaf photo to detect the disease and get knowledge-based treatment "
    "advice — with an optional multilingual, farmer-friendly rewrite."
)

# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
    sample_choice = None
    if SAMPLE_IMAGES:
        names = ["— none —"] + [os.path.basename(p) for p in SAMPLE_IMAGES]
        pick = st.selectbox("…or try a sample", options=names)
        if pick != "— none —":
            sample_choice = os.path.join(DATA_DIR, pick)

    language = st.selectbox("Advisory language", options=LANGUAGES, index=0)

    st.divider()
    ui_key = st.text_input(
        "Gemini API key (optional)", type="password",
        help="Enables a multilingual, farmer-friendly rewrite. Never stored.",
    )
    gemini_key = get_gemini_api_key(ui_key)

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
            **Pipeline:** image → CV model → knowledge base → advisory

            1. A public **PlantVillage** model
               (`transformers.pipeline("image-classification")`) returns the
               **top-3** diseases with confidence.
            2. A **knowledge-based agent** looks up the top match in a hand-authored
               disease knowledge base and assembles structured treatment cards
               (cause / organic / chemical / prevention).
            3. **Reasoning rules:** if the top-1 confidence is low it surfaces the
               top-3 and advises expert confirmation; if the leaf looks *healthy* it
               returns monitoring tips instead of treatment.
            4. **Optional LLM rewrite** (Gemini, gated on a key) restates the advice
               in the chosen Indian language, farmer-friendly. Without a key, the
               raw knowledge-base text is shown.
            """
        )

# ----------------------------- Resolve image --------------------------------
image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif sample_choice:
    image = Image.open(sample_choice)

if image is None:
    st.info("Upload a leaf photo (or pick a sample) from the sidebar to begin.")
    st.stop()

# ------------------------------- Classify -----------------------------------
left, right = st.columns([1, 1.4])
with left:
    st.subheader("Input image")
    st.image(image, use_container_width=True)

try:
    predictions = classify(image, top_k=3)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

adv = build_advisory(predictions, kb=load_kb())

with right:
    st.subheader("Top-3 predictions")
    st.dataframe(
        {"Disease": [p[0] for p in predictions], "Confidence": [f"{p[1]:.1%}" for p in predictions]},
        use_container_width=True, hide_index=True,
    )
    (st.success if adv.healthy else st.warning)(adv.note)

# ------------------------------ Advisory ------------------------------------
st.divider()
st.subheader(f"{'Monitoring' if adv.healthy else 'Treatment'} advisory — {adv.crop}")

if gemini_key and language != "English":
    try:
        with st.spinner(f"Rewriting in {language}…"):
            rewritten = rewrite_advisory(gemini_key, advisory_to_text(adv), language)
        st.markdown(rewritten)
        st.caption("✨ Rewritten by Gemini. Structured cards below are the source advice.")
    except RuntimeError as exc:
        st.warning(f"{exc} Showing the knowledge-base advisory instead.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### 🔎 Symptoms")
    st.write(adv.symptoms or "—")
    st.markdown("##### 🧫 Likely cause")
    st.write(adv.likely_cause or "—")
    st.markdown("##### 🛡️ Prevention")
    st.write(adv.prevention or "—")
with c2:
    st.markdown("##### 🌿 Organic treatment")
    st.write(adv.organic_treatment or "—")
    st.markdown("##### 🧪 Chemical treatment")
    st.write(adv.chemical_treatment or "—")

if not gemini_key and language != "English":
    st.caption(
        "💡 Add a Gemini API key in the sidebar to get this advisory rewritten in "
        f"{language}. Showing English knowledge-base text for now."
    )
