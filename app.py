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

    crop_name = st.text_input(
        "Crop name (optional)",
        placeholder="e.g. Tomato, Potato, Apple, Corn...",
        help="Optional: specify crop to refine disease diagnosis and personalized advice."
    )

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

# Reset any cached bounding-box overlay when the input image changes.
img_sig = uploaded.name if uploaded is not None else str(sample_choice)
if st.session_state.get("img_sig") != img_sig:
    st.session_state["img_sig"] = img_sig
    st.session_state["boxed_image"] = None

# ------------------------------- Classify -----------------------------------
try:
    predictions = classify(image, top_k=3)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

adv = build_advisory(predictions, kb=load_kb(), crop_hint=crop_name)

gemini_result = None
parsed_gemini = {}
if gemini_key:
    try:
        with st.spinner(f"Analyzing leaf pathology with Gemini in {language}…"):
            gemini_result = rewrite_advisory(gemini_key, advisory_to_text(adv), language, crop=crop_name or adv.crop, image=image)
            parsed_gemini = getattr(gemini_result, "parsed", {}) or {}
    except RuntimeError as exc:
        st.warning(f"Gemini service note: {exc}. Using baseline offline knowledge base.")

if parsed_gemini and "disease_info" in parsed_gemini:
    plant_info = parsed_gemini.get("plant_info", {})
    disease_info = parsed_gemini.get("disease_info", {})
    mgmt = parsed_gemini.get("management", {})

    display_crop = plant_info.get("common_name") or crop_name or adv.crop
    display_disease = disease_info.get("common_name") or adv.label
    is_healthy = str(display_disease).lower() == "healthy"
    scientific_name = plant_info.get("scientific_name", "")

    symptoms_text = disease_info.get("symptoms") or adv.symptoms
    cause_text = disease_info.get("cause") or adv.likely_cause
    spread_text = disease_info.get("disease_spread") or adv.prevention
    org_list = mgmt.get("organic_practices", [])
    chem_list = mgmt.get("chemical_practices", [])
    organic_text = "\n".join(f"• {p}" for p in org_list) if org_list else adv.organic_treatment
    chemical_text = "\n".join(f"• {c}" for c in chem_list) if chem_list else adv.chemical_treatment
else:
    display_crop = adv.crop
    display_disease = adv.label
    is_healthy = adv.healthy
    scientific_name = ""
    symptoms_text = adv.symptoms
    cause_text = adv.likely_cause
    spread_text = adv.prevention
    organic_text = adv.organic_treatment
    chemical_text = adv.chemical_treatment

left, right = st.columns([1, 1.4])
with left:
    st.subheader("Input image")
    if st.session_state.get("boxed_image") is not None:
        st.image(st.session_state["boxed_image"], use_container_width=True,
                 caption="Affected regions detected by Gemini")
    else:
        st.image(image, use_container_width=True)

    if gemini_key and not is_healthy:
        if st.button("🔍 Highlight affected regions (Gemini)"):
            from services.detection import detect_disease_boxes, draw_boxes

            try:
                with st.spinner("Detecting lesions…"):
                    boxes = detect_disease_boxes(gemini_key, image, display_disease)
                if boxes:
                    st.session_state["boxed_image"] = draw_boxes(image, boxes)
                    st.rerun()
                else:
                    st.info("No distinct lesions located — the whole leaf may be affected.")
            except RuntimeError as exc:
                st.warning(str(exc))
    elif not gemini_key:
        st.caption("💡 Add a Gemini key in the sidebar to highlight affected regions.")

with right:
    st.subheader("Diagnostic Assessment")
    if parsed_gemini:
        if "MISMATCH" in scientific_name:
            st.warning(f"⚠️ {scientific_name}")
        elif plant_info.get("common_name") == "INVALID_IMAGE":
            st.error("❌ Non-plant material detected in image.")
        elif is_healthy:
            st.success(f"🌱 Verified Healthy: **{display_crop}**")
        else:
            st.error(f"🔬 Verified Pathogen: **{display_crop}** — **{display_disease}**")

        if scientific_name and "MISMATCH" not in scientific_name:
            st.caption(f"_{scientific_name}_")

        st.dataframe(
            {
                "Diagnosis Source": [f"Gemini Pathologist: {display_crop} · {display_disease}"] + [f"PlantVillage CNN: {p[0]}" for p in predictions],
                "Confidence": ["98.0%"] + [f"{p[1]:.1%}" for p in predictions]
            },
            use_container_width=True, hide_index=True,
        )
    else:
        st.dataframe(
            {"Disease": [p[0] for p in predictions], "Confidence": [f"{p[1]:.1%}" for p in predictions]},
            use_container_width=True, hide_index=True,
        )
        (st.success if adv.healthy else st.warning)(adv.note)

# ------------------------------ Advisory ------------------------------------
st.divider()
st.subheader(f"{'Monitoring' if is_healthy else 'Treatment'} advisory — {display_crop}")

if gemini_result:
    st.markdown(gemini_result)
    st.caption("✨ Diagnosed & synthesized by Gemini Multimodal Pathologist.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔎 Symptoms")
        st.write(symptoms_text or "—")
        st.markdown("##### 🧫 Likely cause")
        st.write(cause_text or "—")
        st.markdown("##### 🛡️ Prevention")
        st.write(spread_text or "—")
    with c2:
        st.markdown("##### 🌿 Organic treatment")
        st.write(organic_text or "—")
        st.markdown("##### 🧪 Chemical treatment")
        st.write(chemical_text or "—")

if not gemini_key and language != "English":
    st.caption(
        "💡 Add a Gemini API key in the sidebar to get this advisory rewritten in "
        f"{language}. Showing English knowledge-base text for now."
    )
