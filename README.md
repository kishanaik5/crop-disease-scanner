---
title: Crop Disease Scanner
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# 🩺 Crop Disease Scanner

Upload a leaf photo to **detect the disease** with a public PlantVillage model,
then get **knowledge-based treatment advice** (cause / organic / chemical /
prevention) — with an optional multilingual, farmer-friendly rewrite.

> **Independent, public-data build.** Clean-room reimplementation using a public
> Hugging Face model and a knowledge base authored from general agronomy
> knowledge. It does not reuse, import, or reproduce any private/company code.

## What it does

- Classifies a leaf image into 38 PlantVillage classes → **top-3** with confidence.
- A **knowledge-based agent** assembles structured treatment cards for the top match.
- **Reasoning rules:** low confidence → surface top-3 + advise expert confirmation;
  healthy leaf → monitoring tips instead of treatment.
- **Optional** Gemini rewrite of the advisory in an Indian language.

## Architecture (pipeline)

1. **Detect** — `transformers.pipeline("image-classification")` with
   [`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification)
   (MobileNetV2, 38 classes), cached with `@st.cache_resource`.
2. **Reason** — `services/knowledge_base.py` looks up the top label in
   `data/disease_kb.json` and applies the low-confidence / healthy rules.
3. **Enrich (optional)** — `services/advisory.py` calls the Gemini REST API to
   restate the advisory in the chosen language; falls back to the raw KB text.
4. **Present** — input image, top-3 prediction table, and treatment cards.

## Public data sources

- **Model:** public PlantVillage-trained MobileNetV2 from the Hugging Face Hub.
- **Knowledge base:** `data/disease_kb.json`, hand-authored for all 38 classes
  (`crop, symptoms, likely_cause, organic_treatment, chemical_treatment, prevention`).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
# Optional multilingual rewrite:
export GEMINI_API_KEY="your_key_here"   # otherwise English KB text only
```

The first run downloads the model (~14 MB) from the Hugging Face Hub.

## Set the secret on Hugging Face (optional)

Only needed for the multilingual rewrite. In your Space:
**Settings → Variables and secrets → New secret**

- Name: `GEMINI_API_KEY`
- Value: your Gemini API key (https://aistudio.google.com/app/apikey)

The app reads it via `os.environ`; users can also paste their own key in the
sidebar. The key is never written to disk or logs.

## Disclaimer

Predictions and advice are decision support, not a substitute for a local
agronomist. Always confirm before applying chemical treatments.
