"""FastAPI microservice for Crop Disease Scanner (Hugging Face dual-mode)."""
from __future__ import annotations

import asyncio
import datetime
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from services.advisory import rewrite_advisory
from services.classifier import classify
from services.knowledge_base import advisory_to_text, build_advisory, load_kb
from utils.config import get_gemini_api_key

app = FastAPI(title="Crop Disease Scanner API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def make_log(tag: str, msg: str, log_type: str = "normal") -> dict:
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
    return {"time": time_str, "tag": tag, "msg": msg, "type": log_type}


@app.get("/health")
def health():
    return {"status": "ok", "service": "crop-disease-scanner", "port": int(os.environ.get("PORT", 8000))}


@app.post("/scan/stream")
async def scan_stream(
    file: Optional[UploadFile] = File(None),
    preset: Optional[str] = Form(None),
    crop: Optional[str] = Form(None),
    language: str = Form("English"),
    gemini_key: Optional[str] = Form(None),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
):
    resolved_key = x_gemini_key or gemini_key or get_gemini_api_key()
    crop_name = crop.strip() if crop and crop.strip() else None

    async def generator() -> AsyncGenerator[str, None]:
        crop_info = f" crop='{crop_name}'" if crop_name else ""
        yield format_sse("log", make_log("API", f"POST /scan/stream{crop_info} language='{language}'"))
        await asyncio.sleep(0.1)

        try:
            image = None
            sample_name = preset or "potato_late_blight"
            if file and file.filename:
                yield format_sse("log", make_log("Vision", f"Ingesting uploaded leaf: {file.filename}..."))
                contents = await file.read()
                image = Image.open(BytesIO(contents)).convert("RGB")
            else:
                yield format_sse("log", make_log("Vision", f"Loading preset leaf sample: {sample_name}..."))
                sample_file = DATA_DIR / f"sample_leaf_{sample_name}.jpg"
                if sample_file.exists():
                    image = Image.open(sample_file).convert("RGB")
                else:
                    samples = list(DATA_DIR.glob("sample_leaf_*.jpg"))
                    if samples:
                        image = Image.open(samples[0]).convert("RGB")

            if image is None:
                image = Image.new("RGB", (224, 224), color=(34, 139, 34))

            yield format_sse("log", make_log("ViT/CNN", "Evaluating MobileNetV2 PlantVillage model..."))
            await asyncio.sleep(0.2)

            predictions = classify(image, top_k=3)
            yield format_sse("log", make_log("Inference", f"Top match: '{predictions[0][0]}' ({predictions[0][1] * 100:.1f}%)", "success"))

            yield format_sse("log", make_log("Knowledge", f"Querying agronomic treatment protocols (disease_kb.json){' for ' + crop_name if crop_name else ''}..."))
            kb = load_kb()
            adv = build_advisory(predictions, kb=kb, crop_hint=crop_name)

            symptoms = adv.symptoms
            organic = adv.organic_treatment
            chemical = adv.chemical_treatment
            prevention = adv.prevention

            rewritten_advisory = None
            if resolved_key:
                yield format_sse("log", make_log("GenAI", f"Synthesizing {language} advisory via Gemini LLM{' for ' + adv.crop if adv.crop else ''}...", "normal"))
                try:
                    rewritten_advisory = rewrite_advisory(resolved_key, advisory_to_text(adv), language, crop=crop_name or adv.crop, image=image)
                    yield format_sse("log", make_log("GenAI", f"Synthesized verified advisory in {language}", "success"))
                except Exception as ex:
                    yield format_sse("log", make_log("GenAI", f"LLM note: {ex}", "warn"))
            else:
                yield format_sse("log", make_log("GenAI", "No Gemini API key supplied; utilizing built-in agronomic knowledge base.", "warn"))

            yield format_sse("log", make_log("Pipeline", "Diagnosis and treatment generation complete. Status 200 OK.", "success"))

            result_data = {
                "crop": adv.crop,
                "label": adv.label,
                "healthy": adv.healthy,
                "confidence": round(adv.confidence * 100, 1),
                "predictions": [{"label": p[0], "confidence": round(p[1] * 100, 1)} for p in predictions],
                "symptoms": symptoms,
                "organic_treatment": organic,
                "chemical_treatment": chemical,
                "prevention": prevention,
                "rewritten_advisory": rewritten_advisory,
                "language": language,
                "note": adv.note,
            }
            yield format_sse("result", result_data)

        except Exception as ex:
            yield format_sse("log", make_log("ERR", f"Diagnostic failed: {ex}", "err"))
            yield format_sse("error", {"error": str(ex)})

    return StreamingResponse(generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
