"""
FastAPI Backend Server for Trilingual Speech-to-Text Studio.
Hosts the fine-tuned Meta MMS-300M acoustic model, provides REST endpoints
for real-time audio transcription, language detection, and Levenshtein metric calculations,
and serves the dark STT Studio UI at the root path.
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Add repository root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.inference import STTInferenceEngine, compute_accuracy_metrics

app = FastAPI(
    title="Trilingual STT Engine API",
    description="Automatic Speech Recognition for Gujarati, Hindi, and English powered by fine-tuned MMS-300M",
    version="1.0.0"
)

# Enable CORS for local testing and Hugging Face cross-origin embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = ROOT_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "best_checkpoint"

# Global inference engine instance (lazily or eagerly initialized)
engine_instance: Optional[STTInferenceEngine] = None


def get_engine() -> STTInferenceEngine:
    global engine_instance
    if engine_instance is None:
        print("🚀 Initializing STT Inference Engine...")
        engine_instance = STTInferenceEngine(CHECKPOINT_DIR)
    return engine_instance


@app.on_event("startup")
async def startup_event():
    """Preload model at server launch for instantaneous first-inference latency."""
    try:
        get_engine()
    except Exception as e:
        print(f"⚠️ Warning: Model delayed load or checkpoint missing: {e}")


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health_check():
    """System health check and inference readiness report."""
    engine = get_engine()
    return {
        "status": "healthy",
        "ready": True,
        "model_id": "divy3108/multilingual-stt-model",
        "architecture": "Wav2Vec2ForCTC (Meta MMS-300M)",
        "device": str(engine.device),
        "parameters": sum(p.numel() for p in engine.model.parameters()),
        "vocab_size": len(engine.tokenizer),
        "supported_languages": ["gu", "hi", "en", "auto"]
    }


@app.get("/api/models")
async def list_models():
    """Lists active and available speech models."""
    return {
        "active_model": "divy3108/multilingual-stt-model",
        "models": [
            {
                "id": "divy3108/multilingual-stt-model",
                "name": "Trilingual MMS-300M Master (Gujarati • Hindi • English)",
                "type": "wav2vec2_ctc",
                "vocab_size": 250,
                "target_cer": "10.30%",
                "status": "active_production"
            }
        ]
    }


@app.post("/api/transcribe_upload")
@app.post("/api/transcribe")
@app.post("/api/infer/upload")
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    reference_text: Optional[str] = Form(None)
):
    """
    Transcribes an uploaded audio file (.wav, .mp3, .ogg, .webm, .flac, .m4a).
    Automatically detects spoken language script and computes Levenshtein accuracy if reference text is provided.
    """
    engine = get_engine()
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    temp_path = TEMP_DIR / f"upload_{int(time.time() * 1000)}_{os.getpid()}{suffix}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = engine.transcribe(
            audio_source=temp_path,
            language=language,
            reference_text=reference_text
        )
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


class MetricRequest(BaseModel):
    reference: str
    hypothesis: str


@app.post("/api/metrics/calculate")
async def calculate_metrics_endpoint(req: MetricRequest):
    """
    Calculates exact Character Error Rate (CER) and Word Error Rate (WER)
    via mathematical Levenshtein distance with zero guesswork.
    """
    metrics = compute_accuracy_metrics(req.reference, req.hypothesis)
    return JSONResponse(content=metrics)


# ==============================================================================
# FRONTEND STATIC ASSET SERVING
# ==============================================================================

FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="STT Studio API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (HF Space default: 7860)")
    args = parser.parse_args()

    print(f"\n🎙️ Starting Trilingual STT Studio on http://{args.host}:{args.port} ...")
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=False)
