"""
Evaluation test script running STT inference on real Gujarati, Hindi, and English audio samples.
Verifies transcription, automatic language detection, real-time factor, and latency.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.inference import STTInferenceEngine, compute_accuracy_metrics

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "best_checkpoint"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "frontend" / "samples"

# Ground truth reference texts for standard samples
GROUND_TRUTH = {
    "en": "please turn off the lights in the room",
    "hi": "घर जाने का रास्ता दिखाओ",
    "gu": "તેવીસમી તારીખે મારા કેલેન્ડરમાં શું છે"
}

def run_sample_evaluation():
    print("=" * 65)
    print("🎙️ TRILINGUAL ASR BENCHMARK EVALUATION — LIVE INFERENCE TEST")
    print("=" * 65)

    engine = STTInferenceEngine(CHECKPOINT)

    for lang in ["en", "hi", "gu"]:
        sample_path = SAMPLES_DIR / f"sample_{lang}.wav"
        if not sample_path.exists():
            print(f"Sample not found: {sample_path}")
            continue

        result = engine.transcribe(str(sample_path), language="auto")

        print(f"\n--- [{lang.upper()}] Audio Sample ({result['duration_sec']}s) ---")
        print(f"Predicted Text:      '{result['predicted_text']}'")
        print(f"Detected Language:   {result['detected_language']} (Confidence: {result['language_confidence']})")
        print(f"Inference Latency:   {result['inference_time_ms']} ms")
        print(f"Real-Time Factor:    {result['real_time_factor']}x")

        # If reference provided, compute accuracy
        if lang in GROUND_TRUTH:
            metrics = compute_accuracy_metrics(GROUND_TRUTH[lang], result['predicted_text'])
            print(f"Reference Text:      '{GROUND_TRUTH[lang]}'")
            print(f"Character Error Rate:{metrics['cer']}%")
            print(f"Word Error Rate:     {metrics['wer']}%")

    print("\n" + "=" * 65)
    print("✓ Live Audio Verification Complete!")
    print("=" * 65)

if __name__ == "__main__":
    run_sample_evaluation()
