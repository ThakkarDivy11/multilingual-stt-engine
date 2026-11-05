"""
FastAPI Server Integration Test:
Validates /health, /api/models, /api/metrics/calculate, and /api/transcribe_upload endpoints.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient
from backend.main import app

def test_backend_api():
    print("=" * 60)
    print("🚀 FASTAPI BACKEND INTEGRATION TEST")
    print("=" * 60)

    client = TestClient(app)

    # 1. Test /health
    print("\n1. Testing GET /health ...")
    res_health = client.get("/health")
    print(f"Status: {res_health.status_code}")
    print(f"Payload: {res_health.json()}")
    assert res_health.status_code == 200, "Health check failed"

    # 2. Test /api/models
    print("\n2. Testing GET /api/models ...")
    res_models = client.get("/api/models")
    print(f"Status: {res_models.status_code}")
    print(f"Payload: {res_models.json()}")
    assert res_models.status_code == 200, "Models list failed"

    # 3. Test /api/metrics/calculate
    print("\n3. Testing POST /api/metrics/calculate ...")
    metric_payload = {
        "reference": "hello world",
        "hypothesis": "hello world"
    }
    res_metric = client.post("/api/metrics/calculate", json=metric_payload)
    print(f"Status: {res_metric.status_code}")
    print(f"Payload: {res_metric.json()}")
    assert res_metric.status_code == 200
    assert res_metric.json()["cer"] == 0.0

    # 4. Test POST /api/transcribe_upload with sample_en.wav
    print("\n4. Testing POST /api/transcribe_upload ...")
    sample_wav = Path(__file__).resolve().parent.parent / "frontend" / "samples" / "sample_en.wav"
    with open(sample_wav, "rb") as f:
        res_upload = client.post(
            "/api/transcribe_upload",
            files={"file": ("sample_en.wav", f, "audio/wav")},
            data={"language": "auto", "reference_text": "please turn off the lights in the room"}
        )
    print(f"Status: {res_upload.status_code}")
    data = res_upload.json()
    print(f"Predicted Text:    '{data.get('predicted_text')}'")
    print(f"Detected Language: {data.get('detected_language')}")
    print(f"Inference Latency: {data.get('inference_time_ms')} ms")
    print(f"Accuracy Metrics:  {data.get('benchmark_metrics')}")
    assert res_upload.status_code == 200
    assert data.get("detected_language") == "en"

    print("\n" + "=" * 60)
    print("🎉 ALL API ENDPOINTS VERIFIED & WORKING PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_backend_api()

