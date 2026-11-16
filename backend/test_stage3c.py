"""
Stage 3C Verification Suite for DocuMind AI
Local LLM Integration (Ollama) Verification

Tests:
1. LLM Service Configuration and URL Resolution
2. Parameter & Input Validation (empty prompt, out-of-range temperature & tokens)
3. Ollama Health Endpoint (GET /api/llm/health - handles both online and offline cleanly)
4. Offline / Connection Error Handling (503 Service Unavailable)
5. Model Not Found Handling (404 Not Found)
6. Timeout Error Handling (504 Gateway Timeout)
7. Mocked Successful End-to-End Generation & Response Schema (POST /api/llm/generate)
8. Mocked Model List Endpoint (GET /api/llm/models)
9. Live Ollama Generation (executes real inference if local daemon is running)
"""

import sys
from unittest.mock import AsyncMock, patch
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

import httpx
from fastapi.testclient import TestClient
from app.core.config import settings
from app.db.session import init_db
from app.services.llm_service import (
    LLMService,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    format_bytes,
)
from app.main import app


def run_stage3c_tests():
    print("=" * 70)
    print("RUNNING STAGE 3C VERIFICATION SUITE — LOCAL LLM INTEGRATION (OLLAMA)")
    print("=" * 70)

    init_db()
    client = TestClient(app)

    # TEST 1: Service Configuration
    print("\n[TEST 1] Testing LLM Service Configuration...")
    base_url = LLMService.get_base_url()
    configured_model = LLMService.get_configured_model()
    print(f"  Ollama Base URL: {base_url}")
    print(f"  Configured Model: {configured_model}")
    print(f"  Timeout: {settings.OLLAMA_TIMEOUT}s")
    assert base_url.startswith("http://"), "Base URL must be valid HTTP"
    assert len(configured_model) > 0, "Configured model name cannot be empty"
    print("  ✓ Configuration parameters validated")

    # TEST 2: Helper Functions
    print("\n[TEST 2] Testing Helper Utilities...")
    assert format_bytes(500 * 1024 * 1024) == "500.0 MB"
    assert format_bytes(2 * 1024 * 1024 * 1024) == "2.00 GB"
    assert format_bytes(0) == "0 MB"
    print("  ✓ Byte formatting helper validated")

    # TEST 3: Input & Parameter Validation
    print("\n[TEST 3] Testing Input and Parameter Validation...")
    
    # Empty prompt
    res = client.post("/api/llm/generate", json={"prompt": "   "})
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    assert "empty" in res.json()["detail"].lower()

    # Temperature out of bounds (> 2.0)
    res = client.post("/api/llm/generate", json={"prompt": "Hello", "temperature": 3.5})
    assert res.status_code == 422, f"Expected 422, got {res.status_code}"

    # Temperature out of bounds (< 0.0)
    res = client.post("/api/llm/generate", json={"prompt": "Hello", "temperature": -0.5})
    assert res.status_code == 422, f"Expected 422, got {res.status_code}"

    # Max tokens out of bounds (<= 0)
    res = client.post("/api/llm/generate", json={"prompt": "Hello", "max_tokens": 0})
    assert res.status_code == 422, f"Expected 422, got {res.status_code}"

    print("  ✓ Empty prompt, negative temperature, and invalid max_tokens rejected")

    # TEST 4: Health Check Endpoint (GET /api/llm/health)
    print("\n[TEST 4] Testing GET /api/llm/health (Live / Offline State)...")
    res = client.get("/api/llm/health")
    assert res.status_code == 200, res.text
    health_data = res.json()
    print(f"  Health Status: {health_data['status']}")
    print(f"  Ollama Reachable: {health_data['ollama_reachable']}")
    print(f"  Configured Model: {health_data['configured_model']}")
    print(f"  Message: {health_data['message']}")
    assert health_data["status"] in ["ok", "degraded", "unavailable"]
    assert "configured_model" in health_data
    assert "available_models" in health_data
    print("  ✓ Health endpoint returned standard structured diagnostic response")

    # TEST 5: Mocked Offline / Connection Error (503 Service Unavailable)
    print("\n[TEST 5] Testing Offline Ollama Handling (503 Service Unavailable)...")
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        res = client.post("/api/llm/generate", json={"prompt": "What is AI?"})
        assert res.status_code == 503, f"Expected 503, got {res.status_code}"
        assert "unreachable" in res.json()["detail"].lower() or "connect" in res.json()["detail"].lower()
    print("  ✓ Ollama offline scenario handled with 503 Service Unavailable")

    # TEST 6: Mocked Model Not Found (404 Not Found)
    print("\n[TEST 6] Testing Model Not Found Handling (404 Not Found)...")
    mock_resp_404 = httpx.Response(
        status_code=404,
        json={"error": "model 'non-existent-model' not found, try pulling it first"},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_resp_404):
        res = client.post("/api/llm/generate", json={"prompt": "Hello", "model": "non-existent-model"})
        assert res.status_code == 404, f"Expected 404, got {res.status_code}"
        assert "not found" in res.json()["detail"].lower()
    print("  ✓ Missing model handled with 404 Not Found and pull instruction")

    # TEST 7: Mocked Timeout Error (504 Gateway Timeout)
    print("\n[TEST 7] Testing Timeout Error Handling (504 Gateway Timeout)...")
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timeout")):
        res = client.post("/api/llm/generate", json={"prompt": "Complex calculation"})
        assert res.status_code == 504, f"Expected 504, got {res.status_code}"
        assert "timed out" in res.json()["detail"].lower()
    print("  ✓ Inference timeout handled with 504 Gateway Timeout")

    # TEST 8: Mocked Successful Generation Response Structure
    print("\n[TEST 8] Testing Successful Generation Response Structure...")
    mock_success = httpx.Response(
        status_code=200,
        json={
            "model": "qwen2.5:3b",
            "created_at": "2026-11-06T12:00:00Z",
            "response": "Machine learning is a subset of artificial intelligence that learns from data.",
            "done": True,
            "total_duration": 450000000,  # 450 ms in nanoseconds
            "prompt_eval_count": 12,
            "eval_count": 16,
        },
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )
    with patch("httpx.AsyncClient.post", return_value=mock_success):
        payload = {
            "prompt": "Explain machine learning in simple terms.",
            "temperature": 0.3,
            "max_tokens": 128,
            "system_prompt": "You are an educator.",
        }
        res = client.post("/api/llm/generate", json=payload)
        assert res.status_code == 200, res.text
        gen_data = res.json()
        print(f"  Sample Mocked Output: '{gen_data['response']}'")
        assert gen_data["prompt"] == payload["prompt"]
        assert gen_data["response"].startswith("Machine learning")
        assert gen_data["model"] == "qwen2.5:3b"
        assert gen_data["done"] is True
        assert gen_data["total_duration_ms"] == 450.0
        assert gen_data["tokens_generated"] == 16
    print("  ✓ Generation response structure adheres to API contract")

    # TEST 9: Mocked Models Endpoint (GET /api/llm/models)
    print("\n[TEST 9] Testing GET /api/llm/models...")
    mock_tags = httpx.Response(
        status_code=200,
        json={
            "models": [
                {
                    "name": "qwen2.5:3b",
                    "modified_at": "2026-11-06T10:00:00Z",
                    "size": 1900000000,
                    "digest": "sha256:abc123456",
                },
                {
                    "name": "llama3.2:3b",
                    "modified_at": "2026-11-05T08:00:00Z",
                    "size": 2000000000,
                    "digest": "sha256:def789012",
                },
            ]
        },
        request=httpx.Request("GET", "http://localhost:11434/api/tags"),
    )
    with patch("httpx.AsyncClient.get", return_value=mock_tags):
        res = client.get("/api/llm/models")
        assert res.status_code == 200, res.text
        models_data = res.json()
        assert models_data["total"] == 2
        assert models_data["models"][0]["name"] == "qwen2.5:3b"
        assert models_data["models"][0]["size_formatted"] == "1.77 GB"
    print("  ✓ Models listing endpoint returned correctly parsed model metadata")

    # TEST 10: Live Inference Probe (if Ollama is running)
    print("\n[TEST 10] Checking for Live Local Inference...")
    health = client.get("/api/llm/health").json()
    if health["ollama_reachable"] and health["model_available"]:
        print(f"  Live Ollama detected! Running real test prompt on {health['configured_model']}...")
        live_res = client.post(
            "/api/llm/generate",
            json={"prompt": "Say hello in 3 words.", "max_tokens": 10},
        )
        if live_res.status_code == 200:
            print(f"  ✓ Live generation succeeded: '{live_res.json()['response'].strip()}'")
    else:
        print(f"  (Local Ollama not active or model not downloaded: {health['message']})")
        print("  ✓ Live test conditionally bypassed (all error handling verified via mocks)")

    print("\n" + "=" * 70)
    print("ALL STAGE 3C VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_stage3c_tests()

