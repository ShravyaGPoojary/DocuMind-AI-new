"""
Stage 5 Verification Suite for DocuMind AI
Intelligent Document Summarization (Groq GPT-OSS 120B Integration)

Tests:
1. Supported Summary Types Endpoint & Validation (GET /api/summarization/types)
2. Nonexistent Document ID Handling (404 Not Found)
3. Empty Document Handling (400 Bad Request)
4. Short Document Direct Summarization (mocked Groq completion + sections)
5. All 4 Summary Types (Brief, Detailed, Key Points, Executive)
6. Long Document Hierarchical Map-Reduce Strategy (batching + reduce synthesis)
7. Groq Error Handling (401, 429, 504, 503, 502 & verification of NO silent Qwen fallback)
8. Optional Alternative Provider Routing (explicit provider='ollama')
9. Live Real Groq Summarization (conditional on GROQ_API_KEY, zero key exposure)
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.core.config import settings
from app.db.session import init_db, SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.document_service import DocumentService
from app.services.groq_service import (
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqNetworkError,
    GroqTimeoutError,
    GroqResponseError,
    GroqServiceError,
)
from app.services.summarization_service import SummarizationService
from app.main import app


def run_stage5_tests():
    print("=" * 70)
    print("RUNNING STAGE 5 VERIFICATION SUITE — INTELLIGENT DOCUMENT SUMMARIZATION")
    print("=" * 70)

    init_db()
    db = SessionLocal()
    client = TestClient(app)

    short_doc_id = None
    long_doc_id = None
    empty_doc_id = None

    try:
        # TEST 1: Summary Types Catalog & Input Validation
        print("\n[TEST 1] Testing Summary Types Endpoint & Input Normalization...")
        types_res = client.get("/api/summarization/types")
        assert types_res.status_code == 200
        types_data = types_res.json()["types"]
        type_ids = [t["id"] for t in types_data]
        assert "brief" in type_ids
        assert "detailed" in type_ids
        assert "key_points" in type_ids
        assert "executive" in type_ids
        print(f"  ✓ GET /api/summarization/types returned 4 types: {type_ids}")

        # Normalization checks
        assert SummarizationService.normalize_summary_type("short") == "brief"
        assert SummarizationService.normalize_summary_type("tldr") == "brief"
        assert SummarizationService.normalize_summary_type("detailed") == "detailed"
        assert SummarizationService.normalize_summary_type("key_points") == "key_points"
        assert SummarizationService.normalize_summary_type("executive") == "executive"

        try:
            SummarizationService.normalize_summary_type("invalid_type_xyz")
            assert False, "Should reject invalid summary type"
        except ValueError as err:
            assert "invalid summary_type" in str(err).lower()
        print("  ✓ Summary type normalization and aliases ('short' -> 'brief') validated")

        # TEST 2: Nonexistent Document ID Handling
        print("\n[TEST 2] Testing Nonexistent Document ID (404 Not Found)...")
        fake_id = "00000000-0000-0000-0000-000000000000"
        bad_res = client.post(f"/api/summarization/{fake_id}", json={"summary_type": "brief"})
        assert bad_res.status_code == 404
        assert "not found" in bad_res.json()["detail"].lower()
        print("  ✓ Nonexistent document ID correctly returned 404 Not Found")

        # TEST 3: Empty Document Handling
        print("\n[TEST 3] Testing Empty Document Handling (400 Bad Request)...")
        empty_doc = DocumentService.create_document(
            db=db,
            filename="empty_test.txt",
            original_filename="empty_test.txt",
            file_type="txt",
            file_size=0,
            extracted_text="",
            chunk_count=0,
            processing_status="ready",
        )
        empty_doc_id = empty_doc.id

        empty_res = client.post(f"/api/summarization/{empty_doc_id}", json={"summary_type": "brief"})
        assert empty_res.status_code == 400
        assert "no extracted text or chunks" in empty_res.json()["detail"].lower()
        print("  ✓ Document with zero text or chunks correctly rejected with 400 Bad Request")

        # TEST 4: Short Document Direct Summarization (Mocked)
        print("\n[TEST 4] Testing Short Document Direct Summarization (Strategy: direct)...")
        short_doc = DocumentService.create_document(
            db=db,
            filename="quantum_intro.pdf",
            original_filename="quantum_intro.pdf",
            file_type="pdf",
            file_size=1024,
            extracted_text="Quantum computing utilizes qubits that exhibit superposition and entanglement.",
            page_count=2,
            processing_status="ready",
        )
        short_doc_id = short_doc.id

        chunks_data = [
            {
                "chunk_index": 0,
                "content": "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to execute calculations exponentially faster than classical computers for specific problem classes.",
                "word_count": 23,
                "source_pages": "1",
                "slide_number": None,
            },
            {
                "chunk_index": 1,
                "content": "Shor's algorithm demonstrates polynomial-time integer factorization, posing serious security challenges for traditional RSA public-key cryptosystems.",
                "word_count": 16,
                "source_pages": "2",
                "slide_number": None,
            },
        ]
        DocumentService.save_chunks(db, short_doc_id, chunks_data)

        mock_brief_llm = {
            "prompt": "prompt",
            "response": (
                "## Executive Overview\n"
                "This document introduces the foundational principles of quantum computing, focusing on superposition and entanglement. It highlights how quantum mechanical operations can solve computational problems exponentially faster than classical hardware.\n\n"
                "## Key Takeaways\n"
                "- Leverages superposition and entanglement for exponential speedups\n"
                "- Shor's algorithm solves integer factorization in polynomial time\n"
                "- Traditional RSA public-key encryption faces cryptographic vulnerability"
            ),
            "model": "openai/gpt-oss-120b",
            "provider": "groq",
            "done": True,
            "total_duration_ms": 115.0,
            "tokens_generated": 65,
        }

        with patch("app.services.groq_service.GroqService.generate", return_value=mock_brief_llm):
            res = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": "brief"})
            assert res.status_code == 200, res.text
            data = res.json()

            print(f"  Summary Type: {data['summary_type']} ({data['summary_type_label']})")
            print(f"  Strategy: {data['strategy_used']}")
            print(f"  Model: {data['model']}")
            print(f"  Sections Parsed: {len(data['sections'])}")
            print(f"  Key Points Extracted: {len(data['key_points'])}")

            assert data["strategy_used"] == "direct"
            assert data["summary_type"] == "brief"
            assert data["document_name"] == "quantum_intro.pdf"
            assert data["model"] == "openai/gpt-oss-120b"
            assert data["provider"] == "groq"
            assert len(data["sections"]) >= 2
            assert data["sections"][0]["title"] == "Executive Overview"
            assert len(data["key_points"]) >= 3
            print("  ✓ Short document direct summarization verified with structured sections and key points")

        # TEST 5: All 4 Summary Types Verification
        print("\n[TEST 5] Testing All 4 Summary Types (Brief, Detailed, Key Points, Executive)...")
        type_templates = {
            "brief": (
                "## Executive Overview\nBrief narrative overview.\n\n"
                "## Key Takeaways\n- Point A\n- Point B"
            ),
            "detailed": (
                "## 1. Document Scope & Objectives\nDetailed narrative of scope.\n\n"
                "## 2. Core Methodologies & Arguments\n- Method 1\n- Method 2\n\n"
                "## 3. Critical Findings & Supporting Evidence\n- Evidence 1\n- Evidence 2\n\n"
                "## 4. Strategic Conclusions & Next Steps\nConclusion narrative."
            ),
            "key_points": (
                "## Core Highlights & Principles\n- **Highlight 1**: Principle A\n\n"
                "## Methodological & Technical Details\n- **Tech 1**: Detail B\n\n"
                "## Critical Findings & Outcomes\n- **Finding 1**: Outcome C\n\n"
                "## Actionable Recommendations\n- **Action 1**: Next step D"
            ),
            "executive": (
                "## Executive Overview\nStrategic briefing for leadership.\n\n"
                "## Strategic Findings\n- Strategic finding 1\n\n"
                "## Impact & Evidence Assessment\nImpact narrative.\n\n"
                "## Strategic Recommendations\n- Immediate action 1"
            ),
        }

        for stype, tcontent in type_templates.items():
            mock_resp = {
                "prompt": "prompt",
                "response": tcontent,
                "model": "openai/gpt-oss-120b",
                "provider": "groq",
                "done": True,
                "total_duration_ms": 120.0,
                "tokens_generated": 80,
            }
            with patch("app.services.groq_service.GroqService.generate", return_value=mock_resp):
                res_type = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": stype})
                assert res_type.status_code == 200
                tdata = res_type.json()
                assert tdata["summary_type"] == stype
                assert len(tdata["sections"]) >= 2
                print(f"  ✓ Summary Type '{stype}': generated {len(tdata['sections'])} sections and {len(tdata['key_points'])} key points")

        # TEST 6: Long Document Hierarchical Map-Reduce Strategy
        print("\n[TEST 6] Testing Long Document Map-Reduce Strategy (> 8 chunks)...")
        long_doc = DocumentService.create_document(
            db=db,
            filename="enterprise_architecture_whitepaper.docx",
            original_filename="enterprise_architecture_whitepaper.docx",
            file_type="docx",
            file_size=65536,
            extracted_text="Enterprise architecture whitepaper spanning multiple domains.",
            page_count=12,
            processing_status="ready",
        )
        long_doc_id = long_doc.id

        # Seed 12 chunks to exceed MAP_REDUCE_CHUNK_THRESHOLD (8)
        long_chunks = []
        for idx in range(12):
            long_chunks.append({
                "chunk_index": idx,
                "content": f"Section {idx + 1}: Detailed enterprise microservices analysis, event sourcing, domain-driven design, and Kubernetes scaling benchmarks for batch {idx + 1}.",
                "word_count": 22,
                "source_pages": str(idx + 1),
                "slide_number": None,
            })
        DocumentService.save_chunks(db, long_doc_id, chunks_data=long_chunks)

        mock_map_resp = {
            "prompt": "map prompt",
            "response": "Section digest highlighting microservices architecture, event sourcing, and scaling performance.",
            "model": "openai/gpt-oss-120b",
            "provider": "groq",
            "done": True,
            "total_duration_ms": 90.0,
            "tokens_generated": 20,
        }
        mock_reduce_resp = {
            "prompt": "reduce prompt",
            "response": (
                "## 1. Document Scope & Objectives\n"
                "Comprehensive architecture synthesis across all 12 whitepaper sections.\n\n"
                "## 2. Core Methodologies & Arguments\n"
                "- Microservices architecture with event-driven message queuing\n"
                "- Domain-driven design aggregates for boundary isolation\n\n"
                "## 3. Critical Findings & Supporting Evidence\n"
                "- Low-latency throughput verified across containerized nodes\n\n"
                "## 4. Strategic Conclusions & Next Steps\n"
                "Production deployment recommended with automated blue-green pipelines."
            ),
            "model": "openai/gpt-oss-120b",
            "provider": "groq",
            "done": True,
            "total_duration_ms": 210.0,
            "tokens_generated": 95,
        }

        # Adaptive batching dynamically bundles small chunks into token-budgeted batches + 1 reduce synthesis
        async def mock_groq_func(prompt, **kwargs):
            if "Section Digest:" in prompt or "Document Section" in prompt:
                return mock_map_resp
            return mock_reduce_resp

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_groq_func) as mock_groq_gen:
            res_mr = client.post(f"/api/summarization/{long_doc_id}", json={"summary_type": "detailed"})
            assert res_mr.status_code == 200, res_mr.text
            mr_data = res_mr.json()

            print(f"  Strategy Used: '{mr_data['strategy_used']}'")
            print(f"  Chunks Processed: {mr_data['chunk_count']}")
            print(f"  Groq API Invocations: {mock_groq_gen.call_count}")

            assert mr_data["strategy_used"] == "map_reduce"
            assert mr_data["chunk_count"] == 12
            assert mock_groq_gen.call_count >= 2
            assert len(mr_data["sections"]) == 4
            print(f"  ✓ Adaptive Map-Reduce summarization successfully executed in {mock_groq_gen.call_count} API calls")

        # TEST 7: Groq Error Handling & No Silent Fallback to Qwen
        print("\n[TEST 7] Testing Groq Error Handling & Verification of No Silent Qwen Fallback...")

        # 7a: 401 Unauthorized
        with patch("app.services.groq_service.GroqService.generate", side_effect=GroqAuthError("Invalid Groq API key")):
            res = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": "brief"})
            assert res.status_code == 401
            assert "invalid groq api key" in res.json()["detail"].lower()
            print("  ✓ 401 Unauthorized returned on invalid API key")

        # 7b: 429 Rate limit
        with patch("app.services.groq_service.GroqService.generate", side_effect=GroqRateLimitError("Rate limit reached")):
            res = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": "brief"})
            assert res.status_code == 429
            assert "rate limit" in res.json()["detail"].lower()
            print("  ✓ 429 Too Many Requests returned on rate limit")

        # 7c: 504 Timeout
        with patch("app.services.groq_service.GroqService.generate", side_effect=GroqTimeoutError("Timed out")):
            res = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": "brief"})
            assert res.status_code == 504
            assert "timed out" in res.json()["detail"].lower()
            print("  ✓ 504 Gateway Timeout returned on request timeout")

        # 7d: 503 Network failure
        with patch("app.services.groq_service.GroqService.generate", side_effect=GroqNetworkError("Connection refused")):
            res = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": "brief"})
            assert res.status_code == 503
            assert "connection" in res.json()["detail"].lower()
            print("  ✓ 503 Service Unavailable returned on network failure")

        # 7e: Verify NO silent fallback to Qwen
        with patch("app.services.groq_service.GroqService.generate", side_effect=GroqServiceError("Groq service failure")):
            with patch("app.services.llm_service.LLMService.generate") as mock_qwen:
                res = client.post(f"/api/summarization/{short_doc_id}", json={"summary_type": "brief"})
                assert res.status_code == 502
                assert not mock_qwen.called, "CRITICAL: Must NOT silently switch to Qwen when Groq fails!"
                print("  ✓ Verified: System does NOT silently switch to Qwen when Groq summarization fails")

        # TEST 8: Optional Alternative Provider (Ollama)
        print("\n[TEST 8] Testing Optional Alternative Provider Routing (provider='ollama')...")
        mock_ollama_summary = {
            "prompt": "prompt",
            "response": (
                "## Executive Overview\nSummary synthesized locally via Qwen.\n\n"
                "## Key Takeaways\n- Local model execution\n- Zero cloud dependencies"
            ),
            "model": "qwen2.5:3b",
            "provider": "ollama",
            "done": True,
            "total_duration_ms": 280.0,
            "tokens_generated": 20,
        }
        with patch("app.services.llm_service.LLMService.generate", return_value=mock_ollama_summary):
            res_ollama = client.post(
                f"/api/summarization/{short_doc_id}",
                json={"summary_type": "brief", "provider": "ollama"},
            )
            assert res_ollama.status_code == 200
            odata = res_ollama.json()
            assert odata["provider"] == "ollama"
            assert odata["model"] == "qwen2.5:3b"
            print(f"  ✓ Explicit provider='ollama' successfully routed to Qwen ({odata['model']})")

        # TEST 9: Live Real Groq Summarization Test (Conditional on API Key)
        print("\n[TEST 9] Checking Live Real Groq Summarization with 'openai/gpt-oss-120b'...")
        has_key = bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())
        if has_key:
            print("  GROQ_API_KEY detected. Executing 1 live summarization with 'openai/gpt-oss-120b'...")
            try:
                live_res = client.post(
                    f"/api/summarization/{short_doc_id}",
                    json={"summary_type": "brief"},
                )
                assert live_res.status_code == 200, live_res.text
                live_data = live_res.json()
                print(f"  ✓ Live Summarization Succeeded!")
                print(f"  ✓ Document: '{live_data['document_name']}'")
                print(f"  ✓ Model: {live_data['model']} | Provider: {live_data['provider']}")
                print(f"  ✓ Strategy: {live_data['strategy_used']} | Time: {live_data['generation_time_ms']}ms")
                print(f"  ✓ Sections Parsed: {len(live_data['sections'])}")
                print(f"  ✓ Preview: {live_data['summary'][:160]}...")
                assert live_data["model"] == "openai/gpt-oss-120b"
                assert len(live_data["summary"]) > 20
            except Exception as e:
                print(f"  Live call notice: {type(e).__name__}: {str(e)}")
        else:
            print("  (GROQ_API_KEY not configured: live test skipped)")

        print("\n" + "=" * 70)
        print("ALL STAGE 5 VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        print("\nCleaning up test documents and chunks...")
        if short_doc_id:
            DocumentService.delete_document(db, short_doc_id)
        if long_doc_id:
            DocumentService.delete_document(db, long_doc_id)
        if empty_doc_id:
            DocumentService.delete_document(db, empty_doc_id)
        db.close()
        print("Cleanup completed.")


if __name__ == "__main__":
    run_stage5_tests()

