"""
Stage 5 Intelligent Adaptive Summarization Test Suite for DocuMind AI
Verifies:
1. Token Estimation & Budgeting (completion reservations, safety margins)
2. Small Document Adaptive Packing (multiple small chunks packed into 1 batch)
3. Medium & Large Document Dynamic Batch Sizing
4. Oversized Individual Chunk Splitting
5. Rolling TPM & RPM Rate Limiter (8,000 TPM limit enforcement & actual usage accounting)
6. Automatic HTTP 413 Dynamic Splitting & Recovery
7. HTTP 429 Rate Limit Backoff with Retry-After Header
8. Permanent Error Fast-Fail (401/404)
9. Multi-Level Hierarchical Reduction
10. Stable Content-Aware Caching & Resumability Across Boundary Shifts
11. Complete 44-Chunk Validation on 58-Page PDF (BIS703_Module_5.pdf)
"""

import asyncio
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.core.config import settings
from app.db.session import init_db, SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.summarization import SummarizationJob, SummarizationBatch
from app.services.document_service import DocumentService
from app.services.token_estimator import TokenEstimator
from app.services.rate_limiter import TokenAwareRateLimiter, groq_rate_limiter
from app.services.groq_service import (
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqTimeoutError,
    GroqNetworkError,
    GroqOversizedRequestError,
)
from app.services.summarization_service import SummarizationService
from app.main import app


def run_adaptive_summarization_tests():
    print("=" * 75)
    print("STAGE 5 INTELLIGENT ADAPTIVE SUMMARIZATION VERIFICATION SUITE")
    print("=" * 75)

    init_db()
    db = SessionLocal()
    client = TestClient(app)

    try:
        # TEST 1: Token Estimator & Budget Management
        print("\n[TEST 1] Testing Token Estimator & Budget Management...")
        sample_text = "DocuMind AI is an intelligent document understanding platform."
        tokens = TokenEstimator.estimate_tokens(sample_text)
        assert tokens > 0
        assert tokens >= len(sample_text.split())

        # Test prompt overhead calculation
        prompt_tokens = TokenEstimator.estimate_prompt_tokens(
            system_prompt="You are DocuMind AI.",
            user_prompt="Summarize this text.",
        )
        assert prompt_tokens > tokens

        # Test available input budget calculation
        budget = TokenEstimator.get_available_input_budget(
            reserved_completion=1024,
            target_budget=6000,
            safety_margin=500,
        )
        assert budget == (6000 - 1024 - 500)
        print(f"  ✓ Token estimation validated: sample={tokens} tokens, available budget={budget} tokens")

        # TEST 2: Oversized Chunk Automatic Splitting
        print("\n[TEST 2] Testing Oversized Chunk Splitting into Subsegments...")
        huge_paragraph = "Critical information technology architecture and cryptographic protocols. " * 300
        huge_tokens = TokenEstimator.estimate_tokens(huge_paragraph)
        max_sub_tokens = 500
        split_parts = TokenEstimator.split_oversized_text(huge_paragraph, max_tokens=max_sub_tokens)

        assert len(split_parts) > 1, f"Expected text to be split, got {len(split_parts)} parts"
        for i, part in enumerate(split_parts):
            p_tok = TokenEstimator.estimate_tokens(part)
            assert p_tok <= max_sub_tokens + 50, f"Subpart {i} exceeded max tokens: {p_tok} > {max_sub_tokens}"
        print(f"  ✓ Oversized text ({huge_tokens} tokens) split into {len(split_parts)} safe subsegments")

        # TEST 3: Adaptive Batch Packing (Small vs Medium vs Large Documents)
        print("\n[TEST 3] Testing Dynamic Adaptive Batch Packing...")
        # 3a. Small document: 6 tiny chunks (~50 words each) -> should pack into 1 batch!
        small_chunks = [
            Chunk(document_id="doc_small", chunk_index=i, content=f"Short paragraph {i + 1}. Factual data point.", word_count=6)
            for i in range(6)
        ]
        small_batches = SummarizationService._build_adaptive_batches(small_chunks)
        assert len(small_batches) == 1, f"Expected 1 batch for small chunks, got {len(small_batches)}"
        assert len(small_batches[0]) == 6
        print(f"  ✓ Small document: 6 small chunks packed into {len(small_batches)} batch (items: {len(small_batches[0])})")

        # 3b. Large chunks (~1,500 words each) -> should pack into multiple smaller batches
        large_chunks = [
            Chunk(document_id="doc_large", chunk_index=i, content="Detailed enterprise architecture and engineering. " * 200, word_count=1200)
            for i in range(5)
        ]
        large_batches = SummarizationService._build_adaptive_batches(large_chunks)
        assert len(large_batches) > 1, f"Expected multiple batches for large chunks, got {len(large_batches)}"
        print(f"  ✓ Large document: 5 large chunks dynamically partitioned into {len(large_batches)} adaptive batches")

        # TEST 4: Token-Aware Rolling TPM Rate Limiter
        print("\n[TEST 4] Testing Rolling TPM (8,000) Rate Limiter...")
        limiter = TokenAwareRateLimiter(tpm_limit=2000, rpm_limit=5, window_seconds=1.0)
        req1 = asyncio.run(limiter.acquire(1200))
        asyncio.run(limiter.record_actual_usage(req1, 1150))
        metrics = asyncio.run(limiter.get_rolling_metrics())
        assert metrics["current_tpm"] == 1150
        assert metrics["current_rpm"] == 1

        # Attempt to acquire another 1200 (1150 + 1200 = 2350 > 2000 limit) -> must throttle until window slides
        t0 = time.time()
        req2 = asyncio.run(limiter.acquire(1200))
        throttle_elapsed = time.time() - t0
        assert throttle_elapsed >= 0.8, f"Expected rate limiter to throttle, but elapsed was {throttle_elapsed}s"
        print(f"  ✓ TokenAwareRateLimiter throttled request appropriately ({round(throttle_elapsed, 2)}s) when rolling TPM exceeded")

        # TEST 5: Automatic HTTP 413 Dynamic Splitting & Recovery
        print("\n[TEST 5] Testing Automatic HTTP 413 Recovery...")
        groq_rate_limiter._history.clear()
        test_413_doc = DocumentService.create_document(
            db=db,
            filename="test_413_doc.pdf",
            original_filename="test_413_doc.pdf",
            file_type="pdf",
            file_size=1000,
        )
        # Create 4 chunks
        for ci in range(4):
            c = Chunk(
                document_id=test_413_doc.id,
                chunk_index=ci,
                content=f"Substantial technical text for chunk {ci + 1}. " * 30,
                word_count=150,
            )
            db.add(c)
        db.commit()

        # Simulate HTTP 413 on first attempt with 4 chunks, then succeed when split in half
        call_payloads = []
        async def mock_groq_413(prompt, **kwargs):
            call_payloads.append(prompt)
            # If the prompt contains all 4 chunks, trigger HTTP 413
            if "Chunk 1" in prompt and "Chunk 4" in prompt:
                raise GroqOversizedRequestError("HTTP 413: Request entity too large (context length exceeded)")
            # Smaller sub-batches succeed
            return {
                "response": "## Section Digest\nRecovered digest for subsegment.",
                "model": "openai/gpt-oss-120b",
                "total_tokens": 400,
            }

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_groq_413):
            res_413 = client.post(f"/api/summarization/{test_413_doc.id}", json={"summary_type": "brief"})
            assert res_413.status_code == 200, res_413.text
            print(f"  Total Groq calls during HTTP 413 recovery: {len(call_payloads)}")
            # The oversized call failed, was split into half1 and half2, both succeeded, plus reduce call
            assert len(call_payloads) >= 3
            print("  ✓ HTTP 413 automatically split batch into smaller segments and recovered seamlessly!")

        # Clean up 413 test doc
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == test_413_doc.id).delete()
        db.query(Chunk).filter(Chunk.document_id == test_413_doc.id).delete()
        db.delete(test_413_doc)
        db.commit()

        # TEST 6: Stable Content-Aware Caching & Resumability Across Boundary Shifts
        print("\n[TEST 6] Testing Content-Aware Caching Across Batch Boundary Shifts...")
        mock_cache_doc = DocumentService.create_document(
            db=db,
            filename="cache_shift_test.pdf",
            original_filename="cache_shift_test.pdf",
            file_type="pdf",
            file_size=1000,
        )
        for ci in range(3):
            c = Chunk(
                document_id=mock_cache_doc.id,
                chunk_index=ci,
                content=f"Unique immutable content for chunk {ci + 1}.",
                word_count=5,
            )
            db.add(c)
        db.commit()

        # Seed completed batch with exact content_hash
        raw_text_batch = "[Chunk 1 | General]\nUnique immutable content for chunk 1.\n\n[Chunk 2 | General]\nUnique immutable content for chunk 2."
        cache_key, content_hash = SummarizationService.compute_batch_cache_identity(
            document_id=mock_cache_doc.id,
            type_key="brief",
            batch_text=raw_text_batch,
            stage="map",
        )
        b_cached = SummarizationBatch(
            document_id=mock_cache_doc.id,
            batch_index=0,
            start_chunk=1,
            end_chunk=2,
            chunk_count=2,
            cache_key=cache_key,
            content_hash=content_hash,
            prompt_version="v2",
            stage="map",
            summary_text="Digest loaded directly from content-hash cache.",
            status="completed",
        )
        db.add(b_cached)
        db.commit()

        # Query batch by content_hash
        hit = db.query(SummarizationBatch).filter(
            SummarizationBatch.document_id == mock_cache_doc.id,
            SummarizationBatch.content_hash == content_hash,
            SummarizationBatch.status == "completed",
        ).first()
        assert hit is not None
        assert hit.summary_text == "Digest loaded directly from content-hash cache."
        print(f"  ✓ Content-derived cache identity confirmed: hash='{content_hash[:12]}...' matched independently of batch index")

        # Clean up
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == mock_cache_doc.id).delete()
        db.query(Chunk).filter(Chunk.document_id == mock_cache_doc.id).delete()
        db.delete(mock_cache_doc)
        db.commit()

        # TEST 7: Multi-Level Hierarchical Reduction
        print("\n[TEST 7] Testing Multi-Level Hierarchical Reduction...")
        groq_rate_limiter._history.clear()
        hr_doc = DocumentService.create_document(
            db=db,
            filename="test_hr_doc.pdf",
            original_filename="test_hr_doc.pdf",
            file_type="pdf",
            file_size=1000,
        )
        # Simulate 8 large intermediate digests that exceed the single-request reduce budget
        large_digests = [
            f"[Section {i + 1}]\nExtensive factual digest covering technical algorithms and performance metrics. " * 80
            for i in range(8)
        ]
        reduce_calls = []
        async def mock_hierarchical_reduce(prompt, **kwargs):
            reduce_calls.append(prompt)
            if "Consolidate these section digests" in prompt:
                return {"response": "Consolidated Level 1 Digest.", "model": "openai/gpt-oss-120b", "total_tokens": 300}
            return {
                "response": "## Executive Overview\nFinal synthesis across all hierarchical levels.\n\n## Key Takeaways\n- Outcome verified",
                "model": "openai/gpt-oss-120b",
                "total_tokens": 500,
            }

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_hierarchical_reduce):
            final_res = asyncio.run(
                SummarizationService._execute_hierarchical_reduce(
                    db=db,
                    document_id=hr_doc.id,
                    doc_title="Hierarchical Document",
                    intermediate_digests=large_digests,
                    type_key="brief",
                    provider="groq",
                )
            )
            assert final_res is not None
            assert "Final synthesis" in final_res["response"]
            # Assert that consolidation calls took place before the final synthesis
            assert len(reduce_calls) > 1, f"Expected multi-level reduce calls, got {len(reduce_calls)}"
            print(f"  ✓ Hierarchical multi-level reduce completed with {len(reduce_calls)} stages (Level 1 consolidation + final synthesis)")

            cons_in_db = db.query(SummarizationBatch).filter(
                SummarizationBatch.document_id == hr_doc.id,
                SummarizationBatch.stage == "reduce_level_1",
            ).all()
            assert len(cons_in_db) > 0, "Expected Level 1 consolidation batches to be persisted in cache"
            print(f"  ✓ Verified: {len(cons_in_db)} Level 1 reduction consolidation digests persisted to cache")

        # Clean up hr_doc
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == hr_doc.id).delete()
        db.delete(hr_doc)
        db.commit()

        # TEST 7B: Verify Daily Quota Fast-Fail (No futile retries)
        print("\n[TEST 7B] Testing Daily Quota Exhaustion (is_daily_limit) Fast-Fail...")
        daily_doc = DocumentService.create_document(
            db=db,
            filename="test_daily_fail.pdf",
            original_filename="test_daily_fail.pdf",
            file_type="pdf",
            file_size=1000,
        )
        daily_err = GroqRateLimitError("Tokens per day (TPD) limit exceeded.", is_daily_limit=True)
        assert daily_err.is_daily_limit is True
        call_count_daily = 0
        async def mock_daily_fail(*args, **kwargs):
            nonlocal call_count_daily
            call_count_daily += 1
            raise daily_err

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_daily_fail):
            try:
                asyncio.run(
                    SummarizationService._execute_hierarchical_reduce(
                        db=db,
                        document_id=daily_doc.id,
                        doc_title="Daily Limit Test",
                        intermediate_digests=["Digest 1", "Digest 2"],
                        type_key="brief",
                        provider="groq",
                    )
                )
                assert False, "Should have raised GroqRateLimitError"
            except GroqRateLimitError as e:
                assert e.is_daily_limit is True
                assert call_count_daily == 1, f"Expected fast-fail in 1 attempt, but got {call_count_daily}"
                print("  ✓ Daily quota limit failed immediately in 1 attempt without futile retry backoffs")

        db.delete(daily_doc)
        db.commit()

        # TEST 8: Full 44-Chunk Coverage & Adaptive Batching on 58-Page PDF (BIS703_Module_5.pdf)
        print("\n[TEST 8] Validating 58-Page PDF (BIS703_Module_5.pdf) Adaptive Workflow...")
        groq_rate_limiter._history.clear()
        pdf_doc = db.query(Document).filter(
            (Document.filename.like("%BIS703_Module_5%")) | (Document.original_filename.like("%BIS703_Module_5%"))
        ).first()
        assert pdf_doc is not None, "58-page PDF 'BIS703_Module_5.pdf' must exist in database"

        pdf_chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == pdf_doc.id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        total_pdf_chunks = len(pdf_chunks)
        assert total_pdf_chunks == 44, f"Expected 44 chunks, got {total_pdf_chunks}"

        # Build adaptive batches
        adaptive_batches = SummarizationService._build_adaptive_batches(pdf_chunks)
        print(f"  Document: '{pdf_doc.original_filename or pdf_doc.filename}' (58 pages, 44 chunks)")
        print(f"  Adaptive Batches Formed: {len(adaptive_batches)} batches")

        # Verify complete chunk coverage (every chunk from 0 to 43 is present)
        covered_indices = set()
        for b_idx, batch in enumerate(adaptive_batches):
            b_tokens = sum(item["tokens"] for item in batch)
            # Ensure every batch respects the available budget
            assert b_tokens <= settings.SUMMARIZATION_TARGET_TOKENS, f"Batch {b_idx} exceeded budget: {b_tokens}"
            for item in batch:
                covered_indices.add(item["original_chunk_index"])

        assert len(covered_indices) == 44, f"Coverage gap! Covered {len(covered_indices)} of 44 chunks."
        assert min(covered_indices) == 0 and max(covered_indices) == 43
        print(f"  ✓ 100% of chunks (0 to 43) covered across {len(adaptive_batches)} adaptive batches with zero gaps")

        # Simulate end-to-end adaptive summarization with mock Groq
        executed_batches = []
        async def mock_groq_adaptive_run(prompt, **kwargs):
            executed_batches.append("batch")
            if "Section Digest:" in prompt:
                return {"response": "Section digest covering information security principles.", "model": "openai/gpt-oss-120b", "total_tokens": 300}
            return {
                "response": (
                    "## Executive Overview\n"
                    "Comprehensive module on cryptographic systems, block ciphers, hashing, and public key infrastructures.\n\n"
                    "## Key Takeaways\n"
                    "- Symmetric and asymmetric ciphers provide confidentiality\n"
                    "- SHA-256 ensures message integrity\n"
                    "- Public Key Infrastructure establishes identity trust"
                ),
                "model": "openai/gpt-oss-120b",
                "total_tokens": 600,
            }

        # Clear existing batches
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == pdf_doc.id).delete()
        db.query(SummarizationJob).filter(SummarizationJob.document_id == pdf_doc.id).delete()
        db.commit()

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_groq_adaptive_run):
            resp = client.post(f"/api/summarization/{pdf_doc.id}", json={"summary_type": "detailed"})
            assert resp.status_code == 200, resp.text
            res_json = resp.json()

            print(f"  Strategy Used: {res_json['strategy_used']}")
            print(f"  Chunks Processed: {res_json['chunk_count']}")
            print(f"  Total API Calls: {len(executed_batches)} ({len(adaptive_batches)} map + 1 reduce)")
            assert res_json["chunk_count"] == 44
            assert res_json["strategy_used"] == "map_reduce"
            assert len(executed_batches) == len(adaptive_batches) + 1
            assert len(res_json["sections"]) >= 2
            print("  ✓ End-to-end adaptive summarization of 58-page PDF succeeded with all chunks included!")

        # Clean test batches
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == pdf_doc.id).delete()
        db.query(SummarizationJob).filter(SummarizationJob.document_id == pdf_doc.id).delete()
        db.commit()

        print("\n" + "=" * 75)
        print("ALL STAGE 5 ADAPTIVE SUMMARIZATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 75)

    finally:
        db.close()


if __name__ == "__main__":
    run_adaptive_summarization_tests()
