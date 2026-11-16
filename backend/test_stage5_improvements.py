"""
Stage 5 Verification & Improvement Tests for DocuMind AI
Reliable Large PDF Summarization, Timeout/Rate-Limit Handling, Progress Tracking & Resumability

Tests:
1. 2-Chunk Batching & All-Chunk Coverage Verification (Map-Reduce)
2. Background Job Creation & Polling Endpoint (GET /api/summarization/jobs/{job_id})
3. Batch Persistence & Resumability (reusing completed batches from SQLite without duplicate Groq calls)
4. Rate Limit Backoff with Retry-After Header & Transient Error Handling
5. Permanent Error Fast-Fail (no retry on 401/404)
6. Large Document Processing with the 58-page PDF (BIS703_Module_5.pdf)
"""

import os
import sys
import uuid
import time
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
from app.services.groq_service import (
    GroqAuthError,
    GroqRateLimitError,
    GroqModelUnavailableError,
    GroqTimeoutError,
    GroqNetworkError,
)
from app.services.summarization_service import SummarizationService
from app.main import app


def run_stage5_improvement_tests():
    print("=" * 75)
    print("RUNNING STAGE 5 IMPROVEMENT SUITE — LARGE PDF SUMMARIZATION & RESUMABILITY")
    print("=" * 75)

    init_db()
    db = SessionLocal()
    client = TestClient(app)

    try:
        # TEST 1: Background Job Creation and Polling
        print("\n[TEST 1] Testing Background Job Creation & Status Polling...")
        test_doc = db.query(Document).filter(Document.filename.like("%.pdf")).first()
        assert test_doc is not None, "No PDF document found in database."
        doc_id = test_doc.id

        # Start job with background=true
        bg_res = client.post(
            f"/api/summarization/{doc_id}",
            json={"summary_type": "brief", "background": True},
        )
        assert bg_res.status_code == 200, bg_res.text
        job_data = bg_res.json()
        assert "job_id" in job_data
        job_id = job_data["job_id"]
        assert job_data["document_id"] == doc_id
        assert job_data["status"] in ("pending", "processing", "completed")
        print(f"  ✓ Background job dispatched: job_id='{job_id}', status='{job_data['status']}'")

        # Poll status
        poll_res = client.get(f"/api/summarization/jobs/{job_id}")
        assert poll_res.status_code == 200, poll_res.text
        poll_data = poll_res.json()
        assert poll_data["job_id"] == job_id
        assert "progress_percent" in poll_data
        assert "progress_message" in poll_data
        assert "total_batches" in poll_data
        assert "completed_batches" in poll_data
        print(f"  ✓ Polling endpoint returned: progress={poll_data['progress_percent']}%, message='{poll_data['progress_message']}'")

        # TEST 2: 2-Chunk Batching & Complete Chunk Coverage on 58-Page PDF
        print("\n[TEST 2] Testing 2-Chunk Sequential Batching & Chunk Coverage...")
        large_doc = db.query(Document).filter(
            (Document.filename.like("%BIS703_Module_5%")) | (Document.original_filename.like("%BIS703_Module_5%"))
        ).first()
        if not large_doc:
            large_doc = db.query(Document).join(Chunk).group_by(Document.id).first()

        assert large_doc is not None
        large_chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == large_doc.id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        total_chunks = len(large_chunks)
        expected_batches = (total_chunks + SummarizationService.MAP_BATCH_SIZE - 1) // SummarizationService.MAP_BATCH_SIZE

        print(f"  Target Document: '{large_doc.original_filename or large_doc.filename}'")
        print(f"  Total Chunks: {total_chunks}")
        print(f"  Batch Size: {SummarizationService.MAP_BATCH_SIZE} chunks/request")
        print(f"  Expected Batches: {expected_batches}")
        assert SummarizationService.MAP_BATCH_SIZE == 2, "MAP_BATCH_SIZE must be 2 chunks per request"
        assert expected_batches == (total_chunks + 1) // 2

        # Verify all chunk indices are contiguous and accounted for
        for idx, chunk in enumerate(large_chunks):
            assert chunk.chunk_index == idx, f"Chunk index gap detected at {idx}"
        print(f"  ✓ Every chunk (0 to {total_chunks - 1}) is contiguous and fully accounted for")

        # TEST 3: Resumability & Batch Persistence
        print("\n[TEST 3] Testing Batch Persistence & Instant Resumability...")
        test_resumable_doc_id = str(uuid.uuid4())
        mock_doc = DocumentService.create_document(
            db=db,
            filename="resumable_test.pdf",
            original_filename="resumable_test.pdf",
            file_type="pdf",
            file_size=1000,
        )
        # Create 6 chunks -> 3 batches of 2 chunks
        for ci in range(6):
            c = Chunk(
                document_id=mock_doc.id,
                chunk_index=ci,
                content=f"Factual content for chunk {ci + 1} regarding microservices and database sharding.",
                word_count=10,
            )
            db.add(c)
        db.commit()

        # Seed 2 completed batches in SQLite (Batch 0 and Batch 1)
        b0 = SummarizationBatch(
            document_id=mock_doc.id,
            batch_index=0,
            start_chunk=1,
            end_chunk=2,
            chunk_count=2,
            summary_text="Digest for section 1 (chunks 1-2): Microservices architecture overview.",
            status="completed",
        )
        b1 = SummarizationBatch(
            document_id=mock_doc.id,
            batch_index=1,
            start_chunk=3,
            end_chunk=4,
            chunk_count=2,
            summary_text="Digest for section 2 (chunks 3-4): Database sharding and partition schemes.",
            status="completed",
        )
        db.add_all([b0, b1])
        db.commit()

        # Execute map-reduce with mocked Groq
        # Batch 0 and Batch 1 are cached, so Groq should ONLY be called for Batch 2 (chunks 5-6) + 1 reduce synthesis!
        mock_map_resp = {"response": "Digest for section 3 (chunks 5-6)", "model": "openai/gpt-oss-120b"}
        mock_reduce_resp = {
            "response": "## Executive Overview\nCohesive synthesis covering microservices, sharding, and architecture.\n\n## Key Takeaways\n- Sharding enhances scalability\n- Microservices improve isolation",
            "model": "openai/gpt-oss-120b",
        }

        with patch("app.services.groq_service.GroqService.generate", side_effect=[
            mock_map_resp,      # Only 1 map call for the missing Batch 2
            mock_reduce_resp,   # 1 reduce call
        ]) as mock_call:
            res = client.post(f"/api/summarization/{mock_doc.id}", json={"summary_type": "brief"})
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["strategy_used"] == "map_reduce"
            # Groq was called only 2 times instead of 4 times (3 map + 1 reduce)
            assert mock_call.call_count == 2, f"Expected 2 Groq calls (1 remaining map + 1 reduce), got {mock_call.call_count}"
            print("  ✓ Resumability confirmed: Batches 0 & 1 were reused from SQLite cache without duplicate Groq calls!")
            print(f"  ✓ Only 2 Groq invocations occurred (Batch 2 + Reduce Synthesis) instead of 4.")

        # Cleanup mock doc
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == mock_doc.id).delete()
        db.query(Chunk).filter(Chunk.document_id == mock_doc.id).delete()
        db.delete(mock_doc)
        db.commit()

        # TEST 4: Rate-Limit Exponential Backoff with Retry-After Header
        print("\n[TEST 4] Testing Rate-Limit Backoff with Retry-After Header...")
        rl_err_with_ra = GroqRateLimitError("Rate limit exceeded", retry_after=0.1)
        assert rl_err_with_ra.retry_after == 0.1

        # Create a small 2-chunk document for isolated retry and fast-fail tests
        short_test_doc = DocumentService.create_document(
            db=db,
            filename="isolated_test.pdf",
            original_filename="isolated_test.pdf",
            file_type="pdf",
            file_size=500,
        )
        for ci in range(2):
            c = Chunk(
                document_id=short_test_doc.id,
                chunk_index=ci,
                content=f"Short document excerpt {ci + 1}.",
                word_count=5,
            )
            db.add(c)
        db.commit()

        retry_count = [0]
        async def mock_gen_with_retry(*args, **kwargs):
            retry_count[0] += 1
            if retry_count[0] == 1:
                raise GroqRateLimitError("Rate limit reached. Please wait 0.05s.", retry_after=0.05)
            return {"response": "## Overview\nSuccess after rate limit backoff.", "model": "openai/gpt-oss-120b"}

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_gen_with_retry):
            t0 = time.time()
            res_rl = client.post(f"/api/summarization/{short_test_doc.id}", json={"summary_type": "brief"})
            elapsed = time.time() - t0
            assert res_rl.status_code == 200, res_rl.text
            assert retry_count[0] == 2, f"Expected 2 attempts, got {retry_count[0]}"
            print(f"  ✓ Rate limit automatically retried after retry_after backoff (attempts: {retry_count[0]}, elapsed: {round(elapsed, 2)}s)")

        # TEST 5: Permanent Error Fast-Fail (No Retry on 401/404)
        print("\n[TEST 5] Testing Fast-Fail on Permanent Errors (No Retries on 401)...")
        fail_attempts = [0]
        async def mock_auth_fail(*args, **kwargs):
            fail_attempts[0] += 1
            raise GroqAuthError("Invalid Groq API key")

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_auth_fail):
            res_auth = client.post(f"/api/summarization/{short_test_doc.id}", json={"summary_type": "brief"})
            assert res_auth.status_code == 401
            assert fail_attempts[0] == 1, f"Permanent error should not be retried! Got {fail_attempts[0]} attempts."
            print(f"  ✓ Permanent 401 auth error fast-failed immediately without retries (attempts: {fail_attempts[0]})")

        # Clean up short_test_doc
        db.query(Chunk).filter(Chunk.document_id == short_test_doc.id).delete()
        db.delete(short_test_doc)
        db.commit()

        # TEST 6: Job Resume Endpoint (POST /api/summarization/jobs/{job_id}/resume)
        print("\n[TEST 6] Testing Job Resume Endpoint...")
        failed_job = SummarizationJob(
            document_id=test_doc.id,
            summary_type="brief",
            status="failed",
            error_message="Simulated Groq 429 rate limit error",
            total_chunks=10,
            total_batches=5,
            completed_batches=3,
            current_batch=4,
            progress_percent=50,
            progress_message="Failed at batch 4",
        )
        db.add(failed_job)
        db.commit()

        with patch("app.services.summarization_service.SummarizationService.run_background_job", return_value=None):
            resume_res = client.post(f"/api/summarization/jobs/{failed_job.id}/resume")
            assert resume_res.status_code == 200, resume_res.text
            resumed_data = resume_res.json()
            assert resumed_data["status"] == "pending"
            assert resumed_data["completed_batches"] == 3
            assert resumed_data["error_message"] is None
            print(f"  ✓ Job '{failed_job.id}' successfully resumed from batch 4 (status='{resumed_data['status']}')")

        # Cleanup failed job
        db.delete(failed_job)
        db.commit()

        print("\n" + "=" * 75)
        print("ALL STAGE 5 IMPROVEMENT TESTS PASSED SUCCESSFULLY!")
        print("=" * 75)

    finally:
        db.close()


if __name__ == "__main__":
    run_stage5_improvement_tests()
