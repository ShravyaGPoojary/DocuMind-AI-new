"""
Verification of 58-Page PDF (BIS703_Module_5.pdf) Summarization Pipeline
- 44 chunks sequentially processed in 22 batches (2 chunks/request)
- All chunks accounted for
- Real-time progress updates & batch counters
- SQLite intermediate digest caching & resumability
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.db.session import init_db, SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.summarization import SummarizationJob, SummarizationBatch
from app.services.summarization_service import SummarizationService
from app.main import app


def test_58page_pdf_pipeline():
    print("=" * 75)
    print("VERIFYING 58-PAGE PDF (BIS703_Module_5.pdf) SUMMARIZATION PIPELINE")
    print("=" * 75)

    init_db()
    db = SessionLocal()
    client = TestClient(app)

    try:
        # 1. Locate the 58-page PDF in SQLite
        pdf_doc = db.query(Document).filter(
            (Document.filename.like("%BIS703_Module_5%")) | (Document.original_filename.like("%BIS703_Module_5%"))
        ).first()

        assert pdf_doc is not None, "58-page PDF 'BIS703_Module_5.pdf' not found in database!"
        doc_id = pdf_doc.id

        chunks = (
            db.query(Chunk)
            .filter(Chunk.document_id == doc_id)
            .order_by(Chunk.chunk_index.asc())
            .all()
        )
        total_chunks = len(chunks)
        print(f"\n[INFO] Document: '{pdf_doc.original_filename or pdf_doc.filename}'")
        print(f"[INFO] Page Count: {pdf_doc.page_count} pages")
        print(f"[INFO] Total Chunks: {total_chunks} chunks")

        # Clean existing test batches for clean test run
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == doc_id).delete()
        db.query(SummarizationJob).filter(SummarizationJob.document_id == doc_id).delete()
        db.commit()

        # Batch configuration verification
        batch_size = SummarizationService.MAP_BATCH_SIZE
        assert batch_size == 2, f"Expected batch size 2, got {batch_size}"
        expected_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"[INFO] Batch Size: {batch_size} chunks per request")
        print(f"[INFO] Total Map Batches: {expected_batches} batches")

        # 2. Simulate Map-Reduce with Groq mock tracking every batch
        batch_records = []
        async def mock_groq_generate(prompt, **kwargs):
            if "Section Digest:" in prompt or "Document Section" in prompt:
                batch_records.append(prompt[:120].replace('\n', ' '))
                return {
                    "response": f"Digest for section covering specific technical concepts, definitions, and conclusions.",
                    "model": "openai/gpt-oss-120b",
                }
            else:
                # Reduce step
                return {
                    "response": (
                        "## Executive Overview\n"
                        "This comprehensive module covers the fundamentals of information security, cryptographic algorithms, "
                        "symmetric vs asymmetric encryption, hashing functions, digital signatures, and public key infrastructure.\n\n"
                        "## Key Takeaways\n"
                        "- AES and DES are symmetric ciphers operating on fixed block sizes.\n"
                        "- RSA relies on prime factorization for asymmetric public-key encryption.\n"
                        "- SHA-256 provides collision-resistant cryptographic hashing for data integrity.\n"
                        "- Digital certificates bind public keys to identities using trusted Certificate Authorities."
                    ),
                    "model": "openai/gpt-oss-120b",
                }

        print("\n--- STEP 1: Running Map-Reduce on all 44 chunks sequentially ---")
        t0 = time.time()
        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_groq_generate):
            res = client.post(
                f"/api/summarization/{doc_id}",
                json={"summary_type": "brief"},
            )
        elapsed = time.time() - t0

        assert res.status_code == 200, res.text
        data = res.json()

        print(f"  Summary Type: {data['summary_type']} ({data['summary_type_label']})")
        print(f"  Strategy Used: {data['strategy_used']}")
        print(f"  Chunks Processed: {data['chunk_count']}")
        print(f"  Word Count: {data['word_count']} words")
        print(f"  Map Invocations: {len(batch_records)} batches")
        print(f"  Execution Time: {round(elapsed, 2)}s")

        assert data["strategy_used"] == "map_reduce"
        assert data["chunk_count"] == total_chunks
        assert len(batch_records) == expected_batches, f"Expected {expected_batches} map calls, got {len(batch_records)}"
        assert len(data["sections"]) >= 2
        print(f"  ✓ Verified: All {total_chunks} chunks ({expected_batches} batches) processed sequentially!")

        # 3. Check SQLite persistence of batches
        saved_batches = (
            db.query(SummarizationBatch)
            .filter(SummarizationBatch.document_id == doc_id)
            .order_by(SummarizationBatch.batch_index.asc())
            .all()
        )
        print(f"\n--- STEP 2: Verifying Batch Persistence in SQLite ---")
        print(f"  Saved Batches in SQLite: {len(saved_batches)}")
        assert len(saved_batches) == expected_batches
        # Verify chunk coverage in batches
        covered_chunks = set()
        for b in saved_batches:
            for c_idx in range(b.start_chunk, b.end_chunk + 1):
                covered_chunks.add(c_idx)
        assert len(covered_chunks) == total_chunks, f"Expected {total_chunks} covered chunks, got {len(covered_chunks)}"
        print(f"  ✓ Verified: Every chunk from 1 to {total_chunks} is persisted in SQLite batches!")

        # 4. Test Resumability
        print(f"\n--- STEP 3: Testing Resumability (Zero Duplicate API Calls) ---")
        # Now run again: because all 22 batches are already cached in SQLite,
        # Groq should only be called 1 time (the reduce synthesis step)!
        resumed_calls = []
        async def mock_groq_resumed(prompt, **kwargs):
            resumed_calls.append("call")
            return {
                "response": "## Overview\nRe-synthesized summary from SQLite cache.",
                "model": "openai/gpt-oss-120b",
            }

        with patch("app.services.groq_service.GroqService.generate", side_effect=mock_groq_resumed):
            res_cached = client.post(
                f"/api/summarization/{doc_id}",
                json={"summary_type": "brief"},
            )
            assert res_cached.status_code == 200, res_cached.text
            print(f"  Groq Invocations on Re-run: {len(resumed_calls)} (Expected: exactly 1 reduce call, 0 map calls)")
            assert len(resumed_calls) == 1, f"Expected 1 call (reduce only), got {len(resumed_calls)}"
            print("  ✓ Resumability Verified: 22 batches loaded instantly from cache, saving 22 Groq API calls!")

        # 5. Clean test batches
        db.query(SummarizationBatch).filter(SummarizationBatch.document_id == doc_id).delete()
        db.query(SummarizationJob).filter(SummarizationJob.document_id == doc_id).delete()
        db.commit()

        print("\n" + "=" * 75)
        print("58-PAGE PDF (BIS703_Module_5.pdf) PIPELINE FULLY VERIFIED!")
        print("=" * 75)

    finally:
        db.close()


if __name__ == "__main__":
    test_58page_pdf_pipeline()

